"""
M3 Data Loading — MIMIC-IV Waveform Database (mimic4wdb 0.1.0)
================================================================
Loads multi-channel physiological waveforms directly from the locally
downloaded mimic4wdb tree and windows them for the LSTM-AE anomaly detector
(roadmap M3 core: "Working LSTM-AE on MIMIC-IV Waveforms").

Channel choice: II (single ECG lead), Pleth (PPG/SpO2), Resp (respiration).
Chosen because this is the largest channel combination realistically
matching a portable ambulance monitor (not multi-lead diagnostic ECG, not
invasive lines like ABP/CVP/PAP which are ICU-only) — a survey of the first
usable segment across the 81 fully-downloaded patients found
Resp/II/Pleth present in 80/76/68 of 83 surveyed records respectively,
clearly the broadest common subset.

Sampling frequency is a fixed 62.4725 Hz constant across the whole dataset
(confirmed across all 81 complete patients' master headers), so no
per-record resampling is needed.
"""
import os
import numpy as np
import wfdb

FS = 62.4725
CHANNELS = ["II", "Pleth", "Resp"]

DATASET_ROOT = os.path.join(
    "module 3", "mimic4wdb", "physionet.org", "files", "mimic4wdb", "0.1.0"
)
WAVES_DIR = os.path.join(DATASET_ROOT, "waves")
MANIFEST = os.path.join(DATASET_ROOT, "SHA256SUMS.txt")


def list_complete_patients(project_root):
    """Return patient IDs (e.g. 'p10014354') whose full file set (per the
    dataset's own SHA256SUMS.txt manifest) is present on disk. mimic4wdb is
    being downloaded breadth-first across all 198 patients, so directory
    presence alone does not mean a patient's data is complete."""
    manifest_path = os.path.join(project_root, MANIFEST)
    dataset_root = os.path.join(project_root, DATASET_ROOT)

    files_by_patient = {}
    with open(manifest_path, encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split(None, 1)
            if len(parts) != 2:
                continue
            _, relpath = parts
            if relpath.startswith("waves/"):
                segs = relpath.split("/")
                if len(segs) >= 3:
                    files_by_patient.setdefault(segs[2], []).append(relpath)

    complete = []
    for patient, files in files_by_patient.items():
        if all(os.path.exists(os.path.join(dataset_root, f)) for f in files):
            complete.append(patient)
    return sorted(complete)


def _read_master_segments(record_dir, record_name):
    """Parse the multi-segment master header directly (fast path — avoids
    opening every segment with wfdb just to check its length)."""
    master_hea = os.path.join(record_dir, record_name + ".hea")
    if not os.path.exists(master_hea):
        return []
    with open(master_hea) as f:
        lines = f.readlines()
    start = 1 if lines[0].startswith("#wfdb") else 0
    segments = []
    for line in lines[start + 1:]:
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) != 2:
            continue
        seg_name, seg_len = parts
        seg_len = int(seg_len)
        if seg_len > 0:
            segments.append((seg_name, seg_len))
    return segments


def _segment_channels(record_dir, seg_name):
    seg_hea = os.path.join(record_dir, seg_name + ".hea")
    if not os.path.exists(seg_hea):
        return []
    with open(seg_hea) as f:
        lines = f.readlines()
    chans = []
    for line in lines[1:]:
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        toks = line.split()
        if len(toks) >= 9:
            chans.append(toks[-1])
    return chans


def iter_usable_segments(project_root, patient, channels=CHANNELS, min_len=None):
    """Yield (segment_path_without_ext, seg_len) for every segment of every
    record belonging to `patient` that contains all of `channels`."""
    dataset_root = os.path.join(project_root, DATASET_ROOT)
    group = patient[:4]
    patient_dir = os.path.join(dataset_root, "waves", group, patient)
    if not os.path.isdir(patient_dir):
        return
    for record_name in os.listdir(patient_dir):
        record_dir = os.path.join(patient_dir, record_name)
        if not os.path.isdir(record_dir):
            continue
        for seg_name, seg_len in _read_master_segments(record_dir, record_name):
            if min_len is not None and seg_len < min_len:
                continue
            chans = _segment_channels(record_dir, seg_name)
            if all(c in chans for c in channels):
                yield os.path.join(record_dir, seg_name), seg_len


def _windows_from_signal(sig, window_len, stride=None):
    """Non-overlapping windows by default. Drops any window containing NaN
    (real sensor dropout — MIMIC-IV-WDB encodes invalid samples as NaN)."""
    stride = stride or window_len
    n = sig.shape[0]
    out = []
    for start in range(0, n - window_len + 1, stride):
        w = sig[start:start + window_len]
        if not np.isnan(w).any():
            out.append(w)
    return out


def assign_ambulance_clients(patients, k=10, seed=42):
    """Deterministic patient -> simulated-ambulance-client partition, per
    M3_DP_Threat_Model_Gate.md §5: K disjoint clients, a patient's data lives
    on exactly one client. Returns a dict {client_id (0..k-1): [patient_ids]}.
    Same seed as the rest of the M3 pipeline (see centralized_baseline.py)."""
    rng = np.random.default_rng(seed)
    shuffled = list(rng.permutation(sorted(patients)))
    clients = {i: [] for i in range(k)}
    for idx, patient in enumerate(shuffled):
        clients[idx % k].append(patient)
    return clients


def build_dataset(project_root, patients, channels=CHANNELS, window_sec=30,
                   stride_sec=None, max_windows_per_patient=200):
    """Build a windowed multivariate waveform dataset across `patients`.

    Normalization is per-patient, per-channel (z-score using that patient's
    own available signal), not per-window — so that a window-level amplitude
    anomaly (e.g. a flatline or spike) stays visible relative to that
    patient's own normal range instead of being washed out by per-window
    rescaling.

    Returns (X, patient_ids): X is float32 array [N, window_len, n_channels],
    patient_ids is a list of length N (which patient each window came from —
    needed later to assign windows to simulated per-ambulance clients).
    """
    window_len = int(round(window_sec * FS))
    stride = int(round(stride_sec * FS)) if stride_sec else window_len

    all_X, all_ids = [], []
    for patient in patients:
        patient_windows = []
        for seg_path, seg_len in iter_usable_segments(project_root, patient, channels, min_len=window_len):
            try:
                rec = wfdb.rdrecord(seg_path, channel_names=channels)
            except Exception:
                continue
            sig = rec.p_signal.astype(np.float32)
            patient_windows.extend(_windows_from_signal(sig, window_len, stride))
            if len(patient_windows) >= max_windows_per_patient:
                break
        if not patient_windows:
            continue
        arr = np.stack(patient_windows[:max_windows_per_patient])
        mean = arr.reshape(-1, arr.shape[-1]).mean(axis=0)
        std = arr.reshape(-1, arr.shape[-1]).std(axis=0)
        std[std < 1e-6] = 1e-6
        arr = (arr - mean) / std
        all_X.append(arr)
        all_ids.extend([patient] * arr.shape[0])

    if not all_X:
        return np.empty((0, window_len, len(channels)), dtype=np.float32), []
    X = np.concatenate(all_X, axis=0)
    return X, all_ids
