"""
M3 — DP-SGD model-level collapse diagnostic (per supervisor approval memo v2 §condition
on the retraction: "document the specific change... with the same before/after evidence
standard used for every other fix").

Loads the actual trained checkpoints already backing every F1 number in the primary
contribution report and measures reconstruction std vs. input std directly -- no new
training, a pure read-only diagnostic. Confirmed finding (see M3_Primary_Contribution_Report.pdf
Section 3.1): the LSTM-AE collapses to near-constant output under DP-SGD at every tested K,
and this has never been resolved by any evidenced code change (module 3's git history has
exactly one commit touching dp_federated_client.py, already containing the current
batch-size-cap formula).

Run standalone: `python measure_collapse.py`. Saves results/m3_collapse_diagnostic.json.
"""
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data import list_complete_patients, build_dataset
from lstm_autoencoder import LSTMAutoencoder

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")
SEED = 42

CHECKPOINTS = {
    "Non-DP Federated K=10": r"C:/tmp/nvflare/simulation/m3-non-dp-federated/server/simulate_job/app_server/FL_global_model.pt",
    "DP Federated K=10": r"C:/tmp/nvflare/simulation/m3-dp-federated-k10/server/simulate_job/app_server/FL_global_model.pt",
    "DP Federated K=5": r"C:/tmp/nvflare/simulation/m3-dp-federated-k5/server/simulate_job/app_server/FL_global_model.pt",
    "DP Federated K=3": r"C:/tmp/nvflare/simulation/m3-dp-federated-k3/server/simulate_job/app_server/FL_global_model.pt",
    "DP Federated K=1": r"C:/tmp/nvflare/simulation/m3-dp-federated-k1/server/simulate_job/app_server/FL_global_model.pt",
}
COLLAPSE_THRESHOLD = 0.1  # ratio below which we call a regime "collapsed" -- healthy regimes measured ~0.95-1.00, collapsed ones ~0.002-0.01, no real case falls near this boundary


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    rng = np.random.default_rng(SEED)
    complete = list_complete_patients(PROJECT_ROOT)
    patients = list(rng.permutation(complete))
    n = len(patients)
    n_train = int(n * 0.70)
    n_val = int(n * 0.15)
    val_patients = patients[n_train:n_train + n_val]
    X_val, _ = build_dataset(PROJECT_ROOT, val_patients, window_sec=10, max_windows_per_patient=80)
    X = torch.as_tensor(X_val, dtype=torch.float32)
    input_std = X_val.std(axis=(0, 1))
    print("input_std =", input_std)

    results = {"seed": SEED, "input_std_per_channel": [round(float(v), 6) for v in input_std], "regimes": {}}
    for name, path in CHECKPOINTS.items():
        if not os.path.exists(path):
            print(f"{name}: checkpoint not found at {path}, skipping")
            continue
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        model = LSTMAutoencoder(n_channels=X_val.shape[-1], hidden_size=32, latent_size=16)
        model.load_state_dict(ckpt["model"])
        model.eval()
        with torch.no_grad():
            recon = model(X)
        recon_std = recon.numpy().std(axis=(0, 1))
        ratio = recon_std / input_std
        collapsed = bool(ratio.max() < COLLAPSE_THRESHOLD)
        print(f"{name:25s} recon_std={recon_std}  ratio={ratio}  collapsed={collapsed}")
        results["regimes"][name] = {
            "checkpoint_path": path,
            "recon_std_per_channel": [round(float(v), 6) for v in recon_std],
            "ratio_per_channel": [round(float(v), 6) for v in ratio],
            "collapsed": collapsed,
        }

    out_path = os.path.join(RESULTS_DIR, "m3_collapse_diagnostic.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
