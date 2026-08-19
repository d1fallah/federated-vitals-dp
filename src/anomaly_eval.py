"""
M3 Anomaly Evaluation — synthetic anomaly injection (approved Option A)
=========================================================================
The LSTM-AE is unsupervised (trained on reconstruction only, no labels
needed), but the roadmap requires F1/precision/recall, which need labeled
"this window is anomalous" ground truth. This waveform-only MIMIC-IV-WDB
extract has no such labels natively. Per the supervisor's decision
(17 Aug 2026), synthetic anomaly injection is approved as the primary
evaluation method.

Three synthetic anomaly types are injected into held-out, otherwise-normal
windows (already per-patient z-score normalized by m3_data.build_dataset,
so magnitudes below are in standard-deviation units):
  - spike:    a large-amplitude burst on one channel over a short segment
  - dropout:  a channel forced to zero (the patient's own mean) over a
              segment, simulating a disconnected sensor
  - flatline: a channel held constant at its pre-segment value over a
              segment, simulating a stuck/frozen sensor reading
"""
import numpy as np


ANOMALY_KINDS = ("spike", "dropout", "flatline")


def inject_anomaly(window, kind, rng, min_frac=0.05, max_frac=0.20,
                    spike_magnitude=(4.0, 8.0)):
    """window: [window_len, n_channels], already per-patient normalized."""
    w = window.copy()
    n, c = w.shape
    dur = max(1, int(rng.uniform(min_frac, max_frac) * n))
    start = int(rng.integers(0, n - dur))
    chan = int(rng.integers(0, c))

    if kind == "spike":
        magnitude = rng.uniform(*spike_magnitude) * rng.choice([-1.0, 1.0])
        w[start:start + dur, chan] += magnitude
    elif kind == "dropout":
        w[start:start + dur, chan] = 0.0
    elif kind == "flatline":
        w[start:start + dur, chan] = w[start, chan]
    else:
        raise ValueError(f"unknown anomaly kind: {kind}")
    return w


def build_labeled_eval_set(X_normal, rng, anomaly_frac=0.5, kinds=ANOMALY_KINDS):
    """X_normal: pool of genuinely-normal windows (e.g. from held-out test
    patients, never seen during training or threshold-setting).

    Returns (X_eval, y_eval, kinds_used): y_eval=1 for injected-anomalous
    windows, 0 for untouched normal windows. kinds_used records which
    anomaly type (or 'none') was applied to each row, for breakdown metrics.
    """
    n = X_normal.shape[0]
    perm = rng.permutation(n)
    n_anom = int(n * anomaly_frac)
    anom_idx = set(perm[:n_anom].tolist())

    X_eval, y_eval, kinds_used = [], [], []
    for i in range(n):
        if i in anom_idx:
            kind = rng.choice(kinds)
            X_eval.append(inject_anomaly(X_normal[i], kind, rng))
            y_eval.append(1)
            kinds_used.append(kind)
        else:
            X_eval.append(X_normal[i])
            y_eval.append(0)
            kinds_used.append("none")
    return np.stack(X_eval), np.array(y_eval), kinds_used
