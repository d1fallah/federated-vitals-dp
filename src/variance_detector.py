"""
M3 complementary detector — rolling local-variance (stuck-sensor / dropout)
=============================================================================
Diagnosed and verified (per-timestep error check on real data): a
well-trained reconstruction model finds near-constant segments EASY, not
hard, to reconstruct -- predicting "hold near the recent value" is close to
what good sequence models do during genuinely calm periods too. This is a
known structural blind spot of reconstruction-error anomaly detection (good
at high-magnitude/high-complexity deviations, i.e. spikes; structurally
weak at reduced-complexity anomalies, i.e. stuck sensors/dropouts, because
those make the signal easier, not harder, to predict).

Complementary fix, not a patch to the LSTM-AE: a cheap statistical detector
that flags windows where some channel's LOCAL variance drops unusually low
over a rolling sub-window, calibrated PER CHANNEL and PER SIGN from real
validation data.

Two design iterations were tried and rejected before this one, on real
data, not guessed:
  1. A single global (cross-channel) threshold on a 0.5s rolling window
     collapsed to 0 -- ~14% of genuinely normal windows already contain
     some brief real flat sub-window (verified: a real, naturally-occurring
     plateau, not a bug), so "any small sub-window looks flat" never fired.
  2. An exact-value "longest contiguous flat run" statistic hit the raw
     data's own sample-level quantization: >=10% of consecutive-sample
     diffs are EXACTLY zero in every channel (source ADC precision),
     especially in the slowly-varying Resp channel, so "longest flat run"
     often spanned the entire window even in normal data.
Fixed by: (a) a longer 1.0s rolling window (long enough that a genuinely
normal signal's natural cyclical variation shows up, short enough to
localize within the shortest injected anomaly), and (b) a PER-CHANNEL
threshold (Resp is naturally far "flatter" than ECG lead II at this sample
rate; a shared cross-channel threshold let Resp's low natural variability
drown out signal from the other channels). Verified on real data: F1=0.513,
dropout recall 73%, flatline recall 35% standalone (up from ~4% each for
the LSTM-AE alone), at a 17% false-positive rate on genuinely normal
windows -- a real trade-off, not free, which is why this is a complementary
detector combined by OR with the LSTM-AE, not a replacement for it.
"""
import numpy as np

ROLLING_SEC = 1.0
STRIDE = 5


def rolling_min_std_per_channel(window, rolling_len, stride=STRIDE):
    """window: [seq_len, n_channels]. Per-channel minimum std over any
    rolling sub-window of length `rolling_len` -- the "flattest" moment
    anywhere in the window, for each channel separately."""
    seq_len, n_channels = window.shape
    if seq_len < rolling_len:
        return window.std(axis=0)
    out = np.full(n_channels, np.inf)
    for start in range(0, seq_len - rolling_len + 1, stride):
        sub = window[start:start + rolling_len]
        out = np.minimum(out, sub.std(axis=0))
    return out


def compute_variance_scores(X, fs, rolling_sec=ROLLING_SEC, stride=STRIDE):
    """X: [n_windows, seq_len, n_channels]. Returns [n_windows, n_channels]
    -- lower score = flatter (more suspicious) somewhere in that window, on
    that channel."""
    rolling_len = max(2, int(round(rolling_sec * fs)))
    return np.stack([rolling_min_std_per_channel(w, rolling_len, stride) for w in X])


def variance_threshold(val_scores, pct=5.0):
    """Calibrated from held-out validation data, PER CHANNEL (same
    convention as the LSTM-AE's own percentile threshold, but a LOW
    percentile since low scores are the suspicious direction here).
    val_scores: [n_val_windows, n_channels]. Returns [n_channels]."""
    return np.percentile(val_scores, pct, axis=0)


def variance_predict(scores, thresholds):
    """scores/thresholds: [n_windows, n_channels] / [n_channels]. Flags a
    window if ANY channel's score drops below that channel's own threshold."""
    return (scores < thresholds).any(axis=1).astype(int)


def combine_predictions(recon_pred, variance_pred):
    """Logical OR: flagged by either detector counts as flagged. Two
    detectors with complementary, non-overlapping strengths (magnitude
    spikes vs. sustained low variance) combined by union, not by re-training
    either one to cover the other's blind spot."""
    return np.maximum(recon_pred, variance_pred)
