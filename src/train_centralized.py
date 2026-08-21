"""
M3 Step 1 — Centralized LSTM-AE baseline (approved 17 Aug 2026, no DP/federation)
====================================================================================
Reference performance ceiling for the roadmap's required 3-regime comparison
(centralized / federated-no-DP / federated-with-DP). Patient-level train/
val/test split (no patient's windows appear in more than one split). The
adaptive threshold is set from the held-out validation set's reconstruction
error, not the training set, to avoid an optimistic threshold. Evaluation
uses synthetic anomaly injection (m3_anomaly_eval.py, approved Option A)
on the held-out test patients.
"""
import argparse
import os
import sys
import time
import json
import numpy as np
import torch
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data import list_complete_patients, build_dataset, FS
from lstm_autoencoder import LSTMAutoencoder, train_lstm_ae, reconstruction_error, percentile_threshold
from anomaly_eval import build_labeled_eval_set
from variance_detector import compute_variance_scores, variance_threshold, variance_predict, combine_predictions

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")
MAX_WINDOWS_PER_PATIENT = 80
WINDOW_SEC = 10  # first attempt used 30s (1874 steps): training loss barely moved in
                 # 20 epochs (0.99254 -> 0.98949) -- too long a sequence for a single-
                 # layer LSTM to learn to reconstruct in that budget. 10s (~625 steps)
                 # cuts sequence length ~3x; still long enough to span multiple cardiac/
                 # respiratory cycles.


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42,
                         help="Patient split / anomaly injection seed. Default 42 reproduces the original recorded results exactly; other values (e.g. 123, 7) are for multi-seed confidence-interval reporting.")
    args = parser.parse_args()
    SEED = args.seed
    file_seed_suffix = "" if SEED == 42 else f"_seed{SEED}"

    os.makedirs(RESULTS_DIR, exist_ok=True)
    rng = np.random.default_rng(SEED)

    complete = list_complete_patients(PROJECT_ROOT)
    patients = list(rng.permutation(complete))
    n = len(patients)
    n_train = int(n * 0.70)
    n_val = int(n * 0.15)
    train_patients = patients[:n_train]
    val_patients = patients[n_train:n_train + n_val]
    test_patients = patients[n_train + n_val:]
    print(f"Patients: {n} total -> train {len(train_patients)} / val {len(val_patients)} / test {len(test_patients)}")

    t0 = time.time()
    X_train, ids_train = build_dataset(PROJECT_ROOT, train_patients, window_sec=WINDOW_SEC, max_windows_per_patient=MAX_WINDOWS_PER_PATIENT)
    X_val, ids_val = build_dataset(PROJECT_ROOT, val_patients, window_sec=WINDOW_SEC, max_windows_per_patient=MAX_WINDOWS_PER_PATIENT)
    X_test, ids_test = build_dataset(PROJECT_ROOT, test_patients, window_sec=WINDOW_SEC, max_windows_per_patient=MAX_WINDOWS_PER_PATIENT)
    print(f"Windows: train {X_train.shape[0]} / val {X_val.shape[0]} / test {X_test.shape[0]}  (built in {time.time()-t0:.1f}s)")

    if X_train.shape[0] < 20 or X_val.shape[0] < 10 or X_test.shape[0] < 10:
        print("Not enough windows in one of the splits to run a real baseline. Stopping.")
        sys.exit(1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    model = LSTMAutoencoder(n_channels=X_train.shape[-1], hidden_size=32, latent_size=16)
    print("Training centralized LSTM-AE...")
    t0 = time.time()
    history = train_lstm_ae(model, X_train, epochs=40, lr=3e-3, batch_size=32, device=str(device))
    train_time = time.time() - t0
    print(f"  trained in {train_time:.1f}s, loss[0]={history[0]:.5f} -> loss[-1]={history[-1]:.5f}")
    print(f"  full loss history: {[round(h,5) for h in history]}")

    val_err = reconstruction_error(model, X_val, device=str(device))
    threshold = percentile_threshold(val_err, pct=95.0)
    print(f"  validation reconstruction error: mean={val_err.mean():.5f} std={val_err.std():.5f}")
    print(f"  adaptive threshold (95th percentile of val error): {threshold:.5f}")

    # Complementary variance detector (see m3_variance_detector.py): catches
    # dropout/flatline anomalies the LSTM-AE structurally can't (verified:
    # a well-trained reconstruction model finds sustained near-constant
    # segments EASY, not hard, to reconstruct). Thresholds calibrated from
    # the same held-out validation set, per channel.
    val_var_scores = compute_variance_scores(X_val, FS)
    var_thresh = variance_threshold(val_var_scores, pct=5.0)
    print(f"  variance-detector per-channel thresholds (5th pct of val): {var_thresh}")

    X_eval, y_eval, kinds_used = build_labeled_eval_set(X_test, rng, anomaly_frac=0.5)
    eval_err = reconstruction_error(model, X_eval, device=str(device))
    y_pred_recon = (eval_err > threshold).astype(int)

    eval_var_scores = compute_variance_scores(X_eval, FS)
    y_pred_var = variance_predict(eval_var_scores, var_thresh)

    y_pred_combined = combine_predictions(y_pred_recon, y_pred_var)

    def score_block(name, y_pred):
        f1 = f1_score(y_eval, y_pred, zero_division=0)
        precision = precision_score(y_eval, y_pred, zero_division=0)
        recall = recall_score(y_eval, y_pred, zero_division=0)
        print(f"\n=== {name} — synthetic-anomaly evaluation (real run) ===")
        print(f"  F1={f1:.4f}  Precision={precision:.4f}  Recall={recall:.4f}")
        kinds_arr = np.array(kinds_used)
        breakdown = {}
        for kind in ("spike", "dropout", "flatline"):
            mask = kinds_arr == kind
            if mask.sum() > 0:
                breakdown[kind] = {
                    "n": int(mask.sum()),
                    "detected": int(y_pred[mask].sum()),
                    "recall": float(y_pred[mask].mean()),
                }
        print("  per-anomaly-type recall:", json.dumps(breakdown, indent=2))
        return f1, precision, recall, breakdown

    auroc = roc_auc_score(y_eval, eval_err)  # LSTM-AE's continuous score only; the combined
                                              # detector's decision is a binary OR, no single
                                              # continuous score to rank by.
    print(f"\n  LSTM-AE reconstruction-error AUROC: {auroc:.4f}")

    f1_recon, precision_recon, recall_recon, breakdown_recon = score_block("LSTM-AE only", y_pred_recon)
    f1_var, precision_var, recall_var, breakdown_var = score_block("Variance detector only", y_pred_var)
    f1_comb, precision_comb, recall_comb, breakdown_comb = score_block("LSTM-AE + variance detector (combined)", y_pred_combined)

    results = {
        "regime": "centralized",
        "seed": SEED,
        "window_sec": WINDOW_SEC,
        "threshold_method": "95th percentile of validation error",
        "architecture_note": "per-timestep-bottleneck decoder (fixed from an earlier repeat-vector "
                              "design that collapsed to near-constant output; see lstm_autoencoder.py docstring)",
        "n_patients_total": n,
        "n_patients_train": len(train_patients),
        "n_patients_val": len(val_patients),
        "n_patients_test": len(test_patients),
        "n_windows_train": int(X_train.shape[0]),
        "n_windows_val": int(X_val.shape[0]),
        "n_windows_test": int(X_test.shape[0]),
        "train_time_sec": round(train_time, 1),
        "train_loss_first_epoch": round(history[0], 5),
        "train_loss_last_epoch": round(history[-1], 5),
        "val_error_mean": round(float(val_err.mean()), 5),
        "val_error_std": round(float(val_err.std()), 5),
        "adaptive_threshold": round(threshold, 5),
        "variance_detector_thresholds_per_channel": [round(float(t), 8) for t in var_thresh],
        "eval_n_windows": len(y_eval),
        "eval_n_anomalous": int(y_eval.sum()),
        "auroc_lstm_ae_only": round(float(auroc), 4),
        "lstm_ae_only": {
            "f1": round(float(f1_recon), 4),
            "precision": round(float(precision_recon), 4),
            "recall": round(float(recall_recon), 4),
            "per_anomaly_type_recall": breakdown_recon,
        },
        "variance_detector_only": {
            "f1": round(float(f1_var), 4),
            "precision": round(float(precision_var), 4),
            "recall": round(float(recall_var), 4),
            "per_anomaly_type_recall": breakdown_var,
        },
        "combined_with_variance_detector": {
            "f1": round(float(f1_comb), 4),
            "precision": round(float(precision_comb), 4),
            "recall": round(float(recall_comb), 4),
            "per_anomaly_type_recall": breakdown_comb,
        },
    }
    out_path = os.path.join(RESULTS_DIR, f"m3_centralized_baseline{file_seed_suffix}_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
