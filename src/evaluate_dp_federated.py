"""
M3 Step 3a — evaluate the DP pilot's aggregated global model
================================================================
Same identical protocol as evaluate_federated_model.py (Step 2): same seed,
same patient-level 70/15/15 split (so the held-out test patients are the
same 13 patients used for Steps 1 and 2), same synthetic anomaly injection,
same percentile threshold. Result is PRELIMINARY / pilot-scale (per the
supervisor's guidance): only 3 of the eventual 10 clients participated, and
that model was trained on the current 81-patient pool, not the full
198-patient cohort -- explicitly not the Step 3b full-scale result.
"""
import os
import sys
import json
import numpy as np
import torch
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data import list_complete_patients, build_dataset, FS
from lstm_autoencoder import LSTMAutoencoder, reconstruction_error, percentile_threshold
from anomaly_eval import build_labeled_eval_set
from variance_detector import compute_variance_scores, variance_threshold, variance_predict, combine_predictions

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "Results")
SEED = 42
MAX_WINDOWS_PER_PATIENT = 80
WINDOW_SEC = 10
CHECKPOINT = r"C:/tmp/nvflare/simulation/m3-dp-pilot/server/simulate_job/app_server/FL_global_model.pt"


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    rng = np.random.default_rng(SEED)

    complete = list_complete_patients(PROJECT_ROOT)
    patients = list(rng.permutation(complete))
    n = len(patients)
    n_train = int(n * 0.70)
    n_val = int(n * 0.15)
    val_patients = patients[n_train:n_train + n_val]
    test_patients = patients[n_train + n_val:]
    print(f"Patients: {n} total -> val {len(val_patients)} / test {len(test_patients)} (same split as Steps 1 & 2)")

    X_val, _ = build_dataset(PROJECT_ROOT, val_patients, window_sec=WINDOW_SEC, max_windows_per_patient=MAX_WINDOWS_PER_PATIENT)
    X_test, _ = build_dataset(PROJECT_ROOT, test_patients, window_sec=WINDOW_SEC, max_windows_per_patient=MAX_WINDOWS_PER_PATIENT)
    print(f"Windows: val {X_val.shape[0]} / test {X_test.shape[0]}")

    ckpt = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
    state_dict = ckpt["model"]

    model = LSTMAutoencoder(n_channels=X_val.shape[-1], hidden_size=32, latent_size=16)
    model.load_state_dict(state_dict)
    model.eval()

    val_err = reconstruction_error(model, X_val)
    threshold = percentile_threshold(val_err, pct=95.0)
    print(f"validation reconstruction error: mean={val_err.mean():.5f} std={val_err.std():.5f}")
    print(f"adaptive threshold (95th percentile of val error): {threshold:.5f}")

    val_var_scores = compute_variance_scores(X_val, FS)
    var_thresh = variance_threshold(val_var_scores, pct=5.0)
    print(f"variance-detector per-channel thresholds (5th pct of val): {var_thresh}")

    X_eval, y_eval, kinds_used = build_labeled_eval_set(X_test, rng, anomaly_frac=0.5)
    eval_err = reconstruction_error(model, X_eval)
    y_pred_recon = (eval_err > threshold).astype(int)

    eval_var_scores = compute_variance_scores(X_eval, FS)
    y_pred_var = variance_predict(eval_var_scores, var_thresh)
    y_pred_combined = combine_predictions(y_pred_recon, y_pred_var)

    auroc = roc_auc_score(y_eval, eval_err)
    print(f"\nLSTM-AE reconstruction-error AUROC: {auroc:.4f}")

    def score_block(name, y_pred):
        f1 = f1_score(y_eval, y_pred, zero_division=0)
        precision = precision_score(y_eval, y_pred, zero_division=0)
        recall = recall_score(y_eval, y_pred, zero_division=0)
        print(f"\n=== {name} -- synthetic-anomaly evaluation (real run, PRELIMINARY) ===")
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

    f1_recon, precision_recon, recall_recon, breakdown_recon = score_block("DP Pilot (3 clients) LSTM-AE only", y_pred_recon)
    f1_comb, precision_comb, recall_comb, breakdown_comb = score_block("DP Pilot + variance detector (combined)", y_pred_combined)

    results = {
        "regime": "dp_pilot_3client_preliminary",
        "label": "PRELIMINARY -- 3-client pilot on 81-patient pool, not the full K=10 Step 3b result",
        "seed": SEED,
        "window_sec": WINDOW_SEC,
        "threshold_method": "95th percentile of validation error",
        "architecture_note": "per-timestep-bottleneck decoder (see m3_lstm_ae.py docstring)",
        "n_clients": 3,
        "num_rounds": 5,
        "local_epochs_per_round": 2,
        "target_epsilon": 1.0,
        "target_delta": 1e-4,
        "final_achieved_epsilon": 0.9923976910245224,
        "n_patients_val": len(val_patients),
        "n_patients_test": len(test_patients),
        "n_windows_val": int(X_val.shape[0]),
        "n_windows_test": int(X_test.shape[0]),
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
        "combined_with_variance_detector": {
            "f1": round(float(f1_comb), 4),
            "precision": round(float(precision_comb), 4),
            "recall": round(float(recall_comb), 4),
            "per_anomaly_type_recall": breakdown_comb,
        },
    }
    out_path = os.path.join(RESULTS_DIR, "m3_dp_pilot_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
