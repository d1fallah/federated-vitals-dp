"""
M3 Step 2 — evaluate the FedAvg-aggregated global model
===========================================================
Loads the final FL_global_model.pt persisted by the NVFlare simulator run
(federated_job.py) and evaluates it with the *identical* protocol used for
the centralized baseline (train_centralized.py): same seed, same
patient-level 70/15/15 split (so the held-out test patients are the exact
same set), same synthetic anomaly injection, same percentile threshold
method -- so the two regimes are genuinely comparable, not just similarly
computed.

--network_profile {good,moderate,poor} evaluates one of the connectivity-
robustness ablation runs instead of the original no-degradation baseline:
same utility protocol, plus real (not a-priori-only) participation-rate
reporting pulled from the round logs each client wrote during training.
"""
import argparse
import glob
import os
import shutil
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
from network_sim import load_schedule, participation_rate, breakdown_by_reason, per_round_participation

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")
MAX_WINDOWS_PER_PATIENT = 80
WINDOW_SEC = 10


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--network_profile", type=str, default="none", choices=["none", "good", "moderate", "poor"],
                         help="Evaluate a connectivity-robustness ablation run instead of the no-degradation baseline")
    parser.add_argument("--seed", type=int, default=42,
                         help="Must match the --seed used for the corresponding federated_job.py training run.")
    args = parser.parse_args()
    profile = args.network_profile
    SEED = args.seed
    job_seed_suffix = "" if SEED == 42 else f"-seed{SEED}"
    file_seed_suffix = "" if SEED == 42 else f"_seed{SEED}"

    job_name = f"m3-non-dp-federated{job_seed_suffix}" if profile == "none" else f"m3-non-dp-federated-{profile}{job_seed_suffix}"
    checkpoint = rf"C:/tmp/nvflare/simulation/{job_name}/server/simulate_job/app_server/FL_global_model.pt"

    os.makedirs(RESULTS_DIR, exist_ok=True)
    rng = np.random.default_rng(SEED)

    complete = list_complete_patients(PROJECT_ROOT)
    patients = list(rng.permutation(complete))
    n = len(patients)
    n_train = int(n * 0.70)
    n_val = int(n * 0.15)
    val_patients = patients[n_train:n_train + n_val]
    test_patients = patients[n_train + n_val:]
    print(f"Patients: {n} total -> val {len(val_patients)} / test {len(test_patients)} (same split as train_centralized.py)")

    X_val, _ = build_dataset(PROJECT_ROOT, val_patients, window_sec=WINDOW_SEC, max_windows_per_patient=MAX_WINDOWS_PER_PATIENT)
    X_test, _ = build_dataset(PROJECT_ROOT, test_patients, window_sec=WINDOW_SEC, max_windows_per_patient=MAX_WINDOWS_PER_PATIENT)
    print(f"Windows: val {X_val.shape[0]} / test {X_test.shape[0]}")

    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
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

    # Re-derive the labeled eval set with a fresh rng draw in the exact same
    # sequence train_centralized.py uses (val/test build, then eval-set
    # construction), so the same anomaly injections land on the same test windows.
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
        print(f"\n=== {name} -- synthetic-anomaly evaluation (real run) ===")
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

    label_suffix = "" if profile == "none" else f" [network={profile}]"
    f1_recon, precision_recon, recall_recon, breakdown_recon = score_block(f"Non-DP Federated LSTM-AE only{label_suffix}", y_pred_recon)
    f1_var, precision_var, recall_var, breakdown_var = score_block(f"Non-DP Federated variance detector only{label_suffix}", y_pred_var)
    f1_comb, precision_comb, recall_comb, breakdown_comb = score_block(f"Non-DP Federated + variance detector (combined){label_suffix}", y_pred_combined)

    results = {
        "regime": "non_dp_federated" if profile == "none" else f"non_dp_federated_network_{profile}",
        "network_profile": profile,
        "seed": SEED,
        "window_sec": WINDOW_SEC,
        "threshold_method": "95th percentile of validation error",
        "architecture_note": "per-timestep-bottleneck decoder (see lstm_autoencoder.py docstring)",
        "k_clients": 10,
        "num_rounds": 10,
        "local_epochs_per_round": 4,
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

    if profile != "none":
        # A-priori participation (from the pre-registered schedule federated_job.py
        # generated before training) vs. observed participation (from the round
        # logs each client actually wrote during training) -- these should match
        # exactly, which doubles as a free correctness check on the whole
        # schedule -> client -> FLARE aggregation pipeline.
        schedule_path = os.path.join(RESULTS_DIR, f"network_schedule_{profile}{file_seed_suffix}.json")
        schedule = load_schedule(schedule_path)
        a_priori_rate = participation_rate(schedule)
        a_priori_reasons = breakdown_by_reason(schedule)
        a_priori_per_round = per_round_participation(schedule)

        sandbox_glob = rf"C:/tmp/nvflare/simulation/{job_name}/site-*/simulate_job/app_site-*/custom/network_round_log_{profile}{file_seed_suffix}_*.json"
        client_logs = []
        for src_path in glob.glob(sandbox_glob):
            dst_path = os.path.join(RESULTS_DIR, os.path.basename(src_path))
            shutil.copy(src_path, dst_path)
            with open(dst_path) as f:
                client_logs.append(json.load(f))

        observed_total = sum(len(c["round_log"]) for c in client_logs)
        observed_delivered = sum(1 for c in client_logs for r in c["round_log"] if r["delivered"])
        observed_rate = observed_delivered / observed_total if observed_total else 0.0
        observed_reasons = {}
        for c in client_logs:
            for r in c["round_log"]:
                observed_reasons[r["reason"]] = observed_reasons.get(r["reason"], 0) + 1

        print(f"\nParticipation rate -- a priori (schedule): {a_priori_rate:.4f}  observed (round logs): {observed_rate:.4f}"
              f"  {'[MATCH]' if abs(a_priori_rate - observed_rate) < 1e-9 else '[MISMATCH -- investigate]'}")
        print(f"Reason breakdown -- a priori: {a_priori_reasons}")
        print(f"Reason breakdown -- observed: {observed_reasons}")
        print(f"Per-client logs found: {len(client_logs)} (copied to {RESULTS_DIR})")

        results["connectivity_robustness"] = {
            "a_priori_participation_rate": round(a_priori_rate, 4),
            "a_priori_reason_breakdown": a_priori_reasons,
            "a_priori_per_round_participation": a_priori_per_round,
            "observed_participation_rate": round(observed_rate, 4),
            "observed_reason_breakdown": observed_reasons,
            "n_client_logs_found": len(client_logs),
            "client_logs": client_logs,
        }

    base_name = "m3_non_dp_federated" if profile == "none" else f"m3_non_dp_federated_{profile}"
    out_name = f"{base_name}{file_seed_suffix}_results.json"
    out_path = os.path.join(RESULTS_DIR, out_name)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
