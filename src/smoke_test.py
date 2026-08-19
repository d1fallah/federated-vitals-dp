"""
M3 pipeline smoke test — NOT the official centralized baseline.
==================================================================
Purpose: prove the data loader and LSTM-AE actually run end-to-end on real
mimic4wdb data (correct tensor shapes, loss decreases, no crashes). This is
a code-correctness check, not a result for the paper: the roadmap's DP
threat-model gate is still pending supervisor approval, and the anomaly
*evaluation* methodology (how "abnormal" windows get labeled for
precision/recall/F1) is still an open decision — see the session notes.
Trains on a handful of patients for a few epochs only.
"""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data import list_complete_patients, build_dataset
from lstm_autoencoder import LSTMAutoencoder, train_lstm_ae, reconstruction_error, adaptive_threshold

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

if __name__ == "__main__":
    print("Listing fully-downloaded patients...")
    complete = list_complete_patients(PROJECT_ROOT)
    print(f"  {len(complete)} complete patients found")

    subset = complete[:8]
    print(f"Building windowed dataset from {len(subset)} patients: {subset}")
    t0 = time.time()
    X, patient_ids = build_dataset(PROJECT_ROOT, subset, window_sec=30, max_windows_per_patient=100)
    print(f"  X shape: {X.shape}  (n_windows, window_len, n_channels)")
    print(f"  built in {time.time()-t0:.1f}s from {len(set(patient_ids))} patients with usable data")

    if X.shape[0] < 10:
        print("Not enough windows to smoke-test training. Stopping.")
        sys.exit(0)

    n = X.shape[0]
    n_train = int(n * 0.8)
    X_train, X_val = X[:n_train], X[n_train:]
    print(f"  train windows: {X_train.shape[0]}  val windows: {X_val.shape[0]}")

    model = LSTMAutoencoder(n_channels=X.shape[-1], hidden_size=32, latent_size=16)
    print("Training LSTM-AE (smoke test, 10 epochs)...")
    history = train_lstm_ae(model, X_train, epochs=10, lr=1e-3, batch_size=16)
    print("  training loss per epoch:", [round(h, 5) for h in history])

    train_err = reconstruction_error(model, X_train)
    val_err = reconstruction_error(model, X_val)
    thr = adaptive_threshold(train_err, k=3.0)
    print(f"  train reconstruction error: mean={train_err.mean():.5f} std={train_err.std():.5f}")
    print(f"  val reconstruction error:   mean={val_err.mean():.5f} std={val_err.std():.5f}")
    print(f"  adaptive threshold (mean+3*std of train): {thr:.5f}")
    print(f"  val windows above threshold: {(val_err > thr).sum()} / {len(val_err)}")
    print("\nSmoke test complete — pipeline runs end-to-end on real data.")
