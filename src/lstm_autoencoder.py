"""
M3 LSTM-Autoencoder — unsupervised anomaly detector for ambulance vitals
=========================================================================
Roadmap M3 operational layer: "Sensors -> MQTT -> Local LSTM-AE + Adaptive
Threshold -> Alert / Data-Quality Status". This module implements the
LSTM-AE and the adaptive threshold; MQTT transport and the federated
aggregator are separate, later pieces (federated aggregation is gated on
the DP threat-model gate, see M3_DP_Threat_Model_Gate.md).

Reconstruction-based anomaly detection: train on windows assumed
predominantly normal, score new windows by reconstruction MSE, flag windows
above an adaptive threshold derived from the training error distribution.
"""
import numpy as np
import torch
import torch.nn as nn


class LSTMAutoencoder(nn.Module):
    """Per-timestep-bottleneck LSTM autoencoder.

    Earlier version compressed the whole window to a single latent vector
    from the encoder's final hidden state, then fed the decoder that SAME
    vector repeated at every one of the window's timesteps (a "repeat
    vector" seq2seq pattern). Diagnosed as the root cause of a collapse to
    near-constant reconstruction (measured: reconstruction std ~0.016-0.29
    vs. input std ~1.0-1.15 on real data; got WORSE, not better, with more
    capacity/epochs, ruling out simple undertraining) -- the decoder had no
    time-varying input to work from and MSE optimization found predicting
    near the mean at every step an easier minimum than unfolding real
    dynamics from a constant.

    Fix: bottleneck each encoder timestep individually (latent_size <
    hidden_size still enforces real per-timestep compression) and feed the
    decoder that per-timestep sequence, not one repeated global vector. The
    decoder now has genuine time-varying signal to reconstruct from.
    """

    def __init__(self, n_channels=3, hidden_size=32, latent_size=16, num_layers=1):
        super().__init__()
        self.encoder = nn.LSTM(n_channels, hidden_size, num_layers, batch_first=True)
        self.bottleneck = nn.Linear(hidden_size, latent_size)
        self.expand = nn.Linear(latent_size, hidden_size)
        self.decoder = nn.LSTM(hidden_size, hidden_size, num_layers, batch_first=True)
        self.output_layer = nn.Linear(hidden_size, n_channels)

    def forward(self, x):
        # x: [batch, window_len, n_channels]
        enc_out, _ = self.encoder(x)            # [batch, seq_len, hidden_size], per-timestep
        latent_seq = self.bottleneck(enc_out)    # [batch, seq_len, latent_size] -- per-timestep compression
        decoder_input = self.expand(latent_seq)  # [batch, seq_len, hidden_size]
        dec_out, _ = self.decoder(decoder_input)
        recon = self.output_layer(dec_out)
        return recon


def train_lstm_ae(model, X_train, epochs=20, lr=1e-3, batch_size=32, device="cpu"):
    """Trains in-place. Returns the per-epoch mean training loss (real
    values from the actual run, for logging/verification, not estimates)."""
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    X = torch.as_tensor(X_train, dtype=torch.float32)
    n = X.shape[0]

    history = []
    model.train()
    for epoch in range(epochs):
        perm = torch.randperm(n)
        epoch_loss = 0.0
        n_batches = 0
        for start in range(0, n, batch_size):
            idx = perm[start:start + batch_size]
            batch = X[idx].to(device)
            opt.zero_grad()
            recon = model(batch)
            loss = loss_fn(recon, batch)
            loss.backward()
            opt.step()
            epoch_loss += loss.item()
            n_batches += 1
        history.append(epoch_loss / max(n_batches, 1))
    return history


@torch.no_grad()
def reconstruction_error(model, X, device="cpu", batch_size=64):
    """Per-window mean-squared reconstruction error."""
    model.eval()
    X_t = torch.as_tensor(X, dtype=torch.float32)
    errors = []
    for start in range(0, X_t.shape[0], batch_size):
        batch = X_t[start:start + batch_size].to(device)
        recon = model(batch)
        err = ((recon - batch) ** 2).mean(dim=(1, 2))
        errors.append(err.cpu().numpy())
    return np.concatenate(errors) if errors else np.array([])


def adaptive_threshold(train_errors, k=3.0):
    """Global adaptive threshold: mean + k*std of the training reconstruction
    error distribution. The roadmap's stretch description (per-patient-profile
    adaptive threshold, e.g. a lower SpO2 baseline for a COPD patient) needs
    the patient's clinical profile from the main MIMIC-IV tables, which this
    waveform-only extract does not include — noted as a documented follow-up,
    not implemented here."""
    return float(train_errors.mean() + k * train_errors.std())


def percentile_threshold(val_errors, pct=95.0):
    """Threshold set as a percentile of the held-out validation reconstruction
    error distribution. Preferred over mean+k*std when that distribution is
    skewed/heavy-tailed (common for reconstruction error, which is bounded
    below by 0) — a fixed multiple of std can sit far out in the tail and
    under-flag true anomalies in exactly that regime."""
    return float(np.percentile(val_errors, pct))
