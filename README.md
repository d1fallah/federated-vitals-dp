# 🚑 Privacy-Preserving Federated Anomaly Detection for Ambulance IoT — LSTM-AE + DP + NVFlare

> **Part of the SihhaDZ research project** — An AI-powered smart emergency platform for the Algerian healthcare system.

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-ee4c2c?logo=pytorch)](https://pytorch.org/)
[![NVFlare](https://img.shields.io/badge/NVFlare-2.x-76b900?logo=nvidia)](https://github.com/NVIDIA/NVFlare)
[![Opacus](https://img.shields.io/badge/Opacus-DP--SGD-blue)](https://opacus.ai/)
[![ε=1.0](https://img.shields.io/badge/DP-ε%3D1.0%2C%20δ%3D1e--4-orange)](M3_DP_Threat_Model_Gate.md)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## 🎯 What This Is

This module implements a **privacy-preserving federated learning system** for real-time physiological anomaly detection in ambulance IoT environments. Patient vital-sign streams (ECG, SpO₂, BP, respiratory rate) are processed **locally on each ambulance** — raw data never leaves the vehicle.

The system is built around three research axes evaluated simultaneously:

| Axis | What we measure |
|------|----------------|
| **Privacy** | Differential Privacy (ε, δ) via Opacus DP-SGD |
| **Utility** | F1, Precision, Recall on synthetic anomaly injection |
| **Connectivity robustness** | Packet loss / latency / bandwidth under Good / Moderate / Poor profiles |

This is the first joint empirical characterization of all three axes for an ambulance-IoT federated learning system.

---

## 🔬 Research Contribution

The primary contribution is the **joint empirical analysis** — running the same LSTM-Autoencoder architecture across three training regimes and two distinct anomaly detection methods, with all three axes (privacy, utility, robustness) reported simultaneously:

```
Regime 1: Centralized LSTM-AE            ← performance ceiling
Regime 2: Federated LSTM-AE (no DP)      ← federation cost
Regime 3: Federated LSTM-AE + DP-SGD     ← privacy-utility trade-off
```

Each regime runs under simulated network profiles (Good / Moderate / Poor) using a SimPy discrete-event simulation, producing the first quantified robustness profile for this type of system.

### Anomaly Detection Architecture

Two complementary detectors run in ensemble:

| Detector | Targets | Method |
|----------|---------|--------|
| LSTM-Autoencoder | Spike anomalies, irregular patterns | Reconstruction error > adaptive threshold (95th pct of val set) |
| Variance Detector | Dropout, flatline (signal loss) | Per-channel variance < 5th pct of val set |

> **Why two detectors?** A well-trained reconstruction model finds near-constant flatline segments *easy* to reconstruct — making the LSTM-AE blind to dropout/flatline anomalies. The variance detector closes this structural gap.

### Differential Privacy (Gate-Approved)

Privacy parameters are formally derived and supervisor-approved in [`M3_DP_Threat_Model_Gate.md`](M3_DP_Threat_Model_Gate.md):

- **Privacy unit:** patient (record-level), not window-level
- **Mechanism:** Opacus DP-SGD with one-window-per-patient-per-round sampling
- **ε = 1.0** (fixed by roadmap), **δ = 1e-4** (derived: δ ≪ 1/N for N=198 patients)
- **Aggregation:** NVFlare FedAvg, local DP (raw data never leaves client)

---

## 📁 Repository Structure

```
federated-iot-anomaly-detection-dp/
├── src/                                  # All original research code
│   ├── train_centralized.py             # Step 1: Centralized LSTM-AE (approved 17 Aug 2026)
│   ├── lstm_autoencoder.py              # LSTM-Autoencoder model + training
│   ├── data.py                          # Patient-level windowing & dataset construction
│   ├── anomaly_eval.py                  # Synthetic anomaly injection evaluation
│   ├── variance_detector.py             # Complementary variance-based detector
│   ├── dp_federated_client.py           # DP-SGD federated client (NVFlare + Opacus)
│   ├── dp_federated_job.py              # NVFlare job configuration for DP pilot
│   ├── federated_client.py              # Non-DP federated client baseline
│   ├── federated_job.py                 # NVFlare job configuration (non-DP)
│   ├── evaluate_dp_federated.py         # DP federated evaluation
│   ├── evaluate_non_dp_federated.py     # Non-DP federated evaluation
│   └── smoke_test.py                    # Quick sanity check (no data required)
│
├── results/                             # Real experiment outputs (JSON)
│   ├── m3_centralized_baseline_results.json
│   ├── m3_non_dp_federated_results.json
│   ├── m3_dp_pilot_results.json
│   └── dp_pilot_round_log_site-*.json
│
├── dp-threat-model-gate.md              # Supervisor-approved DP threat model
├── dp-gate-progress-report.pdf
├── supervisor-decision-v1.docx
├── supervisor-decision-v2.docx
└── README.md
```

---

## 🏗️ System Architecture

```
┌──────────────────────────────────────────────────────┐
│                   Ambulance Fleet                     │
│                                                      │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  │
│  │ Ambulance 1 │  │ Ambulance 2 │  │ Ambulance K │  │
│  │             │  │             │  │             │  │
│  │ Sensors     │  │ Sensors     │  │ Sensors     │  │
│  │ → MQTT QoS1 │  │ → MQTT QoS1 │  │ → MQTT QoS1 │  │
│  │ → LSTM-AE   │  │ → LSTM-AE   │  │ → LSTM-AE   │  │
│  │ → Alerts    │  │ → Alerts    │  │ → Alerts    │  │
│  │             │  │             │  │             │  │
│  │ Local       │  │ Local       │  │ Local       │  │
│  │ DP-SGD      │  │ DP-SGD      │  │ DP-SGD      │  │
│  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘  │
│         │ gradient updates (DP-protected)  │         │
└─────────┼──────────────────────────────────┼─────────┘
          │                                  │
          ▼                                  ▼
    ┌─────────────────────────────────────────┐
    │     NVFlare Aggregator (FedAvg)          │
    │     (honest-but-curious threat model)    │
    └─────────────────────────────────────────┘
```

### Network Simulation (SimPy)

Three profiles are simulated using SimPy discrete-event simulation:

| Profile | Packet Loss | Latency | Bandwidth |
|---------|-------------|---------|-----------|
| Good | < 1% | < 50ms | > 10 Mbps |
| Moderate | 1–5% | 50–200ms | 1–10 Mbps |
| Poor | > 5% | > 200ms | < 1 Mbps |

---

## ⚙️ Setup

### Prerequisites
- Python 3.9+
- PyTorch 2.x
- NVFlare 2.x
- Opacus
- MIMIC-IV Waveform Database access ([PhysioNet credentialed](https://physionet.org/content/mimic4wdb/))

### Installation

```bash
git clone https://github.com/YOUR_USERNAME/federated-iot-anomaly-detection-dp
cd federated-iot-anomaly-detection-dp

pip install torch torchvision nvflare opacus paho-mqtt simpy scikit-learn numpy
```

### Data

This repository does **not** include any patient data. Download MIMIC-IV Waveform Database from PhysioNet (credentialed access required) and place it in `data/mimic4wdb/`.

---

## 🚀 Reproducing Results

```bash
# Quick sanity check (no data needed)
python src/smoke_test.py

# Step 1 — Centralized baseline (performance ceiling)
python src/train_centralized.py

# Step 2 — Non-DP Federated baseline
python src/federated_job.py

# Step 3 — DP Pilot (2–3 client subset, privacy–utility verification)
python src/dp_federated_job.py

# Evaluate all regimes
python src/evaluate_dp_federated.py
python src/evaluate_non_dp_federated.py
```

All scripts use `SEED = 42` for reproducibility. Results are written to `Results/` as JSON.

---

## 📊 Preliminary Results (Centralized Baseline)

> Full ablation results (all 3 regimes × 3 network profiles) are pending the DP pilot approval run.

The centralized LSTM-AE + variance detector ensemble (Step 1, approved 17 Aug 2026):

See [`results/m3_centralized_baseline_results.json`](results/m3_centralized_baseline_results.json) for exact numbers (produced by a real experiment run, seed=42).

**Key design decisions documented in results:**
- 10-second windows (not 30s) — 30s sequences caused near-flat training loss in 20 epochs; 10s cuts sequence length ~3× while still spanning multiple cardiac/respiratory cycles
- Per-timestep decoder (not repeat-vector) — the repeat-vector architecture collapsed to near-constant output
- Variance detector added after discovering LSTM-AE structurally cannot flag flatline segments (it reconstructs them easily)

---

## 📋 Gate Documents

| Gate | Status | Document |
|------|--------|----------|
| DP Threat-Model Gate | ✅ Revised & resubmitted (17 Aug 2026) | [`dp-threat-model-gate.md`](dp-threat-model-gate.md) |
| DP Pilot (§6.3) | ⏳ Next step | Pending supervisor approval of gate |

---

## 📚 References

- **NVFlare:** NVIDIA Federated Learning Application Runtime Environment — [github.com/NVIDIA/NVFlare](https://github.com/NVIDIA/NVFlare)
- **Opacus:** Yousefpour et al., *Opacus: User-Friendly Differential Privacy Library in PyTorch*, 2021
- **DP-SGD:** Abadi et al., *Deep Learning with Differential Privacy*, CCS 2016
- **McMahan et al.:** *Learning Differentially Private Recurrent Language Models*, ICLR 2018 (client-level DP reference)
- **Dwork & Roth:** *The Algorithmic Foundations of Differential Privacy*, 2014 (δ derivation)
- **Paho MQTT:** Eclipse Foundation — [github.com/eclipse/paho.mqtt.python](https://github.com/eclipse/paho.mqtt.python)
- **MIMIC-IV Waveform DB:** Johnson et al., PhysioNet 2023

---

## ⚠️ Data & Ethics Notice

This work uses the MIMIC-IV Waveform Database, which contains de-identified physiological recordings. Access requires PhysioNet credentialed data use agreement.

**No patient data is included in this repository.**

---

## 🔗 Related Modules

This is Module 3 of the SihhaDZ research stack:

| Module | Topic |
|--------|-------|
| M1 | Secure Homomorphic Patient Profiling (CKKS) |
| M2 | Conformal Risk Prediction (GRU-D + ECI, ICLR 2025) |
| **M3** | **← You are here** — Federated IoT Anomaly Detection (DP + NVFlare) |
| M4 | Hospital Resource Forecasting (N-BEATSx + NCC) |
| M5 | Multi-Agent Ambulance Routing (MAPPO + GAT) |
