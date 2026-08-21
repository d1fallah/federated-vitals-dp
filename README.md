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
│   ├── federated_client.py              # Non-DP federated client (+ network-profile ablation)
│   ├── federated_job.py                 # NVFlare job configuration (non-DP)
│   ├── network_sim.py                   # SimPy connectivity-robustness ablation (Good/Moderate/Poor)
│   ├── evaluate_dp_federated.py         # DP federated evaluation (K-sweep: 10, 5, 3, 1)
│   ├── evaluate_non_dp_federated.py     # Non-DP federated evaluation (+ network-profile ablation)
│   ├── analyze_multiseed.py             # 5-seed aggregation: mean+/-std, anomaly breakdown, TOST test
│   ├── measure_collapse.py              # DP-SGD collapse diagnostic (recon_std vs input_std)
│   ├── generate_latex_tables.py         # -> paper_assets/tables/*.tex (booktabs, real data)
│   ├── generate_figures.py              # -> paper_assets/figures/*.pdf (300 DPI vector, real data)
│   └── smoke_test.py                    # Quick sanity check (no data required)
│
├── results/                             # Real experiment outputs (JSON) -- 45 files: 9 regimes x 5 seeds
│   ├── m3_centralized_baseline{,_seed123,_seed7,_seed456,_seed789}_results.json
│   ├── m3_non_dp_federated{,_good,_moderate,_poor}{,_seed123,_seed7,_seed456,_seed789}_results.json
│   ├── m3_dp_federated_k{10,5,3,1}{,_seed123,_seed7,_seed456,_seed789}_results.json
│   ├── m3_multiseed_analysis.json       # analyze_multiseed.py's full output (tables + TOST results)
│   ├── m3_collapse_diagnostic.json      # measure_collapse.py's output
│   ├── network_schedule_{good,moderate,poor}{,_seed*}.json
│   ├── network_round_log_{profile}{,_seed*}_site-*.json
│   └── dp_pilot_round_log_k{K}{,_seed*}_site-*.json
│
├── paper_assets/                        # Publication-ready assets (not a paper draft -- raw material)
│   ├── tables/*.tex                     # 6 LaTeX booktabs tables, real data, IEEE-template-ready
│   └── figures/*.pdf                    # 4 vector figures, 300 DPI, real data
│
├── M3_Findings_Dossier.md               # Comprehensive findings write-up for future paper assembly
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

Three profiles are simulated using SimPy discrete-event simulation ([`src/network_sim.py`](src/network_sim.py)), at the roadmap's fixed parameter values (not ranges — these exact numbers are what makes the ablation comparable across the paper's Good/Moderate/Poor axis):

| Profile | Packet Loss | Latency | Bandwidth | Dropout |
|---------|-------------|---------|-----------|---------|
| Good | 1% | 50ms | 10 Mbps | 5% |
| Moderate | 5% | 200ms | 2 Mbps | 20% |
| Poor | 10% | 500ms | 0.5 Mbps | 40% |

Each (client, round) pair independently rolls dropout and packet-loss gates, then contends for the profile's shared bandwidth with every other client active that round (fair-share contention) against a fixed 3.0s round deadline. Integrated into the non-DP federated pipeline only ([`src/federated_client.py`](src/federated_client.py) / [`src/federated_job.py`](src/federated_job.py) via `--network_profile {good,moderate,poor}`) — not the DP pipeline, since the DP client's privacy accountant is constructed once at round 0 and persists across rounds, so silently skipping a round there would reopen already-approved DP-accounting questions.

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

# Step 2 — Non-DP Federated baseline (K=10, 10 rounds, 4 local epochs)
python src/federated_job.py --n_clients 10 --num_rounds 10 --epochs 4
python src/evaluate_non_dp_federated.py

# Step 2b — Connectivity-robustness ablation (same K/rounds/epochs, only network profile varies)
python src/federated_job.py --network_profile good     --n_clients 10 --num_rounds 10 --epochs 4
python src/federated_job.py --network_profile moderate --n_clients 10 --num_rounds 10 --epochs 4
python src/federated_job.py --network_profile poor     --n_clients 10 --num_rounds 10 --epochs 4
python src/evaluate_non_dp_federated.py --network_profile good
python src/evaluate_non_dp_federated.py --network_profile moderate
python src/evaluate_non_dp_federated.py --network_profile poor

# Step 3 — DP-Federated K-sweep (K=10, 5, 3, 1; all clients in each partition participate)
python src/dp_federated_job.py --k_clients 10 --n_clients 10
python src/dp_federated_job.py --k_clients 5  --n_clients 5
python src/dp_federated_job.py --k_clients 3  --n_clients 3
python src/dp_federated_job.py --k_clients 1  --n_clients 1
python src/evaluate_dp_federated.py --k_clients 10
python src/evaluate_dp_federated.py --k_clients 5
python src/evaluate_dp_federated.py --k_clients 3
python src/evaluate_dp_federated.py --k_clients 1

# Repeat all of the above with --seed 123/7/456/789, then aggregate + generate publication assets:
python src/analyze_multiseed.py
python src/measure_collapse.py
python src/generate_latex_tables.py
python src/generate_figures.py
```

All scripts accept `--seed` (default 42, reproduces the numbers below exactly; other values are for multi-seed confidence-interval runs — outputs get a `_seed{N}`/`-seed{N}` suffix so they never overwrite the seed-42 results). Results are written to `results/` as JSON. See [`M3_Findings_Dossier.md`](M3_Findings_Dossier.md) for the complete, organized findings write-up (raw material for the eventual paper, not a draft) and `paper_assets/` for ready-to-use LaTeX tables and vector figures.

---

## 📊 Results — All Three Primary-Contribution Axes (Real Runs, 5-seed mean ± std)

Full write-up with discussion: [`M3_Primary_Contribution_Report.pdf`](M3_Primary_Contribution_Report.pdf). Updated per the supervisor's 20 Aug approval memo (`M3_Supervisor_Final_Approval_Memo_v2.docx`) with the 4 required validation analyses: 2 more seeds, the 5-seed table, a per-anomaly-type breakdown, and a formal equivalence test.

**Utility** (centralized vs. non-DP-federated vs. DP-federated, 5 seeds — 42, 123, 7, 456, 789 — mean ± std):

| Regime | LSTM-AE-only F1 | Variance-only F1 | Combined F1 |
|---|---|---|---|
| Centralized (ceiling) | 0.370 ± 0.050 | 0.483 ± 0.067 | 0.641 ± 0.023 |
| Non-DP Federated (K=10) | 0.398 ± 0.058 | 0.483 ± 0.067 | 0.648 ± 0.019 |
| DP Federated K=10 / 5 / 3 / 1 | 0.393 / 0.392 / 0.392 / 0.395 (± 0.025–0.030) | 0.483 ± 0.067 (all four) | 0.645 / 0.645 / 0.646 / 0.647 (± 0.027–0.029) |

**The key statistical result**: across all 9 regimes (this table plus connectivity below), the standard deviation *between* regime means is 0.005 combined F1 — roughly 5× smaller than the *average* seed-to-seed standard deviation *within* a single regime (0.027), a stronger separation than the earlier 3-seed draft found (3×). A formal **TOST equivalence test** (margin Δ=0.05 F1, the supervisor's own pre-specified value, scope: all 9 regimes) confirms 7 of 8 non-reference regimes — including the entire DP K-sweep — are statistically equivalent to the centralized reference (p=0.007–0.019); only Connectivity Poor falls just short (p=0.054, 90% CI extends 0.001 past the margin), reported honestly rather than rounded into significance. Full table: PDF report §5.

**Per-anomaly-type mechanism** (mean recall ± std across 5 seeds, consistent across every regime): the LSTM-AE-only detector is strong on spike (0.63–0.76) and nearly blind to dropout/flatline (0.03–0.05); the variance detector is the mirror image (spike 0.18, dropout 0.61, flatline 0.38). The combined detector inherits the stronger of the two on every type — the literal mechanistic reason overall F1 stays stable regardless of what happens to the LSTM-AE under DP or federation. Full table: PDF report §3.2.

**Important correction, checked directly rather than assumed**: the LSTM-AE genuinely collapses to near-constant output under DP-SGD at every tested K — measured on the actual checkpoints behind the numbers above (recon_std/input_std ratio 0.002–0.010 for every DP-federated regime, vs. 0.95–1.00 for the healthy non-DP model). This project's git history (`module 3` is its own repo, but the DP client file has exactly one commit) contains no before/after diff showing a fix, and none was found — an earlier retraction of the original "DP collapses" diagnosis was itself inaccurate. **The precise claim: DP degrades reconstruction fidelity but does not produce a statistically distinguishable degradation in downstream ensemble anomaly-detection F1 under the tested conditions** — not "DP has no effect." The variance-detector-only score (F1=0.483±0.067, identical within each seed across every regime) supplies a collapse-invariant floor, and even the collapsed LSTM-AE's reconstruction error still weakly ranks large-magnitude anomalies above baseline, adding ≈+0.16 F1 on average — real, but well below what a non-collapsed model would provide. Full evidence table and discussion: PDF report §3.1.

**Privacy**: real accountant-derived (ε, δ) at every K, one-window-per-patient-per-round verified every round (`all_rounds_ok=True` in every saved log) — see [`results/m3_dp_federated_k10_results.json`](results/m3_dp_federated_k10_results.json) (and `k5`/`k3`/`k1`).

**Connectivity robustness** (Good/Moderate/Poor, roadmap-fixed parameters, real SimPy schedule verified against observed FLARE round logs — a-priori and observed participation rates match exactly in every profile, at all 5 seeds):

| Profile | Participation | Combined F1 (5-seed mean ± std) |
|---|---|---|
| Good | 91–97% | 0.647 ± 0.031 |
| Moderate | 70–77% | 0.648 ± 0.029 |
| Poor | 10% | 0.660 ± 0.030 |

Poor's mean is nominally highest — it's also the one regime the formal equivalence test couldn't establish as statistically indistinguishable from centralized (p=0.054, borderline). See [`results/m3_non_dp_federated_good_results.json`](results/m3_non_dp_federated_good_results.json) (and `moderate`/`poor`).

**Key design decisions documented in results:**
- 10-second windows (not 30s) — 30s sequences caused near-flat training loss in 20 epochs; 10s cuts sequence length ~3× while still spanning multiple cardiac/respiratory cycles
- Per-timestep decoder (not repeat-vector) — the repeat-vector architecture collapsed to near-constant output
- Variance detector added after discovering LSTM-AE structurally cannot flag flatline segments (it reconstructs them easily) — this same detector is also what makes utility robust across the privacy and connectivity axes, since its contribution never depends on how well the LSTM-AE trained
- GPU (RTX 3050) is used only in `train_centralized.py` (single-threaded, ~6.8× real speedup measured). The federated/DP-federated jobs stay on CPU deliberately, enforced with `CUDA_VISIBLE_DEVICES=-1` set before any `torch` import: NVFlare's in-process simulator shares one CUDA context across concurrently-running client threads, and `device="cpu"` call sites alone weren't sufficient to prevent reproducible CUDA out-of-memory crashes at K=10 (library code, most likely Opacus/DPLSTM setup, appears to touch CUDA whenever it's merely available)

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
