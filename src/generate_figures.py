"""
M3 -- generate publication-quality vector figures from real saved results, for reuse
when the paper is eventually written (CLAUDE.md Step 6: "high-resolution vector figures,
300 DPI, PDF/SVG").

Reads only results/m3_multiseed_analysis.json, results/m3_collapse_diagnostic.json, and
the per-K DP-federated result files -- no invented numbers, no synthetic data. Writes one
PDF (vector) per figure to paper_assets/figures/.
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(SRC_DIR, "..", "results")
FIGURES_DIR = os.path.join(SRC_DIR, "..", "paper_assets", "figures")

plt.rcParams.update({
    "font.size": 9,
    "font.family": "sans-serif",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "pdf.fonttype": 42,  # embed TrueType, not Type 3, for camera-ready compatibility
})

REGIME_ORDER = ["Centralized", "Non-DP Federated (K=10)", "DP Federated K=10", "DP Federated K=5",
                "DP Federated K=3", "DP Federated K=1", "Connectivity Good", "Connectivity Moderate",
                "Connectivity Poor"]
REGIME_LABELS = {"Centralized": "Central-\nized", "Non-DP Federated (K=10)": "Non-DP\nFed (K=10)",
                  "DP Federated K=10": "DP\nK=10", "DP Federated K=5": "DP\nK=5", "DP Federated K=3": "DP\nK=3",
                  "DP Federated K=1": "DP\nK=1", "Connectivity Good": "Conn.\nGood",
                  "Connectivity Moderate": "Conn.\nModerate", "Connectivity Poor": "Conn.\nPoor"}


def load(name):
    with open(os.path.join(RESULTS_DIR, name)) as f:
        return json.load(f)


def save(fig, name):
    os.makedirs(FIGURES_DIR, exist_ok=True)
    path = os.path.join(FIGURES_DIR, name)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {path}")


def fig1_combined_f1_stability(analysis):
    fm = analysis["full_metrics_by_regime"]
    means = [fm[r]["combined_f1"]["mean"] for r in REGIME_ORDER]
    stds = [fm[r]["combined_f1"]["std"] for r in REGIME_ORDER]
    labels = [REGIME_LABELS[r] for r in REGIME_ORDER]

    fig, ax = plt.subplots(figsize=(7.2, 3.2))
    x = np.arange(len(REGIME_ORDER))
    ax.errorbar(x, means, yerr=stds, fmt="o", color="#0b3d5c", ecolor="#0b3d5c",
                elinewidth=1.3, capsize=3, markersize=5)
    grand_mean = np.mean(means)
    ax.axhline(grand_mean, color="#c77b00", linestyle="--", linewidth=1, label=f"Grand mean ({grand_mean:.3f})")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=7.5)
    ax.set_ylabel("Combined F1 (5-seed mean ± std)")
    ax.set_title("Combined F1 is stable across privacy, federation, and connectivity conditions")
    ax.legend(frameon=False, loc="lower right", fontsize=8)
    ax.set_ylim(0.55, 0.75)
    save(fig, "fig1_combined_f1_stability.pdf")


def fig2_sigma_vs_k():
    ks = [10, 5, 3, 1]
    sigma_ranges = []
    for k in ks:
        d = load(f"m3_dp_federated_k{k}_results.json")
        sigmas = [c["noise_multiplier"] for c in d["client_logs"]]
        sigma_ranges.append((min(sigmas), max(sigmas), np.mean(sigmas)))

    fig, ax = plt.subplots(figsize=(4.2, 3.2))
    x = np.arange(len(ks))
    means = [s[2] for s in sigma_ranges]
    lo = [s[2] - s[0] for s in sigma_ranges]
    hi = [s[1] - s[2] for s in sigma_ranges]
    ax.bar(x, means, yerr=[lo, hi], capsize=4, color="#16506f", width=0.55)
    ax.set_xticks(x)
    ax.set_xticklabels([f"K={k}" for k in ks])
    ax.set_ylabel("Noise multiplier $\\sigma$")
    ax.set_title("$\\sigma$ is non-monotonic in $K$\n(batch cap + step-count interaction)")
    save(fig, "fig2_sigma_vs_k.pdf")


def fig3_anomaly_type_mechanism(analysis):
    # One representative regime (DP Federated K=10) makes the mechanism visible without
    # a 9-regime wall of bars; the pattern is confirmed identical in direction across all
    # 9 regimes (see anomaly_breakdown.tex / the Findings Dossier for the full table).
    regime = "DP Federated K=10"
    row = analysis["anomaly_type_breakdown"][regime]
    kinds = ["spike", "dropout", "flatline"]
    detectors = ["LSTM-AE-only", "Variance-only", "Combined"]
    colors = {"LSTM-AE-only": "#8faec4", "Variance-only": "#c77b00", "Combined": "#1e7a3d"}

    fig, ax = plt.subplots(figsize=(5.0, 3.2))
    width = 0.25
    x = np.arange(len(kinds))
    for i, det in enumerate(detectors):
        means = [row[det][k]["mean"] for k in kinds]
        stds = [row[det][k]["std"] for k in kinds]
        ax.bar(x + (i - 1) * width, means, width, yerr=stds, capsize=3, label=det, color=colors[det])
    ax.set_xticks(x)
    ax.set_xticklabels([k.capitalize() for k in kinds])
    ax.set_ylabel("Recall (5-seed mean ± std)")
    ax.set_title(f"Detector complementarity by anomaly type\n({regime})")
    ax.legend(frameon=False, fontsize=8)
    save(fig, "fig3_anomaly_type_mechanism.pdf")


def fig4_collapse_evidence():
    d = load("m3_collapse_diagnostic.json")
    names = list(d["regimes"].keys())
    ratios = [np.mean(d["regimes"][n]["ratio_per_channel"]) for n in names]
    colors = ["#1e7a3d" if not d["regimes"][n]["collapsed"] else "#b3261e" for n in names]

    fig, ax = plt.subplots(figsize=(5.5, 3.2))
    x = np.arange(len(names))
    ax.bar(x, ratios, color=colors, width=0.6)
    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels([n.replace(" Federated", "\nFederated").replace(" (", "\n(") for n in names], fontsize=7)
    ax.set_ylabel("recon_std / input_std (log scale)")
    ax.set_title("DP-SGD collapse: real and unresolved at every tested $K$")
    ax.axhline(1.0, color="#888", linestyle=":", linewidth=0.8)
    save(fig, "fig4_collapse_evidence.pdf")


def main():
    analysis = load("m3_multiseed_analysis.json")
    fig1_combined_f1_stability(analysis)
    fig2_sigma_vs_k()
    fig3_anomaly_type_mechanism(analysis)
    fig4_collapse_evidence()
    print("\nAll figures generated from real saved results.")


if __name__ == "__main__":
    main()
