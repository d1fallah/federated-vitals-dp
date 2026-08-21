"""
M3 — generate LaTeX (booktabs) tables for every real result table in the primary
contribution report, for reuse when the paper is eventually written (per CLAUDE.md's
Step 6 deliverable spec: "LaTeX booktabs comparison tables... mean +/- std over >=5 seeds").

Reads only already-saved real results (results/m3_multiseed_analysis.json,
results/m3_collapse_diagnostic.json, results/m3_dp_federated_k{K}_results.json) -- no new
computation, no invented numbers. Writes one .tex file per table to paper_assets/tables/.
"""
import json
import os

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(SRC_DIR, "..", "results")
TABLES_DIR = os.path.join(SRC_DIR, "..", "paper_assets", "tables")

PRIMARY_REGIMES = ["Centralized", "Non-DP Federated (K=10)", "DP Federated K=10",
                   "DP Federated K=5", "DP Federated K=3", "DP Federated K=1"]
CONNECTIVITY_REGIMES = ["Non-DP Federated (K=10)", "Connectivity Good", "Connectivity Moderate", "Connectivity Poor"]
CONNECTIVITY_LABELS = {"Non-DP Federated (K=10)": "None (baseline)", "Connectivity Good": "Good",
                        "Connectivity Moderate": "Moderate", "Connectivity Poor": "Poor"}
CONNECTIVITY_PARTICIPATION = {"Non-DP Federated (K=10)": "100\\%", "Connectivity Good": "91--97\\%",
                               "Connectivity Moderate": "70--77\\%", "Connectivity Poor": "10\\%"}


def load(name):
    with open(os.path.join(RESULTS_DIR, name)) as f:
        return json.load(f)


def pm(stat):
    return f"{stat['mean']:.3f} $\\pm$ {stat['std']:.3f}"


def rng(lo, hi, fmt="{:.2f}"):
    """Formats a min--max range, collapsing to a single value when lo == hi (e.g. K=1's single client)."""
    return fmt.format(lo) if lo == hi else f"{fmt.format(lo)}--{fmt.format(hi)}"


def write(filename, content):
    os.makedirs(TABLES_DIR, exist_ok=True)
    path = os.path.join(TABLES_DIR, filename)
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
    print(f"wrote {path}")


def table_privacy_k_sweep():
    rows = []
    for k in [10, 5, 3, 1]:
        d = load(f"m3_dp_federated_k{k}_results.json")
        logs = d["client_logs"]
        n_patients = [c["n_patients"] for c in logs]
        sigmas = [c["noise_multiplier"] for c in logs]
        epsilons = [c["final_achieved_epsilon"] for c in logs]
        all_ok = all(r["ok"] for c in logs for r in c["round_log"])
        n_clients = len(logs)
        rows.append(
            f"{k} & {rng(min(n_patients), max(n_patients), '{:d}')} & {rng(min(sigmas), max(sigmas))} & "
            f"{rng(min(epsilons), max(epsilons))} & {'Yes' if all_ok else 'NO'} ({n_clients}/{n_clients}) \\\\"
        )
    content = (
        "% M3 Axis 1 -- Privacy: DP K-sweep (seed 42; target epsilon=1.0, delta=1e-4)\n"
        "\\begin{table}[t]\n\\centering\n\\caption{DP-SGD privacy mechanism across the K-sweep. All rounds "
        "protocol-compliant: one window per patient per round, verified every round, every client, every K.}\n"
        "\\label{tab:m3-privacy-k-sweep}\n"
        "\\begin{tabular}{@{}lllll@{}}\n\\toprule\n"
        "$K$ (clients) & Patients/client & Noise mult. $\\sigma$ & Achieved $\\varepsilon$ & Protocol-compliant \\\\\n"
        "\\midrule\n" + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n\\end{table}\n"
    )
    write("privacy_k_sweep.tex", content)


def table_utility_primary(analysis):
    fm = analysis["full_metrics_by_regime"]
    rows = []
    for name in PRIMARY_REGIMES:
        m = fm[name]
        rows.append(f"{name} & {pm(m['auroc'])} & {pm(m['lstm_ae_only_f1'])} & {pm(m['variance_only_f1'])} & {pm(m['combined_f1'])} \\\\")
    content = (
        "% M3 Axis 2 -- Utility: primary 5-seed comparison table (seeds 42,123,7,456,789)\n"
        "\\begin{table}[t]\n\\centering\n\\caption{Utility comparison across the three required regimes "
        "(centralized, non-DP federated, DP federated K-sweep), 5-seed mean $\\pm$ std.}\n"
        "\\label{tab:m3-utility-primary}\n"
        "\\begin{tabular}{@{}lllll@{}}\n\\toprule\n"
        "Regime & AUROC (LSTM-AE) & LSTM-AE-only F1 & Variance-only F1 & Combined F1 \\\\\n"
        "\\midrule\n" + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n\\end{table}\n"
    )
    write("utility_primary.tex", content)


def table_connectivity(analysis):
    fm = analysis["full_metrics_by_regime"]
    rows = []
    for name in CONNECTIVITY_REGIMES:
        m = fm[name]
        label = CONNECTIVITY_LABELS[name]
        participation = CONNECTIVITY_PARTICIPATION[name]
        rows.append(f"{label} & {participation} & {pm(m['lstm_ae_only_f1'])} & {pm(m['variance_only_f1'])} & {pm(m['combined_f1'])} \\\\")
    content = (
        "% M3 Axis 3 -- Connectivity robustness: 5-seed comparison table\n"
        "\\begin{table}[t]\n\\centering\n\\caption{Connectivity-robustness ablation (roadmap-fixed Good/Moderate/"
        "Poor network profiles), 5-seed mean $\\pm$ std. Participation: a-priori schedule matched observed round "
        "logs exactly at every seed.}\n"
        "\\label{tab:m3-connectivity}\n"
        "\\begin{tabular}{@{}lllll@{}}\n\\toprule\n"
        "Profile & Participation & LSTM-AE-only F1 & Variance-only F1 & Combined F1 \\\\\n"
        "\\midrule\n" + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n\\end{table}\n"
    )
    write("connectivity.tex", content)


def table_anomaly_breakdown(analysis):
    breakdown = analysis["anomaly_type_breakdown"]
    rows = []
    for name in PRIMARY_REGIMES:
        regime_rows = breakdown[name]
        first = True
        for det_label in ("LSTM-AE-only", "Variance-only", "Combined"):
            row = regime_rows[det_label]
            regime_cell = f"\\multirow{{3}}{{*}}{{{name}}}" if first else ""
            first = False
            rows.append(
                f"{regime_cell} & {det_label} & "
                f"{row['spike']['mean']:.3f} $\\pm$ {row['spike']['std']:.3f} & "
                f"{row['dropout']['mean']:.3f} $\\pm$ {row['dropout']['std']:.3f} & "
                f"{row['flatline']['mean']:.3f} $\\pm$ {row['flatline']['std']:.3f} \\\\"
            )
        rows.append("\\midrule")
    if rows and rows[-1] == "\\midrule":
        rows.pop()
    content = (
        "% M3 -- per-anomaly-type, per-detector recall breakdown, 5-seed mean +/- std\n"
        "% Requires \\usepackage{multirow} in the paper preamble.\n"
        "\\begin{table}[t]\n\\centering\n\\caption{Recall by anomaly type and detector variant, 5-seed mean "
        "$\\pm$ std. The mechanistic evidence for ensemble stability: the variance detector covers "
        "dropout/flatline, the LSTM-AE covers spike, regardless of DP/federation condition.}\n"
        "\\label{tab:m3-anomaly-breakdown}\n"
        "\\begin{tabular}{@{}llllll@{}}\n\\toprule\n"
        "Regime & Detector & Spike recall & Dropout recall & Flatline recall \\\\\n"
        "\\midrule\n" + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n\\end{table}\n"
    )
    write("anomaly_breakdown.tex", content)


def table_equivalence_test(analysis):
    tost = analysis["equivalence_test_vs_centralized"]
    rows = []
    order = ["Non-DP Federated (K=10)", "DP Federated K=10", "DP Federated K=5", "DP Federated K=3",
             "DP Federated K=1", "Connectivity Good", "Connectivity Moderate", "Connectivity Poor"]
    for name in order:
        r = tost[name]
        verdict = "Equivalent" if r["equivalent_at_alpha_0.05"] else "Not established"
        rows.append(
            f"{name} & {r['mean_diff']:+.4f} & [{r['ci_90'][0]:+.3f}, {r['ci_90'][1]:+.3f}] & "
            f"{r['tost_p']:.4f} & {verdict} \\\\"
        )
    content = (
        "% M3 -- formal TOST equivalence test, margin Delta=0.05 F1, vs. Centralized reference\n"
        "\\begin{table}[t]\n\\centering\n\\caption{TOST equivalence test ($\\Delta=0.05$ F1, supervisor-specified "
        "margin), each regime vs. the centralized reference, 5-seed combined F1. Welch two-sample formulation; "
        "90\\% CI equivalent to two one-sided $\\alpha=0.05$ tests.}\n"
        "\\label{tab:m3-equivalence-test}\n"
        "\\begin{tabular}{@{}lllll@{}}\n\\toprule\n"
        "Regime (vs. Centralized) & Mean diff. & 90\\% CI & TOST $p$ & Verdict \\\\\n"
        "\\midrule\n" + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n\\end{table}\n"
    )
    write("equivalence_test.tex", content)


def table_collapse_evidence():
    d = load("m3_collapse_diagnostic.json")
    rows = []
    for name, r in d["regimes"].items():
        ratio = r["ratio_per_channel"]
        status = "\\textbf{Collapsed}" if r["collapsed"] else "Not collapsed"
        rows.append(f"{name} & {min(ratio):.4f}--{max(ratio):.4f} & {status} \\\\")
    content = (
        "% M3 -- DP-SGD model-level collapse evidence (recon_std / input_std), real checkpoints, seed 42\n"
        "\\begin{table}[t]\n\\centering\n\\caption{Reconstruction std vs.\\ input std, measured directly on "
        "the actual checkpoints behind every reported F1 (pure diagnostic, no new training). DP-SGD collapse "
        "is real and unresolved at every tested $K$; no evidenced code fix exists.}\n"
        "\\label{tab:m3-collapse-evidence}\n"
        "\\begin{tabular}{@{}lll@{}}\n\\toprule\n"
        "Regime & recon\\_std / input\\_std & Status \\\\\n"
        "\\midrule\n" + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n\\end{table}\n"
    )
    write("collapse_evidence.tex", content)


def main():
    analysis = load("m3_multiseed_analysis.json")
    table_privacy_k_sweep()
    table_utility_primary(analysis)
    table_connectivity(analysis)
    table_anomaly_breakdown(analysis)
    table_equivalence_test(analysis)
    table_collapse_evidence()
    print("\nAll LaTeX tables generated from real saved results.")


if __name__ == "__main__":
    main()
