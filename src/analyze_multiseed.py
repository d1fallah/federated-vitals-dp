"""
M3 — multi-seed analysis: mean+/-std tables, per-anomaly-type breakdown, and a
formal TOST equivalence test, per the supervisor's 20 Aug approval memo
(M3_Supervisor_Final_Approval_Memo_v2.docx, section 4).

Reads only already-saved results/*.json (no new training). Three outputs:
  1. Combined-F1 mean+/-std per regime (extends the earlier 3-seed summary to
     however many seed result files actually exist per regime -- SEEDS below).
  2. Per-anomaly-type (spike/dropout/flatline), per-detector (LSTM-AE-only /
     variance-only / combined) recall, mean+/-std across seeds.
  3. TOST (two one-sided tests) equivalence test, margin Delta=0.05 F1 (the
     supervisor's own pre-specified value, not chosen from the data), each of
     the other 8 regimes vs. Centralized as reference, on combined F1. Uses a
     real Welch (unequal-variance) two-sample formulation -- see tost() below.

Scope note: equivalence test covers all 9 regimes (privacy K-sweep + utility
baselines + connectivity), per the memo's explicit instruction not to limit it
to the K-sweep alone.
"""
import json
import math
import os
import statistics

from scipy import stats

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")
SEEDS = [42, 123, 7, 456, 789]  # extend/shrink to match whichever seed result files actually exist
DELTA = 0.05  # supervisor-specified equivalence margin, F1

REGIMES = {
    "Centralized":              "m3_centralized_baseline{suffix}_results.json",
    "Non-DP Federated (K=10)":  "m3_non_dp_federated{suffix}_results.json",
    "Connectivity Good":        "m3_non_dp_federated_good{suffix}_results.json",
    "Connectivity Moderate":    "m3_non_dp_federated_moderate{suffix}_results.json",
    "Connectivity Poor":        "m3_non_dp_federated_poor{suffix}_results.json",
    "DP Federated K=10":        "m3_dp_federated_k10{suffix}_results.json",
    "DP Federated K=5":         "m3_dp_federated_k5{suffix}_results.json",
    "DP Federated K=3":         "m3_dp_federated_k3{suffix}_results.json",
    "DP Federated K=1":         "m3_dp_federated_k1{suffix}_results.json",
}

# Memo's primary 5-seed table (Centralized, Non-DP FL, DP-FL K=10/5/3/1); the
# connectivity regimes are additional, needed for the equivalence test's
# required all-9-regime scope, and reported as a secondary table.
PRIMARY_TABLE_REGIMES = ["Centralized", "Non-DP Federated (K=10)", "DP Federated K=10",
                          "DP Federated K=5", "DP Federated K=3", "DP Federated K=1"]


def seed_suffix(seed):
    return "" if seed == 42 else f"_seed{seed}"


def load_regime_seeds(filename_template):
    """Returns list of loaded result dicts, one per seed that actually exists on disk."""
    docs = []
    for seed in SEEDS:
        path = os.path.join(RESULTS_DIR, filename_template.format(suffix=seed_suffix(seed)))
        if os.path.exists(path):
            with open(path) as f:
                docs.append((seed, json.load(f)))
    return docs


def mean_std(vals):
    m = statistics.mean(vals)
    s = statistics.stdev(vals) if len(vals) > 1 else 0.0
    return m, s


def welch_tost(sample_a, sample_b, delta, alpha=0.05):
    """Two one-sided tests (TOST) for equivalence of two independent-sample
    means within +/-delta, Welch (unequal-variance) formulation.

    Returns dict with mean_diff, se, df, p_upper, p_lower, tost_p (max of the
    two -- both must be < alpha to conclude equivalence), and the (1-2*alpha)
    CI on the mean difference (numerically equivalent conclusion: reject H0
    of non-equivalence iff this CI is fully inside [-delta, delta]).
    """
    n1, n2 = len(sample_a), len(sample_b)
    m1, m2 = statistics.mean(sample_a), statistics.mean(sample_b)
    v1 = statistics.variance(sample_a) if n1 > 1 else 0.0
    v2 = statistics.variance(sample_b) if n2 > 1 else 0.0
    diff = m1 - m2
    se = math.sqrt(v1 / n1 + v2 / n2) if (v1 > 0 or v2 > 0) else 1e-12
    if v1 == 0 and v2 == 0:
        df = n1 + n2 - 2
    else:
        df = (v1 / n1 + v2 / n2) ** 2 / (
            (v1 / n1) ** 2 / (n1 - 1) + (v2 / n2) ** 2 / (n2 - 1)
        )

    t_upper = (diff - delta) / se   # H0: diff >= delta ; reject (p small) supports diff < delta
    t_lower = (diff + delta) / se   # H0: diff <= -delta; reject (p small) supports diff > -delta
    p_upper = stats.t.cdf(t_upper, df)
    p_lower = 1 - stats.t.cdf(t_lower, df)
    tost_p = max(p_upper, p_lower)

    t_crit = stats.t.ppf(1 - alpha, df)
    ci_low, ci_high = diff - t_crit * se, diff + t_crit * se

    return {
        "mean_diff": round(diff, 5),
        "se": round(se, 5),
        "df": round(df, 2),
        "p_upper": round(p_upper, 5),
        "p_lower": round(p_lower, 5),
        "tost_p": round(tost_p, 5),
        "ci_90": [round(ci_low, 5), round(ci_high, 5)],
        "equivalent_at_alpha_0.05": bool(tost_p < alpha),
    }


def main():
    print(f"Seeds considered: {SEEDS}   Equivalence margin Delta={DELTA} F1\n")

    # --- 1. Full metric set (AUROC, LSTM-only, variance-only, combined) mean+/-std per regime ---
    combined_f1_by_regime = {}
    full_metrics_by_regime = {}
    print(f"{'Regime':28s} {'n_seeds':8s} {'combined F1 (mean +/- std)':30s} {'per-seed values'}")
    for name, template in REGIMES.items():
        docs = load_regime_seeds(template)
        f1s = [d["combined_with_variance_detector"]["f1"] for _, d in docs]
        combined_f1_by_regime[name] = f1s
        m, s = mean_std(f1s)
        print(f"{name:28s} {len(f1s):<8d} {m:.4f} +/- {s:.4f}{'':10s} {f1s}")

        def stat(vals):
            mm, ss = mean_std(vals)
            return {"mean": round(mm, 4), "std": round(ss, 4), "values": [round(v, 4) for v in vals]}

        full_metrics_by_regime[name] = {
            "n_seeds": len(docs),
            "auroc": stat([d["auroc_lstm_ae_only"] for _, d in docs]),
            "lstm_ae_only_f1": stat([d["lstm_ae_only"]["f1"] for _, d in docs]),
            "lstm_ae_only_precision": stat([d["lstm_ae_only"]["precision"] for _, d in docs]),
            "lstm_ae_only_recall": stat([d["lstm_ae_only"]["recall"] for _, d in docs]),
            "variance_only_f1": stat([d["variance_detector_only"]["f1"] for _, d in docs]),
            "combined_f1": stat(f1s),
            "combined_precision": stat([d["combined_with_variance_detector"]["precision"] for _, d in docs]),
            "combined_recall": stat([d["combined_with_variance_detector"]["recall"] for _, d in docs]),
        }

    # --- 2. Per-anomaly-type, per-detector breakdown, mean+/-std across seeds ---
    print("\n\n=== Per-anomaly-type, per-detector recall (mean +/- std across seeds) ===")
    detector_keys = {"LSTM-AE-only": "lstm_ae_only", "Variance-only": "variance_detector_only", "Combined": "combined_with_variance_detector"}
    anomaly_breakdown = {}
    for name, template in REGIMES.items():
        docs = load_regime_seeds(template)
        anomaly_breakdown[name] = {}
        print(f"\n{name}:")
        for det_label, det_key in detector_keys.items():
            row = {}
            for kind in ("spike", "dropout", "flatline"):
                vals = []
                for _, d in docs:
                    breakdown = d[det_key]["per_anomaly_type_recall"]
                    if kind in breakdown:
                        vals.append(breakdown[kind]["recall"])
                if vals:
                    m, s = mean_std(vals)
                    row[kind] = {"mean": round(m, 4), "std": round(s, 4), "n": len(vals)}
            anomaly_breakdown[name][det_label] = row
            print(f"  {det_label:16s} " + "  ".join(f"{k}={v['mean']:.3f}+/-{v['std']:.3f}" for k, v in row.items()))

    # --- 3. TOST equivalence test, all 8 non-reference regimes vs Centralized ---
    print(f"\n\n=== TOST equivalence test (margin Delta={DELTA} F1), each regime vs. Centralized ===")
    reference = combined_f1_by_regime["Centralized"]
    tost_results = {}
    for name, f1s in combined_f1_by_regime.items():
        if name == "Centralized":
            continue
        result = welch_tost(f1s, reference, DELTA)
        tost_results[name] = result
        verdict = "EQUIVALENT" if result["equivalent_at_alpha_0.05"] else "NOT ESTABLISHED"
        print(f"{name:28s} diff={result['mean_diff']:+.4f}  90% CI=[{result['ci_90'][0]:+.4f}, {result['ci_90'][1]:+.4f}]  "
              f"TOST p={result['tost_p']:.4f}  -> {verdict}")

    # --- Save everything ---
    out = {
        "seeds_considered": SEEDS,
        "delta_margin": DELTA,
        "full_metrics_by_regime": full_metrics_by_regime,
        "combined_f1_by_regime": {k: {"f1_values": v, "mean": round(statistics.mean(v), 4),
                                       "std": round(statistics.stdev(v), 4) if len(v) > 1 else 0.0,
                                       "n_seeds": len(v)} for k, v in combined_f1_by_regime.items()},
        "anomaly_type_breakdown": anomaly_breakdown,
        "equivalence_test_vs_centralized": tost_results,
        "primary_table_regimes": PRIMARY_TABLE_REGIMES,
    }
    out_path = os.path.join(RESULTS_DIR, "m3_multiseed_analysis.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nFull analysis saved to {out_path}")


if __name__ == "__main__":
    main()
