"""Paired DeLong + paired bootstrap: EchoNext S6 vs MIMIC S10 on the same 2022 patients."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.evaluation.metrics import metrics_from_predictions

N_BOOT = 2000
SEED = 42
THRESHOLD = 0.5
METRIC_KEYS = ("auroc", "auprc", "accuracy", "f1", "sensitivity", "specificity")


def extract(metrics: dict) -> dict[str, float]:
    return {
        "auroc": float(metrics["auroc"]),
        "auprc": float(metrics["auprc"]),
        "accuracy": float(metrics["accuracy"]),
        "f1": float(metrics["f1"]),
        "sensitivity": float(metrics["recall"]),
        "specificity": float(metrics["specificity"]),
    }


def compute_midrank(x: np.ndarray) -> np.ndarray:
    order = np.argsort(x)
    sorted_x = x[order]
    n = len(x)
    ranks = np.empty(n, dtype=np.float64)
    i = 0
    while i < n:
        j = i
        while j < n and sorted_x[j] == sorted_x[i]:
            j += 1
        ranks[i:j] = 0.5 * (i + j - 1)
        i = j
    out = np.empty(n, dtype=np.float64)
    out[order] = ranks + 1.0
    return out


def fast_delong(predictions_sorted_transposed: np.ndarray, n_pos: int) -> tuple[np.ndarray, np.ndarray]:
    m = n_pos
    n = predictions_sorted_transposed.shape[1] - m
    positives = predictions_sorted_transposed[:, :m]
    negatives = predictions_sorted_transposed[:, m:]
    k = predictions_sorted_transposed.shape[0]
    tx = np.empty((k, m), dtype=np.float64)
    ty = np.empty((k, n), dtype=np.float64)
    tz = np.empty((k, m + n), dtype=np.float64)
    for r in range(k):
        tx[r] = compute_midrank(positives[r])
        ty[r] = compute_midrank(negatives[r])
        tz[r] = compute_midrank(predictions_sorted_transposed[r])
    aucs = tz[:, :m].sum(axis=1) / m / n - float(m + 1.0) / 2.0 / n
    v01 = (tz[:, :m] - tx) / n
    v10 = 1.0 - (tz[:, m:] - ty) / m
    sx = np.cov(v01)
    sy = np.cov(v10)
    cov = sx / m + sy / n
    return aucs, cov


def paired_delong(y_true: np.ndarray, scores_a: np.ndarray, scores_b: np.ndarray) -> dict:
    y_true = np.asarray(y_true, dtype=np.int64)
    order = np.argsort(-y_true)
    y_sorted = y_true[order]
    n_pos = int(y_sorted.sum())
    stacked = np.vstack((scores_a[order], scores_b[order]))
    aucs, cov = fast_delong(stacked, n_pos)
    diff = float(aucs[0] - aucs[1])
    var = float((np.array([[1.0, -1.0]]) @ cov @ np.array([[1.0], [-1.0]])).item())
    se = float(np.sqrt(max(var, 0.0)))
    z = diff / se if se > 0 else np.inf
    p = float(2.0 * norm.sf(abs(z)))
    ci_lo = diff - 1.96 * se
    ci_hi = diff + 1.96 * se
    sklearn_a = float(roc_auc_score(y_true, scores_a))
    sklearn_b = float(roc_auc_score(y_true, scores_b))
    return {
        "method": "paired DeLong (Sun & Xu / structural components)",
        "auroc_s6_delong": float(aucs[0]),
        "auroc_s10_delong": float(aucs[1]),
        "auroc_s6_sklearn": sklearn_a,
        "auroc_s10_sklearn": sklearn_b,
        "difference_s6_minus_s10": diff,
        "variance": var,
        "std_error": se,
        "z_statistic": float(z),
        "wald_ci_lower_95": float(ci_lo),
        "wald_ci_upper_95": float(ci_hi),
        "p_value_two_sided": p,
        "covariance_matrix": np.asarray(cov, dtype=float).tolist(),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred-s6", required=True)
    parser.add_argument("--pred-s10", required=True)
    parser.add_argument("--output-dir", default="outputs/paired_2022")
    parser.add_argument("--expected-n", type=int, default=0, help="Optional exact paired count check.")
    args = parser.parse_args()
    out_dir = Path(args.output_dir)
    s10 = pd.read_csv(args.pred_s10)
    s6 = pd.read_csv(args.pred_s6)
    s10["patient_id"] = s10["patient_id"].astype(str)
    s6["patient_id"] = s6["patient_id"].astype(str)
    checks = {
        "s10_n": int(len(s10)),
        "s6_n": int(len(s6)),
        "s10_unique_patients": int(s10["patient_id"].nunique()),
        "s6_unique_patients": int(s6["patient_id"].nunique()),
        "s10_duplicate_patients": bool(s10["patient_id"].duplicated().any()),
        "s6_duplicate_patients": bool(s6["patient_id"].duplicated().any()),
        "patient_set_equal": set(s10["patient_id"]) == set(s6["patient_id"]),
    }
    merged = s6.merge(s10, on="patient_id", suffixes=("_s6", "_s10"), how="inner", validate="one_to_one")
    checks["inner_join_n"] = int(len(merged))
    checks["true_label_all_equal"] = bool((merged["true_label_s6"] == merged["true_label_s10"]).all())
    checks["n_case"] = int((merged["true_label_s6"] == 1).sum())
    checks["n_control"] = int((merged["true_label_s6"] == 0).sum())
    if not checks["patient_set_equal"] or not checks["true_label_all_equal"]:
        raise RuntimeError(f"Patient alignment failed: {checks}")
    if args.expected_n and len(merged) != int(args.expected_n):
        raise RuntimeError(f"expected {args.expected_n} paired patients, got {len(merged)}")

    y = merged["true_label_s6"].to_numpy(dtype=np.int64)
    p6 = merged["predicted_probability_s6"].to_numpy(dtype=np.float64)
    p10 = merged["predicted_probability_s10"].to_numpy(dtype=np.float64)
    point_s6 = extract(metrics_from_predictions(y, p6, THRESHOLD))
    point_s10 = extract(metrics_from_predictions(y, p10, THRESHOLD))
    delong = paired_delong(y, p6, p10)

    rng = np.random.RandomState(SEED)
    n = len(y)
    boot = {key: [] for key in METRIC_KEYS}
    skipped = 0
    while len(boot["auprc"]) < N_BOOT:
        idx = rng.randint(0, n, size=n)
        y_b = y[idx]
        if np.unique(y_b).size < 2:
            skipped += 1
            continue
        m6 = extract(metrics_from_predictions(y_b, p6[idx], THRESHOLD))
        m10 = extract(metrics_from_predictions(y_b, p10[idx], THRESHOLD))
        if not all(np.isfinite(m6[k]) and np.isfinite(m10[k]) for k in METRIC_KEYS):
            skipped += 1
            continue
        for key in METRIC_KEYS:
            boot[key].append(m6[key] - m10[key])

    bootstrap_rows = []
    for key in METRIC_KEYS:
        deltas = np.asarray(boot[key], dtype=np.float64)
        obs = point_s6[key] - point_s10[key]
        lo, hi = np.percentile(deltas, [2.5, 97.5])
        n_boot = len(deltas)
        if obs >= 0:
            tail = int(np.sum(deltas <= 0))
        else:
            tail = int(np.sum(deltas >= 0))
        p_boot = min(1.0, 2.0 * (tail + 1.0) / (n_boot + 1.0))
        bootstrap_rows.append(
            {
                "metric": key,
                "mimic_s10": point_s10[key],
                "echonext_s6": point_s6[key],
                "difference_s6_minus_s10": obs,
                "bootstrap_ci_lower_95": float(lo),
                "bootstrap_ci_upper_95": float(hi),
                "bootstrap_p_two_sided": float(p_boot),
                "n_valid_bootstrap": int(len(deltas)),
                "n_skipped": int(skipped),
                "method": "paired patient-level bootstrap (same indices for both models)",
            }
        )

    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(bootstrap_rows).to_csv(out_dir / "paired_bootstrap_differences.csv", index=False)
    with (out_dir / "delong_auroc.json").open("w", encoding="utf-8") as file:
        json.dump({"alignment": checks, "delong": delong}, file, indent=2)
    with (out_dir / "alignment_check.json").open("w", encoding="utf-8") as file:
        json.dump(checks, file, indent=2)

    table_rows = []
    for row in bootstrap_rows:
        key = row["metric"]
        item = {
            "metric": key,
            "mimic_s10": row["mimic_s10"],
            "echonext_s6": row["echonext_s6"],
            "difference_s6_minus_s10": row["difference_s6_minus_s10"],
            "ci_lower_95": row["bootstrap_ci_lower_95"] if key != "auroc" else delong["wald_ci_lower_95"],
            "ci_upper_95": row["bootstrap_ci_upper_95"] if key != "auroc" else delong["wald_ci_upper_95"],
            "p_value": row["bootstrap_p_two_sided"] if key != "auroc" else delong["p_value_two_sided"],
            "test": "paired DeLong (Wald 95% CI on delta)" if key == "auroc" else "paired bootstrap",
        }
        if key == "auroc":
            item["delong_z"] = delong["z_statistic"]
            item["delong_variance"] = delong["variance"]
            item["delong_difference"] = delong["difference_s6_minus_s10"]
        table_rows.append(item)
    pd.DataFrame(table_rows).to_csv(out_dir / "paired_comparison_summary.csv", index=False)
    print(json.dumps({"alignment": checks, "delong": delong}, indent=2, default=str))
    print(pd.DataFrame(table_rows).to_string(index=False))


if __name__ == "__main__":
    main()
