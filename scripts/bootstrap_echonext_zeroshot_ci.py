"""Patient-level bootstrap 95% CI on frozen EchoNext zero-shot predictions."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.evaluation.metrics import metrics_from_predictions

N_VALID = 2000
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--output-dir", default="outputs/mimic/echonext_zeroshot")
    args = parser.parse_args()
    pred_path = Path(args.predictions)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    frame = pd.read_csv(pred_path)
    if frame["patient_id"].duplicated().any():
        groups = {
            pid: sub.reset_index(drop=True)
            for pid, sub in frame.groupby("patient_id", sort=False)
        }
        patient_ids = np.array(list(groups.keys()), dtype=object)
        use_groups = True
    else:
        y_true_all = frame["true_label"].to_numpy(dtype=np.int64)
        y_prob_all = frame["predicted_probability"].to_numpy(dtype=np.float64)
        patient_ids = frame["patient_id"].to_numpy()
        use_groups = False

    if use_groups:
        y_true_all = frame["true_label"].to_numpy(dtype=np.int64)
        y_prob_all = frame["predicted_probability"].to_numpy(dtype=np.float64)

    point = extract(metrics_from_predictions(y_true_all, y_prob_all, threshold=THRESHOLD))
    rng = np.random.RandomState(SEED)
    n_patients = len(patient_ids)
    collected = {key: [] for key in METRIC_KEYS}
    skipped = 0
    attempts = 0
    while len(collected["auroc"]) < N_VALID:
        attempts += 1
        idx = rng.randint(0, n_patients, size=n_patients)
        if use_groups:
            parts = [groups[patient_ids[i]] for i in idx]
            boot = pd.concat(parts, ignore_index=True)
            y_true = boot["true_label"].to_numpy(dtype=np.int64)
            y_prob = boot["predicted_probability"].to_numpy(dtype=np.float64)
        else:
            y_true = y_true_all[idx]
            y_prob = y_prob_all[idx]
        if np.unique(y_true).size < 2:
            skipped += 1
            continue
        metrics = extract(metrics_from_predictions(y_true, y_prob, threshold=THRESHOLD))
        if not all(np.isfinite(metrics[key]) for key in METRIC_KEYS):
            skipped += 1
            continue
        for key in METRIC_KEYS:
            collected[key].append(metrics[key])

    rows = []
    for key in METRIC_KEYS:
        values = np.asarray(collected[key], dtype=np.float64)
        lo, hi = np.percentile(values, [2.5, 97.5])
        rows.append(
            {
                "metric": key,
                "point_estimate": point[key],
                "ci_lower_95": float(lo),
                "ci_upper_95": float(hi),
                "bootstrap_mean": float(values.mean()),
                "n_valid_bootstrap": int(len(values)),
                "n_skipped_single_class_or_nan": int(skipped),
                "n_draw_attempts": int(attempts),
                "seed": SEED,
                "threshold": THRESHOLD,
                "bootstrap_unit": "patient_id",
                "n_patients": int(n_patients),
                "n_rows": int(len(frame)),
            }
        )
    out_csv = out_dir / "echonext_s10_zeroshot_metrics_with_ci.csv"
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    summary = {
        "n_valid_bootstrap": N_VALID,
        "n_skipped_single_class_or_nan": skipped,
        "n_draw_attempts": attempts,
        "seed": SEED,
        "threshold": THRESHOLD,
        "prediction_file": str(pred_path),
        "metrics": {row["metric"]: row for row in rows},
    }
    with (out_dir / "echonext_s10_zeroshot_metrics_with_ci.json").open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2)
    print(json.dumps(summary, indent=2))
    print(f"[csv] {out_csv}")


if __name__ == "__main__":
    main()
