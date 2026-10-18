"""Paired DeLong and patient-level bootstrap as used for EchoNext 2022 comparisons."""
from __future__ import annotations

from typing import Any

import numpy as np
from scipy.stats import norm
from sklearn.metrics import roc_auc_score

from .metrics import metrics_from_predictions

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
    cov = np.cov(v01) / m + np.cov(v10) / n
    return aucs, cov


def paired_delong(y_true: np.ndarray, scores_a: np.ndarray, scores_b: np.ndarray) -> dict[str, Any]:
    y_true = np.asarray(y_true, dtype=np.int64)
    order = np.argsort(-y_true)
    n_pos = int(y_true[order].sum())
    stacked = np.vstack((scores_a[order], scores_b[order]))
    aucs, cov = fast_delong(stacked, n_pos)
    diff = float(aucs[0] - aucs[1])
    var = float((np.array([[1.0, -1.0]]) @ cov @ np.array([[1.0], [-1.0]])).item())
    se = float(np.sqrt(max(var, 0.0)))
    z = diff / se if se > 0 else np.inf
    return {
        "method": "paired DeLong (Sun & Xu / structural components)",
        "auroc_a": float(aucs[0]),
        "auroc_b": float(aucs[1]),
        "sklearn_a": float(roc_auc_score(y_true, scores_a)),
        "sklearn_b": float(roc_auc_score(y_true, scores_b)),
        "difference_a_minus_b": diff,
        "std_error": se,
        "z_statistic": float(z),
        "wald_ci_lower_95": diff - 1.96 * se,
        "wald_ci_upper_95": diff + 1.96 * se,
        "p_value_two_sided": float(2.0 * norm.sf(abs(z))),
    }


def paired_bootstrap(
    y: np.ndarray,
    scores_a: np.ndarray,
    scores_b: np.ndarray,
    n_boot: int = 2000,
    seed: int = 42,
    threshold: float = 0.5,
) -> list[dict[str, Any]]:
    """Shared patient indices. Two-sided p = min(1, 2*(1+#opposite)/(B+1))."""
    rng = np.random.RandomState(seed)
    n = len(y)
    point_a = extract(metrics_from_predictions(y, scores_a, threshold))
    point_b = extract(metrics_from_predictions(y, scores_b, threshold))
    boot = {key: [] for key in METRIC_KEYS}
    while len(boot["auprc"]) < n_boot:
        idx = rng.randint(0, n, size=n)
        y_b = y[idx]
        if np.unique(y_b).size < 2:
            continue
        ma = extract(metrics_from_predictions(y_b, scores_a[idx], threshold))
        mb = extract(metrics_from_predictions(y_b, scores_b[idx], threshold))
        for key in METRIC_KEYS:
            boot[key].append(ma[key] - mb[key])
    rows = []
    for key in METRIC_KEYS:
        deltas = np.asarray(boot[key], dtype=np.float64)
        obs = point_a[key] - point_b[key]
        lo, hi = np.percentile(deltas, [2.5, 97.5])
        tail = int(np.sum(deltas <= 0)) if obs >= 0 else int(np.sum(deltas >= 0))
        p = min(1.0, 2.0 * (tail + 1.0) / (len(deltas) + 1.0))
        rows.append(
            {
                "metric": key,
                "model_a": point_a[key],
                "model_b": point_b[key],
                "difference_a_minus_b": obs,
                "bootstrap_ci_lower_95": float(lo),
                "bootstrap_ci_upper_95": float(hi),
                "bootstrap_p_two_sided": float(p),
                "n_valid_bootstrap": int(len(deltas)),
            }
        )
    return rows
