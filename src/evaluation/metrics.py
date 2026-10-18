from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    recall_score,
    roc_auc_score,
)


def metrics_from_predictions(
    y_true: np.ndarray,
    y_probability: np.ndarray,
    threshold: float = 0.5,
) -> dict[str, Any]:
    y_true = np.asarray(y_true, dtype=np.int64)
    y_probability = np.asarray(y_probability, dtype=np.float64)
    y_pred = (y_probability >= threshold).astype(np.int64)
    matrix = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = matrix.ravel()
    try:
        auroc = float(roc_auc_score(y_true, y_probability))
        auprc = float(average_precision_score(y_true, y_probability))
    except ValueError:
        auroc = float("nan")
        auprc = float("nan")
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "specificity": float(tn / max(tn + fp, 1)),
        "auroc": auroc,
        "auprc": auprc,
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "n": int(len(y_true)),
    }
