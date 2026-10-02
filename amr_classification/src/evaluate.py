"""
evaluate.py — Evaluation utilities for the AMR classification pipeline.
"""

import numpy as np
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    precision_recall_curve,
)
from sklearn.inspection import permutation_importance


def tune_classification_threshold(y_val: np.ndarray,
                                   y_val_proba: np.ndarray) -> float:
    """
    Find the threshold that maximises F1 on the inner validation set.
    Threshold is selected entirely within the inner fold to prevent
    threshold leakage into outer-fold evaluation.
    """
    precisions, recalls, thresholds = precision_recall_curve(y_val, y_val_proba)

    with np.errstate(divide="ignore", invalid="ignore"):
        f1_scores = (
            2 * (precisions[:-1] * recalls[:-1])
            / (precisions[:-1] + recalls[:-1])
        )
        f1_scores = np.nan_to_num(f1_scores)

    best_idx = int(np.argmax(f1_scores))
    return float(thresholds[best_idx]) if best_idx < len(thresholds) else 0.5


def calculate_metrics(y_true: np.ndarray,
                       y_proba: np.ndarray,
                       threshold: float) -> dict:
    """
    Compute AUROC, PR-AUC, and threshold-dependent metrics
    (F1, Precision, Recall) on the outer test fold.
    """
    y_pred = (y_proba >= threshold).astype(int)
    return {
        "AUROC":     roc_auc_score(y_true, y_proba),
        "PR-AUC":    average_precision_score(y_true, y_proba),
        "F1":        f1_score(y_true, y_pred, zero_division=0),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall":    recall_score(y_true, y_pred, zero_division=0),
        "Threshold": threshold,
    }


def get_permutation_importance(model, X_test, y_test) -> np.ndarray:
    """
    Compute permutation importance on the held-out test set using
    average precision (PR-AUC) as the scoring metric.
    Averaged over 10 repeats to reduce variance.
    """
    result = permutation_importance(
        model, X_test, y_test,
        n_repeats=10,
        random_state=42,
        scoring="average_precision",
    )
    return result.importances_mean
