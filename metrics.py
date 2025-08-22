# metrics.py
# Common metrics: accuracy, precision/recall/F1 (binary or macro), ROC-AUC
from __future__ import annotations
import numpy as np
from typing import Dict, Tuple, Optional
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score, roc_auc_score, confusion_matrix


def classification_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_prob: Optional[np.ndarray] = None,
                           average: str = "macro") -> Dict[str, float]:
    """
    y_true: [N]
    y_pred: [N] discrete labels
    y_prob: [N, C] class probabilities or scores for AUC (optional)
    """
    res = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, average=average, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, average=average, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, average=average, zero_division=0)),
    }
    if y_prob is not None:
        try:
            if y_prob.ndim == 1 or y_prob.shape[1] == 1:
                # binary
                res["auc"] = float(roc_auc_score(y_true, y_prob))
            else:
                res["auc"] = float(roc_auc_score(y_true, y_prob, multi_class="ovr"))
        except Exception:
            res["auc"] = float("nan")
    return res


def confusion(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    return confusion_matrix(y_true, y_pred)
1111