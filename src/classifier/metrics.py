"""Shared multi-label evaluation for CRADLEBench crisis-type models.

Used by both the RoBERTa classifier (train_classifier.py) and the
bag-of-words baseline (baseline.py) so the two are scored identically and are
directly comparable, per project_framework.md.txt's planned "trained
classifier vs. simple baseline" comparison.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import f1_score, precision_recall_fscore_support

from src.classifier.labels import CRISIS_TYPES


def compute_multilabel_metrics(labels: np.ndarray, preds: np.ndarray) -> dict[str, float]:
    """labels/preds: (n_examples, len(CRISIS_TYPES)) 0/1 arrays, same column order."""
    metrics = {
        "f1_micro": float(f1_score(labels, preds, average="micro", zero_division=0)),
        "f1_macro": float(f1_score(labels, preds, average="macro", zero_division=0)),
    }

    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average=None, zero_division=0)
    for crisis_type, p, r, f in zip(CRISIS_TYPES, precision, recall, f1):
        metrics[f"precision_{crisis_type}"] = float(p)
        metrics[f"recall_{crisis_type}"] = float(r)
        metrics[f"f1_{crisis_type}"] = float(f)

    flagged_true = labels.any(axis=1)
    flagged_pred = preds.any(axis=1)
    metrics["flagged_accuracy"] = float((flagged_true == flagged_pred).mean())
    metrics["flagged_f1"] = float(f1_score(flagged_true, flagged_pred, zero_division=0))
    return metrics
