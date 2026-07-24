import unittest

import numpy as np

from src.classifier.labels import CRISIS_TYPES
from src.classifier.metrics import compute_multilabel_metrics


class ComputeMultilabelMetricsTests(unittest.TestCase):
    def test_perfect_predictions_score_one(self) -> None:
        # Every crisis type needs at least one positive example, or macro-F1
        # averages in a 0 for types with no support under zero_division=0.
        labels = np.eye(len(CRISIS_TYPES), dtype=int)
        metrics = compute_multilabel_metrics(labels, labels.copy())
        self.assertEqual(metrics["f1_micro"], 1.0)
        self.assertEqual(metrics["f1_macro"], 1.0)
        self.assertEqual(metrics["flagged_accuracy"], 1.0)
        self.assertEqual(metrics["flagged_f1"], 1.0)

    def test_all_zero_predictions_on_flagged_row_score_zero_recall(self) -> None:
        labels = np.array([[1, 0, 0, 0, 0, 0, 0]])
        preds = np.array([[0, 0, 0, 0, 0, 0, 0]])
        metrics = compute_multilabel_metrics(labels, preds)
        self.assertEqual(metrics[f"recall_{CRISIS_TYPES[0]}"], 0.0)
        self.assertEqual(metrics["flagged_f1"], 0.0)

    def test_multi_label_row_credits_partial_overlap(self) -> None:
        # true: selfharm + rape both present; pred: only selfharm.
        labels = np.array([[1, 0, 0, 0, 1, 0, 0]])
        preds = np.array([[1, 0, 0, 0, 0, 0, 0]])
        metrics = compute_multilabel_metrics(labels, preds)
        self.assertEqual(metrics[f"recall_{CRISIS_TYPES[0]}"], 1.0)
        self.assertEqual(metrics[f"recall_{CRISIS_TYPES[4]}"], 0.0)
        # still "flagged" correctly since at least one true positive
        self.assertEqual(metrics["flagged_accuracy"], 1.0)

    def test_output_covers_every_crisis_type(self) -> None:
        labels = np.zeros((2, len(CRISIS_TYPES)), dtype=int)
        preds = np.zeros((2, len(CRISIS_TYPES)), dtype=int)
        metrics = compute_multilabel_metrics(labels, preds)
        for crisis_type in CRISIS_TYPES:
            self.assertIn(f"precision_{crisis_type}", metrics)
            self.assertIn(f"recall_{crisis_type}", metrics)
            self.assertIn(f"f1_{crisis_type}", metrics)


if __name__ == "__main__":
    unittest.main()
