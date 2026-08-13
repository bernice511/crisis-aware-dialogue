import unittest

from src.classifier.labels import (
    CRISIS_TYPES,
    is_flagged,
    normalize_label,
    parse_labels,
    split_type_temporal,
    temporal_by_type,
    to_multihot,
)


class NormalizeLabelTests(unittest.TestCase):
    def test_no_crisis_variants(self) -> None:
        self.assertEqual(normalize_label("No crisis"), "no_crisis")
        self.assertEqual(normalize_label("no_crisis"), "no_crisis")

    def test_suicideideation_parenthetical_variants(self) -> None:
        self.assertEqual(
            normalize_label("suicideideation(passive)_ongoing"),
            "suicideideation_passive_ongoing",
        )
        self.assertEqual(
            normalize_label("suicideideation(active)_past"),
            "suicideideation_active_past",
        )

    def test_childabuseendangerment_spelling(self) -> None:
        self.assertEqual(
            normalize_label("childabuseendangerment_past"),
            "childabuse_endangerment_past",
        )


class ParseLabelsTests(unittest.TestCase):
    def test_splits_and_normalizes_comma_separated(self) -> None:
        self.assertEqual(
            parse_labels("suicideideation_passive_ongoing, selfharm_ongoing"),
            ["suicideideation_passive_ongoing", "selfharm_ongoing"],
        )

    def test_empty_and_missing(self) -> None:
        self.assertEqual(parse_labels(""), [])
        self.assertEqual(parse_labels(None), [])
        self.assertEqual(parse_labels(float("nan")), [])


class SplitTypeTemporalTests(unittest.TestCase):
    def test_ongoing_and_past(self) -> None:
        self.assertEqual(split_type_temporal("domesticviolence_ongoing"), ("domesticviolence", "ongoing"))
        self.assertEqual(split_type_temporal("rape_past"), ("rape", "past"))

    def test_no_crisis(self) -> None:
        self.assertEqual(split_type_temporal("no_crisis"), ("no_crisis", "none"))


class MultihotTests(unittest.TestCase):
    def test_no_crisis_is_all_zero(self) -> None:
        vector = to_multihot(["no_crisis"])
        self.assertEqual(vector, [0] * len(CRISIS_TYPES))
        self.assertFalse(is_flagged(vector))

    def test_single_type_sets_one_index(self) -> None:
        vector = to_multihot(["selfharm_ongoing"])
        self.assertEqual(sum(vector), 1)
        self.assertEqual(vector[CRISIS_TYPES.index("selfharm")], 1)
        self.assertTrue(is_flagged(vector))

    def test_multi_label_row_sets_both_indices(self) -> None:
        vector = to_multihot(["suicideideation_passive_ongoing", "selfharm_ongoing"])
        self.assertEqual(vector[CRISIS_TYPES.index("suicideideation_passive")], 1)
        self.assertEqual(vector[CRISIS_TYPES.index("selfharm")], 1)
        self.assertEqual(sum(vector), 2)

    def test_temporal_ignores_but_records_by_type(self) -> None:
        temporal = temporal_by_type(["domesticviolence_ongoing", "rape_past"])
        self.assertEqual(temporal, {"domesticviolence": "ongoing", "rape": "past"})


if __name__ == "__main__":
    unittest.main()
