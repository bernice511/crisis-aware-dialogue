import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from src.classifier.labels import CRISIS_TYPES
from src.classifier.preprocess import preprocess, validate_and_clean


def rows() -> list[dict[str, str]]:
    return [
        {"question_id": 1, "question_text": "  I feel hopeless and want to hurt myself.  ", "final_labels": "selfharm_ongoing"},
        {"question_id": 2, "question_text": "Just asking about therapy options in general.", "final_labels": "No crisis"},
        {"question_id": 3, "question_text": "He hit me again last night and I'm scared.", "final_labels": "domesticviolence_ongoing"},
        {
            "question_id": 4,
            "question_text": "I think about ending things and I've been cutting.",
            "final_labels": "suicideideation(passive)_ongoing, selfharm_ongoing",
        },
    ]


class ValidateAndCleanTests(unittest.TestCase):
    def test_missing_column_raises(self) -> None:
        df = pd.DataFrame([{"question_id": 1, "final_labels": "no_crisis"}])
        with self.assertRaises(ValueError):
            validate_and_clean(df, source=Path("dummy.csv"))

    def test_duplicate_ids_raise(self) -> None:
        df = pd.DataFrame(
            [
                {"question_id": 1, "question_text": "a", "final_labels": "no_crisis"},
                {"question_id": 1, "question_text": "b", "final_labels": "no_crisis"},
            ]
        )
        with self.assertRaises(ValueError):
            validate_and_clean(df, source=Path("dummy.csv"))

    def test_normalizes_whitespace(self) -> None:
        df = pd.DataFrame([{"question_id": 1, "question_text": "  a   b  ", "final_labels": "no_crisis"}])
        cleaned = validate_and_clean(df, source=Path("dummy.csv"))
        self.assertEqual(cleaned["question_text"].iloc[0], "a b")


class PreprocessEndToEndTests(unittest.TestCase):
    def test_writes_expected_files_and_multihot_vectors(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            train_csv = root / "train.csv"
            pd.DataFrame(rows()).to_csv(train_csv, index=False)

            counts = preprocess(
                {"train": train_csv, "validation": train_csv, "test": train_csv},
                root / "out",
            )
            self.assertEqual(counts, {"train": 4, "validation": 4, "test": 4})

            records = [json.loads(line) for line in (root / "out/train.jsonl").read_text().splitlines()]
            self.assertEqual(len(records), 4)

            no_crisis_record = next(r for r in records if r["id"] == 2)
            self.assertEqual(no_crisis_record["labels"], [0] * len(CRISIS_TYPES))
            self.assertFalse(no_crisis_record["flagged"])

            multi_label_record = next(r for r in records if r["id"] == 4)
            self.assertEqual(sorted(multi_label_record["label_names"]), ["selfharm", "suicideideation_passive"])
            self.assertTrue(multi_label_record["flagged"])
            self.assertEqual(multi_label_record["temporal"], {"selfharm": "ongoing", "suicideideation_passive": "ongoing"})

            manifest = json.loads((root / "out/manifest.json").read_text())
            self.assertEqual(manifest["crisis_types"], list(CRISIS_TYPES))
            self.assertEqual(manifest["counts"], counts)

    def test_missing_source_file_raises(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            missing = root / "does_not_exist.csv"
            with self.assertRaises(FileNotFoundError):
                preprocess({"train": missing, "validation": missing, "test": missing}, root / "out")


if __name__ == "__main__":
    unittest.main()
