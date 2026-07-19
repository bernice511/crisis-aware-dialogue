import json
import tempfile
import unittest
from pathlib import Path

from src.llama.preprocess import format_example, preprocess, validate_and_clean


def sample(index: int, intention: str = "Death-Me") -> dict[str, str]:
    return {
        "synthetic_text": f"  Example   narrative {index}.  ",
        "Scenario": "Hopelessness",
        "Negative Core Belief": "Overgeneralization",
        "Intention Category": intention,
    }


class PreprocessTests(unittest.TestCase):
    def test_clean_and_deduplicate(self) -> None:
        cleaned = validate_and_clean([sample(1), sample(1)])
        self.assertEqual(len(cleaned), 1)
        self.assertEqual(cleaned[0]["synthetic_text"], "Example narrative 1.")

    def test_conversational_prompt_completion(self) -> None:
        formatted = format_example(validate_and_clean([sample(1)])[0])
        self.assertEqual(formatted["prompt"][0]["role"], "system")
        self.assertEqual(formatted["completion"][0]["role"], "assistant")
        answer = json.loads(formatted["completion"][0]["content"])
        self.assertEqual(answer["scenario"], "Hopelessness")

    def test_end_to_end_files_and_counts(self) -> None:
        rows = [sample(i, "Death-Me" if i % 2 else "Death-Others") for i in range(20)]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.json"
            source.write_text(json.dumps(rows), encoding="utf-8")
            counts = preprocess(source, root / "out", seed=7)
            self.assertEqual(sum(counts.values()), 20)
            self.assertEqual(counts, {"train": 16, "validation": 2, "test": 2})
            first = json.loads((root / "out/train.jsonl").read_text().splitlines()[0])
            self.assertIn("prompt", first)
            self.assertIn("completion", first)


if __name__ == "__main__":
    unittest.main()

