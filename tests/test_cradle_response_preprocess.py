import unittest

from src.cradle_response.inference import build_messages, build_risk_context
from src.cradle_response.preprocess import (
    build_response_examples,
    collapse_continuations,
    ensure_dialogue_ids,
    redact_pii,
)


ROWS = [
    {"dialogue_id": "d1", "turn_id": 0, "text": "User: I feel unsafe. mail me at a@b.com", "labels": "alert_ongoing"},
    {"dialogue_id": "d1", "turn_id": 1, "text": "Listener: I hear how difficult this is. Are you in immediate danger?", "labels": "none"},
    {"dialogue_id": "d1", "turn_id": 2, "text": "User: I have been thinking about suicide.", "labels": "confirm_suicide_ongoing"},
    {"dialogue_id": "d1", "turn_id": 3, "text": "Listener: Thank you for telling me. Can you move away from anything you could use to hurt yourself?", "labels": "none"},
]


class CradleResponsePreprocessTests(unittest.TestCase):
    def test_redacts_common_contact_information(self):
        cleaned = redact_pii("Email me at test@example.com or call 617-555-0123")
        self.assertEqual(cleaned, "Email me at [EMAIL] or call [PHONE]")

    def test_builds_listener_targets_with_cumulative_risk(self):
        examples = build_response_examples(ROWS)
        self.assertEqual(len(examples), 2)
        first_context = examples[0]["metadata"]["risk_context"]
        second_context = examples[1]["metadata"]["risk_context"]
        self.assertEqual(first_context["current_signals"], ["alert_ongoing"])
        self.assertEqual(
            second_context["known_events"],
            ["alert_ongoing", "confirm_suicide_ongoing"],
        )
        self.assertEqual(examples[0]["prompt"][-1]["role"], "user")
        self.assertEqual(examples[1]["completion"][0]["role"], "assistant")
        self.assertNotIn("a@b.com", examples[0]["prompt"][-1]["content"])

    def test_short_history_still_starts_with_user(self):
        example = build_response_examples(ROWS, max_history_turns=2)[1]
        roles = [message["role"] for message in example["prompt"]]
        self.assertEqual(roles, ["system", "user"])

    def test_classifier_interface_builds_generation_prompt(self):
        context = build_risk_context(["alert_ongoing"], ["alert_ongoing"])
        messages = build_messages([], "I do not know what to do.", context, 14)
        self.assertEqual(context["source"], "classifier_prediction")
        self.assertEqual(messages[-1]["role"], "user")
        self.assertIn("alert_ongoing", messages[0]["content"])

    def test_reconstructs_missing_hub_dialogue_ids(self):
        rows = [
            {"turn_id": 0, "text": "User: First", "labels": ""},
            {"turn_id": 1, "text": "Listener: Reply", "labels": ""},
            {"turn_id": 0, "text": "User: Second", "labels": ""},
        ]
        restored = ensure_dialogue_ids(rows, split_name="train")
        self.assertEqual(restored[0]["dialogue_id"], "train-000000")
        self.assertEqual(restored[1]["dialogue_id"], "train-000000")
        self.assertEqual(restored[2]["dialogue_id"], "train-000001")

    def test_merges_unprefixed_response_continuations(self):
        rows = [
            {"turn_id": 7, "text": "Listener: Here are some steps:", "labels": ""},
            {"turn_id": 8, "text": "- First step", "labels": ""},
            {"turn_id": 9, "text": "- Second step", "labels": ""},
        ]
        collapsed = collapse_continuations(rows)
        self.assertEqual(len(collapsed), 1)
        self.assertIn("First step", collapsed[0]["text"])
        self.assertIn("Second step", collapsed[0]["text"])

    def test_skips_orphan_opening_listener(self):
        rows = [
            {"dialogue_id": "d2", "turn_id": 0, "text": "Listener: What brings you in?", "labels": ""},
            {"dialogue_id": "d2", "turn_id": 1, "text": "User: I need some help.", "labels": ""},
            {"dialogue_id": "d2", "turn_id": 2, "text": "Listener: I am listening.", "labels": ""},
        ]
        examples = build_response_examples(rows)
        self.assertEqual(len(examples), 1)
        self.assertEqual(examples[0]["metadata"]["target_turn_id"], 2)


if __name__ == "__main__":
    unittest.main()
