import unittest

from src.pipeline import append_turn, build_classifier_input, build_context, update_known_events


class UpdateKnownEventsTests(unittest.TestCase):
    def test_appends_new_labels_in_order(self):
        updated = update_known_events(["selfharm"], ["rape", "selfharm"])
        self.assertEqual(updated, ["selfharm", "rape"])

    def test_empty_predicted_types_is_a_no_op(self):
        self.assertEqual(update_known_events(["selfharm"], []), ["selfharm"])


class BuildContextTests(unittest.TestCase):
    def test_flagged_result_becomes_current_signals_and_known_event(self):
        classifier_result = {
            "flagged": True,
            "predicted_types": ["suicideideation_passive"],
            "scores": {"suicideideation_passive": 0.82},
            "num_windows": 1,
        }
        context, known_events = build_context(classifier_result, [])
        self.assertEqual(context["current_signals"], ["suicideideation_passive"])
        self.assertEqual(context["source"], "classifier_prediction")
        self.assertEqual(known_events, ["suicideideation_passive"])

    def test_unflagged_result_falls_back_to_none_but_keeps_history(self):
        classifier_result = {"flagged": False, "predicted_types": [], "scores": {}, "num_windows": 1}
        context, known_events = build_context(classifier_result, ["selfharm"])
        self.assertEqual(context["current_signals"], ["none"])
        self.assertEqual(known_events, ["selfharm"])

    def test_known_events_accumulate_across_turns(self):
        first_result = {"flagged": True, "predicted_types": ["selfharm"], "scores": {}, "num_windows": 1}
        _, known_events = build_context(first_result, [])
        second_result = {"flagged": True, "predicted_types": ["rape"], "scores": {}, "num_windows": 1}
        context, known_events = build_context(second_result, known_events)
        self.assertEqual(known_events, ["selfharm", "rape"])
        self.assertEqual(context["known_events"], ["selfharm", "rape"])


class AppendTurnTests(unittest.TestCase):
    def test_redacts_and_appends(self):
        history = []
        turn = append_turn(history, "user", "reach me at test@example.com")
        self.assertEqual(turn, {"role": "user", "content": "reach me at [EMAIL]"})
        self.assertEqual(history, [turn])


class BuildClassifierInputTests(unittest.TestCase):
    def test_only_user_turns_are_included(self):
        history = [
            {"role": "user", "content": "I've been struggling lately"},
            {"role": "assistant", "content": "I'm here with you, tell me more"},
        ]
        combined = build_classifier_input(history, "it's been rough")
        self.assertNotIn("I'm here with you", combined)
        self.assertIn("I've been struggling lately", combined)
        self.assertIn("it's been rough", combined)

    def test_limits_to_max_context_turns_when_explicitly_capped(self):
        history = [{"role": "user", "content": f"turn {i}"} for i in range(5)]
        combined = build_classifier_input(history, "current", max_context_turns=2)
        self.assertNotIn("turn 0", combined)
        self.assertNotIn("turn 1", combined)
        self.assertNotIn("turn 2", combined)
        self.assertIn("turn 3", combined)
        self.assertIn("turn 4", combined)
        self.assertIn("current", combined)

    def test_default_is_uncapped_and_keeps_an_early_disclosure_in_view(self):
        # Regression test: a capped window previously let an early disclosure
        # scroll out of view after enough later turns, which flipped a
        # still-in-crisis conversation's badge back to "clear."
        history = [{"role": "user", "content": "turn 0 the actual disclosure"}]
        history += [{"role": "user", "content": f"turn {i}"} for i in range(1, 30)]
        combined = build_classifier_input(history, "current")
        self.assertIn("the actual disclosure", combined)

    def test_empty_history_is_just_the_current_text(self):
        self.assertEqual(build_classifier_input([], "hello"), "hello")


if __name__ == "__main__":
    unittest.main()
