"""Tests for frozen prediction records and error summaries."""

import unittest

from freeze_primary import (
    add_transitions,
    prediction_entry,
    reproduced_changed_set,
    summarize_predictions,
    transition_entry,
)


class FreezePrimaryTests(unittest.TestCase):
    def test_transition_classifies_correction_and_introduced_error(self):
        wrong = prediction_entry(-0.5, 0.0, True)
        corrected = prediction_entry(0.5, 0.0, True)
        self.assertEqual(
            transition_entry(wrong, corrected),
            {
                "changed": True,
                "outcome": "corrected",
                "direction": "no-clickbait -> clickbait",
                "score_delta": 1.0,
            },
        )
        self.assertEqual(
            transition_entry(corrected, wrong)["outcome"], "introduced"
        )
        self.assertEqual(
            transition_entry(corrected, corrected)["outcome"], "unchanged"
        )

    def test_summary_and_change_set_cover_every_condition(self):
        rows = []
        for index, truth in enumerate((False, True)):
            row = {
                "id": str(index),
                "group_id": f"g{index}",
                "truth_class": "clickbait" if truth else "no-clickbait",
                "models": {},
            }
            for model_name in ("logistic_regression", "linear_svm"):
                row["models"][f"A/{model_name}"] = prediction_entry(
                    -1.0, 0.0, truth
                )
                row["models"][f"B/{model_name}"] = prediction_entry(
                    1.0, 0.0, truth
                )
                row["models"][f"C/{model_name}"] = prediction_entry(
                    1.0 if truth else -1.0, 0.0, truth
                )
            rows.append(row)
        add_transitions(rows)
        summary = summarize_predictions(rows)
        self.assertEqual(
            summary["models"]["C/logistic_regression"]["correct"], 2
        )
        changes = reproduced_changed_set(rows)
        self.assertIn(
            ("0", "logistic_regression", "B-A", "introduced"), changes
        )
        self.assertIn(
            ("1", "linear_svm", "C-A", "corrected"), changes
        )


if __name__ == "__main__":
    unittest.main()
