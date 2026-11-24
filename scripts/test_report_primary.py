"""Tests for formal result tables, figures, and timing summaries."""

import unittest

import numpy as np
from scipy.sparse import csr_matrix

from report_primary import (
    REPORT_FILENAMES,
    build_result_rows,
    macro_f1_svg,
    result_markdown,
    timing_stats,
    transform_condition,
)
from tfidf_preexperiment import CONDITIONS, MODELS


class ReportPrimaryTests(unittest.TestCase):
    def metrics(self):
        values = {}
        for condition_index, condition in enumerate(CONDITIONS):
            for model_index, model_name in enumerate(MODELS):
                score = 0.70 + condition_index * 0.02 + model_index * 0.01
                values[f"{condition}/{model_name}"] = {
                    "C": 1.0,
                    "threshold": -0.5,
                    "validation_macro_f1": score - 0.01,
                    "macro_f1": score,
                    "clickbait_precision": 0.6,
                    "clickbait_recall": 0.5,
                    "clickbait_f1": 0.55,
                    "average_precision": 0.65,
                    "confusion_matrix": [[10, 2], [3, 5]],
                }
        return values

    def test_result_rows_cover_six_models_and_confusion_counts(self):
        rows = build_result_rows(self.metrics())
        self.assertEqual(len(rows), 6)
        self.assertEqual(rows[0]["condition"], "A")
        self.assertEqual(rows[0]["false_positive"], 2)
        self.assertEqual(rows[-1]["false_negative"], 3)

    def test_markdown_and_svg_include_exact_scores(self):
        rows = build_result_rows(self.metrics())
        markdown = result_markdown(rows)
        figure = macro_f1_svg(rows)
        self.assertIn("0.7000", markdown)
        self.assertIn("0.7500", figure)
        self.assertIn("Post text + title + body", figure)

    def test_timing_stats_retain_sorted_raw_measurements(self):
        summary = timing_stats([4.0, 1.0, 3.0, 2.0])
        self.assertEqual(summary["median"], 2.5)
        self.assertEqual(summary["raw"], [1.0, 2.0, 3.0, 4.0])
        self.assertAlmostEqual(summary["mean"], 2.5)

    def test_declared_report_files_include_tables_figures_and_timing(self):
        self.assertIn("primary_results.csv", REPORT_FILENAMES)
        self.assertIn("primary_macro_f1.svg", REPORT_FILENAMES)
        self.assertIn("inference_time.json", REPORT_FILENAMES)
        self.assertIn("inference_latency.svg", REPORT_FILENAMES)

    def test_condition_blocks_use_training_weight_normalization(self):
        class Vectorizer:
            def __init__(self, value):
                self.value = value

            def transform(self, texts):
                return csr_matrix([[self.value] for _ in texts])

        bundle = {
            "settings": {"weights": [1, 1, 1]},
            "conditions": CONDITIONS,
            "vectorizers": {
                "post_text": Vectorizer(1),
                "title": Vectorizer(2),
                "body_1000": Vectorizer(3),
            },
        }
        texts = {field: ["example"] for field in bundle["vectorizers"]}
        matrix = transform_condition(bundle, "C", texts).toarray()[0]
        np.testing.assert_allclose(matrix, np.array([1, 2, 3]) / np.sqrt(3))


if __name__ == "__main__":
    unittest.main()
