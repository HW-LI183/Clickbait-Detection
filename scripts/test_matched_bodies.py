"""Regression checks for leakage, body matching and paired resampling."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from evaluate_matched_bodies import (
    bootstrap_gap, match_articles, output_paths, run, save_results,
    sha256, verify_reference_score,
)


class WrongBodyTests(unittest.TestCase):
    def test_swap_preserves_domain_and_excludes_duplicates(self):
        titles = ["senate reform tax", "senate reform budget", "senate reform election", "senate reform vote"]
        bodies = ["senate tax rates", "senate budget debate", "senate election polls", "senate vote results"]
        records = {}
        train = []
        for i in range(8):
            key = f"train_{i}"
            train.append(key)
            records[key] = dict(title=titles[i % 4], body_1000=bodies[i % 4])
        ids = [str(i) for i in range(6)]
        records.update({key: dict(title=titles[i % 4], body_1000=bodies[i % 4],
                                  body_full=bodies[i % 4]) for i, key in enumerate(ids)})
        sources = {key: dict(domain="news.example", canonical_sha256=key) for key in ids}
        sources["4"]["domain"] = "different.example"
        sources["5"]["domain"] = ""
        sources["3"]["canonical_sha256"] = sources["2"]["canonical_sha256"]
        manifest = {key: dict(group_id=key) for key in ids}
        manifest["1"]["group_id"] = manifest["0"]["group_id"]

        pairs, _, _ = match_articles(train, ids, records, manifest, sources)
        self.assertEqual(len(pairs), 1)
        i, j, cosine, ratio = pairs[0]
        self.assertIn(i, (0, 1))
        self.assertIn(j, (2, 3))
        self.assertGreaterEqual(cosine, 0.10)
        self.assertEqual(ratio, 1)

    def test_bootstrap_keeps_both_directions_together(self):
        truth = np.array([True, False] * 4)
        self.assertEqual(bootstrap_gap(truth, truth, ~truth), [1, 1])
        self.assertEqual(bootstrap_gap(truth, truth, truth), [0, 0])

    def test_reference_score_requires_absolute_agreement(self):
        verify_reference_score("B/logistic_regression", 0.75, 0.75)
        verify_reference_score("C/linear_svm", 0.75 + 5e-13, 0.75)
        for value in (0.750001, float("nan"), float("inf")):
            with self.subTest(value=value), self.assertRaisesRegex(
                ValueError, "C/linear_svm.*differs"
            ):
                verify_reference_score("C/linear_svm", value, 0.75)

    def test_existing_either_output_stops_before_loading_data(self):
        for topic_only in (False, True):
            prefix = "matched_body_topic_only" if topic_only else "matched_body"
            for index in (0, 1):
                with self.subTest(topic_only=topic_only, index=index):
                    with tempfile.TemporaryDirectory() as temporary:
                        directory = Path(temporary)
                        existing = output_paths(directory, prefix)[index]
                        existing.write_bytes(b"published")
                        with patch("evaluate_matched_bodies.load_training") as load:
                            with self.assertRaises(FileExistsError):
                                run(topic_only, directory)
                            load.assert_not_called()
                        self.assertEqual(existing.read_bytes(), b"published")
                        self.assertEqual(list(directory.iterdir()), [existing])

    def test_save_results_records_hash_and_refuses_overwrite(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary) / "new"
            save_results(directory, "matched_body", [{"id": "example"}], {"models": {}})
            pairs, result = output_paths(directory, "matched_body")
            self.assertEqual(json.loads(result.read_text())["pairs_sha256"], sha256(pairs))
            before = {path: path.read_bytes() for path in (pairs, result)}
            with self.assertRaises(FileExistsError):
                save_results(directory, "matched_body", [{"id": "changed"}], {})
            self.assertEqual(before, {path: path.read_bytes() for path in (pairs, result)})
            self.assertEqual(set(directory.iterdir()), {pairs, result})

    def test_serialization_failure_leaves_no_output_files(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            with self.assertRaises(TypeError):
                save_results(directory, "matched_body", [{"id": "example"}],
                             {"not_json": object()})
            self.assertEqual(list(directory.iterdir()), [])

    def test_publication_conflict_preserves_other_writer_and_cleans_partial_output(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            pairs, result = output_paths(directory, "matched_body")
            original_open = Path.open

            def conflicting_open(path, mode="r", *args, **kwargs):
                if path == result and mode == "xb":
                    path.write_bytes(b"other writer")
                return original_open(path, mode, *args, **kwargs)

            with patch.object(Path, "open", conflicting_open):
                with self.assertRaises(FileExistsError):
                    save_results(directory, "matched_body", [{"id": "example"}], {})
            self.assertFalse(pairs.exists())
            self.assertEqual(result.read_bytes(), b"other writer")
            self.assertEqual(list(directory.iterdir()), [result])


if __name__ == "__main__":
    unittest.main()
