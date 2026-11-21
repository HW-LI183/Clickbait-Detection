"""Checks for literal feature definitions and train-only shared IDF."""

import json
import tempfile
import unittest
from pathlib import Path

import joblib
from engineer_features import (
    FEATURE_NAMES,
    extract_batch,
    fit_relation_vectorizer,
    lexical_relation,
    load_inputs,
    run,
    tokens,
    writing_features,
)
from tfidf_preexperiment import sha256


class EngineeredFeatureTests(unittest.TestCase):
    def test_writing_and_lexical_definitions(self):
        text = "You and your 42 friends: what is this?!"
        post = tokens(text)
        values = writing_features(text, post)
        self.assertEqual(values["post_word_count"], 8)
        self.assertEqual(values["post_question_count"], 1)
        self.assertEqual(values["post_exclamation_count"], 1)
        self.assertEqual(values["post_digit_token_count"], 1)
        self.assertEqual(values["post_second_person_count"], 2)
        self.assertEqual(values["post_demonstrative_count"], 1)
        self.assertEqual(values["post_interrogative_count"], 1)
        relation = lexical_relation(
            ["you", "see", "this"], ["this", "is", "you"], "post_title",
        )
        self.assertAlmostEqual(relation["post_title_overlap_recall"], 2 / 3)
        self.assertAlmostEqual(relation["post_title_jaccard"], 1 / 2)
        self.assertEqual(relation["post_title_length_ratio"], 1)
        self.assertEqual(
            lexical_relation(["you"], [], "post_body")["post_body_length_ratio"], 0,
        )

    def test_shared_encoder_does_not_fit_on_validation_or_test(self):
        records = {
            "a": {"post_text": "known common", "title": "common",
                  "body_1000": "article common"},
            "b": {"post_text": "known common", "title": "common",
                  "body_1000": "article common"},
            "c": {"post_text": "hiddenmarker", "title": "common",
                  "body_1000": "article common"},
        }
        settings = {"ngram": [1, 2], "limits": [5, 5, 5]}
        vectorizer = fit_relation_vectorizer(records, ["a", "b"], settings)
        self.assertNotIn("hiddenmarker", vectorizer.vocabulary_)
        rows, masks = extract_batch(records, ["c"], vectorizer)
        self.assertEqual(rows[0]["post_title_tfidf_cosine"], 0)
        self.assertEqual(rows[0]["post_body_tfidf_cosine"], 0)
        self.assertTrue(masks["post_text"][0])
        self.assertEqual(set(rows[0]) - {"id"}, set(FEATURE_NAMES))

    def test_full_sidecar_is_label_free_and_reproducible(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            processed = root / "processed"
            selected = root / "selected"
            output = root / "engineered"
            processed.mkdir()
            selected.mkdir()
            rows = [
                {"id": row_id, "post_text": post, "title": "common headline",
                 "body_1000": "article common body"}
                for row_id, post in (
                    ("a", "known common"), ("b", "known common"),
                    ("c", "hiddenmarker"), ("d", "unseenword"),
                )
            ]
            manifest = [
                {"id": row_id, "split": split, "truthClass": "clickbait"}
                for row_id, split in (
                    ("a", "train"), ("b", "train"),
                    ("c", "validation"), ("d", "test"),
                )
            ]
            for name, data in (
                ("records.jsonl", rows), ("split_manifest.jsonl", manifest),
            ):
                (processed / name).write_text(
                    "".join(json.dumps(row) + "\n" for row in data),
                    encoding="utf-8",
                )
            hashes = {
                name: sha256(processed / name)
                for name in ("records.jsonl", "split_manifest.jsonl")
            }
            (processed / "audit.json").write_text(
                json.dumps({"output_sha256": hashes}), encoding="utf-8",
            )
            (selected / "selection.json").write_text(
                json.dumps({
                    "inputs_sha256": hashes,
                    "selected": {"settings": {
                        "ngram": [1, 2], "limits": [5, 5, 5],
                    }},
                }),
                encoding="utf-8",
            )
            audit = run(processed, selected, output, batch_size=2)
            self.assertEqual(audit["rows_by_split"], {
                "test": 1, "train": 2, "validation": 1,
            })
            sidecar_hash = sha256(output / "features.jsonl")
            vectorizer = joblib.load(output / "relation_vectorizer.joblib")
            self.assertNotIn("hiddenmarker", vectorizer.vocabulary_)
            sidecar = [
                json.loads(line)
                for line in (output / "features.jsonl").read_text(
                    encoding="utf-8",
                ).splitlines()
            ]
            self.assertEqual([row["id"] for row in sidecar], ["a", "b", "c", "d"])
            self.assertTrue(all("truthClass" not in row for row in sidecar))
            self.assertEqual(sidecar[2]["post_body_tfidf_cosine"], 0)
            run(processed, selected, output, batch_size=3)
            self.assertEqual(sha256(output / "features.jsonl"), sidecar_hash)
            rows[0]["post_text"] = "changed"
            (processed / "records.jsonl").write_text(
                "".join(json.dumps(row) + "\n" for row in rows),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "hash mismatch"):
                load_inputs(processed, selected)


if __name__ == "__main__":
    unittest.main()
