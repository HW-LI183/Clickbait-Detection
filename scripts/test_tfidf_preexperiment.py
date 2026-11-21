"""Focused checks for the training-only vectorization and selection rules."""

import unittest

import numpy

from tfidf_preexperiment import (
    CONDITIONS,
    FIELDS,
    FOLDS,
    MODELS,
    candidates,
    candidate_key,
    combine_blocks,
    make_folds,
    select_candidate,
    vectorize_fold,
)


class TfidfPreexperimentTests(unittest.TestCase):
    def test_fold_keeps_groups_together(self):
        labels = numpy.asarray(["clickbait", "no-clickbait"] * 40)
        groups = numpy.asarray([f"g{i // 2}" for i in range(80)])
        folds = make_folds(labels, groups)
        self.assertEqual(set(folds), set(range(FOLDS)))
        for group in set(groups):
            self.assertEqual(len(set(folds[groups == group])), 1)

    def test_vocabulary_is_fit_on_training_rows_only(self):
        texts = {
            field: ["known token", "known token", "hiddenmarker unique"]
            for field in FIELDS
        }
        vectors, fitted, dimensions = vectorize_fold(
            texts, numpy.asarray([0, 1]), numpy.asarray([2]),
            (100, 100, 100), (1, 1),
        )
        for field in FIELDS:
            self.assertNotIn("hiddenmarker", fitted[field].vocabulary_)
            self.assertEqual(dimensions[field], 2)
            self.assertEqual(vectors[field][1].nnz, 0)
        empty_vectors, _, _ = vectorize_fold(
            texts, numpy.asarray([0, 1]), numpy.asarray([], dtype=int),
            (100, 100, 100), (1, 1),
        )
        self.assertEqual(empty_vectors["post_text"][1].shape, (0, 2))
        both = combine_blocks(vectors, CONDITIONS["B"], (2, 1, 0.5))
        self.assertEqual(both[0].shape, (2, 4))
        self.assertTrue(numpy.allclose(
            numpy.sqrt(both[0].multiply(both[0]).sum(axis=1)), 1,
        ))

    def test_selection_uses_all_cells_and_complexity_tolerance(self):
        rows = []
        for candidate in candidates():
            name = candidate_key(candidate)
            score = 0.70
            if name == "large_ng2_w0":
                score = 0.704
            for fold in range(FOLDS):
                for condition in CONDITIONS:
                    for model in MODELS:
                        rows.append({
                            "candidate": name, "fold": fold, "condition": condition,
                            "model": model, "macro_f1": score,
                            **{f"vocab_{field}": limit for field, limit in
                               zip(FIELDS, candidate["limits"])},
                        })
        selected, ranking = select_candidate(rows)
        self.assertEqual(selected["candidate"], "small_ng1_w0")
        self.assertEqual(len(ranking), 24)
        rows.pop()
        with self.assertRaises(ValueError):
            select_candidate(rows)


if __name__ == "__main__":
    unittest.main()
