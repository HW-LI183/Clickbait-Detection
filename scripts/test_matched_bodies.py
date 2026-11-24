"""Regression checks for leakage, body matching and paired resampling."""
import unittest

import numpy as np

from evaluate_matched_bodies import bootstrap_gap, match_articles


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


if __name__ == "__main__":
    unittest.main()
