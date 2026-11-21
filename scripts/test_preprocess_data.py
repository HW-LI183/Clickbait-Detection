import unittest

from preprocess_data import (
    BODY_CAP,
    field_audit,
    group_records,
    normalize_field,
    prepare,
    split_groups,
)


class PreprocessingTests(unittest.TestCase):
    def test_missing_required_text(self):
        instances = {
            "a": {
                "id": "a",
                "postText": ["  headline  ", "\tcontinued "],
                "targetTitle": " title ",
                "targetParagraphs": [" ".join(f"body{n}" for n in range(20))],
            },
            "b": {
                "id": "b",
                "postText": [" \t "],
                "targetTitle": "title",
                "targetParagraphs": ["body"],
            },
        }
        truth = {
            "a": {"id": "a", "truthClass": "clickbait"},
            "b": {"id": "b", "truthClass": "no-clickbait"},
        }
        audit = field_audit(instances, ("postText",))
        self.assertEqual(tuple(audit), ("postText",))
        self.assertEqual(audit["postText"]["status"]["whitespace_only"], 1)
        rows, excluded = prepare(instances, truth)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["post_text"], "headline continued")
        self.assertEqual(
            excluded,
            [{"id": "b", "reasons": ["post_text", "body_under_20_words"]}],
        )

    def test_body_cap_preserves_full_text(self):
        words = [f"word{index}" for index in range(BODY_CAP + 5)]
        instances = {
            "a": {
                "id": "a",
                "postText": ["teaser"],
                "targetTitle": "title",
                "targetParagraphs": [" ".join(words)],
            }
        }
        rows, _ = prepare(instances, {"a": {"truthClass": "clickbait"}})
        self.assertEqual(len(rows[0]["body_1000"].split()), BODY_CAP)
        self.assertEqual(len(rows[0]["body_full"].split()), BODY_CAP + 5)
        self.assertEqual(normalize_field([" a  ", "b "], "list"), "a b")

    def test_exact_and_near_duplicates_stay_in_one_split(self):
        base = " ".join(f"token{index}" for index in range(120))
        changed = base.replace("token50", "changed50")
        rows = [
            {
                "id": f"{index:02d}",
                "post_text": f"teaser {index}",
                "title": f"title {index}",
                "body_full": body,
                "body_words": len(body.split()),
                "truthClass": "clickbait" if index % 2 else "no-clickbait",
            }
            for index, body in enumerate(
                [base, changed, base]
                + [" ".join(f"other{index}_{word}" for word in range(30))
                   for index in range(40)]
            )
        ]
        groups, diagnostics = group_records(rows)
        joined = next(indices for indices in groups.values() if 0 in indices)
        self.assertEqual(set(joined), {0, 1, 2})
        self.assertGreaterEqual(diagnostics["near_duplicate_group_merges"], 1)
        first, _ = split_groups(groups, rows)
        second, _ = split_groups(groups, rows)
        self.assertEqual(first, second)
        group_id = next(name for name, indices in groups.items() if 0 in indices)
        self.assertIn(first[group_id], ("train", "validation", "test"))

    def test_unmatched_ids_fail(self):
        with self.assertRaisesRegex(ValueError, "Unmatched IDs"):
            prepare({"a": {"id": "a"}}, {})

    def test_short_nonempty_body_is_excluded(self):
        instance = {
            "x": {
                "id": "x",
                "postText": ["teaser"],
                "targetTitle": "title",
                "targetParagraphs": ["Still Watching? Rotate Device"],
            }
        }
        rows, excluded = prepare(
            instance, {"x": {"id": "x", "truthClass": "clickbait"}}
        )
        self.assertFalse(rows)
        self.assertEqual(excluded[0]["reasons"], ["body_under_20_words"])


if __name__ == "__main__":
    unittest.main()
