import unittest

from preprocess_data import field_audit
from preprocess_optional_features import clean_optional, identify_duplicates


class OptionalFeatureTests(unittest.TestCase):
    def test_missing_and_blank_values_do_not_exclude_rows(self):
        instances = {
            "a": {
                "id": "a",
                "targetCaptions": [" picture.jpg ", "  ", " caption  text "],
                "targetKeywords": "  news,   science ",
                "targetDescription": "  one\n two  ",
            },
            "b": {
                "id": "b",
                "targetCaptions": [],
                "targetKeywords": "",
                "targetDescription": None,
            },
            "c": {"id": "c"},
        }
        rows = clean_optional(instances, instances)
        self.assertEqual(len(rows), 3)
        self.assertEqual(rows[0]["target_captions"], "picture.jpg caption text")
        self.assertEqual(rows[0]["target_keywords"], "news, science")
        self.assertEqual(rows[0]["target_description"], "one two")
        self.assertEqual(
            [rows[1][field] for field in ("target_captions", "target_keywords", "target_description")],
            ["", "", ""],
        )
        self.assertEqual(rows[2]["target_captions"], "")
        audit = field_audit(
            instances, ("targetCaptions", "targetKeywords", "targetDescription")
        )
        self.assertEqual(audit["targetCaptions"]["status"]["empty"], 1)
        self.assertEqual(audit["targetCaptions"]["status"]["absent"], 1)
        self.assertEqual(
            audit["targetCaptions"]["list_elements"]["whitespace_only"], 1
        )
        self.assertEqual(audit["targetDescription"]["status"]["null"], 1)

    def test_exact_and_near_duplicates_are_separate_by_feature(self):
        base = " ".join(f"token{index}" for index in range(100))
        changed = base.replace("token40", "changed40")
        rows = [
            {"id": "a", "target_description": base},
            {"id": "b", "target_description": changed},
            {"id": "c", "target_description": base},
            {"id": "d", "target_description": "unique short"},
            {"id": "e", "target_description": ""},
        ]
        splits = {"a": "train", "b": "test", "c": "train", "d": "test", "e": "test"}
        groups, audit = identify_duplicates(rows, "target_description", splits)
        self.assertEqual(groups["a"], groups["b"])
        self.assertEqual(groups["a"], groups["c"])
        self.assertNotIn("d", groups)
        self.assertNotIn("e", groups)
        self.assertEqual(audit["cross_split_groups"], 1)
        self.assertEqual(audit["duplicate_records"], 3)
        self.assertEqual(
            groups, identify_duplicates(rows, "target_description", splits)[0]
        )

    def test_wrong_optional_type_fails(self):
        with self.assertRaisesRegex(ValueError, "targetKeywords"):
            clean_optional({"a": {"id": "a", "targetKeywords": ["wrong"]}}, {"a"})


if __name__ == "__main__":
    unittest.main()
