"""Summarize blinded text reviews using the saved validation predictions."""
import csv
import json
from collections import Counter

import numpy as np

from evaluate_matched_bodies import OUTPUT
from evaluate_primary import macro_f1
from tfidf_preexperiment import MODELS


def read(name):
    with (OUTPUT / name).open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def main():
    review = read("matched_body_topic_review.csv")
    results = {}
    for scope in ("matched_body", "matched_body_topic_only"):
        audit = [r for r in review if r["scope"] == scope]
        rows = read(scope + "_pairs.csv")
        ranked = sorted(rows[::2], key=lambda r: (-float(r["cosine"]), r["recipient"]))
        if scope.endswith("topic_only"):
            positions = np.rint(np.linspace(0, len(ranked) // 2 - 1, 48)).astype(int)
            ranked = [ranked[i] for i in positions]
        expected = {(r["pair"], r["recipient"], r["donor"]) for r in ranked}
        observed = {(r["pair"], r["first"], r["second"]) for r in audit}
        if expected != observed or len(observed) != len(audit):
            raise ValueError("Text reviews do not match the declared pair sample")
        chosen = {r["pair"] for r in audit if r["assessment"] == "specific"}
        selected = [r for r in rows if r["pair"] in chosen]
        y = np.array([r["label"] == "clickbait" for r in selected])
        out = {"assessed_pairs": len(audit), "assessments": dict(Counter(r["assessment"] for r in audit)),
               "specific_rows": len(selected), "clickbait_rows": int(y.sum()), "models": {}}
        for name in MODELS:
            predicted = {c: np.array([bool(int(r[f"{name}_{c}"])) for r in selected])
                         for c in ("B", "C", "swapped")}
            f1 = {c: macro_f1(y, v) for c, v in predicted.items()}
            orig, wrong = predicted["C"], predicted["swapped"]
            out["models"][name] = {"macro_f1": f1, "gap": f1["C"] - f1["swapped"],
                                   "correct_only_with_original": int(((orig == y) & (wrong != y)).sum()),
                                   "correct_only_with_swap": int(((orig != y) & (wrong == y)).sum())}
        results[scope] = out
    (OUTPUT / "matched_body_reviewed_validation.json").write_text(
        json.dumps(results, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
