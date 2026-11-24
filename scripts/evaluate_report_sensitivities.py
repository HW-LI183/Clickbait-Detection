"""Validation-only sensitivity analyses for the course report.

Use the C classifier settings selected in the primary study for every variant.
Fit new body vocabularies for length variants and fit all numeric transformations
on training records. No new test predictions are made.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import sklearn
from scipy.sparse import csr_matrix, hstack
from sklearn.preprocessing import StandardScaler

from engineer_features import extract_batch, fit_relation_vectorizer
from evaluate_primary import metrics
from preprocess_data import INPUT_HASHES
from tfidf_preexperiment import (
    FIELDS, MODELS, ROOT, combine_blocks, load_training, pilot_model,
    read_jsonl, sha256, vectorize_fold,
)

CAPS = (200, 500, 1000, 1500)
RELATIONS = tuple("post_body_" + suffix for suffix in
                  ("tfidf_cosine", "overlap_recall", "jaccard", "length_ratio"))


def relation_matrix(records, ids, vectorizer):
    values = []
    for start in range(0, len(ids), 256):
        rows, _ = extract_batch(records, ids[start:start + 256], vectorizer)
        values.extend([row[k] for k in RELATIONS] for row in rows)
    return np.asarray(values)


def run(output_path):
    if output_path.exists():
        raise FileExistsError(f"Result already exists: {output_path}. Choose a fresh --output path.")
    out = ROOT / "output"
    primary = json.loads((out / "primary_evaluation.json").read_text(encoding="utf-8"))
    processed = ROOT / "data/processed"
    train, _, labels, _, hashes = load_training(processed)
    if hashes != primary["inputs_sha256"]:
        raise ValueError("Unexpected training corpus or split")
    manifest = read_jsonl(processed / "split_manifest.jsonl")
    records = read_jsonl(processed / "records.jsonl")
    val = sorted(key for key, row in manifest.items() if row["split"] == "validation")
    ids = train + val
    val_y = np.array([manifest[k]["truthClass"] == "clickbait" for k in val])
    texts = {f: [records[k][f] for k in ids] for f in FIELDS}
    settings = primary["settings"]
    report = {"split": "validation", "n": len(val), "inputs_sha256": hashes,
              "sklearn": sklearn.__version__, "caps": {}, "relations": list(RELATIONS),
              "relation_block_scale": 1 / np.sqrt(len(RELATIONS)), "augmented": {},
              "annotation_subgroups": {}}

    truth_path = ROOT / "data/truth.jsonl"
    if sha256(truth_path) != INPUT_HASHES["truth.jsonl"]:
        raise ValueError("Annotator judgments differ from the frozen source")
    judgments = read_jsonl(truth_path)
    votes = [judgments[k]["truthJudgments"] for k in val]
    if any(len(v) != 5 for v in votes):
        raise ValueError("Expected five judgments per post")
    upper = np.array([sum(score >= 0.5 for score in v) for v in votes])
    close = np.minimum(upper, 5 - upper) == 2
    for cap in CAPS:
        texts["body_1000"] = [" ".join(records[k]["body_full"].split()[:cap]) for k in ids]
        if cap == 1000 and any(t != records[k]["body_1000"] for k, t in zip(ids, texts["body_1000"])):
            raise ValueError("Body reconstruction differs from the primary text")
        vectors, _, _ = vectorize_fold(texts, np.arange(len(train)), np.arange(len(train), len(ids)),
                                      settings["limits"], tuple(settings["ngram"]))
        xt, xv = combine_blocks(vectors, FIELDS, settings["weights"])
        report["caps"][str(cap)] = {}
        if cap == 1000:
            encoder = fit_relation_vectorizer(records, train, settings)
            tr_num = relation_matrix(records, train, encoder)
            va_num = relation_matrix(records, val, encoder)
            scaler = StandardScaler().fit(tr_num)
            tr_aug = hstack((xt, csr_matrix(scaler.transform(tr_num) * report["relation_block_scale"])), format="csr")
            va_aug = hstack((xv, csr_matrix(scaler.transform(va_num) * report["relation_block_scale"])), format="csr")
        for name in MODELS:
            conf = primary["models"][f"C/{name}"]
            model = pilot_model(name).set_params(C=conf["C"]).fit(xt, labels == "clickbait")
            scores = model.decision_function(xv)
            pred = scores >= conf["threshold"]
            data = metrics(val_y, pred, scores)
            report["caps"][str(cap)][name] = data
            if cap == 1000:
                if abs(data["macro_f1"] - conf["validation_macro_f1"]) > 1e-10:
                    raise ValueError("The recorded reference validation score was not reproduced")
                for tag, mask in (("split_3_to_2", close), ("agreement_4_or_5", ~close)):
                    sub = report["annotation_subgroups"].setdefault(tag, {"n": int(mask.sum()), "clickbait": int(val_y[mask].sum())})
                    sub[name] = metrics(val_y[mask], pred[mask], scores[mask])
                augmented = pilot_model(name).set_params(C=conf["C"]).fit(tr_aug, labels == "clickbait")
                aug_scores = augmented.decision_function(va_aug)
                report["augmented"][name] = metrics(val_y, aug_scores >= conf["threshold"], aug_scores)
        print("Completed cap", cap, flush=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(report, indent=2) + "\n")
    print(f"Saved validation results to {output_path}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path,
        default=ROOT / "data/reproductions/report_validation_sensitivities.json",
        help="Destination JSON for this run. Choose a path that does not exist.",
    )
    run(parser.parse_args().output)
