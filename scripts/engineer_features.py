"""Extract transparent writing and cross-text features without fitting a model."""

import argparse
import json
import math
import os
import platform
import re
import sys
import tempfile
from collections import Counter
from pathlib import Path

import joblib
import numpy
import scipy
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer

from tfidf_preexperiment import ROOT, read_jsonl, sha256


FIELDS = ("post_text", "title", "body_1000")
SPLITS = ("train", "validation", "test")
SECOND_PERSON = frozenset(("you", "your", "yours", "yourself", "yourselves"))
DEMONSTRATIVES = frozenset(("this", "that", "these", "those"))
INTERROGATIVES = frozenset((
    "who", "whom", "whose", "what", "which", "when", "where", "why", "how",
))
WORD_RE = re.compile(r"\b\w+\b", re.UNICODE)
SEED = 20261121
FEATURE_NAMES = (
    "post_word_count",
    "post_question_count",
    "post_exclamation_count",
    "post_digit_token_count",
    "post_second_person_count",
    "post_demonstrative_count",
    "post_interrogative_count",
    "post_title_tfidf_cosine",
    "post_title_overlap_recall",
    "post_title_jaccard",
    "post_title_length_ratio",
    "post_body_tfidf_cosine",
    "post_body_overlap_recall",
    "post_body_jaccard",
    "post_body_length_ratio",
)


def tokens(text):
    return [match.group().casefold() for match in WORD_RE.finditer(text)]


def writing_features(text, post_tokens):
    return {
        "post_word_count": len(post_tokens),
        "post_question_count": text.count("?") + text.count("\uff1f"),
        "post_exclamation_count": text.count("!") + text.count("\uff01"),
        "post_digit_token_count": sum(
            any(character.isdecimal() for character in token) for token in post_tokens
        ),
        "post_second_person_count": sum(token in SECOND_PERSON for token in post_tokens),
        "post_demonstrative_count": sum(
            token in DEMONSTRATIVES for token in post_tokens
        ),
        "post_interrogative_count": sum(
            token in INTERROGATIVES for token in post_tokens
        ),
    }


def lexical_relation(post_tokens, target_tokens, prefix):
    post_set, target_set = set(post_tokens), set(target_tokens)
    intersection = len(post_set & target_set)
    union = len(post_set | target_set)
    return {
        f"{prefix}_overlap_recall": intersection / len(post_set) if post_set else 0.0,
        f"{prefix}_jaccard": intersection / union if union else 0.0,
        f"{prefix}_length_ratio": (
            len(post_tokens) / len(target_tokens) if target_tokens else 0.0
        ),
    }


def load_inputs(processed_dir, selection_dir):
    audit_path = processed_dir / "audit.json"
    with audit_path.open("r", encoding="utf-8") as stream:
        audit = json.load(stream)
    hashes = {}
    for name in ("records.jsonl", "split_manifest.jsonl"):
        hashes[name] = sha256(processed_dir / name)
        if hashes[name] != audit["output_sha256"][name]:
            raise ValueError(f"{name}: frozen preprocessing hash mismatch")
    records = read_jsonl(processed_dir / "records.jsonl")
    manifest = read_jsonl(processed_dir / "split_manifest.jsonl")
    if records.keys() != manifest.keys():
        raise ValueError("Records and manifest IDs differ")
    for row_id, row in records.items():
        if manifest[row_id].get("split") not in SPLITS:
            raise ValueError(f"{row_id}: invalid split")
        for field in FIELDS:
            if not isinstance(row.get(field), str) or not row[field].strip():
                raise ValueError(f"{row_id}: missing {field}")
    selection_path = selection_dir / "selection.json"
    with selection_path.open("r", encoding="utf-8") as stream:
        selection = json.load(stream)
    if selection["inputs_sha256"] != hashes:
        raise ValueError("Selected TF-IDF was fitted against different inputs")
    settings = selection["selected"]["settings"]
    ngram = tuple(settings["ngram"])
    limits = settings["limits"]
    if len(ngram) != 2 or len(limits) != len(FIELDS):
        raise ValueError("Invalid selected TF-IDF settings")
    return records, manifest, hashes, settings, sha256(selection_path)


def fit_relation_vectorizer(records, train_ids, settings):
    vectorizer = TfidfVectorizer(
        max_features=sum(settings["limits"]),
        ngram_range=tuple(settings["ngram"]),
        min_df=2,
        sublinear_tf=True,
        norm="l2",
        dtype=numpy.float64,
    )
    vectorizer.fit(
        " ".join(records[row_id][field] for field in FIELDS)
        for row_id in train_ids
    )
    return vectorizer


def extract_batch(records, ids, vectorizer):
    text = {field: [records[row_id][field] for row_id in ids] for field in FIELDS}
    vectors = {field: vectorizer.transform(text[field]) for field in FIELDS}
    post_tokens = [tokens(value) for value in text["post_text"]]
    target_tokens = {
        field: [tokens(value) for value in text[field]]
        for field in FIELDS[1:]
    }
    similarities = {}
    for field, short_name in (("title", "title"), ("body_1000", "body")):
        values = numpy.asarray(
            vectors["post_text"].multiply(vectors[field]).sum(axis=1)
        ).ravel()
        similarities[short_name] = numpy.clip(values, 0.0, 1.0)
    result = []
    for index, row_id in enumerate(ids):
        features = writing_features(text["post_text"][index], post_tokens[index])
        for field, short_name in (("title", "title"), ("body_1000", "body")):
            prefix = f"post_{short_name}"
            features[f"{prefix}_tfidf_cosine"] = float(similarities[short_name][index])
            features.update(lexical_relation(
                post_tokens[index], target_tokens[field][index], prefix,
            ))
        if set(features) != set(FEATURE_NAMES) or not all(
            isinstance(value, (int, float)) and math.isfinite(value)
            for value in features.values()
        ):
            raise ValueError(f"{row_id}: invalid feature output")
        result.append({"id": row_id, **features})
    zero_masks = {
        field: vectors[field].getnnz(axis=1) == 0
        for field in FIELDS
    }
    return result, zero_masks


def write_json(path, value):
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, sort_keys=True)
        stream.write("\n")


def run(processed_dir, selection_dir, output_dir, batch_size=512):
    if batch_size <= 0:
        raise ValueError("Batch size must be positive")
    records, manifest, hashes, settings, selection_hash = load_inputs(
        processed_dir, selection_dir,
    )
    ids = sorted(records)
    train_ids = [row_id for row_id in ids if manifest[row_id]["split"] == "train"]
    if not train_ids:
        raise ValueError("No training records")
    vectorizer = fit_relation_vectorizer(records, train_ids, settings)
    output_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=output_dir.parent) as temporary:
        stage = Path(temporary)
        joblib.dump(vectorizer, stage / "relation_vectorizer.joblib", compress=3)
        zero_counts = {split: Counter() for split in SPLITS}
        split_counts = Counter()
        feature_path = stage / "features.jsonl"
        with feature_path.open("w", encoding="utf-8", newline="\n") as stream:
            for start in range(0, len(ids), batch_size):
                batch_ids = ids[start:start + batch_size]
                rows, zero_masks = extract_batch(records, batch_ids, vectorizer)
                for row in rows:
                    stream.write(
                        json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
                    )
                    split_counts[manifest[row["id"]]["split"]] += 1
                for field in FIELDS:
                    for row_id, is_zero in zip(
                        batch_ids, zero_masks[field]
                    ):
                        zero_counts[manifest[row_id]["split"]][field] += int(is_zero)
                print(f"Extracted {min(start + batch_size, len(ids))}/{len(ids)}", flush=True)
        if sum(split_counts.values()) != len(ids):
            raise ValueError("Incomplete output")
        audit = {
            "seed": SEED,
            "training_rows_for_relation_idf": len(train_ids),
            "rows": len(ids),
            "rows_by_split": dict(sorted(split_counts.items())),
            "feature_names": list(FEATURE_NAMES),
            "feature_count": len(FEATURE_NAMES),
            "relation_vectorizer": {
                "purpose": "cross_field_similarity_only",
                "max_features": sum(settings["limits"]),
                "actual_vocabulary_size": len(vectorizer.vocabulary_),
                "ngram_range": list(settings["ngram"]),
                "min_df": 2,
                "sublinear_tf": True,
                "smooth_idf": True,
                "norm": "l2",
                "fit_split": "train",
            },
            "zero_shared_vectors_by_split": {
                split: dict(sorted(zero_counts[split].items())) for split in SPLITS
            },
            "inputs_sha256": hashes,
            "selection_sha256": selection_hash,
            "script_sha256": sha256(Path(__file__)),
            "protocol_sha256": sha256(
                ROOT / "docs" / "protocols" / "engineered_features_protocol.md"
            ),
            "features_sha256": sha256(feature_path),
            "relation_vectorizer_sha256": sha256(
                stage / "relation_vectorizer.joblib"
            ),
            "environment": {
                "python": sys.version,
                "platform": platform.platform(),
                "numpy": numpy.__version__,
                "scipy": scipy.__version__,
                "scikit_learn": sklearn.__version__,
                "joblib": joblib.__version__,
                "OPENBLAS_NUM_THREADS": os.environ.get("OPENBLAS_NUM_THREADS"),
                "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS"),
            },
        }
        write_json(stage / "audit.json", audit)
        names = ("features.jsonl", "relation_vectorizer.joblib", "audit.json")
        for name in names:
            destination = output_dir / name
            source = stage / name
            if destination.exists() and sha256(destination) != sha256(source):
                raise ValueError(f"{destination}: existing output differs; not overwritten")
        for name in names:
            destination = output_dir / name
            if not destination.exists():
                (stage / name).replace(destination)
    print(
        f"Saved {len(ids)} rows, {len(FEATURE_NAMES)} features to {output_dir}",
        flush=True,
    )
    return audit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--processed-dir", type=Path, default=ROOT / "data" / "processed",
    )
    parser.add_argument(
        "--selection-dir", type=Path,
        default=ROOT / "data" / "processed" / "tfidf_preexperiment",
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=ROOT / "data" / "processed" / "engineered",
    )
    parser.add_argument("--batch-size", type=int, default=512)
    args = parser.parse_args()
    run(args.processed_dir, args.selection_dir, args.output_dir, args.batch_size)


if __name__ == "__main__":
    main()
