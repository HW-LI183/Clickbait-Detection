"""Select one field-wise TF-IDF configuration using training groups only."""

import argparse
import csv
import hashlib
import json
import math
import os
import platform
import sys
from collections import Counter, defaultdict
from pathlib import Path

import joblib
import numpy
import scipy
import sklearn
from scipy.sparse import csr_matrix, hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.svm import LinearSVC


ROOT = Path(__file__).resolve().parent.parent
SEED = 20261121
FOLDS = 4
SIZE_LIMITS = {
    "small": (5000, 5000, 15000),
    "medium": (10000, 10000, 30000),
    "large": (20000, 20000, 60000),
}
NGRAMS = ((1, 1), (1, 2))
WEIGHTS = ((1, 1, 1), (2, 1, 1), (1, 1, 0.5), (2, 1, 0.5))
FIELDS = ("post_text", "title", "body_1000")
CONDITIONS = {"A": FIELDS[:1], "B": FIELDS[:2], "C": FIELDS}
MODELS = ("logistic_regression", "linear_svm")
TOLERANCE = 0.005
LABELS = ("clickbait", "no-clickbait")


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_jsonl(path):
    rows = {}
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            row = json.loads(line)
            if not isinstance(row, dict) or not isinstance(row.get("id"), str):
                raise ValueError(f"{path}:{line_number}: invalid row")
            if row["id"] in rows:
                raise ValueError(f"{path}:{line_number}: duplicate ID")
            rows[row["id"]] = row
    return rows


def load_training(processed_dir):
    with (processed_dir / "audit.json").open("r", encoding="utf-8") as stream:
        audit = json.load(stream)
    for name in ("records.jsonl", "split_manifest.jsonl"):
        actual = sha256(processed_dir / name)
        if actual != audit["output_sha256"][name]:
            raise ValueError(f"{name}: hash differs from frozen preprocessing audit")
    records = read_jsonl(processed_dir / "records.jsonl")
    manifest = read_jsonl(processed_dir / "split_manifest.jsonl")
    if records.keys() != manifest.keys():
        raise ValueError("Records and split manifest IDs differ")
    ids = sorted(row_id for row_id, row in manifest.items() if row["split"] == "train")
    if not ids:
        raise ValueError("No training records")
    texts = {field: [] for field in FIELDS}
    labels, groups = [], []
    for row_id in ids:
        row, split = records[row_id], manifest[row_id]
        if split["truthClass"] not in LABELS:
            raise ValueError(f"{row_id}: invalid label")
        if not isinstance(split.get("group_id"), str) or not split["group_id"]:
            raise ValueError(f"{row_id}: missing group")
        labels.append(split["truthClass"])
        groups.append(split["group_id"])
        for field in FIELDS:
            value = row.get(field)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{row_id}: missing {field}")
            texts[field].append(value)
    return ids, texts, numpy.asarray(labels), numpy.asarray(groups), {
        name: sha256(processed_dir / name)
        for name in ("records.jsonl", "split_manifest.jsonl")
    }


def make_folds(labels, groups):
    splitter = StratifiedGroupKFold(n_splits=FOLDS, shuffle=True, random_state=SEED)
    assignments = numpy.full(len(labels), -1, dtype=int)
    for fold, (train_indices, held_indices) in enumerate(
        splitter.split(numpy.zeros(len(labels)), labels, groups)
    ):
        if set(groups[train_indices]) & set(groups[held_indices]):
            raise ValueError("A duplicate group crosses the pilot fold boundary")
        if set(labels[train_indices]) != set(LABELS) or set(labels[held_indices]) != set(LABELS):
            raise ValueError("Both labels must occur in every fold")
        assignments[held_indices] = fold
    if (assignments < 0).any():
        raise ValueError("Unassigned training row")
    return assignments


def vectorize_fold(texts, train_indices, held_indices, limits, ngram):
    vectors, fitted, dimensions = {}, {}, {}
    for field, limit in zip(FIELDS, limits):
        vectorizer = TfidfVectorizer(
            max_features=limit,
            ngram_range=ngram,
            min_df=2,
            max_df=1.0,
            sublinear_tf=True,
            norm="l2",
            dtype=numpy.float64,
        )
        train = vectorizer.fit_transform([texts[field][i] for i in train_indices])
        held = (
            vectorizer.transform([texts[field][i] for i in held_indices])
            if len(held_indices) else csr_matrix((0, train.shape[1]))
        )
        vectors[field] = (train, held)
        fitted[field] = vectorizer
        dimensions[field] = len(vectorizer.vocabulary_)
    return vectors, fitted, dimensions


def combine_blocks(vectors, fields, weights):
    selected = [weights[FIELDS.index(field)] for field in fields]
    norm = math.sqrt(sum(weight * weight for weight in selected))
    return tuple(
        hstack(
            [vectors[field][side] * (weight / norm) for field, weight in zip(fields, selected)],
            format="csr",
        )
        for side in (0, 1)
    )


def pilot_model(name):
    if name == "logistic_regression":
        return LogisticRegression(
            C=1.0, solver="liblinear", class_weight=None,
            max_iter=2000, random_state=SEED,
        )
    return LinearSVC(
        C=1.0, penalty="l2", loss="squared_hinge", dual="auto",
        class_weight=None, max_iter=5000, random_state=SEED,
    )


def candidates():
    for size, limits in SIZE_LIMITS.items():
        for ngram in NGRAMS:
            for weight_index, weights in enumerate(WEIGHTS):
                yield {
                    "size": size,
                    "limits": limits,
                    "ngram": ngram,
                    "weight_index": weight_index,
                    "weights": weights,
                }


def candidate_key(candidate):
    return f"{candidate['size']}_ng{candidate['ngram'][1]}_w{candidate['weight_index']}"


def select_candidate(rows):
    by_candidate = defaultdict(list)
    for row in rows:
        by_candidate[row["candidate"]].append(row)
    expected = FOLDS * len(CONDITIONS) * len(MODELS)
    if len(by_candidate) != len(SIZE_LIMITS) * len(NGRAMS) * len(WEIGHTS):
        raise ValueError("Incomplete candidate grid")
    summaries = []
    for candidate in candidates():
        key = candidate_key(candidate)
        group = by_candidate[key]
        if len(group) != expected:
            raise ValueError(f"{key}: incomplete fold scores")
        cells = defaultdict(list)
        for row in group:
            cells[(row["condition"], row["model"])].append(row["macro_f1"])
        if len(cells) != len(CONDITIONS) * len(MODELS) or any(
            len(values) != FOLDS for values in cells.values()
        ):
            raise ValueError(f"{key}: incomplete condition/model scores")
        cell_means = {
            f"{condition}/{model}": float(numpy.mean(cells[(condition, model)]))
            for condition in CONDITIONS for model in MODELS
        }
        cell_fold_sd = {
            f"{condition}/{model}": float(
                numpy.std(cells[(condition, model)], ddof=1)
            )
            for condition in CONDITIONS for model in MODELS
        }
        dimensions = {
            field: float(numpy.mean([
                row[f"vocab_{field}"] for row in group if row["condition"] == "C"
                and row["model"] == MODELS[0]
            ]))
            for field in FIELDS
        }
        summaries.append({
            "candidate": key,
            "mean_macro_f1": float(numpy.mean(list(cell_means.values()))),
            "cell_means": cell_means,
            "cell_fold_sd": cell_fold_sd,
            "mean_vocabulary_dimensions": dimensions,
            "mean_total_vocabulary": sum(dimensions.values()),
            "settings": candidate,
        })
    best_score = max(item["mean_macro_f1"] for item in summaries)
    eligible = [
        item for item in summaries
        if item["mean_macro_f1"] >= best_score - TOLERANCE - 1e-12
    ]
    selected = min(
        eligible,
        key=lambda item: (
            item["mean_total_vocabulary"],
            item["settings"]["ngram"][1],
            item["settings"]["weight_index"],
        ),
    )
    return selected, sorted(summaries, key=lambda item: item["mean_macro_f1"], reverse=True)


def write_json(path, payload):
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2, sort_keys=True)
        stream.write("\n")


def check_existing_scores(score_path, fold_path, ids, labels, groups, folds):
    saved_folds = read_jsonl(fold_path)
    if set(saved_folds) != set(ids):
        raise ValueError("Saved fold assignment IDs differ from training IDs")
    for row_id, label, group, fold in zip(ids, labels, groups, folds):
        saved = saved_folds[row_id]
        if (saved["truthClass"], saved["group_id"], saved["fold"]) != (
            label, group, int(fold)
        ):
            raise ValueError(f"{row_id}: saved fold assignment changed")
    with score_path.open("r", encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    keys = set()
    grid = {candidate_key(item): item for item in candidates()}
    for row in rows:
        key = (row["candidate"], int(row["fold"]), row["condition"], row["model"])
        if (
            key in keys or key[0] not in grid or key[1] not in range(FOLDS)
            or key[2] not in CONDITIONS or key[3] not in MODELS
        ):
            raise ValueError(f"Unexpected or duplicate score key: {key}")
        keys.add(key)
        row["fold"] = int(row["fold"])
        row["macro_f1"] = float(row["macro_f1"])
        if not math.isfinite(row["macro_f1"]) or not 0 <= row["macro_f1"] <= 1:
            raise ValueError(f"{key}: invalid Macro-F1")
        for field, limit in zip(FIELDS, grid[key[0]]["limits"]):
            row[f"vocab_{field}"] = int(row[f"vocab_{field}"])
            if not 0 < row[f"vocab_{field}"] <= limit:
                raise ValueError(f"{key}: invalid {field} vocabulary size")
    expected = {
        (key, fold, condition, model)
        for key in grid for fold in range(FOLDS)
        for condition in CONDITIONS for model in MODELS
    }
    if keys != expected:
        raise ValueError(f"Incomplete saved scores: {len(keys)}/{len(expected)}")
    return rows


def run(processed_dir, output_dir):
    ids, texts, labels, groups, hashes = load_training(processed_dir)
    folds = make_folds(labels, groups)
    output_dir.mkdir(parents=True, exist_ok=True)
    fold_path = output_dir / "folds.jsonl"
    with fold_path.open("w", encoding="utf-8", newline="\n") as stream:
        for row_id, label, group, fold in zip(ids, labels, groups, folds):
            stream.write(json.dumps({
                "id": row_id, "group_id": str(group), "truthClass": str(label),
                "fold": int(fold),
            }, sort_keys=True) + "\n")
    score_path = output_dir / "scores.csv"
    rows = []
    with score_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=[
                "candidate", "fold", "condition", "model", "macro_f1",
                *(f"vocab_{field}" for field in FIELDS),
            ],
        )
        writer.writeheader()
        for fold in range(FOLDS):
            train_indices = numpy.flatnonzero(folds != fold)
            held_indices = numpy.flatnonzero(folds == fold)
            print(
                f"Fold {fold + 1}/{FOLDS}: {len(train_indices)} fit, "
                f"{len(held_indices)} held out; classes "
                f"{dict(Counter(labels[held_indices]))}", flush=True,
            )
            for size, limits in SIZE_LIMITS.items():
                for ngram in NGRAMS:
                    vectors, _, dimensions = vectorize_fold(
                        texts, train_indices, held_indices, limits, ngram
                    )
                    print(f"  {size}, {ngram}: {dimensions}", flush=True)
                    for weight_index, weights in enumerate(WEIGHTS):
                        candidate = {
                            "size": size, "ngram": ngram,
                            "weight_index": weight_index,
                        }
                        for condition, fields in CONDITIONS.items():
                            x_train, x_held = combine_blocks(vectors, fields, weights)
                            for model_name in MODELS:
                                model = pilot_model(model_name)
                                model.fit(x_train, labels[train_indices])
                                prediction = model.predict(x_held)
                                row = {
                                    "candidate": candidate_key(candidate),
                                    "fold": fold,
                                    "condition": condition,
                                    "model": model_name,
                                    "macro_f1": f1_score(
                                        labels[held_indices], prediction,
                                        labels=LABELS, average="macro", zero_division=0,
                                    ),
                                    **{
                                        f"vocab_{field}": dimensions[field]
                                        for field in FIELDS
                                    },
                                }
                                rows.append(row)
                                writer.writerow(row)
                        stream.flush()
    return finalize_scores(output_dir, ids, texts, labels, groups, folds, hashes, rows)


def finalize_scores(output_dir, ids, texts, labels, groups, folds, hashes, rows):
    fold_path = output_dir / "folds.jsonl"
    score_path = output_dir / "scores.csv"
    selected, ranking = select_candidate(rows)
    settings = selected["settings"]
    all_indices = numpy.arange(len(ids))
    _, fitted, full_dimensions = vectorize_fold(
        texts, all_indices, numpy.asarray([], dtype=int),
        settings["limits"], settings["ngram"],
    )
    bundle_path = output_dir / "vectorizers.joblib"
    joblib.dump({
        "vectorizers": fitted,
        "settings": settings,
        "training_ids_sha256": hashlib.sha256(
            ("\n".join(ids) + "\n").encode("utf-8")
        ).hexdigest(),
    }, bundle_path, compress=3)
    protocol_path = ROOT / "docs" / "protocols" / "tfidf_preexperiment_protocol.md"
    result = {
        "selected": selected,
        "ranking": ranking,
        "full_training_vocabulary_dimensions": full_dimensions,
        "parameters": {
            "seed": SEED, "folds": FOLDS, "metric": "macro_f1",
            "selection_tolerance": TOLERANCE,
            "models": {
                "logistic_regression": "liblinear, L2, C=1, max_iter=2000",
                "linear_svm": "LinearSVC, squared_hinge, L2, C=1, dual=auto, max_iter=5000",
            },
        },
        "environment": {
            "python": sys.version, "platform": platform.platform(),
            "numpy": numpy.__version__, "scipy": scipy.__version__,
            "scikit_learn": sklearn.__version__, "joblib": joblib.__version__,
            "OPENBLAS_NUM_THREADS": os.environ.get("OPENBLAS_NUM_THREADS"),
            "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS"),
        },
        "inputs_sha256": hashes,
        "script_sha256": sha256(Path(__file__)),
        "protocol_sha256": sha256(protocol_path),
        "folds_sha256": sha256(fold_path),
        "scores_sha256": sha256(score_path),
        "training_rows": len(ids),
        "training_groups": len(set(groups)),
        "folds_distribution": [
            {"fold": fold, "rows": int((folds == fold).sum()),
             "labels": dict(Counter(labels[folds == fold]))}
            for fold in range(FOLDS)
        ],
    }
    write_json(output_dir / "selection.json", result)
    print(
        f"Selected {selected['candidate']}, mean Macro-F1 "
        f"{selected['mean_macro_f1']:.5f}; saved {output_dir}", flush=True,
    )
    return result


def finalize_existing(processed_dir, output_dir):
    ids, texts, labels, groups, hashes = load_training(processed_dir)
    folds = make_folds(labels, groups)
    rows = check_existing_scores(
        output_dir / "scores.csv", output_dir / "folds.jsonl",
        ids, labels, groups, folds,
    )
    print(f"Validated {len(rows)} complete saved pilot scores", flush=True)
    return finalize_scores(output_dir, ids, texts, labels, groups, folds, hashes, rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--processed-dir", type=Path, default=ROOT / "data" / "processed",
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=ROOT / "data" / "processed" / "tfidf_preexperiment",
    )
    parser.add_argument(
        "--finalize-existing", action="store_true",
        help="Validate all saved scores and finish fitting without rerunning pilots",
    )
    args = parser.parse_args()
    if args.finalize_existing:
        finalize_existing(args.processed_dir, args.output_dir)
    else:
        run(args.processed_dir, args.output_dir)


if __name__ == "__main__":
    main()
