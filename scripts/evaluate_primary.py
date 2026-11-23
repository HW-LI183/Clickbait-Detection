"""Evaluate the frozen A/B/C experiment on the reserved splits."""

import json
import platform
from collections import Counter
from time import perf_counter

import numpy as np
import sklearn
from sklearn.metrics import (
    average_precision_score, confusion_matrix, f1_score,
    precision_recall_fscore_support,
)

from tfidf_preexperiment import (
    CONDITIONS, FIELDS, MODELS, ROOT, SEED, combine_blocks, pilot_model,
    load_training, read_jsonl, vectorize_fold,
)

C_VALUES = (0.01, 0.1, 1.0, 10.0, 100.0)
THRESHOLDS = (0.0, -0.25, 0.25, -0.5, 0.5)


def macro_f1(y, prediction):
    return f1_score(y, prediction, labels=[False, True], average="macro", zero_division=0)


def metrics(y, prediction, scores):
    precision, recall, f1, _ = precision_recall_fscore_support(
        y, prediction, average="binary", zero_division=0,
    )
    return {
        "macro_f1": macro_f1(y, prediction),
        "clickbait_precision": precision,
        "clickbait_recall": recall,
        "clickbait_f1": f1,
        "average_precision": average_precision_score(y, scores),
        "confusion_matrix": confusion_matrix(y, prediction, labels=[False, True]).tolist(),
    }


def group_intervals(y, groups, predictions, repetitions=2000):
    _, group_indices = np.unique(groups, return_inverse=True)
    group_count = int(group_indices.max()) + 1
    # Store each group's TN, FP, FN and TP so resampled groups retain all rows.
    counts = np.stack([
        np.bincount(4 * group_indices + 2 * y + pred, minlength=4 * group_count)
        .reshape(group_count, 4)
        for pred in predictions
    ])
    rng = np.random.default_rng(SEED)
    differences = []
    for _ in range(repetitions):
        draw = rng.integers(group_count, size=group_count)
        tn, fp, fn, tp = counts[:, draw].sum(axis=1).T
        errors = fp + fn
        negative = np.divide(2 * tn, 2 * tn + errors, out=np.zeros(3), where=2 * tn + errors != 0)
        positive = np.divide(2 * tp, 2 * tp + errors, out=np.zeros(3), where=2 * tp + errors != 0)
        f1 = (negative + positive) / 2
        differences.append(f1[2] - f1[:2])
    intervals = np.percentile(differences, [2.5, 97.5], axis=0).T
    observed = [macro_f1(y, pred) for pred in predictions]
    return {
        f"C-{condition}": {"delta": observed[2] - observed[i], "ci95": intervals[i].tolist()}
        for i, condition in enumerate(("A", "B"))
    }


def run(processed_dir, output_path):
    train_ids, texts, train_labels, _, hashes = load_training(processed_dir)
    manifest = read_jsonl(processed_dir / "split_manifest.jsonl")
    records = read_jsonl(processed_dir / "records.jsonl")
    validation_ids = sorted(key for key, row in manifest.items() if row["split"] == "validation")
    test_ids = sorted(key for key, row in manifest.items() if row["split"] == "test")
    if not validation_ids or not test_ids:
        raise ValueError("The frozen validation and test splits are required")

    y_train = train_labels == "clickbait"
    y_validation = np.array([manifest[key]["truthClass"] == "clickbait" for key in validation_ids])
    # Refit the documented selected configuration on the frozen training rows.
    limits, ngram, weights = (5000, 5000, 15000), (1, 2), (1, 1, 1)
    for field in FIELDS:
        texts[field].extend(records[key][field] for key in validation_ids)
    train_end = len(train_ids)
    vectors, encoders, _ = vectorize_fold(
        texts, np.arange(train_end), np.arange(train_end, train_end + len(validation_ids)),
        limits, ngram,
    )

    selected = {}
    for condition, fields in CONDITIONS.items():
        train, validation = combine_blocks(vectors, fields, weights)
        for name in MODELS:
            best_score = -1
            for c in C_VALUES:
                model = pilot_model(name).set_params(C=c).fit(train, y_train)
                scores = model.decision_function(validation)
                for threshold in THRESHOLDS:
                    score = macro_f1(y_validation, scores >= threshold)
                    if score > best_score:
                        best_score = score
                        selected[condition, name] = (model, threshold, score)

    # All tuning has finished before labels or features from test are used.
    y_test = np.array([manifest[key]["truthClass"] == "clickbait" for key in test_ids])
    groups = [manifest[key]["group_id"] for key in test_ids]
    started = perf_counter()
    test_blocks = {
        field: (encoder.transform([records[key][field] for key in test_ids]), vectors[field][1][:0])
        for field, encoder in encoders.items()
    }
    encoding_seconds = perf_counter() - started
    majority = np.full(len(test_ids), Counter(y_train).most_common(1)[0][0])
    result = {
        "inputs_sha256": hashes,
        "split_sizes": [len(train_ids), len(validation_ids), len(test_ids)],
        "settings": {"limits": limits, "ngram": ngram, "weights": weights},
        "runtime": {
            "platform": platform.platform(), "processor": platform.processor(),
            "sklearn": sklearn.__version__, "batch_size": len(test_ids),
            "shared_tfidf_transform_seconds": encoding_seconds,
        },
        "majority": metrics(y_test, majority, majority),
        "models": {},
    }
    predictions = {}
    for condition, fields in CONDITIONS.items():
        test, _ = combine_blocks(test_blocks, fields, weights)
        for name in MODELS:
            model, threshold, validation_score = selected[condition, name]
            started = perf_counter()
            for _ in range(20):
                scores = model.decision_function(test)
                predicted = scores >= threshold
            seconds = (perf_counter() - started) / 20
            predictions[condition, name] = predicted
            result["models"][f"{condition}/{name}"] = {
                "C": model.C, "threshold": threshold, "validation_macro_f1": validation_score,
                "batch_classifier_seconds": seconds, **metrics(y_test, predicted, scores),
            }

    result["primary_contrasts"] = group_intervals(
        y_test, groups, [predictions[condition, "logistic_regression"] for condition in CONDITIONS],
    )
    changes = []
    for name in MODELS:
        base = predictions["A", name]
        for condition in ("B", "C"):
            current = predictions[condition, name]
            for i in np.flatnonzero(base != current):
                changes.append({
                    "id": test_ids[i], "model": name, "contrast": f"{condition}-A",
                    "outcome": "corrected" if current[i] == y_test[i] else "introduced",
                })
    result["changed_predictions"] = changes
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2)
        stream.write("\n")
    return result


if __name__ == "__main__":
    processed = ROOT / "data" / "processed"
    output = processed / "primary_evaluation.json"
    if output.exists():
        raise SystemExit(f"Evaluation already exists: {output}")
    run(processed, output)
    print(output)
