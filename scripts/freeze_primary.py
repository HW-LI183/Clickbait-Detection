"""Freeze the published primary models and export auditable test predictions."""

import argparse
import json
import math
import os
import platform
import sys
import tempfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import scipy
import sklearn

from evaluate_primary import macro_f1, metrics
from tfidf_preexperiment import (
    CONDITIONS,
    FIELDS,
    MODELS,
    ROOT,
    combine_blocks,
    load_training,
    pilot_model,
    read_jsonl,
    sha256,
    vectorize_fold,
)


LABELS = {False: "no-clickbait", True: "clickbait"}
PRIMARY_RESULT_COMMIT = "9200f194464e990c1a376a29e3f096a239c5c75e"
FLOAT_TOLERANCE = 1e-12


def read_json(path):
    with path.open("r", encoding="utf-8") as stream:
        return json.load(stream)


def write_json(path, value):
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, sort_keys=True)
        stream.write("\n")


def write_jsonl(path, rows):
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def repository_path(path):
    path = path.resolve()
    try:
        return path.relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        return str(path)


def assert_close(name, actual, expected, tolerance=FLOAT_TOLERANCE):
    if not math.isclose(float(actual), float(expected), rel_tol=0.0, abs_tol=tolerance):
        raise ValueError(f"{name}: reproduced {actual!r}, published {expected!r}")


def verify_model_metrics(model_key, reproduced, published):
    drift = {}
    for name in (
        "validation_macro_f1",
        "macro_f1",
        "clickbait_precision",
        "clickbait_recall",
        "clickbait_f1",
        "average_precision",
    ):
        tolerance = 1e-5 if name == "average_precision" else FLOAT_TOLERANCE
        assert_close(
            f"{model_key}.{name}", reproduced[name], published[name], tolerance
        )
        drift[name] = float(reproduced[name]) - float(published[name])
    if reproduced["confusion_matrix"] != published["confusion_matrix"]:
        raise ValueError(
            f"{model_key}.confusion_matrix: reproduced "
            f"{reproduced['confusion_matrix']!r}, published "
            f"{published['confusion_matrix']!r}"
        )
    return drift


def prediction_entry(score, threshold, truth):
    predicted = bool(score >= threshold)
    return {
        "score": float(score),
        "threshold": float(threshold),
        "prediction": LABELS[predicted],
        "correct": predicted == bool(truth),
    }


def transition_entry(base, current):
    changed = base["prediction"] != current["prediction"]
    if not changed:
        outcome = "unchanged"
    elif current["correct"]:
        outcome = "corrected"
    else:
        outcome = "introduced"
    return {
        "changed": changed,
        "outcome": outcome,
        "direction": f"{base['prediction']} -> {current['prediction']}",
        "score_delta": current["score"] - base["score"],
    }


def add_transitions(rows):
    for row in rows:
        transitions = {}
        for model_name in MODELS:
            base = row["models"][f"A/{model_name}"]
            for condition in ("B", "C"):
                key = f"{condition}-A/{model_name}"
                transitions[key] = transition_entry(
                    base, row["models"][f"{condition}/{model_name}"]
                )
        row["transitions"] = transitions
    return rows


def summarize_predictions(rows):
    model_summary = {}
    for condition in CONDITIONS:
        for model_name in MODELS:
            key = f"{condition}/{model_name}"
            counts = Counter()
            for row in rows:
                item = row["models"][key]
                truth = row["truth_class"]
                if item["correct"]:
                    counts["correct"] += 1
                elif truth == "no-clickbait":
                    counts["false_positive"] += 1
                else:
                    counts["false_negative"] += 1
                counts[f"predicted_{item['prediction']}"] += 1
            model_summary[key] = dict(sorted(counts.items()))

    transition_summary = {}
    for model_name in MODELS:
        for condition in ("B", "C"):
            key = f"{condition}-A/{model_name}"
            counts = Counter()
            score_deltas = []
            for row in rows:
                item = row["transitions"][key]
                counts[item["outcome"]] += 1
                if item["changed"]:
                    counts[f"direction:{item['direction']}"] += 1
                    counts[f"truth:{row['truth_class']}"] += 1
                    score_deltas.append(item["score_delta"])
            transition_summary[key] = {
                **dict(sorted(counts.items())),
                "mean_score_delta_among_changes": (
                    float(np.mean(score_deltas)) if score_deltas else 0.0
                ),
            }
    standard_deviations = np.array([
        row["truth_judgment_std"] for row in rows
        if row.get("truth_judgment_std") is not None
    ])
    disagreement = None
    if len(standard_deviations):
        low_threshold, high_threshold = np.quantile(
            standard_deviations, [0.25, 0.75]
        )
        disagreement = {
            "definition": (
                "Low and high disagreement use the bottom and top quartiles "
                "of the five-annotator judgment standard deviation. Ties at "
                "the thresholds are retained. Annotation statistics are used "
                "only for descriptive analysis, never as model inputs."
            ),
            "low_threshold_max": float(low_threshold),
            "high_threshold_min": float(high_threshold),
            "groups": {},
        }
        for name, predicate in (
            ("low", lambda value: value <= low_threshold),
            ("high", lambda value: value >= high_threshold),
        ):
            group = [row for row in rows if predicate(row["truth_judgment_std"])]
            truth = np.array([row["truth_class"] == "clickbait" for row in group])
            condition_scores = {}
            for condition in ("A", "C"):
                predicted = np.array([
                    row["models"][f"{condition}/logistic_regression"]["prediction"]
                    == "clickbait"
                    for row in group
                ])
                condition_scores[condition] = macro_f1(truth, predicted)
            disagreement["groups"][name] = {
                "rows": len(group),
                "A_logistic_macro_f1": condition_scores["A"],
                "C_logistic_macro_f1": condition_scores["C"],
                "C_minus_A": condition_scores["C"] - condition_scores["A"],
            }
    return {
        "models": model_summary,
        "transitions": transition_summary,
        "annotator_disagreement": disagreement,
    }


def select_error_cases(rows, records, truth_details, cases_per_bucket):
    selected = []
    for model_name in MODELS:
        for condition in ("B", "C"):
            transition_key = f"{condition}-A/{model_name}"
            for outcome in ("corrected", "introduced"):
                candidates = [
                    row for row in rows
                    if row["transitions"][transition_key]["outcome"] == outcome
                ]
                candidates.sort(
                    key=lambda row: (
                        -abs(row["transitions"][transition_key]["score_delta"]),
                        row["id"],
                    )
                )
                for row in candidates[:cases_per_bucket]:
                    source = records[row["id"]]
                    annotation = truth_details.get(row["id"], {})
                    body_words = len(source["body_full"].split())
                    selected.append({
                        "case_type": "prediction_change",
                        "id": row["id"],
                        "group_id": row["group_id"],
                        "truth_class": row["truth_class"],
                        "model": model_name,
                        "contrast": f"{condition}-A",
                        "outcome": outcome,
                        "transition": row["transitions"][transition_key],
                        "baseline": row["models"][f"A/{model_name}"],
                        "current": row["models"][f"{condition}/{model_name}"],
                        "post_text": source["post_text"],
                        "title": source["title"],
                        "body_excerpt": source["body_full"][:800],
                        "body_words": body_words,
                        "body_truncated_at_1000": body_words > 1000,
                        "truth_mean": annotation.get("truthMean"),
                        "truth_judgments": annotation.get("truthJudgments"),
                        "truth_judgment_std": row.get("truth_judgment_std"),
                    })

    primary_key = "C/logistic_regression"
    for error_type in ("false_positive", "false_negative"):
        candidates = []
        for row in rows:
            item = row["models"][primary_key]
            is_false_positive = (
                row["truth_class"] == "no-clickbait"
                and item["prediction"] == "clickbait"
            )
            is_false_negative = (
                row["truth_class"] == "clickbait"
                and item["prediction"] == "no-clickbait"
            )
            if (
                error_type == "false_positive" and is_false_positive
                or error_type == "false_negative" and is_false_negative
            ):
                margin = abs(item["score"] - item["threshold"])
                candidates.append((margin, row))
        candidates.sort(key=lambda item: (-item[0], item[1]["id"]))
        for margin, row in candidates[:cases_per_bucket]:
            source = records[row["id"]]
            annotation = truth_details.get(row["id"], {})
            body_words = len(source["body_full"].split())
            selected.append({
                "case_type": "primary_classification_error",
                "error_type": error_type,
                "id": row["id"],
                "group_id": row["group_id"],
                "truth_class": row["truth_class"],
                "model": primary_key,
                "prediction": row["models"][primary_key],
                "confidence_margin": margin,
                "post_text": source["post_text"],
                "title": source["title"],
                "body_excerpt": source["body_full"][:800],
                "body_words": body_words,
                "body_truncated_at_1000": body_words > 1000,
                "truth_mean": annotation.get("truthMean"),
                "truth_judgments": annotation.get("truthJudgments"),
                "truth_judgment_std": row.get("truth_judgment_std"),
            })
    return selected


def reproduced_changed_set(rows):
    changes = set()
    for row in rows:
        for key, item in row["transitions"].items():
            if not item["changed"]:
                continue
            contrast, model_name = key.split("/", 1)
            changes.add((row["id"], model_name, contrast, item["outcome"]))
    return changes


def published_changed_set(primary):
    return {
        (row["id"], row["model"], row["contrast"], row["outcome"])
        for row in primary["changed_predictions"]
    }


def freeze(processed_dir, primary_path, output_dir, cases_per_bucket=3):
    primary = read_json(primary_path)
    train_ids, texts, train_labels, _, input_hashes = load_training(processed_dir)
    manifest = read_jsonl(processed_dir / "split_manifest.jsonl")
    records = read_jsonl(processed_dir / "records.jsonl")
    truth_path = processed_dir.parent / "truth.jsonl"
    truth_details = read_jsonl(truth_path) if truth_path.is_file() else {}
    validation_ids = sorted(
        row_id for row_id, row in manifest.items() if row["split"] == "validation"
    )
    test_ids = sorted(
        row_id for row_id, row in manifest.items() if row["split"] == "test"
    )
    split_sizes = [len(train_ids), len(validation_ids), len(test_ids)]
    if split_sizes != primary["split_sizes"]:
        raise ValueError(
            f"Frozen split sizes differ: reproduced {split_sizes}, "
            f"published {primary['split_sizes']}"
        )
    if input_hashes != primary["inputs_sha256"]:
        raise ValueError("Processed input hashes differ from the published evaluation")

    settings = primary["settings"]
    limits = tuple(settings["limits"])
    ngram = tuple(settings["ngram"])
    weights = tuple(settings["weights"])
    for field in FIELDS:
        texts[field].extend(records[row_id][field] for row_id in validation_ids)
    train_end = len(train_ids)
    vectors, encoders, dimensions = vectorize_fold(
        texts,
        np.arange(train_end),
        np.arange(train_end, train_end + len(validation_ids)),
        limits,
        ngram,
    )

    test_blocks = {
        field: (
            encoder.transform([records[row_id][field] for row_id in test_ids]),
            vectors[field][1][:0],
        )
        for field, encoder in encoders.items()
    }
    y_train = train_labels == "clickbait"
    y_validation = np.array([
        manifest[row_id]["truthClass"] == "clickbait" for row_id in validation_ids
    ])
    y_test = np.array([
        manifest[row_id]["truthClass"] == "clickbait" for row_id in test_ids
    ])

    rows = []
    for row_id in test_ids:
        judgments = truth_details.get(row_id, {}).get("truthJudgments")
        rows.append({
            "id": row_id,
            "group_id": manifest[row_id]["group_id"],
            "truth_class": manifest[row_id]["truthClass"],
            "truth_mean": truth_details.get(row_id, {}).get("truthMean"),
            "truth_judgment_std": (
                float(np.std(judgments)) if judgments else None
            ),
            "models": {},
        })
    frozen_models = {}
    reproduced_metrics = {}
    metric_drift = {}
    for condition, fields in CONDITIONS.items():
        train, validation = combine_blocks(vectors, fields, weights)
        test, _ = combine_blocks(test_blocks, fields, weights)
        for model_name in MODELS:
            model_key = f"{condition}/{model_name}"
            published = primary["models"][model_key]
            model = pilot_model(model_name).set_params(C=published["C"])
            model.fit(train, y_train)
            threshold = float(published["threshold"])
            validation_scores = model.decision_function(validation)
            validation_macro_f1 = macro_f1(
                y_validation, validation_scores >= threshold
            )
            test_scores = model.decision_function(test)
            test_predictions = test_scores >= threshold
            reproduced = {
                "C": float(model.C),
                "threshold": threshold,
                "validation_macro_f1": validation_macro_f1,
                **metrics(y_test, test_predictions, test_scores),
            }
            metric_drift[model_key] = verify_model_metrics(
                model_key, reproduced, published
            )
            reproduced_metrics[model_key] = reproduced
            frozen_models[model_key] = model
            for index, score in enumerate(test_scores):
                rows[index]["models"][model_key] = prediction_entry(
                    score, threshold, y_test[index]
                )

    add_transitions(rows)
    reproduced_changes = reproduced_changed_set(rows)
    published_changes = published_changed_set(primary)
    if reproduced_changes != published_changes:
        missing = sorted(published_changes - reproduced_changes)[:5]
        extra = sorted(reproduced_changes - published_changes)[:5]
        raise ValueError(
            "Per-record decisions do not reproduce the published change list; "
            f"missing examples={missing!r}, extra examples={extra!r}"
        )

    summary = summarize_predictions(rows)
    error_cases = select_error_cases(
        rows, records, truth_details, cases_per_bucket
    )
    output_dir = output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(f"Frozen output already exists: {output_dir}")
    output_dir.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(
        prefix=".freeze-primary-", dir=output_dir.parent
    ) as temporary:
        stage = Path(temporary)
        predictions_path = stage / "test_predictions.jsonl"
        metrics_path = stage / "reproduced_metrics.json"
        summary_path = stage / "error_analysis_summary.json"
        cases_path = stage / "error_case_candidates.jsonl"
        models_path = stage / "frozen_models.joblib"
        write_jsonl(predictions_path, rows)
        write_json(metrics_path, reproduced_metrics)
        write_json(summary_path, summary)
        write_jsonl(cases_path, error_cases)
        joblib.dump({
            "status": "frozen",
            "conditions": CONDITIONS,
            "fields": FIELDS,
            "vectorizers": encoders,
            "models": frozen_models,
            "settings": settings,
            "thresholds": {
                key: value["threshold"] for key, value in primary["models"].items()
            },
            "input_sha256": input_hashes,
            "primary_evaluation_sha256": sha256(primary_path),
        }, models_path, compress=3)

        artifacts = {
            path.name: {"sha256": sha256(path), "bytes": path.stat().st_size}
            for path in (
                predictions_path,
                metrics_path,
                summary_path,
                cases_path,
                models_path,
            )
        }
        manifest_value = {
            "status": "frozen",
            "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
            "primary_result_commit": PRIMARY_RESULT_COMMIT,
            "policy": {
                "primary_result_may_not_be_reselected_from_test": True,
                "allowed": [
                    "exact reproducibility checks",
                    "descriptive error analysis",
                    "predeclared sensitivity analyses labelled as secondary",
                ],
                "prohibited": [
                    "retuning from test performance",
                    "replacing the primary result with a post-test best model",
                    "changing the frozen split or sample exclusions",
                ],
            },
            "source": {
                "primary_evaluation": repository_path(primary_path),
                "primary_evaluation_sha256": sha256(primary_path),
                "processed_inputs_sha256": input_hashes,
                "freeze_script_sha256": sha256(Path(__file__)),
            },
            "reproduction": {
                "classification_metrics_and_confusion_matrices_match": True,
                "published_changed_decision_list_matches": True,
                "metric_drift_reproduced_minus_published": metric_drift,
                "average_precision_tolerance": 1e-5,
                "note": (
                    "Continuous decision scores can differ by floating-point "
                    "roundoff across binary runtimes; the largest allowed "
                    "aggregate AP drift is 1e-5. Class decisions, validation "
                    "Macro-F1, test classification metrics, confusion matrices, "
                    "and the published changed-decision list must match."
                ),
            },
            "settings": settings,
            "vocabulary_dimensions": dimensions,
            "models": {
                key: {
                    "C": value["C"],
                    "threshold": value["threshold"],
                    "validation_macro_f1": value["validation_macro_f1"],
                    "test_macro_f1": value["macro_f1"],
                }
                for key, value in reproduced_metrics.items()
            },
            "split_sizes": split_sizes,
            "test_group_count": len({manifest[row_id]["group_id"] for row_id in test_ids}),
            "artifacts": artifacts,
            "environment": {
                "python": sys.version,
                "platform": platform.platform(),
                "numpy": np.__version__,
                "scipy": scipy.__version__,
                "scikit_learn": sklearn.__version__,
                "joblib": joblib.__version__,
                "OPENBLAS_NUM_THREADS": os.environ.get("OPENBLAS_NUM_THREADS"),
                "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS"),
            },
        }
        write_json(stage / "freeze_manifest.json", manifest_value)
        stage.rename(output_dir)

    print(f"Frozen six models and wrote {len(rows):,} test predictions to {output_dir}")
    return manifest_value, rows, summary, error_cases


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--processed-dir", type=Path, default=ROOT / "data" / "processed"
    )
    parser.add_argument(
        "--primary-result", type=Path,
        default=ROOT / "output" / "primary_evaluation.json",
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=ROOT / "output" / "frozen_primary",
    )
    parser.add_argument("--cases-per-bucket", type=int, default=3)
    args = parser.parse_args()
    if args.cases_per_bucket <= 0:
        parser.error("--cases-per-bucket must be positive")
    freeze(
        args.processed_dir,
        args.primary_result,
        args.output_dir,
        args.cases_per_bucket,
    )


if __name__ == "__main__":
    main()
