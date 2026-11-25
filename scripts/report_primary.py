"""Verify frozen test inference and build poster-ready result assets."""

import argparse
import csv
import gc
import hashlib
import html
import json
import math
import os
import platform
import statistics
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter_ns

# Keep the formal timing run single-threaded unless the caller explicitly chose
# another setting before starting Python.
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import joblib
import numpy as np
import scipy
import sklearn
from scipy.sparse import hstack

from tfidf_preexperiment import CONDITIONS, FIELDS, MODELS, ROOT, read_jsonl


CONDITION_LABELS = {
    "A": "Post text",
    "B": "Post text + title",
    "C": "Post text + title + body",
}
MODEL_LABELS = {
    "logistic_regression": "Logistic regression",
    "linear_svm": "Linear SVM",
}
MODEL_COLORS = {
    "logistic_regression": "#0072B2",
    "linear_svm": "#D55E00",
}
REPORT_FILENAMES = (
    "primary_results.csv",
    "primary_results.md",
    "primary_results_table.svg",
    "primary_macro_f1.svg",
    "primary_error_counts.svg",
    "inference_time.json",
    "inference_time.csv",
    "inference_time.md",
    "inference_latency.svg",
)


def read_json(path):
    with path.open("r", encoding="utf-8") as stream:
        return json.load(stream)


def read_jsonl_rows(path):
    with path.open("r", encoding="utf-8") as stream:
        return [json.loads(line) for line in stream]


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_write_text(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", newline="\n", delete=False, dir=path.parent
    ) as stream:
        stream.write(text)
        temporary = Path(stream.name)
    os.replace(temporary, path)


def atomic_write_json(path, value):
    atomic_write_text(
        path,
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )


def model_order():
    return [
        f"{condition}/{model_name}"
        for condition in CONDITIONS
        for model_name in MODELS
    ]


def build_result_rows(metrics):
    rows = []
    for model_key in model_order():
        condition, model_name = model_key.split("/", 1)
        item = metrics[model_key]
        tn, fp = item["confusion_matrix"][0]
        fn, tp = item["confusion_matrix"][1]
        rows.append({
            "condition": condition,
            "input": CONDITION_LABELS[condition],
            "model": MODEL_LABELS[model_name],
            "C": item["C"],
            "threshold": item["threshold"],
            "validation_macro_f1": item["validation_macro_f1"],
            "test_macro_f1": item["macro_f1"],
            "clickbait_precision": item["clickbait_precision"],
            "clickbait_recall": item["clickbait_recall"],
            "clickbait_f1": item["clickbait_f1"],
            "average_precision": item["average_precision"],
            "true_negative": tn,
            "false_positive": fp,
            "false_negative": fn,
            "true_positive": tp,
        })
    return rows


def csv_text(rows, columns):
    from io import StringIO

    stream = StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=columns, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue()


def result_markdown(rows):
    lines = [
        "# Frozen primary results",
        "",
        "The table reports the held-out test split with 2,894 records. "
        "Condition A uses post text, B adds the linked title, and C also adds "
        "the first 1,000 article-body words.",
        "",
        "| Condition | Model | Validation Macro-F1 | Test Macro-F1 | Precision | Recall | Clickbait F1 | AP | FP | FN |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            "| {condition} | {model} | {validation_macro_f1:.4f} | "
            "{test_macro_f1:.4f} | {clickbait_precision:.4f} | "
            "{clickbait_recall:.4f} | {clickbait_f1:.4f} | "
            "{average_precision:.4f} | {false_positive} | "
            "{false_negative} |".format(**row)
        )
    lines.extend([
        "",
        "Precision, recall, and Clickbait F1 treat clickbait as the positive "
        "class. AP means average precision. FP and FN mean false positives and "
        "false negatives.",
        "",
    ])
    return "\n".join(lines)


def svg_document(width, height, body, title):
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" '
        f'height="{height}" viewBox="0 0 {width} {height}" role="img" '
        f'aria-labelledby="title">\n<title id="title">{html.escape(title)}</title>\n'
        f'<rect width="100%" height="100%" fill="#FFFFFF"/>\n{body}\n</svg>\n'
    )


def text_svg(x, y, value, size=24, weight=400, anchor="start", fill="#222222"):
    return (
        f'<text x="{x}" y="{y}" font-family="Arial, Helvetica, sans-serif" '
        f'font-size="{size}" font-weight="{weight}" text-anchor="{anchor}" '
        f'fill="{fill}">{html.escape(str(value))}</text>'
    )


def results_table_svg(rows):
    width, height = 1760, 640
    pieces = [text_svg(70, 60, "Frozen primary test results", 34, 700)]
    pieces.append(text_svg(70, 96, "Held-out test set, n = 2,894", 20, 400, fill="#555555"))
    columns = [
        (70, "Condition", "condition"),
        (220, "Model", "model"),
        (600, "Macro-F1", "test_macro_f1"),
        (800, "Precision", "clickbait_precision"),
        (1000, "Recall", "clickbait_recall"),
        (1190, "Clickbait F1", "clickbait_f1"),
        (1430, "FP", "false_positive"),
        (1570, "FN", "false_negative"),
    ]
    header_y = 155
    pieces.append(f'<rect x="55" y="115" width="1650" height="62" rx="8" fill="#E8F1F7"/>')
    for x, label, _ in columns:
        pieces.append(text_svg(x, header_y, label, 21, 700))
    for index, row in enumerate(rows):
        top = 177 + index * 69
        if index % 2:
            pieces.append(f'<rect x="55" y="{top}" width="1650" height="69" fill="#F7F7F7"/>')
        y = top + 44
        for x, _, key in columns:
            value = row[key]
            if key in {
                "test_macro_f1", "clickbait_precision", "clickbait_recall", "clickbait_f1"
            }:
                value = f"{value:.4f}"
            pieces.append(text_svg(x, y, value, 21))
    pieces.append(text_svg(70, 617, "Positive class: clickbait. FP = false positives; FN = false negatives.", 18, fill="#555555"))
    return svg_document(width, height, "\n".join(pieces), "Frozen primary test results table")


def macro_f1_svg(rows):
    width, height = 1500, 900
    left, right, top, bottom = 300, 1350, 130, 735
    minimum, maximum = 0.70, 0.80
    pieces = [
        text_svg(70, 62, "Test Macro-F1 by input condition", 36, 700),
        text_svg(70, 100, "Dots show the frozen held-out test result", 20, fill="#555555"),
    ]
    for tick in np.arange(minimum, maximum + 0.0001, 0.02):
        x = left + (tick - minimum) / (maximum - minimum) * (right - left)
        pieces.append(f'<line x1="{x:.1f}" y1="{top}" x2="{x:.1f}" y2="{bottom}" stroke="#D9D9D9" stroke-width="2"/>')
        pieces.append(text_svg(f"{x:.1f}", bottom + 42, f"{tick:.2f}", 20, anchor="middle", fill="#555555"))
    pieces.append(text_svg((left + right) / 2, 830, "Macro-F1", 24, 700, anchor="middle"))
    row_by_key = {
        (row["condition"], row["model"]): row for row in rows
    }
    model_names = list(MODEL_LABELS)
    for condition_index, condition in enumerate(CONDITIONS):
        center_y = 225 + condition_index * 190
        pieces.append(text_svg(70, center_y - 10, f"{condition}: {CONDITION_LABELS[condition]}", 22, 700))
        for model_index, model_name in enumerate(model_names):
            model_label = MODEL_LABELS[model_name]
            row = row_by_key[(condition, model_label)]
            value = row["test_macro_f1"]
            y = center_y + (model_index * 58)
            x = left + (value - minimum) / (maximum - minimum) * (right - left)
            pieces.append(text_svg(255, y + 7, model_label, 19, anchor="end", fill="#444444"))
            pieces.append(f'<line x1="{left}" y1="{y}" x2="{x:.1f}" y2="{y}" stroke="{MODEL_COLORS[model_name]}" stroke-width="5" opacity="0.45"/>')
            pieces.append(f'<circle cx="{x:.1f}" cy="{y}" r="12" fill="{MODEL_COLORS[model_name]}"/>')
            pieces.append(text_svg(f"{x + 22:.1f}", y + 7, f"{value:.4f}", 20, 700, fill=MODEL_COLORS[model_name]))
    pieces.append(text_svg(70, 875, "Displayed axis: 0.70 to 0.80. Exact values are printed beside each point.", 18, fill="#555555"))
    return svg_document(width, height, "\n".join(pieces), "Test Macro-F1 by input condition")


def error_counts_svg(rows):
    width, height = 1500, 900
    left, right, top = 400, 1360, 145
    maximum = max(row["false_positive"] + row["false_negative"] for row in rows)
    axis_max = int(np.ceil(maximum / 100.0) * 100)
    pieces = [
        text_svg(70, 62, "Test classification errors", 36, 700),
        text_svg(70, 100, "False positives and false negatives, n = 2,894", 20, fill="#555555"),
    ]
    for tick in range(0, axis_max + 1, 100):
        x = left + tick / axis_max * (right - left)
        pieces.append(f'<line x1="{x:.1f}" y1="{top}" x2="{x:.1f}" y2="760" stroke="#E2E2E2" stroke-width="2"/>')
        pieces.append(text_svg(f"{x:.1f}", 800, tick, 19, anchor="middle", fill="#555555"))
    for index, row in enumerate(rows):
        y = top + 35 + index * 96
        label = f'{row["condition"]}  {row["model"]}'
        pieces.append(text_svg(left - 25, y + 25, label, 20, anchor="end"))
        fp_width = row["false_positive"] / axis_max * (right - left)
        fn_width = row["false_negative"] / axis_max * (right - left)
        pieces.append(f'<rect x="{left}" y="{y}" width="{fp_width:.1f}" height="42" fill="#E69F00"/>')
        pieces.append(f'<rect x="{left + fp_width:.1f}" y="{y}" width="{fn_width:.1f}" height="42" fill="#CC79A7"/>')
        total = row["false_positive"] + row["false_negative"]
        pieces.append(text_svg(f"{left + fp_width + fn_width + 15:.1f}", y + 29, total, 19, 700))
    pieces.extend([
        '<rect x="505" y="850" width="26" height="26" fill="#E69F00"/>',
        text_svg(545, 871, "False positive", 19),
        '<rect x="755" y="850" width="26" height="26" fill="#CC79A7"/>',
        text_svg(795, 871, "False negative", 19),
    ])
    return svg_document(width, height, "\n".join(pieces), "False-positive and false-negative counts")


def transform_condition(bundle, condition, test_texts):
    weights = dict(zip(FIELDS, bundle["settings"]["weights"]))
    selected_fields = bundle["conditions"][condition]
    norm = math.sqrt(sum(weights[field] ** 2 for field in selected_fields))
    blocks = []
    for field in selected_fields:
        block = bundle["vectorizers"][field].transform(test_texts[field])
        block = block * (weights[field] / norm)
        blocks.append(block)
    if len(blocks) == 1:
        return blocks[0].tocsr()
    return hstack(blocks, format="csr")


def timing_stats(values):
    ordered = sorted(float(value) for value in values)
    return {
        "median": statistics.median(ordered),
        "mean": statistics.fmean(ordered),
        "min": ordered[0],
        "max": ordered[-1],
        "stdev": statistics.stdev(ordered) if len(ordered) > 1 else 0.0,
        "raw": ordered,
    }


def run_inference_once(bundle, model_key, test_texts):
    condition, _ = model_key.split("/", 1)
    started = perf_counter_ns()
    matrix = transform_condition(bundle, condition, test_texts)
    transformed = perf_counter_ns()
    scores = bundle["models"][model_key].decision_function(matrix)
    predictions = scores >= bundle["thresholds"][model_key]
    finished = perf_counter_ns()
    return {
        "feature_ms": (transformed - started) / 1_000_000,
        "classifier_ms": (finished - transformed) / 1_000_000,
        "end_to_end_ms": (finished - started) / 1_000_000,
        "scores": scores,
        "predictions": predictions,
    }


def assert_frozen_predictions(model_key, result, expected_rows):
    expected_scores = np.array([
        row["models"][model_key]["score"] for row in expected_rows
    ])
    expected_predictions = np.array([
        row["models"][model_key]["prediction"] == "clickbait"
        for row in expected_rows
    ])
    if not np.allclose(result["scores"], expected_scores, rtol=0.0, atol=1e-12):
        difference = float(np.max(np.abs(result["scores"] - expected_scores)))
        raise ValueError(f"{model_key}: decision scores differ by up to {difference}")
    if not np.array_equal(result["predictions"], expected_predictions):
        raise ValueError(f"{model_key}: predictions differ from frozen export")


def benchmark_models(bundle, test_texts, expected_rows, warmups, repeats):
    model_results = {}
    sample_count = len(expected_rows)
    for model_key in model_order():
        for _ in range(warmups):
            result = run_inference_once(bundle, model_key, test_texts)
            assert_frozen_predictions(model_key, result, expected_rows)
        feature_values = []
        classifier_values = []
        total_values = []
        for _ in range(repeats):
            gc.collect()
            result = run_inference_once(bundle, model_key, test_texts)
            assert_frozen_predictions(model_key, result, expected_rows)
            feature_values.append(result["feature_ms"])
            classifier_values.append(result["classifier_ms"])
            total_values.append(result["end_to_end_ms"])
        total_stats = timing_stats(total_values)
        model_results[model_key] = {
            "feature_transform_ms": timing_stats(feature_values),
            "classifier_and_threshold_ms": timing_stats(classifier_values),
            "end_to_end_ms": total_stats,
            "median_ms_per_sample": total_stats["median"] / sample_count,
            "median_samples_per_second": sample_count / (total_stats["median"] / 1000),
        }
    return model_results


def inference_rows(timing):
    rows = []
    for model_key in model_order():
        condition, model_name = model_key.split("/", 1)
        item = timing["models"][model_key]
        rows.append({
            "condition": condition,
            "input": CONDITION_LABELS[condition],
            "model": MODEL_LABELS[model_name],
            "feature_transform_median_ms": item["feature_transform_ms"]["median"],
            "classifier_median_ms": item["classifier_and_threshold_ms"]["median"],
            "end_to_end_median_ms": item["end_to_end_ms"]["median"],
            "end_to_end_stdev_ms": item["end_to_end_ms"]["stdev"],
            "median_ms_per_sample": item["median_ms_per_sample"],
            "median_samples_per_second": item["median_samples_per_second"],
        })
    return rows


def inference_markdown(timing, rows):
    environment = timing["environment"]
    cold = timing["cold_start"]
    lines = [
        "# Frozen primary inference time",
        "",
        f'This run used {timing["protocol"]["warmups"]} warm-ups and '
        f'{timing["protocol"]["repeats"]} measured repeats on the complete '
        f'{timing["protocol"]["batch_size"]}-record test batch. Each model was '
        "timed independently. End-to-end time includes TF-IDF transformation, "
        "sparse block assembly, classifier scoring, and thresholding. It excludes "
        "file loading.",
        "",
        f'- Python: `{environment["python"]}`',
        f'- Platform: `{environment["platform"]}`',
        f'- Processor: `{environment["processor"]}`',
        f'- Logical CPUs: `{environment["logical_cpu_count"]}`',
        f'- Model bundle load: `{cold["model_bundle_load_ms"]:.2f} ms`',
        f'- Test data load and preparation: `{cold["test_data_load_and_prepare_ms"]:.2f} ms`',
        "",
        "| Condition | Model | TF-IDF ms | Classifier ms | End-to-end ms | ms/sample | Samples/s |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            "| {condition} | {model} | {feature_transform_median_ms:.2f} | "
            "{classifier_median_ms:.2f} | {end_to_end_median_ms:.2f} | "
            "{median_ms_per_sample:.4f} | {median_samples_per_second:.1f} |".format(**row)
        )
    lines.extend([
        "",
        "Times are machine-specific. `inference_time.json` retains every measured "
        "repeat, the software versions, and the thread settings. The one-time load "
        "measurements are single observations and may depend on the operating-system "
        "file cache.",
        "",
    ])
    return "\n".join(lines)


def inference_svg(rows, batch_size):
    width, height = 1500, 900
    left, right, top = 430, 1360, 150
    maximum = max(row["end_to_end_median_ms"] for row in rows)
    axis_max = int(np.ceil(maximum / 250.0) * 250)
    pieces = [
        text_svg(70, 62, "Frozen model inference latency", 36, 700),
        text_svg(70, 100, f"Median end-to-end time for a {batch_size:,}-record batch", 20, fill="#555555"),
    ]
    tick_step = max(250, int(np.ceil(axis_max / 5 / 250.0) * 250))
    for tick in range(0, axis_max + 1, tick_step):
        x = left + tick / axis_max * (right - left)
        pieces.append(f'<line x1="{x:.1f}" y1="{top}" x2="{x:.1f}" y2="760" stroke="#E2E2E2" stroke-width="2"/>')
        pieces.append(text_svg(f"{x:.1f}", 800, tick, 19, anchor="middle", fill="#555555"))
    for index, row in enumerate(rows):
        y = top + 35 + index * 96
        model_name = next(key for key, value in MODEL_LABELS.items() if value == row["model"])
        label = f'{row["condition"]}  {row["model"]}'
        pieces.append(text_svg(left - 25, y + 25, label, 20, anchor="end"))
        bar_width = row["end_to_end_median_ms"] / axis_max * (right - left)
        pieces.append(f'<rect x="{left}" y="{y}" width="{bar_width:.1f}" height="42" rx="4" fill="{MODEL_COLORS[model_name]}"/>')
        pieces.append(text_svg(f"{left + bar_width + 15:.1f}", y + 29, f'{row["end_to_end_median_ms"]:.1f} ms', 19, 700))
    pieces.append(text_svg((left + right) / 2, 855, "Milliseconds per complete test batch", 22, 700, anchor="middle"))
    return svg_document(width, height, "\n".join(pieces), "Frozen model inference latency")


def verify_bundle(bundle, primary, freeze_manifest):
    if bundle.get("status") != "frozen" or freeze_manifest.get("status") != "frozen":
        raise ValueError("The model bundle and freeze manifest must both be frozen")
    if set(bundle["models"]) != set(primary["models"]):
        raise ValueError("Frozen model keys differ from the primary evaluation")
    if set(bundle["vectorizers"]) != set(FIELDS):
        raise ValueError("Frozen vectorizer fields differ from the formal protocol")


def run(frozen_dir, processed_dir, output_dir, warmups, repeats):
    frozen_dir = frozen_dir.resolve()
    processed_dir = processed_dir.resolve()
    output_dir = output_dir.resolve()
    primary_path = ROOT / "output" / "primary_evaluation.json"

    load_started = perf_counter_ns()
    bundle = joblib.load(frozen_dir / "frozen_models.joblib")
    bundle_load_ms = (perf_counter_ns() - load_started) / 1_000_000
    primary = read_json(primary_path)
    metrics = read_json(frozen_dir / "reproduced_metrics.json")
    freeze_manifest = read_json(frozen_dir / "freeze_manifest.json")
    verify_bundle(bundle, primary, freeze_manifest)

    data_started = perf_counter_ns()
    records = read_jsonl(processed_dir / "records.jsonl")
    manifest = read_jsonl(processed_dir / "split_manifest.jsonl")
    test_ids = sorted(
        row_id for row_id, row in manifest.items() if row["split"] == "test"
    )
    expected_rows = read_jsonl_rows(frozen_dir / "test_predictions.jsonl")
    if [row["id"] for row in expected_rows] != test_ids:
        raise ValueError("Frozen prediction IDs differ from the formal test split")
    test_texts = {
        field: [records[row_id][field] for row_id in test_ids]
        for field in FIELDS
    }
    data_load_ms = (perf_counter_ns() - data_started) / 1_000_000

    timing_models = benchmark_models(
        bundle, test_texts, expected_rows, warmups, repeats
    )
    timing = {
        "measured_at_utc": datetime.now(timezone.utc).isoformat(),
        "protocol": {
            "batch_size": len(test_ids),
            "warmups": warmups,
            "repeats": repeats,
            "clock": "time.perf_counter_ns",
            "independent_model_runs": True,
            "end_to_end_definition": (
                "TF-IDF transformation of the required fields, sparse block "
                "assembly, classifier decision_function, and frozen thresholding"
            ),
            "excluded_from_end_to_end": [
                "model bundle loading",
                "processed JSONL reading",
                "test text list preparation",
            ],
            "verification": (
                "Every warm-up and measured repeat must reproduce all frozen "
                "decision scores within absolute tolerance 1e-12 and every class "
                "decision exactly."
            ),
        },
        "cold_start": {
            "model_bundle_load_ms": bundle_load_ms,
            "test_data_load_and_prepare_ms": data_load_ms,
            "combined_load_and_prepare_ms": bundle_load_ms + data_load_ms,
            "note": "Single observations; operating-system file caching can affect them.",
        },
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "machine": platform.machine(),
            "processor": platform.processor() or os.environ.get("PROCESSOR_IDENTIFIER", "unknown"),
            "logical_cpu_count": os.cpu_count(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "scikit_learn": sklearn.__version__,
            "joblib": joblib.__version__,
            "OPENBLAS_NUM_THREADS": os.environ.get("OPENBLAS_NUM_THREADS"),
            "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS"),
        },
        "models": timing_models,
    }

    result_rows = build_result_rows(metrics)
    timing_rows = inference_rows(timing)
    result_columns = list(result_rows[0])
    timing_columns = list(timing_rows[0])
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = {
        "primary_results.csv": csv_text(result_rows, result_columns),
        "primary_results.md": result_markdown(result_rows),
        "primary_results_table.svg": results_table_svg(result_rows),
        "primary_macro_f1.svg": macro_f1_svg(result_rows),
        "primary_error_counts.svg": error_counts_svg(result_rows),
        "inference_time.csv": csv_text(timing_rows, timing_columns),
        "inference_time.md": inference_markdown(timing, timing_rows),
        "inference_latency.svg": inference_svg(timing_rows, len(test_ids)),
    }
    for filename, content in outputs.items():
        atomic_write_text(output_dir / filename, content)
    atomic_write_json(output_dir / "inference_time.json", timing)

    artifact_paths = [output_dir / name for name in REPORT_FILENAMES]
    error_report = output_dir / "primary_error_analysis.md"
    if error_report.is_file():
        artifact_paths.append(error_report)
    report_manifest = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "source": {
            "primary_evaluation": str(primary_path.relative_to(ROOT)).replace("\\", "/"),
            "primary_evaluation_sha256": sha256(primary_path),
            "freeze_manifest_sha256": sha256(frozen_dir / "freeze_manifest.json"),
            "report_script_sha256": sha256(Path(__file__)),
        },
        "test_rows": len(test_ids),
        "artifacts": {
            path.name: {"bytes": path.stat().st_size, "sha256": sha256(path)}
            for path in artifact_paths
        },
    }
    atomic_write_json(output_dir / "report_assets_manifest.json", report_manifest)
    print(
        f"Verified six frozen models and wrote {len(artifact_paths)} report assets "
        f"to {output_dir}"
    )
    return result_rows, timing, report_manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--frozen-dir", type=Path, default=ROOT / "output" / "frozen_primary"
    )
    parser.add_argument(
        "--processed-dir", type=Path, default=ROOT / "data" / "processed"
    )
    parser.add_argument(
        "--output-dir", type=Path, default=ROOT / "output" / "report_assets"
    )
    parser.add_argument("--warmups", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=10)
    args = parser.parse_args()
    if args.warmups < 0:
        parser.error("--warmups cannot be negative")
    if args.repeats < 2:
        parser.error("--repeats must be at least 2")
    run(
        args.frozen_dir,
        args.processed_dir,
        args.output_dir,
        args.warmups,
        args.repeats,
    )


if __name__ == "__main__":
    main()
