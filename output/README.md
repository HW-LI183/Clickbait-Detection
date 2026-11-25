# Output index

All final evaluation, poster, and report files are under this directory.

## Frozen primary evidence

`frozen_primary/` contains the model bundle and the complete machine-readable
test evidence. These files define the frozen primary result and must not be
edited by hand.

| File | Contents |
| --- | --- |
| `frozen_primary/freeze_manifest.json` | Frozen settings, source hashes, environment, and artifact hashes. |
| `frozen_primary/frozen_models.joblib` | Six fitted classifiers, three fitted TF-IDF vectorizers, and thresholds. |
| `frozen_primary/reproduced_metrics.json` | Reproduced aggregate scores for all six models. |
| `frozen_primary/test_predictions.jsonl` | Prediction, score, threshold, and correctness for every test record. |
| `frozen_primary/error_analysis_summary.json` | Error and prediction-change counts. |
| `frozen_primary/error_case_candidates.jsonl` | Selected examples for manual error analysis. |

## Poster and report assets

`report_assets/` contains the files intended for direct use in the poster and
written report.

| File | Contents |
| --- | --- |
| `report_assets/primary_results.csv` | Complete main result table for spreadsheet or plotting software. |
| `report_assets/primary_results.md` | Main result table formatted for the report. |
| `report_assets/primary_results_table.svg` | Vector result table for the poster. |
| `report_assets/primary_macro_f1.svg` | Vector comparison of test Macro-F1. |
| `report_assets/primary_error_counts.svg` | Vector comparison of false positives and false negatives. |
| `report_assets/inference_time.json` | Full timing protocol, environment, and raw repeat measurements. |
| `report_assets/inference_time.csv` | Compact timing table for further analysis. |
| `report_assets/inference_time.md` | Human-readable timing table and measurement definition. |
| `report_assets/inference_latency.svg` | Vector inference-latency chart for the poster. |
| `report_assets/primary_error_analysis.md` | Qualitative and quantitative frozen error analysis. |
| `report_assets/report_assets_manifest.json` | Source and artifact hashes for this report-asset set. |

SVG is a vector format, so the figures can be resized for a poster without
becoming blurry. The CSV files retain unrounded numeric values.

`primary_evaluation.json` is the original published evaluation result used by
the freeze step.
