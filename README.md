# Webis-Clickbait-17 data

The proposal uses the publicly labeled `clickbait17-train-170630.zip` from
Webis-Clickbait-17, Zenodo record 5530410. It does not require the much larger
web-page archive files in the same record.

Run with Python 3 (standard library only):

```powershell
python scripts/download_data.py --extract
```

This downloads about 937 MB into `data/`, verifies the archive against the
size and MD5 published by Zenodo, and extracts `instances.jsonl` and
`truth.jsonl`. A failed transfer leaves a `.part` file for the next run to
resume. Use `--dry-run` to inspect the current Zenodo metadata without a
download, or `--output-dir PATH` to choose a different location. Repeated
runs verify and reuse existing files; a mismatched file is never overwritten.

The files under `data/` are third-party research data, not generated code.

## Document index

The research documents are organized below. Paths in the protocols are
interpreted from the repository root unless they name a sibling protocol.

| Document | Purpose |
| --- | --- |
| [README.md](README.md) | Data access, commands, current configuration, and later ablation ideas. |
| [Group_Proposal.pdf](Group_Proposal.pdf) | Original group proposal and research motivation. |
| [plan.md](plan.md) | Revised study design, A/B/C contrasts, metrics, and schedule. |
| [Preprocessing protocol](docs/protocols/protocol.md) | Frozen source hashes, cleaning, grouping, split, seed, and environment. |
| [Optional-feature protocol](docs/protocols/optional_features_protocol.md) | Cleaning and duplicate audit for captions, keywords, and description. |
| [TF-IDF pre-experiment protocol](docs/protocols/tfidf_preexperiment_protocol.md) | Training-only candidate grid and selection rule. |
| [Engineered-feature protocol](docs/protocols/engineered_features_protocol.md) | Definitions and train-only fitting for interpretable writing/relationship features. |
| [TF-IDF pre-experiment results](docs/results/tfidf_preexperiment_results.md) | Selected settings, pilot scores, and integrity checks. |

Machine-readable audits and detailed scores are under
`data/processed/`, `data/processed/optional/`,
`data/processed/tfidf_preexperiment/`, and
`data/processed/engineered/`. They are generated locally and excluded
from Git.

## Preprocessing

The frozen environment, input hashes, seed, text rules, duplicate-grouping
method, and split procedure are documented in
`docs/protocols/protocol.md`. From the
project root, run:

```powershell
python scripts/preprocess_data.py
python -m unittest discover -s scripts -p "test_*.py" -v
```

Preprocessing verifies the raw SHA-256 hashes before reading the files.
It audits missing and blank values in the required text fields and label.
It excludes records with an empty required teaser/title/body or fewer than
20 normalized body words, then saves the full body plus its first 1,000
words. Duplicate article
and teaser groups are kept intact in a seeded 70/15/15 split.

The validated outputs are `data/processed/records.jsonl`,
`data/processed/split_manifest.jsonl`, and `data/processed/audit.json`.
The first file contains only model-text inputs and IDs; labels and split
assignments are in the separate manifest. A rerun verifies identical
output bytes and refuses to overwrite different records or splits. A
documented protocol-only audit change requires an explicit existing-audit
hash with `--refresh-audit-if-sha256`. The earlier
diagnostic output, produced before the documented short-body rule, is
preserved in `data/processed-v1-diagnostic/` and must not be used for
modeling.

## Optional article metadata

`targetCaptions`, `targetKeywords`, and `targetDescription` are processed
separately under `docs/protocols/optional_features_protocol.md`. Run:

```powershell
python scripts/preprocess_optional_features.py
```

The sidecar in `data/processed/optional/` contains one cleaned row per
retained primary ID, duplicate-group IDs, and a field-level audit.
Missing optional text becomes an empty string and never changes the
frozen A/B/C sample or split. The audit reports repeated optional text
that spans train/validation/test; review that risk before using these
fields in a separate exploratory model. The primary results must not
be reselected based on this sidecar.

## Training-only TF-IDF pre-experiment

The candidate grid, group-aware four-fold procedure, pilot models, and
selection rule are fixed in
`docs/protocols/tfidf_preexperiment_protocol.md`. Install
`numpy`, `scipy`, `scikit-learn`, and `joblib` for the same Python interpreter.
For a local installation in `.python_deps`, run from the project root:

```powershell
py -3.11 -m pip install --target .python_deps numpy scipy scikit-learn joblib
$env:PYTHONPATH = (Resolve-Path .python_deps).Path
$env:OPENBLAS_NUM_THREADS = "1"
$env:OMP_NUM_THREADS = "1"
py -3.11 scripts/tfidf_preexperiment.py
```

The script verifies the frozen processed-file hashes and uses only training
rows. It writes fold assignments, all pilot Macro-F1 scores, the selected
configuration, and fitted training-only vectorizers under
`data/processed/tfidf_preexperiment/`. Validation/test rows are not
vectorized or evaluated here. Only load the saved `joblib` artifact from
this trusted local run. If scoring completed but finalization was interrupted,
`--finalize-existing` verifies that every saved fold/condition/model score
is present before refitting the selected vectorizers.
The selected configuration and its training-only scores are summarized in
`docs/results/tfidf_preexperiment_results.md`.

This is the best **under the pre-registered selection rule**, not a global
optimum: only 24 vocabulary/ngram/weight candidates were considered, with
fixed-C pilot LR and linear SVM and four grouped training folds. The
six-cell mean Macro-F1 was 0.694241 for the frozen 5k/5k/15k, word
1-2 gram, 1:1:1 setting. Another candidate scored 0.696655, but its
0.002415 advantage was inside the pre-specified 0.005 complexity
tolerance. No validation/test evidence favors this choice, and its
ranking need not transfer to other models, corpora, preprocessing, or
additional features. Do not reinterpret pilot scores as final results.

## Interpretable feature sidecar

The definitions and train-only shared TF-IDF fitting rule are in
`docs/protocols/engineered_features_protocol.md`. With the local dependencies above:

```powershell
py -3.11 scripts/engineer_features.py
```

This writes 15 numeric writing/relationship features per retained ID to
`data/processed/engineered/features.jsonl`, with a hash audit and a
separate shared-space relation encoder. It includes no label and does
not fit a classifier. The main A/B/C TF-IDF vectorizers and frozen
comparison remain unchanged; any future feature scaling must be fitted
on training rows only, in a separately stated ablation.

## Data for later ablations

Join sidecars by `id`, never by row position. Keep the same retained IDs
and the group-preserving assignments in `data/processed/split_manifest.jsonl`
for every comparable run. The following inputs are already available:

| Source | Available content | Intended use |
| --- | --- | --- |
| `data/processed/records.jsonl` | `post_text`, `title`, `body_1000`, `body_full` | Frozen A/B/C text; `body_full` can supply alternative body caps. |
| `data/processed/tfidf_preexperiment/vectorizers.joblib` | Three train-fitted, field-specific word TF-IDF encoders | Reproduce the frozen word-feature baseline; load only this trusted local artifact. |
| `data/processed/engineered/features.jsonl` | Seven post writing counts; post-title and post-body cosine, overlap recall, Jaccard, and length ratio (15 columns total) | Separate, interpretable feature ablations. Definitions and zero-vector audit are in `docs/protocols/engineered_features_protocol.md` and `data/processed/engineered/audit.json`. |
| `data/processed/engineered/relation_vectorizer.joblib` | Shared-space TF-IDF encoder fitted on training text only | Reproduce the two cross-field cosine columns; **not** a replacement for the main field-specific encoders. |
| `data/processed/optional/optional_features.jsonl` | Cleaned captions, keywords, and description, with empty strings for missing values | Exploratory metadata ablations only; consult the optional audit and duplicate groups first. |

Recommended experiments, after completing and freezing the six primary
word-TF-IDF A/B/C comparisons:

1. **Writing cues:** Compare C against C plus the seven `post_*_count`
   features from the engineered sidecar. A writing-cues-only teaser model
   can be a descriptive reference, but not a replacement for A.
2. **Text alignment:** Compare C against C plus the four `post_body_*`
   relation features; separately compare B against B plus the four
   `post_title_*` features. Then try C plus both sets, and finally C plus
   all 15 sidecar columns. This isolates body alignment, title alignment,
   and writing cues rather than attributing a combined gain to one source.
   Never give A a title/body relation feature it cannot observe.
3. **Character n-grams:** Add a new **post-only** character TF-IDF block
   with `analyzer="char"` and `ngram_range=(3, 5)` to each of A/B/C,
   keeping the original word blocks and the same character settings in
   all three conditions. This representation has **not** been generated
   yet. Predeclare its vocabulary cap and relative block weight using
   training-only group folds (or validation), fit vocabulary/IDF on train,
   and compare against each condition's unchanged word baseline. A
   `char_wb` variant is a *separate* experiment, not an implicit swap.
4. **Body length:** Rebuild C with the first 200, 500, and 1,500 words
   from `body_full`, against the frozen 1,000-word C reference. Fit each
   new body vocabulary on train only. Do not reuse the `body_1000`
   relation columns for another cap; recompute body relations if needed.
   A within-split shuffled-body control is optional and exploratory.
5. **Optional metadata:** Append captions, keywords, and description
   one field at a time to C, with explicit missingness indicators. These
   are not part of the main comparison. The optional audit reports
   cross-split duplicate groups; review and disclose that risk before
   interpreting any apparent gain. Do not resplit or select the primary
   result based on these fields.

For numeric sidecars, fit scaling on training rows only and apply it
unchanged to validation/test; choose its relative scale against sparse
TF-IDF before inspecting test results. Any new character vocabulary,
body-cap vocabulary, IDF, or learned feature selection likewise fits on
train only. Keep classifier family, records, split, metric, and tuning
budget matched within each ablation; record feature settings and paired
Macro-F1 differences. Use validation for predeclared ablation decisions,
reserve test for the final evaluation, and label additional comparisons
exploratory rather than using them to revise the primary C-A/C-B claims.
