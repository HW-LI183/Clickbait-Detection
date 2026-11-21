# Training-only TF-IDF pre-experiment protocol

Version 1.0, fixed on 2026-11-21 (Asia/Shanghai) before fitting.
This is a selection stage for the frozen `plan.md` A/B/C experiment, not
an evaluation on the validation or test partitions. The preprocessing and
split in `protocol.md` remain unchanged.

## Inputs and environment

- Use only `train` rows from `data/processed/records.jsonl` and
  `split_manifest.jsonl`. Check their SHA-256 values against
  `data/processed/audit.json`. Never fit or score on validation/test rows.
- Inputs: `post_text` for A, plus `title` for B, plus `body_1000` for C.
  The three optional metadata fields and annotation statistics are excluded.
- Python 3.11, NumPy, SciPy, and scikit-learn. Record actual versions,
  platform, input hashes, script/protocol hashes, and seed in the output.
- Seed: 20261121. Use four `StratifiedGroupKFold` folds with `group_id`
  so no frozen duplicate group is split within this pre-experiment. Save
  the training-ID-to-fold assignment and its hash.

## Pre-registered candidates

Fit a separate word `TfidfVectorizer` per field **inside each training
fold**. Common settings: lowercase, Unicode-aware default word token
pattern (two or more word characters), `min_df=2`, `max_df=1.0`,
`sublinear_tf=True`, `use_idf=True`, `smooth_idf=True`, `norm="l2"`,
no stop-word removal. Vocabulary limits for (post, title, body):

| Size | post | title | body |
| --- | ---: | ---: | ---: |
| small | 5,000 | 5,000 | 15,000 |
| medium | 10,000 | 10,000 | 30,000 |
| large | 20,000 | 20,000 | 60,000 |

Try word `ngram_range` `(1, 1)` and `(1, 2)` for all fields. Try
relative block weights (post, title, body): `(1, 1, 1)`,
`(2, 1, 1)`, `(1, 1, 0.5)`, `(2, 1, 0.5)`. For a condition, omit absent
fields and divide its included weights by their Euclidean norm before
concatenating the separately L2-normalized sparse blocks. Thus the total
feature-vector scale is comparable across A/B/C. There are 24 candidates;
the weights are shared, not independently tuned for each condition.

## Scoring and selection

- In each fold and candidate, fit six pilot classifiers: A/B/C paired
  with logistic regression (`liblinear`, L2, C=1, no class weighting,
  max_iter=2000) and linear SVM (`LinearSVC`, L2, C=1, no class weighting,
  max_iter=5000). Do not tune their parameters in this stage. Score the
  out-of-fold predictions using Macro-F1 at their default decision boundary.
  Record every score and the actual vocabulary dimensions.
- Primary selection score is the unweighted mean of the six condition/model
  mean Macro-F1 scores across four folds. Within 0.005 of the highest
  score, select the simplest candidate: smaller total realized vocabulary
  first, then unigrams before bigrams, then weights in the listed order.
  This is a practical complexity tolerance, not a significance test.
  Report each condition/model score and fold spread alongside the aggregate.
- Once selected, refit the three vectorizers on **all training rows** and
  save the fitted transformers and selected settings. Do not encode or
  evaluate validation/test in this script. Future formal comparisons may
  tune each model's own hyperparameters and threshold on validation as
  planned, but must reuse this one frozen TF-IDF configuration for all
  six conditions. Do not revisit this selection using validation/test.

Run `python scripts/tfidf_preexperiment.py` from the project root. Outputs
are under `data/processed/tfidf_preexperiment/`: `folds.jsonl`,
`scores.csv`, `selection.json`, and `vectorizers.joblib`. Treat the
joblib file as a local trusted artifact; do not load untrusted pickles.
