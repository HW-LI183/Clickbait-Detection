# Training-only TF-IDF pre-experiment results

Run on 2026-11-21 (Asia/Shanghai). The pre-registered method is in
`docs/protocols/tfidf_preexperiment_protocol.md`; complete numeric results and provenance
are in `data/processed/tfidf_preexperiment/selection.json`, `scores.csv`,
and `folds.jsonl`. The fitted training-only encoders are in
`vectorizers.joblib` in that same directory.

## Selected configuration

- Each field has its own word TF-IDF vocabulary: `post_text` 5,000 terms,
  `title` 5,000 terms, `body_1000` 15,000 terms. All three reached their
  limits when refitted on the full 13,506-row training split.
- Word 1-2 grams; `min_df=2`, `sublinear_tf=True`, smoothed IDF, no
  stop-word removal, L2 normalization per field.
- Relative weights (post, title, body): **1:1:1**. For each condition,
  include only its available fields and normalize their weights to unit
  Euclidean length before sparse concatenation.

The six-condition/model mean four-fold Macro-F1 is **0.694241**. The
unconstrained best candidate, with the same size/ngram but relative
weights **2:1:1**, scored **0.696655**. The difference, **0.002415**, is
within the pre-registered 0.005 tolerance; equal weights win the stated
simplicity tie-break. The selected candidate's individual fold means are:

| Input | Logistic regression | Linear SVM |
| --- | ---: | ---: |
| A: post | 0.670153 | 0.698166 |
| B: post + title | 0.665410 | 0.696695 |
| C: post + title + body | 0.707258 | 0.727762 |

These are **pilot out-of-fold training scores**, not validation or test
scores and not the final six model comparisons. Differences between A/B/C
here are descriptive and must not be reported as held-out effects.

## Integrity and next use

- Four group-disjoint stratified folds: 3,377 / 3,377 / 3,377 / 3,375
  held-out training rows. All 576 expected scores are present.
- The full run was repeated after repairing an empty-transform
  finalization edge case. The complete `scores.csv` was byte-identical
  between runs (SHA-256
  `b91bae854aab44b8f6b3bb2996ccc9911ff5bec24a35a0fe79fecebc727ba76f`).
  The final run completed the fitted encoders and metadata with the
  current script hash.
- Reuse the saved encoders and relative weights for all primary A/B/C
  comparisons. Tune only the subsequent classifier parameters and
  thresholds on the reserved validation split, then evaluate the test
  split once under `plan.md`. Do not reconsider TF-IDF settings using
  those splits. The three optional metadata fields remain separate.
