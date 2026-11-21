# Interpretable feature extraction protocol

Version 1.0, fixed on 2026-11-21 (Asia/Shanghai), before extracting.
This sidecar does not change the frozen A/B/C TF-IDF configuration,
sample, split, or primary comparisons. No classifier is fitted here.

## Inputs and fitted state

- Verify the hashes of `data/processed/records.jsonl` and
  `split_manifest.jsonl` against the frozen preprocessing audit. Verify
  that `tfidf_preexperiment/selection.json` refers to the same inputs.
- Use `post_text`, `title`, and `body_1000` only. IDs align one-to-one
  with the frozen records. Neither labels nor optional article metadata
  are read as features.
- The main model's field-specific TF-IDF vectorizers cannot compare
  different fields directly because their coordinates differ. For
  relationship features only, fit a **separate shared** `TfidfVectorizer`
  on one joined post + title + capped body document per *training* row.
  Reuse the selected main word n-gram range and the sum of its three
  vocabulary caps (25,000); use `min_df=2`, `sublinear_tf=True`,
  lowercase, smoothed IDF, default Unicode word token pattern, and L2
  normalization. There is no hyperparameter search for this sidecar.
  Transform all splits in the same shared space. Save the encoder.
- Seed 20261121 is recorded for alignment, although feature extraction
  uses no random operations. Record Python/scikit-learn versions, input
  and selection hashes, code/protocol hashes, and output hashes.

## Feature definitions

Unicode `\b\w+\b` tokens are casefolded for lexical counts (including
one-character tokens). Punctuation counts include ASCII and fullwidth
`?` / `!`. A digit token contains at least one Unicode decimal digit.
These are deliberately literal counts, not POS tags or semantic claims.

- Writing features: `post_word_count`, `post_question_count`,
  `post_exclamation_count`, `post_digit_token_count`,
  `post_second_person_count` (`you`, `your`, `yours`, `yourself`,
  `yourselves`), `post_demonstrative_count` (`this`, `that`, `these`,
  `those`), and `post_interrogative_count` (`who`, `whom`, `whose`,
  `what`, `which`, `when`, `where`, `why`, `how`).
- For each target field (`title`, `body_1000`), compute
  `post_{title|body}_tfidf_cosine` as row-wise dot product of the
  L2-normalized shared-space vectors (zero if either vector is zero);
  `post_{title|body}_overlap_recall` as intersection of distinct
  tokens divided by distinct post tokens;
  `post_{title|body}_jaccard` as intersection divided by union; and
  `post_{title|body}_length_ratio` as post token count divided by
  target token count. Set a zero denominator's result to zero.
  No stop-word filtering is applied. These overlap measures are
  surface-form proxies, not semantic equivalence.

Output `data/processed/engineered/features.jsonl` holds the ID and
15 numeric features, sorted by ID, without labels or split membership.
`audit.json` records hashes, counts and zero-vector diagnostics;
`relation_vectorizer.joblib` holds only the train-fitted shared encoder.
To use numeric features in later models, fit any scaler on training
rows only and perform a separately specified ablation. Do not
silently add this sidecar to the six primary comparisons.
