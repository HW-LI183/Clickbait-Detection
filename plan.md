# Clickbait Detection Beyond Headlines

## Research objective

**Question:** On the held-out Webis-Clickbait-17 posts, does adding the linked
article improve binary clickbait classification beyond the social-media teaser
(`postText`)? How much of any change comes from the article title versus the
body, and which predictions change for better or worse?

The outcome is agreement with the corpus's `truthClass` judgments, not factual
truth or the reliability of a human reader. "Improvement" means a positive
held-out difference in the pre-specified primary metric, with its uncertainty
reported. We do not assume that more content must help.

The **primary contrast** is teaser + title + body (C) versus teaser alone (A)
using logistic regression. The **incremental-body contrast** is C versus teaser
+ title (B) with the same classifier. Linear SVM repeats both comparisons as
a robustness check. These contrasts, rather than the best result selected
after seeing the test set, will answer the research question.

## Data and scope

- Use the labeled `clickbait17-train-170630.zip` from Webis-Clickbait-17
  (Zenodo record 5530410). Join `data/instances.jsonl` and
  `data/truth.jsonl` by `id`; assert one-to-one IDs and use `truthClass` as
  the binary target.
- Primary A/B/C model inputs are `postText`, `targetTitle`, and
  `targetParagraphs` only. Captions, keywords, and descriptions are
  separately cleaned as optional features for a later exploratory
  comparison; they do not alter the primary contrasts or split. Inspect
  their cross-split duplicate groups before any such model. Never use
  `truthJudgments`, `truthMean`, other annotation statistics, timestamps,
  or images as predictors. The deleted `media` directory is not needed.
- A preliminary audit found 19,538 unique instance IDs: 4,761 clickbait
  and 14,777 non-clickbait labels. All records have nonempty titles and
  paragraph fields under a basic whitespace check. Repeat this audit with
  explicit text-quality checks before freezing the sample; report any
  exclusions and the resulting class counts. All experimental conditions
  must use the identical set of records.
- Inspect a small, documented sample for extraction artifacts such as
  navigation text, repeated headlines, and very short or irrelevant bodies.
  Define any exclusion rule before model evaluation, and retain an audit
  table of missing/invalid text and duplicate groups.

## Leakage-resistant split

1. Normalize Unicode and whitespace for comparison while preserving the
   original text for analysis. Detect exact duplicate IDs, teaser texts,
   titles, and article bodies. Identify near-duplicate articles using a
   documented text-similarity rule chosen without consulting test labels.
2. Form connected article groups from exact or near-duplicate bodies and
   repeated article titles. Also ensure identical teaser text cannot appear
   on both sides of a split; inspect unexpectedly large connected groups
   rather than silently accepting a broad grouping rule.
3. Make one fixed, approximately label-stratified **70/15/15 split by group**,
   targeting proportions of posts rather than groups. Save the seed, group
   assignments, IDs, counts, and class balance. Check that no defined
   duplicate group crosses train, validation, and test.
4. Fit all vocabulary, IDF, scaling, feature selection, and model parameters
   on training data only. Use validation data for pre-specified tuning and
   threshold selection. Inspect the test labels only for the final evaluation.

The preliminary data check found 1,236 records with a repeated nonempty
article title and 802 with repeated teaser text. These are reasons to
validate the grouping, not a claim that every repeated string is the same
article.

## Controlled models and text inputs

| Condition | Inputs |
| --- | --- |
| A | Social-media teaser (`postText`) |
| B | Teaser + linked article title (`targetTitle`) |
| C | Teaser + title + article body (`targetParagraphs`) |

- Include a majority-class predictor as a reference. Pair each of A, B,
  and C with word-level TF-IDF plus logistic regression and word-level
  TF-IDF plus linear SVM: six main model/input combinations.
- Keep each field separate in the feature pipeline. Use the same
  preprocessing and field-specific vectorizer settings whenever that field
  appears, then L2-normalize feature blocks and apply the single TF-IDF
  vocabulary/ngram/relative-weight configuration selected by the
  training-only grouped cross-validation in
  `docs/protocols/tfidf_preexperiment_protocol.md`. Freeze this configuration before
  formal validation tuning; do not retune field weights separately for
  A/B/C or for the two classifiers. Report the selected settings. Give
  all six combinations the same subsequent model-tuning budget.
- The **main body cap is the first 1,000 whitespace-delimited words** after
  normalization, concatenating paragraphs in their original order. This
  replaces the original 500-word main cap. Report how many articles are
  truncated; the preliminary audit counted 2,825 above 1,000 words.
- Hold the split, samples, classifier family, and non-body fields fixed
  while comparing conditions. As secondary sensitivity analyses, repeat C
  with 200-, 500-, and 1,500-word body caps. A within-split shuffled-body
  control may be added if time permits to probe whether gains reflect
  article-specific information rather than simply more text; label it
  exploratory and never use it to select the main result.
- As a separate ablation, add teaser/body TF-IDF cosine similarity, lexical
  overlap, and length ratio to C. Compute features without test-set fitting
  and compare against the same C model without these features. The core
  project does not require a large language model.

## Evaluation and interpretation

- **Primary metric:** test Macro-F1 for the pre-specified logistic-regression
  contrasts C - A and C - B. Select hyperparameters and any classification
  threshold using validation Macro-F1 only; freeze them before one final
  test evaluation. Report each model's score, both paired differences, and
  95% confidence intervals from paired bootstrap resampling of article
  groups. An interval crossing zero is not evidence of a clear gain.
- Also report clickbait precision, recall, F1, average precision from
  continuous scores, confusion matrices, and inference time under a stated
  hardware/batch setting. Present the SVM results as a robustness check,
  not as a post-hoc replacement for the primary model.
- Inspect cases where B or C changes A's prediction, separating corrections
  from new errors and noting extraction noise or truncation where relevant.
  Analyze high- versus low-annotator-disagreement subsets descriptively;
  define the grouping rule and report subgroup sizes. Annotation statistics
  are for this analysis only, never model inputs.
- A null or negative result still answers the question. Limit conclusions
  to clickbait-label prediction in this historical English-news corpus;
  do not generalize them to factual verification, human reading behavior,
  or modern news without additional evidence.

## Deliverables and remaining schedule

Keep new code and tests under `scripts/`. Deliver the frozen split manifest,
reproducible preprocessing/training/evaluation commands, a README with data
access and environment details, result tables with uncertainty, error
examples, a poster, and the final report. Do not commit the large dataset
or extracted media.

| Dates in 2026 | Milestone |
| --- | --- |
| November 21-22 | Finish quality/duplicate audit; document grouping and freeze the split. |
| November 23-24 | Implement and validate the majority baseline and six main comparisons. |
| November 25-26 | Run the primary evaluation, bootstrap intervals, and focused sensitivity/error analysis; prepare the poster. |
| November 27 | Submit the poster. |
| November 28-December 3 | Complete alignment and length ablations, inspect failures, and rehearse explanations. |
| December 4 | Present at the poster session. |
| December 5-11 | Incorporate feedback, rerun reproducibility checks, and submit the final package. |

## Source proposal and data

This plan revises `Group_Proposal.pdf` (15 October 2026) while preserving its
original research question, dataset, classical-model scope, and deliverables.
The data source is Webis-Clickbait-17, version 2, Zenodo record 5530410.
