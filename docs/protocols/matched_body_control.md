# Matched wrong-body control

This exploratory validation experiment asks whether the frozen C classifier scores
better with the original article body than with another, topically similar body
from the same archived news host. The automated matching rules and endpoint
were set before scoring the swaps. Scope and review amendments are documented
below. The earlier A/B/C test evaluation had already been completed.

**Source audit amendment.** The full-validation topic/length comparison was run
first. Archive requests for outlet verification repeatedly timed out. Before
scoring any same-host swaps, the source audit was bounded to the first complete
calendar month, December 2016 UTC (595 validation recipients). The same-host
result therefore refers only to matchable December records. The independent
five-month topic/length matching covers the full validation pool.

## Eligible pairs

Use validation rows from the primary split. The release's instances contain
neither publisher nor URL. Use the canonical article URL, falling back to
`og:url`, in the archived article HTML linked to each Webis ID. Drop the
leading `www.` and require identical hosts. A host is a proxy for the outlet.
Unknown sources do not match other unknowns. A normalized host/path fingerprint
identifies multiple captures of the same article. Keep this metadata out of
classifier features.

Encode each article as its title twice followed by its first 100 body words.
Use a separate train-fitted English-stopword-filtered word TF-IDF encoder
(1–2 grams, min_df 2, max_features 30,000, sublinear TF, L2), exclusively for
matching. An edge requires:

- the same nonempty source host, different canonical article and frozen
  duplicate group, and different normalized capped bodies;
- cosine similarity from 0.10 through 0.80 inclusive, to avoid unrelated
  articles and very close copies;
- the shorter full body has at least two-thirds as many words as the longer.

Visit eligible edges in descending cosine order, then descending length ratio,
then in ascending validation ID order. Take disjoint pairs; never reuse an
article fingerprint or frozen duplicate group, even across different pairs.
Exchange bodies in **both** directions. This preserves exactly the same set of
body texts across the two conditions and avoids reliance on a few favorite
donors. Hold the recipient's teaser, title and label fixed. Pairing uses no
labels, predictions or classifier scores. Report coverage, host distribution,
similarity and length ratios. This estimates an effect for the matchable subset.

For a qualitative check, sort the same-host pairs by cosine, break ties by
recipient ID, and inspect 12 evenly spaced ranks from lowest through highest
(or all pairs if fewer). Review both headlines and leads without labels or
predictions. Mark a pair specific if it shares a concrete issue, event or
subject, broad if only a beat or an entity overlaps, and weak if boilerplate
or unrelated content dominates. These descriptive judgments do not change
the scoring sample.

The initial inspection found template-driven matches. A post-hoc sensitivity
review therefore covered every same-host pair and 48 additional pairs in the
five-month run. For the latter, sort descending by stored cosine and ascending
by recipient ID, take the upper half, and sample 48 evenly spaced ranks
(round to nearest even integer). The AI-assisted reviewer saw headlines and
lead text with individual labels and predictions hidden; aggregate automated
results were already known. The review is a single judgment per pair. Mark
close coverage of the same news development as `overlap`. Only `specific`
pairs enter the descriptive reviewed subset. The automated results keep every
originally selected pair. A subgroup comparison also counts prediction changes
by donor/recipient label agreement after pairing.

## Scoring and uncertainty

Refit B and C on all frozen training rows, reusing the primary field encoders'
settings, model families, selected regularization strengths and decision
thresholds from `output/primary_evaluation.json`. Use the same C model instance
for both body assignments. Score B on identical recipients as a reference.
The main endpoint is correct-body minus swapped-body Macro-F1 for logistic
regression. Repeat with linear SVM. Record confusion matrices, class metrics,
average precision and individual changes. Encode validation text only after
train fitting. No new test predictions are made.

Draw 2,000 paired bootstrap samples of entire two-article pairs, with
replacement and seed 20261121. Report percentile 95% intervals conditional on
the fixed validation matching and fitted models. They omit uncertainty in
matching rules, training and hyperparameter selection. All matches remain
lexical approximations. A positive difference can reflect article-specific
lexical predictors and cannot establish semantic understanding.

As a coverage check, repeat the same fixed protocol across the entire validation
pool without using host metadata. In that secondary run, frozen group and body
checks exclude duplicates; a canonical-URL check is unavailable. Its matched
sample can differ from the same-host sample. Differences between the two effect
sizes do not isolate a publisher effect.
