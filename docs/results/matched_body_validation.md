# Does the classifier need the correct article body?

Replacing the body with a lexically similar, similarly long article reduced
logistic-regression Macro-F1 by 10.38 points across 1,552 matched validation
posts. Holding the website fixed gave a smaller, imprecise estimate. When we
also checked the actual subject of the articles, only seven of the 38 same-host
pairs clearly shared a specific topic while reporting distinct content. On
those 14 posts both classifiers had the same Macro-F1 before and after swapping.

## Controlled comparison

Each pair exchanged its bodies in both directions. The teaser, headline, label,
model and decision threshold stayed fixed for each recipient. Each body appeared
once in each condition, and no frozen duplicate group appeared in two pairs.
The B reference uses the same recipients and sees only teaser and headline.
C uses the first 1,000 body words at most. All scores below are validation
Macro-F1. Gap is correct C minus swapped C in percentage points. The intervals
resample 2,000 sets of entire pairs.

| Pair selection | Posts (pairs) | Model | B | Correct C | Swapped C | Gap [95% CI], points |
|---|---:|---|---:|---:|---:|---:|
| Similar topic words and length, five months | 1,552 (776) | Logistic regression | .7490 | .7859 | .6822 | +10.38 [7.73, 13.24] |
| Same sample | 1,552 (776) | Linear SVM | .7421 | .7940 | .6925 | +10.15 [7.71, 12.86] |
| Same archived host, similar topic words and length, December | 76 (38) | Logistic regression | .7318 | .8042 | .7317 | +7.26 [−1.55, 16.42] |
| Same sample | 76 (38) | Linear SVM | .7741 | .8212 | .7741 | +4.71 [−3.74, 13.05] |

Of 2,894 validation posts, the five-month rule matched 1,552, including 343
clickbait posts. The December source audit checked all 595 validation posts
from that month. Archived HTML supplied 593 canonical URLs and two Open Graph
URLs. The same-host rule matched 76 posts, including 24 clickbait posts, on
19 hosts. One pair linked to pages on twitter.com. Exact host is a proxy for
outlet, and the two automated samples differ. Their gaps cannot isolate the
effect of controlling for publisher. Median shorter/longer body-length ratios
were .823 for the five-month sample and .822 for December.

## Are the articles actually similar?

The text review considered article headlines and leads with individual labels
and predictions hidden. The original automatic scores above include weak
matches. For example, two BBC pages about a snake in a Christmas tree and a
missile launch shared a long sequence of social-sharing links. An example of
a genuine matched swap is the IndiaTimes pair covering India's currency change.
One article reports postponed weddings following cash shortages. The other
reports a bank raid that recovered new notes. These are distinct reports on
the same specific issue (December pair 36).

| Review pool | Specific topic, distinct content | Broad overlap | Weak | Potentially substitutable accounts |
|---|---:|---:|---:|---:|
| All 38 same-host pairs | 7 | 18 | 12 | 1 |
| 48 evenly spaced pairs in the more lexically similar half of the five-month pool | 21 | 17 | 8 | 2 |

| Specific-topic subset | Model | Correct C | Swapped C | Right only with original | Right only with swap |
|---|---|---:|---:|---:|---:|
| Same host, 14 posts | Logistic regression | .8571 | .8571 | 1 | 1 |
| Same host, 14 posts | Linear SVM | .8571 | .8571 | 1 | 1 |
| Host unrestricted, 42 posts | Logistic regression | .9631 | .8894 | 2 | 0 |
| Host unrestricted, 42 posts | Linear SVM | .8980 | .9293 | 0 | 1 |

The small reviewed subsets are descriptive. Selection was exploratory, and
one AI-assisted reviewer assessed topic fit without independent adjudication.
The 48-pair review preferentially samples stronger lexical matches, so its
acceptance fraction cannot be extrapolated to all 776 pairs.

## What the result explains

In the five-month sample, 231 pairs had opposite clickbait labels. For logistic
regression on those pairs, 106 predictions were right only with the original
body and seven only with the swap. Among the 545 same-label pairs those counts
were 24 and 29. Donor labels were examined only after matching. A body can
carry vocabulary predictive of its own label, so a score drop after swapping
can arise in a linear model without any explicit check that the teaser's
promise was fulfilled.

The larger experiment establishes sensitivity to body assignment under the
specified lexical matching rule. The December interval includes zero, and the
reviewed same-topic, same-host evidence is too small to establish an advantage
for the correct body. The original held-out A/B/C result remains a separate
estimate of the benefit of *adding* body text.

## Reproduce and inspect

From the project root, use Python 3.11 with scikit-learn 1.9.1. This run used
NumPy 2.4.6 and SciPy 1.17.1. The four full-validation B/C reference scores
matched the recorded primary evaluation exactly. The experiment made no new
test predictions.

```sh
python scripts/download_data.py --extract
python scripts/preprocess_data.py
python scripts/evaluate_matched_bodies.py --output-dir output/reproduction
python scripts/evaluate_matched_bodies.py --topic-only --output-dir output/reproduction
python scripts/summarize_topic_review.py
```

The evaluators refuse to overwrite either destination file. Choose a fresh
output directory for each rerun; the commands above leave published artifacts
unchanged. They stop if any B/C full-validation Macro-F1 differs from its
primary reference by more than `1e-12` (absolute tolerance only).
The review-summary command reads the published pair audits in `output/`,
not the independent evaluator outputs in `output/reproduction/`.

The published source table supports offline scoring once preprocessing is
complete. To regenerate source provenance, install `requests` and run
`python scripts/recover_validation_sources.py`. It reads ZIP indexes and
bounded HTML prefixes from the [official Webis archives](https://zenodo.org/records/5530410)
and caches successful audits under `data/`. Raw article text is excluded from
the submitted outputs.

Data credit: Martin Potthast, Tim Gollub, Matti Wiegmann and Benno Stein,
[Webis-Clickbait-17](https://doi.org/10.5281/zenodo.5530410).

- [Matching protocol](../protocols/matched_body_control.md) and [frozen primary evaluation](../../output/primary_evaluation.json)
- [December source audit](../../output/validation_article_sources.csv), [reciprocal predictions](../../output/matched_body_pairs.csv) and [December metrics](../../output/matched_body_validation.json)
- [Five-month reciprocal predictions](../../output/matched_body_topic_only_pairs.csv) and [metrics](../../output/matched_body_topic_only_validation.json)
- [Pair-level topic judgments and rationales](../../output/matched_body_topic_review.csv) and [reviewed-subset metrics](../../output/matched_body_reviewed_validation.json)
