# Frozen primary error analysis

## Scope

This analysis uses the frozen 2,894-row test split and the six published
A/B/C models. It does not retune a classifier, threshold, vectorizer, sample
rule, or split. The primary model remains logistic regression, with condition
C using the teaser, linked article title, and first 1,000 body words.

The test contains 698 clickbait and 2,196 no-clickbait records. Condition C
with logistic regression makes 2,433 correct decisions, 168 false positives,
and 293 false negatives. Its Macro-F1 is 0.7676.

## What article context changes

Relative to teaser-only condition A, condition C changes 261 logistic-
regression decisions. It corrects 158 errors and introduces 103 errors, a net
reduction of 55 classification errors. The same pattern appears with linear
SVM: C corrects 157 errors and introduces 102.

Article title alone is not enough. Condition B corrects 110 logistic-
regression errors but introduces 163, which is consistent with B's lower
Macro-F1. The main improvement therefore comes from the body-inclusive C
condition rather than simply adding the linked title.

## Representative corrected cases

The cases below were selected deterministically by the largest absolute
score change. They illustrate mechanisms; they do not estimate how often each
mechanism occurs.

| Test ID | Teaser excerpt | What the article adds | Interpretation |
| --- | --- | --- | --- |
| `822193532484001792` | `Live now:` | The title and body identify a Trump inauguration celebration and provide the withheld event details. | The teaser is almost content-free. Article context supplies the subject and the attention-oriented event framing, changing A's no-clickbait decision to a correct clickbait decision. |
| `827234224461713409` | `Primary rigger says what?` | The title identifies Debbie Wasserman Schultz and the body supplies the quoted claim about a dictator. | The teaser uses an unresolved referent and a question. C can connect that teaser to the article's person and claim, correcting A. |
| `838056579954065408` | `Back on offense.` | The article identifies Trump's wiretapping accusation against Obama. | The teaser withholds the event. The body resolves the vague phrase, although annotators also disagreed on this example, with a mean judgment of 0.60. |

## Representative introduced errors

| Test ID | Teaser excerpt | What the article adds | Interpretation |
| --- | --- | --- | --- |
| `833535559607975938` | `Brian Williams... complaining about damage to journalism's reputation.` | The body contains political criticism, quotations, and emotionally loaded discussion of media credibility. | C changes a correct no-clickbait decision into clickbait. The model appears sensitive to provocative political language even when the corpus label is no-clickbait. |
| `839510441143463937` | `Paul Ryan the type of gymbro...` | The article contains partisan language about Obamacare and political opposition. | The teaser is sarcastic, while the body adds topic words associated with contentious stories. Those cues raise the clickbait score but do not match the gold class. |
| `846496451467317249` | `Is there a more family-friendly headline writer than Drudge?` | The article discusses censorship and sensitive-content filtering. | The rhetorical teaser and charged article language resemble clickbait style, producing a false positive after context is added. |

## High-confidence remaining errors

Some errors remain even when condition C is far from its decision threshold.

- False positive `828466458476687361` is a legitimate fitness-advice article,
  but its `How to ... and why` construction, rhetorical questions, and phrases
  such as `truth bomb` resemble common clickbait writing patterns.
- False positive `846748454210748416` uses the vague teaser `This Is the End
  (2013)` and a title about a `huge subscriber decline`. Its annotator mean is
  0.40, so the binary no-clickbait label hides substantial disagreement.
- False negative `824484292814204928` is written as a direct factual report
  about two football players arrested in a murder investigation. Its
  annotator mean is 0.467, showing that the gold clickbait class is itself
  borderline.
- False negative `805006685735649280` uses nearly the same direct factual
  wording in the teaser and title. It lacks the surface patterns the model
  often associates with clickbait, despite a clickbait gold label and an
  annotator mean of 0.667.

These examples identify two main failure modes. First, rhetorical, sarcastic,
or politically charged language can create false positives even when the
article is labeled no-clickbait. Second, factual-looking teasers can create
false negatives when the crowd label treats the post as clickbait for reasons
that are not captured by obvious surface wording.

## Annotator disagreement

Annotator disagreement is measured as the standard deviation of the five
`truthJudgments` values. Low and high groups use the bottom and top quartiles
of that standard deviation, retaining ties at the thresholds.

| Disagreement group | Rows | A logistic Macro-F1 | C logistic Macro-F1 | C minus A |
| --- | ---: | ---: | ---: | ---: |
| Low | 736 | 0.8321 | 0.8863 | +0.0542 |
| High | 728 | 0.6736 | 0.7088 | +0.0352 |

Absolute performance is much lower on high-disagreement records, which is
consistent with ambiguous or subjective labels. Article context improves
Macro-F1 in both descriptive groups, but this subgroup comparison has no
confidence interval and should not be presented as a new significance claim.
Annotation statistics were never used as model inputs.

## Reproducible evidence

`output/frozen_primary/test_predictions.jsonl` contains the truth label,
continuous score, threshold, predicted class, correctness flag, and A-to-B/C
transition for every test ID. `error_analysis_summary.json` contains the
aggregate counts above. `error_case_candidates.jsonl` contains the
deterministically selected examples and short local text excerpts used for
human review.

The selected examples are deliberately extreme cases. The report and poster
should pair them with the complete aggregate counts and should not imply that
six examples describe the entire corpus.
