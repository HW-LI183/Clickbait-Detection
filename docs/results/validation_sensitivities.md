# How does article context affect clickbait detection?

We evaluated three questions on the same 2,894 validation posts from Webis-Clickbait-17. Both classifiers used condition C (teaser, article headline, and body), with the regularization settings and decision thresholds from the [primary evaluation](../../output/primary_evaluation.json). Vocabularies, feature scaling, and model coefficients were fitted on training records. All results on this page are exploratory validation measurements.

## How much of the body helps?

We kept the teaser and headline fixed, truncated each body to the first *k* words, and refitted the model for each cap. The 1,000-word run reproduced the original condition C validation Macro-F1 for both classifiers.

| Maximum body words | Logistic regression Macro-F1 | Linear SVM Macro-F1 |
| ---: | ---: | ---: |
| 200 | 0.7403 | 0.7490 |
| 500 | 0.7548 | 0.7656 |
| 1,000 | 0.7690 | 0.7723 |
| 1,500 | 0.7689 | 0.7737 |

Extending the cap from 200 to 1,000 words raised Macro-F1 by 2.86 percentage points for logistic regression and 2.33 points for SVM. The next 500 words changed the scores by −0.01 and +0.13 points, respectively.

## Do explicit teaser–body features help?

At the 1,000-word cap, we added four features to condition C: cosine similarity in a shared TF-IDF vocabulary, the fraction of distinct teaser words found in the body, Jaccard overlap, and the teaser-to-body token-count ratio. We standardized each feature using training statistics and multiplied the four-feature block by 0.5. Each augmented model used its original decision threshold.

| Model | Macro-F1, original | Macro-F1, augmented | Average precision, original | Average precision, augmented |
| --- | ---: | ---: | ---: | ---: |
| Logistic regression | 0.7690 | 0.7724 | 0.6946 | 0.6862 |
| Linear SVM | 0.7723 | 0.7837 | 0.6976 | 0.6928 |

The added features increased Macro-F1 at the fixed thresholds. Average precision fell for both models, so the gain in classification at those thresholds did not extend to how well the models ranked clickbait posts.

## What happens when annotators disagree?

We grouped each post's five ratings into lower and upper halves at 0.5, counting ratings of 0.5 or above in the upper half. The scores below use the original 1,000-word model predictions and the published binary label (`truthClass`).

| Agreement on rating half | Posts | Clickbait posts | Logistic regression Macro-F1 | Linear SVM Macro-F1 |
| --- | ---: | ---: | ---: | ---: |
| 3–2 split | 801 | 329 (41.1%) | 0.6275 | 0.6383 |
| 4 or 5 agree | 2,093 | 370 (17.7%) | 0.8407 | 0.8377 |

Both classifiers scored lower on closely divided posts. The groups also differed in clickbait prevalence, which limits how much of the gap can be attributed to disagreement alone.

## Reproduce the results

From the repository root, use CPython 3.11.9. The recorded run used NumPy 2.4.6, SciPy 1.17.1, scikit-learn 1.9.1, joblib 1.6.0, and threadpoolctl 3.8.0.

```sh
python -m pip install numpy==2.4.6 scipy==1.17.1 scikit-learn==1.9.1 joblib==1.6.0 threadpoolctl==3.8.0
python scripts/download_data.py --extract
python scripts/preprocess_data.py
python scripts/evaluate_report_sensitivities.py
```

The script verifies the frozen record, split, and annotation inputs and writes `data/reproductions/report_validation_sensitivities.json`. To repeat the run, pass `--output` with a fresh file path. The [published JSON](../../output/report_validation_sensitivities.json) contains full-precision metrics and confusion matrices for every table above.
