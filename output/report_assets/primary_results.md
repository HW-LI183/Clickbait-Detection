# Frozen primary results

The table reports the held-out test split with 2,894 records. Condition A uses post text, B adds the linked title, and C also adds the first 1,000 article-body words.

| Condition | Model | Validation Macro-F1 | Test Macro-F1 | Precision | Recall | Clickbait F1 | AP | FP | FN |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| A | Logistic regression | 0.7405 | 0.7409 | 0.6569 | 0.5458 | 0.5962 | 0.6489 | 199 | 317 |
| A | Linear SVM | 0.7464 | 0.7462 | 0.6408 | 0.5802 | 0.6090 | 0.6504 | 227 | 293 |
| B | Logistic regression | 0.7337 | 0.7270 | 0.5970 | 0.5688 | 0.5825 | 0.6143 | 268 | 301 |
| B | Linear SVM | 0.7390 | 0.7394 | 0.6424 | 0.5559 | 0.5960 | 0.6523 | 216 | 310 |
| C | Logistic regression | 0.7690 | 0.7676 | 0.7068 | 0.5802 | 0.6373 | 0.7027 | 168 | 293 |
| C | Linear SVM | 0.7723 | 0.7727 | 0.6852 | 0.6175 | 0.6496 | 0.7054 | 198 | 267 |

Precision, recall, and Clickbait F1 treat clickbait as the positive class. AP means average precision. FP and FN mean false positives and false negatives.
