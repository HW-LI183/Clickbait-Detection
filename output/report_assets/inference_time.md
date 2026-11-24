# Frozen primary inference time

This run used 2 warm-ups and 10 measured repeats on the complete 2894-record test batch. Each model was timed independently. End-to-end time includes TF-IDF transformation, sparse block assembly, classifier scoring, and thresholding. It excludes file loading.

- Python: `3.11.9 (tags/v3.11.9:de54cf5, Apr  2 2024, 10:12:12) [MSC v.1938 64 bit (AMD64)]`
- Platform: `Windows-10-10.0.26300-SP0`
- Processor: `Intel64 Family 6 Model 186 Stepping 2, GenuineIntel`
- Logical CPUs: `16`
- Model bundle load: `112.48 ms`
- Test data load and preparation: `536.50 ms`

| Condition | Model | TF-IDF ms | Classifier ms | End-to-end ms | ms/sample | Samples/s |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| A | Logistic regression | 31.46 | 0.22 | 31.68 | 0.0109 | 91359.5 |
| A | Linear SVM | 32.08 | 0.22 | 32.37 | 0.0112 | 89408.3 |
| B | Logistic regression | 61.92 | 0.25 | 62.17 | 0.0215 | 46551.2 |
| B | Linear SVM | 62.25 | 0.26 | 62.49 | 0.0216 | 46311.3 |
| C | Logistic regression | 1210.65 | 1.94 | 1212.43 | 0.4189 | 2386.9 |
| C | Linear SVM | 1186.83 | 1.99 | 1188.79 | 0.4108 | 2434.4 |

Times are machine-specific. `inference_time.json` retains every measured repeat, the software versions, and the thread settings. The one-time load measurements are single observations and may depend on the operating-system file cache.
