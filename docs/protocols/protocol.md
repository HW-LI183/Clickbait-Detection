# Preprocessing protocol

Version 1.2, updated on 2026-11-21 (Asia/Shanghai), before any modeling.
This document specifies the input, environment, decisions, and outputs for
the Webis-Clickbait-17 experiment in `plan.md`. Any later change to these
rules must be documented with a new protocol revision and regenerated
outputs; do not silently change the frozen split.

Revision history: version 1.0 preceded a diagnostic preprocessing run.
That run found 190 retained records with fewer than 20 body words, including
page placeholders such as "Still Watching? Rotate Device". Version 1.1
defines these as unusable article content and excludes them. The version
1.0 split was not used for training or evaluation. Version 1.2 moves the
three optional article-metadata fields to a separate preprocessing and
audit protocol. It changes only the field-audit scope, not the retained
records, grouping, or split.

## Environment and source

- Host as reported by Python: `Windows-10-10.0.26300-SP0`, AMD64, little-endian.
- Interpreter: CPython 3.11.9, executable
  `C:\Users\spide\AppData\Local\Programs\Python\Python311\python.exe`.
- Unicode database: 14.0.0. Text I/O is explicit UTF-8 with LF output.
- Preprocessing dependencies: Python standard library only; no external
  package, stochastic library, or downloaded model is used.
- Input: labeled Webis-Clickbait-17, Zenodo record 5530410,
  `clickbait17-train-170630.zip`, already extracted to:

| File | SHA-256 before preprocessing |
| --- | --- |
| `data/instances.jsonl` | `88fb4dbbb4008c15048d750aa4d6ee954e09c5fd757db6dc5dd8d65222a60884` |
| `data/truth.jsonl` | `9ba97792ff81f604a9c3b1abbe8471d2e619be2acd6742d48e5a588f611c14b0` |

The script must reject inputs whose hashes differ. Raw files remain
unchanged. The `media` directory is not needed.

## Frozen parameters and rules

- Random seed: **20261121**, passed to local `random.Random` instances;
  do not rely on Python's process-randomized `hash()` or global RNG state.
- Main body cap: **1,000 words**, defined as the first 1,000
  whitespace-delimited tokens after normalization and joining paragraphs
  in source order. Preserve the full normalized body for later 200-, 500-,
  and 1,500-word sensitivity analyses.
- Required model text: `postText` (list of strings), `targetTitle` (string),
  `targetParagraphs` (list of strings). Normalize each string to Unicode
  NFC, collapse all whitespace to one ASCII space, strip, then join
  nonempty list elements with one space. Preserve punctuation and case in
  the model text. No stemming, stop-word removal, or TF-IDF fitting here.
- `instances.jsonl` and `truth.jsonl` must be valid JSON objects with
  unique, matching nonempty string IDs and valid binary `truthClass`
  values. Fail on malformed records, unexpected required-field types,
  duplicate IDs, or unmatched ID sets; do not silently discard them.
- Exclude a joined record when one or more required normalized model texts
  are empty, or the full normalized article body has fewer than 20
  whitespace-delimited words. Report its ID and reasons. This threshold
  removes obvious page placeholders; it does not claim that all longer
  bodies are clean. Use the same retained records for A, B, and C. Empty
  optional fields never cause exclusion.
- Audit the three required instance text fields and `truthClass` for absent
  keys, explicit `null`, empty strings/list values, and whitespace-only
  text. For required list fields, also count empty, whitespace-only, null,
  and nonempty elements. Optional article metadata is processed separately
  under `optional_features_protocol.md`; it is not copied into the
  primary model-text output. Post media, timestamps, and judgment
  statistics remain outside the model input.
- For duplicate grouping only, casefold and collapse the normalized text.
  Union records with identical nonempty article bodies, article titles,
  or teaser texts. For near-duplicate bodies, tokenize casefolded text
  into Unicode word tokens and hash consecutive five-token shingles with
  BLAKE2b (64-bit). Use the 16 smallest distinct shingle hashes as
  deterministic candidate anchors. Only anchors appearing in at most
  100 records are considered; candidate pairs must share at least two
  such anchors. Union a candidate pair when its complete shingle-set
  Jaccard similarity is at least 0.90 and its shorter/longer word-count
  ratio is at least 0.80. Bodies shorter than five words can only be
  linked by exact text/title/teaser rules. Record candidate and match
  counts and inspect the largest connected groups in the audit report.
- Assign stable group IDs by sorting components on their smallest member
  ID. Split whole groups into train/validation/test with target post
  fractions 70/15/15. Run 64 deterministic seeded greedy assignments
  (seed plus attempt number), selecting the one with the lowest
  normalized squared error in total and per-label counts; ties use the
  earliest attempt. Sort groups by descending size with seeded
  random tie order before each assignment. Save the seed, assignments,
  counts, and achieved class proportions. A split with a group on two
  sides is invalid. Labels may be used for stratification and audit,
  never for text processing or duplicate detection.
- No vocabulary, IDF, feature weighting, model fitting, or threshold
  selection occurs here. Subsequent learning must fit on train only,
  tune on validation only, and reserve test labels for final evaluation.

## Outputs and verification

The script `scripts/preprocess_data.py` writes under `data/processed/`:

- `records.jsonl`: `id`, `post_text`, `title`, `body_full`, `body_1000`.
- `split_manifest.jsonl`: `id`, `group_id`, `split`, `truthClass`.
- `audit.json`: input/output hashes, field-level missing counts,
  exclusions, class/split distributions, body-length and grouping
  diagnostics, and protocol parameters.

Write deterministic UTF-8/LF JSONL in sorted ID order. The output directory
is replaced only after all validation passes; an existing valid output may
be reproduced byte-for-byte. Verify one-to-one retained IDs, the 1,000-word
cap, nonempty required texts, expected label set, no group spanning splits,
and a 70/15/15 group split as close as feasible. Record any material
deviation in `audit.json` and the final report.
