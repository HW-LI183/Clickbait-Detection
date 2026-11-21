# Optional article-metadata preprocessing protocol

Version 1.0, recorded on 2026-11-21 (Asia/Shanghai), before generating
optional features. The source environment and raw input SHA-256 values are
fixed in `protocol.md`. This is a sidecar to the version 1.2 primary
preprocessing output, not a change to its retained IDs, groups, or split.

## Inputs and cleaning

- Verify both raw JSONL files against the primary protocol SHA-256 values,
  the primary records and split manifest against the hashes recorded in
  its audit, and the audit's protocol hash before joining by `id`.
  Process exactly the retained IDs in the manifest, sorted by ID. No
  optional-field missingness excludes a record.
- `targetCaptions` is a list of strings; `targetKeywords` and
  `targetDescription` are strings. For each element/string, use the same
  Unicode NFC and whitespace normalization as the primary script, then
  join nonempty caption elements with one space in their source order.
  Missing keys and explicit `null` become `""`; an unexpected non-null
  type fails rather than being silently stringified. Preserve case and
  punctuation in model-ready text. Caption strings are text metadata;
  no image files are read.
- Audit absent, null, empty, whitespace-only, and present values for each
  field in both the full raw set and the retained set. For the caption
  list, also count blank/null/nonstring elements. Report cleaned empty
  counts. Do not combine these optional fields with primary `records.jsonl`.

## Duplicate identification

For each of the three fields separately, ignore empty cleaned values
when grouping. Union identical nonempty casefolded strings. For potential
near duplicates, use the primary script's Unicode word tokenizer,
five-word BLAKE2b shingle hashes, 16 bottom-hash anchors, a maximum
anchor frequency of 100, at least two shared anchors, complete shingle
Jaccard >= 0.90, and shorter/longer word-count ratio >= 0.80. Values
shorter than five tokens only participate in exact matching. Connected
components of at least two records receive deterministic IDs sorted by
their smallest member ID; singletons and empty values have no duplicate
group ID.

Report exact and near-duplicate merges, repeated-record counts, largest
groups, and groups spanning the frozen train/validation/test split.
These are diagnostics only: never resplit based on optional metadata
after inspecting test labels. Any later model using a feature with
cross-split duplicate groups must address and disclose that risk in a
separate, predeclared exploratory evaluation. The A/B/C primary results
remain unchanged.

## Outputs

`scripts/preprocess_optional_features.py` writes deterministically to
`data/processed/optional/`:

- `optional_features.jsonl`: `id`, `target_captions`,
  `target_keywords`, `target_description` for every retained ID.
- `duplicate_groups.jsonl`: `id` and three corresponding optional
  duplicate-group IDs (or `null`).
- `audit.json`: source and output hashes, missingness, duplicates,
  cross-split diagnostics, and both protocol hashes.

Use UTF-8 and LF with rows sorted by ID. Verify the ID set matches the
primary manifest, and on rerun verify identical bytes rather than
overwriting different files. No label, annotation score, or primary
model text is copied into the optional feature table.
