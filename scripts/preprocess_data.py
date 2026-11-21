"""Audit, normalize, group, and split the labeled Webis-Clickbait-17 data."""

import argparse
import hashlib
import heapq
import json
import random
import re
import tempfile
import unicodedata
from collections import Counter, defaultdict
from functools import lru_cache
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
SEED = 20261121
BODY_CAP = 1000
MIN_BODY_WORDS = 20
INPUT_HASHES = {
    "instances.jsonl": "88fb4dbbb4008c15048d750aa4d6ee954e09c5fd757db6dc5dd8d65222a60884",
    "truth.jsonl": "9ba97792ff81f604a9c3b1abbe8471d2e619be2acd6742d48e5a588f611c14b0",
}
LABELS = ("clickbait", "no-clickbait")
SPLITS = ("train", "validation", "test")
FRACTIONS = (0.70, 0.15, 0.15)
WORD_RE = re.compile(r"\w+", re.UNICODE)


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_jsonl(path):
    records = {}
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            record = json.loads(line)
            if not isinstance(record, dict):
                raise ValueError(f"{path}:{line_number}: expected JSON object")
            record_id = record.get("id")
            if not isinstance(record_id, str) or not record_id:
                raise ValueError(f"{path}:{line_number}: invalid ID")
            if record_id in records:
                raise ValueError(f"{path}:{line_number}: duplicate ID {record_id}")
            records[record_id] = record
    return records


def normalize(text):
    if not isinstance(text, str):
        raise TypeError(f"Expected string, got {type(text).__name__}")
    return " ".join(unicodedata.normalize("NFC", text).split())


def normalize_field(value, expected_type):
    if expected_type == "list":
        if not isinstance(value, list):
            raise TypeError(f"Expected list, got {type(value).__name__}")
        if any(not isinstance(part, str) for part in value):
            raise TypeError("Expected a list of strings")
        return " ".join(part for item in value if (part := normalize(item)))
    return normalize(value)


def field_audit(records, fields):
    result = {}
    for field in sorted(set(fields)):
        statuses = Counter()
        types = Counter()
        elements = Counter()
        for row in records.values():
            if field not in row:
                statuses["absent"] += 1
                continue
            value = row[field]
            types[type(value).__name__] += 1
            if isinstance(value, list):
                for item in value:
                    if item is None:
                        elements["null"] += 1
                    elif item == "":
                        elements["empty"] += 1
                    elif isinstance(item, str) and not item.strip():
                        elements["whitespace_only"] += 1
                    elif isinstance(item, str):
                        elements["present"] += 1
                    else:
                        elements["nonstring"] += 1
            if value is None:
                statuses["null"] += 1
            elif value == "" or value == []:
                statuses["empty"] += 1
            elif isinstance(value, str) and not value.strip():
                statuses["whitespace_only"] += 1
            elif isinstance(value, list) and all(
                item is None or isinstance(item, str) and not item.strip()
                for item in value
            ):
                statuses["whitespace_only"] += 1
            else:
                statuses["present"] += 1
        result[field] = {
            "status": {
                key: statuses[key]
                for key in ("absent", "null", "empty", "whitespace_only", "present")
            },
            "types": dict(types),
            "list_elements": {
                key: elements[key]
                for key in ("null", "empty", "whitespace_only", "present", "nonstring")
            },
        }
    return result


def prepare(instances, truth):
    if set(instances) != set(truth):
        raise ValueError(
            f"Unmatched IDs: {len(set(instances) - set(truth))} instance-only, "
            f"{len(set(truth) - set(instances))} truth-only"
        )
    rows = []
    exclusions = []
    for record_id in sorted(instances):
        instance = instances[record_id]
        label = truth[record_id].get("truthClass")
        if label not in LABELS:
            raise ValueError(f"Invalid truthClass for {record_id}: {label!r}")
        values = {}
        for source, output, kind in (
            ("postText", "post_text", "list"),
            ("targetTitle", "title", "string"),
            ("targetParagraphs", "body_full", "list"),
        ):
            if source not in instance or instance[source] is None:
                values[output] = ""
            else:
                try:
                    values[output] = normalize_field(instance[source], kind)
                except TypeError as error:
                    raise ValueError(f"{record_id}.{source}: {error}") from error
        missing = [name for name, value in values.items() if not value]
        body_words = len(values["body_full"].split())
        if values["body_full"] and body_words < MIN_BODY_WORDS:
            missing.append("body_under_20_words")
        if missing:
            exclusions.append({"id": record_id, "reasons": missing})
            continue
        words = values["body_full"].split()
        rows.append(
            {
                "id": record_id,
                **values,
                "body_1000": " ".join(words[:BODY_CAP]),
                "truthClass": label,
                "body_words": len(words),
            }
        )
    return rows, exclusions


class UnionFind:
    def __init__(self, count):
        self.parent = list(range(count))
        self.size = [1] * count

    def find(self, item):
        while self.parent[item] != item:
            self.parent[item] = self.parent[self.parent[item]]
            item = self.parent[item]
        return item

    def union(self, first, second):
        a, b = self.find(first), self.find(second)
        if a == b:
            return False
        if self.size[a] < self.size[b]:
            a, b = b, a
        self.parent[b] = a
        self.size[a] += self.size[b]
        return True


def shingle_hashes(text):
    tokens = WORD_RE.findall(text.casefold())
    return {
        int.from_bytes(
            hashlib.blake2b(
                "\x1f".join(tokens[index : index + 5]).encode("utf-8"),
                digest_size=8,
            ).digest(),
            "big",
        )
        for index in range(len(tokens) - 4)
    }


def group_records(rows):
    union = UnionFind(len(rows))
    exact_links = Counter()
    for field in ("body_full", "title", "post_text"):
        seen = {}
        for index, row in enumerate(rows):
            key = row[field].casefold()
            if key in seen:
                if union.union(index, seen[key]):
                    exact_links[field] += 1
            else:
                seen[key] = index

    anchors = defaultdict(list)
    for index, row in enumerate(rows):
        for anchor in heapq.nsmallest(16, shingle_hashes(row["body_full"])):
            anchors[anchor].append(index)
    candidate_counts = Counter()
    ignored_anchors = 0
    for indices in anchors.values():
        if len(indices) > 100:
            ignored_anchors += 1
            continue
        for position, first in enumerate(indices):
            for second in indices[position + 1 :]:
                candidate_counts[(first, second)] += 1
    del anchors

    @lru_cache(maxsize=256)
    def shingles(index):
        return shingle_hashes(rows[index]["body_full"])

    matched = 0
    verified = 0
    for (first, second), shared in sorted(candidate_counts.items()):
        if shared < 2 or union.find(first) == union.find(second):
            continue
        shorter = min(rows[first]["body_words"], rows[second]["body_words"])
        longer = max(rows[first]["body_words"], rows[second]["body_words"])
        if shorter / longer < 0.80:
            continue
        a, b = shingles(first), shingles(second)
        verified += 1
        if len(a & b) / len(a | b) >= 0.90:
            union.union(first, second)
            matched += 1

    components = defaultdict(list)
    for index, row in enumerate(rows):
        components[union.find(index)].append(index)
    groups = {}
    for number, indices in enumerate(
        sorted(components.values(), key=lambda group: rows[group[0]]["id"]), 1
    ):
        groups[f"g{number:06d}"] = indices
    diagnostics = {
        "exact_group_merges": dict(exact_links),
        "near_duplicate_candidate_pairs": sum(
            count >= 2 for count in candidate_counts.values()
        ),
        "near_duplicate_pairs_verified": verified,
        "near_duplicate_group_merges": matched,
        "common_anchors_ignored": ignored_anchors,
        "groups": len(groups),
        "largest_groups": [
            {"group_id": group_id, "size": len(indices)}
            for group_id, indices in sorted(
                groups.items(), key=lambda item: (-len(item[1]), item[0])
            )[:10]
        ],
    }
    return groups, diagnostics


def split_groups(groups, rows):
    totals = Counter(row["truthClass"] for row in rows)
    targets = {
        split: {
            "total": len(rows) * fraction,
            **{label: totals[label] * fraction for label in LABELS},
        }
        for split, fraction in zip(SPLITS, FRACTIONS)
    }
    descriptions = [
        (
            group_id,
            {
                "total": len(indices),
                **dict(Counter(rows[index]["truthClass"] for index in indices)),
            },
        )
        for group_id, indices in groups.items()
    ]

    def score(counts):
        return sum(
            (counts[split][key] - targets[split][key]) ** 2
            / max(targets[split][key], 1)
            for split in SPLITS
            for key in ("total", *LABELS)
        )

    best = None
    for attempt in range(64):
        rng = random.Random(SEED + attempt)
        ordered = descriptions.copy()
        rng.shuffle(ordered)
        ordered.sort(key=lambda item: -item[1]["total"])
        counts = {split: Counter() for split in SPLITS}
        assignment = {}
        for group_id, group_counts in ordered:
            possibilities = []
            for split in SPLITS:
                delta = sum(
                    (
                        (counts[split][key] + group_counts.get(key, 0) - targets[split][key]) ** 2
                        - (counts[split][key] - targets[split][key]) ** 2
                    )
                    / max(targets[split][key], 1)
                    for key in ("total", *LABELS)
                )
                possibilities.append((delta, rng.random(), split))
            selected = min(possibilities)[2]
            assignment[group_id] = selected
            counts[selected].update(group_counts)
        result = score(counts)
        if best is None or result < best[0]:
            best = (result, attempt, assignment, counts)
    _, attempt, assignment, counts = best
    if any(counts[split]["total"] == 0 for split in SPLITS):
        raise ValueError("Group split produced an empty partition")
    return assignment, {
        "seed": SEED,
        "selected_attempt": attempt,
        "targets": targets,
        "actual": {
            split: {key: counts[split][key] for key in ("total", *LABELS)}
            for split in SPLITS
        },
    }


def write_jsonl(path, rows):
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def process(root=ROOT, expected_audit_sha256=None):
    root = Path(root).resolve()
    data_dir = root / "data"
    input_hashes = {name: sha256(data_dir / name) for name in INPUT_HASHES}
    if input_hashes != INPUT_HASHES:
        raise ValueError(f"Raw data SHA-256 differs from protocol: {input_hashes}")
    instances = read_jsonl(data_dir / "instances.jsonl")
    truth = read_jsonl(data_dir / "truth.jsonl")
    print(f"Loaded {len(instances):,} instances and {len(truth):,} labels")
    missing = {
        "instances": field_audit(instances, ("postText", "targetTitle", "targetParagraphs")),
        "truth": field_audit(truth, ("truthClass",)),
    }
    rows, exclusions = prepare(instances, truth)
    print(f"Retained {len(rows):,} records; excluded {len(exclusions):,}")
    groups, grouping = group_records(rows)
    print(f"Formed {len(groups):,} leakage groups")
    assignments, split_info = split_groups(groups, rows)
    group_for_index = {
        index: group_id for group_id, indices in groups.items() for index in indices
    }
    records = [
        {key: row[key] for key in ("id", "post_text", "title", "body_full", "body_1000")}
        for row in rows
    ]
    manifest = [
        {
            "id": row["id"],
            "group_id": group_for_index[index],
            "split": assignments[group_for_index[index]],
            "truthClass": row["truthClass"],
        }
        for index, row in enumerate(rows)
    ]
    if set(group_for_index.values()) != set(groups) or set(assignments) != set(groups):
        raise ValueError("Not all groups assigned")
    if any(
        not all(row[field] for field in ("post_text", "title", "body_full", "body_1000"))
        or len(row["body_1000"].split()) > BODY_CAP
        for row in records
    ):
        raise ValueError("Invalid model text or body cap in processed records")
    if len({item["id"] for item in records}) != len(records) or len(records) != len(manifest):
        raise ValueError("Processed records and manifest do not align")
    group_splits = defaultdict(set)
    for item in manifest:
        group_splits[item["group_id"]].add(item["split"])
    if any(len(splits) != 1 for splits in group_splits.values()):
        raise ValueError("A duplicate group crosses split boundaries")
    lengths = sorted(row["body_words"] for row in rows)
    audit = {
        "protocol_sha256": sha256(root / "docs" / "protocols" / "protocol.md"),
        "input_sha256": input_hashes,
        "source_records": len(instances),
        "retained_records": len(rows),
        "exclusions": exclusions,
        "field_audit": missing,
        "labels": dict(Counter(row["truthClass"] for row in rows)),
        "body_words": {
            "median": lengths[len(lengths) // 2] if lengths else 0,
            "p90": lengths[int(len(lengths) * 0.90)] if lengths else 0,
            "above_1000": sum(length > BODY_CAP for length in lengths),
        },
        "grouping": grouping,
        "split": split_info,
        "parameters": {
            "seed": SEED,
            "body_cap": BODY_CAP,
            "minimum_body_words": MIN_BODY_WORDS,
            "near_duplicate_jaccard": 0.90,
            "near_duplicate_min_length_ratio": 0.80,
            "candidate_anchors": 16,
            "maximum_anchor_frequency": 100,
            "split_attempts": 64,
        },
    }
    output = data_dir / "processed"
    with tempfile.TemporaryDirectory(prefix=".preprocess-", dir=data_dir) as temporary:
        temporary = Path(temporary).resolve()
        if temporary.parent != data_dir.resolve():
            raise ValueError("Temporary output escaped data directory")
        write_jsonl(temporary / "records.jsonl", records)
        write_jsonl(temporary / "split_manifest.jsonl", manifest)
        audit["output_sha256"] = {
            name: sha256(temporary / name)
            for name in ("records.jsonl", "split_manifest.jsonl")
        }
        with (temporary / "audit.json").open(
            "w", encoding="utf-8", newline="\n"
        ) as stream:
            json.dump(audit, stream, ensure_ascii=False, indent=2, sort_keys=True)
            stream.write("\n")
        if output.exists():
            if not output.is_dir() or any(
                not (output / name).is_file()
                or sha256(output / name) != sha256(temporary / name)
                for name in ("records.jsonl", "split_manifest.jsonl")
            ):
                raise FileExistsError(
                    f"{output} contains different records or split; "
                    "refusing to overwrite"
                )
            old_audit = output / "audit.json"
            if not old_audit.is_file():
                raise FileExistsError(f"Missing existing audit: {old_audit}")
            old_hash = sha256(old_audit)
            new_hash = sha256(temporary / "audit.json")
            if old_hash != new_hash:
                if not expected_audit_sha256 or old_hash != expected_audit_sha256.lower():
                    raise FileExistsError(
                        f"Audit differs at {old_audit}; pass "
                        "--refresh-audit-if-sha256 with its verified old hash"
                    )
                (temporary / "audit.json").replace(old_audit)
                print(f"Refreshed audit only; records and split unchanged: {output}")
            else:
                print(f"Verified existing deterministic output: {output}")
        else:
            temporary.rename(output)
            print(f"Wrote validated output: {output}")
    return audit


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--refresh-audit-if-sha256",
        metavar="SHA256",
        help="Replace only a changed audit when its current hash matches this value",
    )
    args = parser.parse_args()
    process(expected_audit_sha256=args.refresh_audit_if_sha256)
