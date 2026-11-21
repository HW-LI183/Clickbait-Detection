"""Clean optional article metadata without changing the primary data split."""

import argparse
import json
import tempfile
from pathlib import Path

from preprocess_data import (
    INPUT_HASHES,
    ROOT,
    WORD_RE,
    field_audit,
    group_records,
    normalize_field,
    read_jsonl,
    sha256,
    write_jsonl,
)


FIELDS = (
    ("targetCaptions", "target_captions", "list"),
    ("targetKeywords", "target_keywords", "string"),
    ("targetDescription", "target_description", "string"),
)


def clean_optional(instances, retained_ids):
    rows = []
    for record_id in sorted(retained_ids):
        if record_id not in instances:
            raise ValueError(f"Retained ID missing from raw instances: {record_id}")
        source = instances[record_id]
        row = {"id": record_id}
        for raw_name, output_name, kind in FIELDS:
            value = source.get(raw_name)
            if value is None:
                row[output_name] = ""
            else:
                try:
                    row[output_name] = normalize_field(value, kind)
                except TypeError as error:
                    raise ValueError(f"{record_id}.{raw_name}: {error}") from error
        rows.append(row)
    return rows


def identify_duplicates(rows, field, split_by_id):
    present = [
        {
            "id": row["id"],
            "body_full": row[field],
            "body_words": len(WORD_RE.findall(row[field].casefold())),
            "title": row["id"],
            "post_text": row["id"],
        }
        for row in rows
        if row[field]
    ]
    if not present:
        return {}, {
            "nonempty_records": 0,
            "duplicate_groups": 0,
            "duplicate_records": 0,
            "cross_split_groups": 0,
            "cross_split_records": 0,
            "cross_split_examples": [],
            "largest_groups": [],
            "matching": {},
        }
    groups, matching = group_records(present)
    assignments = {}
    duplicates = []
    cross_split = []
    for indices in groups.values():
        if len(indices) < 2:
            continue
        group_id = f"{field}-{len(duplicates) + 1:06d}"
        ids = [present[index]["id"] for index in indices]
        for record_id in ids:
            assignments[record_id] = group_id
        duplicates.append({"group_id": group_id, "size": len(ids)})
        splits = {split_by_id[record_id] for record_id in ids}
        if len(splits) > 1:
            cross_split.append(
                {"group_id": group_id, "size": len(ids), "splits": sorted(splits)}
            )
    return assignments, {
        "nonempty_records": len(present),
        "duplicate_groups": len(duplicates),
        "duplicate_records": sum(item["size"] for item in duplicates),
        "cross_split_groups": len(cross_split),
        "cross_split_records": sum(item["size"] for item in cross_split),
        "cross_split_examples": cross_split[:20],
        "largest_groups": sorted(
            duplicates, key=lambda item: (-item["size"], item["group_id"])
        )[:10],
        "matching": {
            key: value
            for key, value in matching.items()
            if key not in ("groups", "largest_groups")
        },
    }


def process(root=ROOT, expected_audit_sha256=None):
    root = Path(root).resolve()
    data_dir = root / "data"
    primary_dir = data_dir / "processed"
    primary_audit = json.loads((primary_dir / "audit.json").read_text(encoding="utf-8"))
    raw_hashes = {name: sha256(data_dir / name) for name in INPUT_HASHES}
    if raw_hashes != INPUT_HASHES or primary_audit["input_sha256"] != raw_hashes:
        raise ValueError("Raw inputs do not match the frozen primary protocol")
    if primary_audit["protocol_sha256"] != sha256(
        root / "docs" / "protocols" / "protocol.md"
    ):
        raise ValueError("Primary protocol changed after its audit was generated")
    primary_hashes = {
        name: sha256(primary_dir / name)
        for name in ("records.jsonl", "split_manifest.jsonl")
    }
    if primary_audit["output_sha256"] != primary_hashes:
        raise ValueError("Primary processed data differs from its audit")
    manifest = read_jsonl(primary_dir / "split_manifest.jsonl")
    if len(manifest) != primary_audit["retained_records"]:
        raise ValueError("Primary manifest count does not match its audit")
    split_by_id = {record_id: row["split"] for record_id, row in manifest.items()}
    if set(split_by_id.values()) - {"train", "validation", "test"}:
        raise ValueError("Unknown primary split name")

    instances = read_jsonl(data_dir / "instances.jsonl")
    rows = clean_optional(instances, manifest)
    if len(rows) != len(manifest) or {row["id"] for row in rows} != set(manifest):
        raise ValueError("Optional feature IDs do not align with primary manifest")

    group_by_field = {}
    duplicate_audit = {}
    for _, field, _ in FIELDS:
        group_by_field[field], duplicate_audit[field] = identify_duplicates(
            rows, field, split_by_id
        )
        print(
            f"{field}: {duplicate_audit[field]['duplicate_groups']:,} "
            f"duplicate groups, {duplicate_audit[field]['cross_split_groups']:,} "
            "cross-split groups"
        )
    duplicate_rows = [
        {
            "id": row["id"],
            **{
                field + "_duplicate_group": group_by_field[field].get(row["id"])
                for _, field, _ in FIELDS
            },
        }
        for row in rows
    ]
    audit = {
        "primary_protocol_sha256": sha256(
            root / "docs" / "protocols" / "protocol.md"
        ),
        "optional_protocol_sha256": sha256(
            root / "docs" / "protocols" / "optional_features_protocol.md"
        ),
        "raw_input_sha256": raw_hashes,
        "primary_output_sha256": primary_hashes,
        "retained_records": len(rows),
        "raw_field_audit": field_audit(instances, (item[0] for item in FIELDS)),
        "retained_field_audit": field_audit(
            {record_id: instances[record_id] for record_id in manifest},
            (item[0] for item in FIELDS),
        ),
        "cleaned_empty_counts": {
            field: sum(not row[field] for row in rows) for _, field, _ in FIELDS
        },
        "duplicate_audit": duplicate_audit,
    }
    output = primary_dir / "optional"
    with tempfile.TemporaryDirectory(prefix=".optional-", dir=primary_dir) as temporary:
        temporary = Path(temporary).resolve()
        if temporary.parent != primary_dir.resolve():
            raise ValueError("Temporary output escaped processed data directory")
        write_jsonl(temporary / "optional_features.jsonl", rows)
        write_jsonl(temporary / "duplicate_groups.jsonl", duplicate_rows)
        audit["output_sha256"] = {
            name: sha256(temporary / name)
            for name in ("optional_features.jsonl", "duplicate_groups.jsonl")
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
                for name in ("optional_features.jsonl", "duplicate_groups.jsonl")
            ):
                raise FileExistsError(
                    f"{output} exists with different contents; refusing to overwrite"
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
                print(f"Refreshed optional audit only; feature files unchanged: {output}")
            else:
                print(f"Verified existing deterministic optional output: {output}")
        else:
            temporary.rename(output)
            print(f"Wrote optional feature output: {output}")
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
