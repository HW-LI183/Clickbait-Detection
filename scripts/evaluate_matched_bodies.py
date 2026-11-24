"""Score the original and a matched wrong article body on the validation split."""
import argparse
import csv
import hashlib
import json
import platform
from collections import Counter, defaultdict

import numpy as np
import scipy
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer

from evaluate_primary import metrics
from tfidf_preexperiment import (
    CONDITIONS, FIELDS, MODELS, ROOT, SEED, combine_blocks, load_training,
    pilot_model, read_jsonl, sha256, vectorize_fold,
)

OUTPUT = ROOT / "output"
SOURCES = OUTPUT / "validation_article_sources.csv"


def topic_text(row):
    return " ".join([row["title"], row["title"], *row["body_1000"].split()[:100]])


def match_articles(train_ids, ids, records, manifest, sources):
    encoder = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=30000,
                              stop_words="english", sublinear_tf=True)
    encoder.fit(topic_text(records[key]) for key in train_ids)
    topics = encoder.transform(topic_text(records[key]) for key in ids)
    by_domain = defaultdict(list)
    for i, key in enumerate(ids):
        if sources is None:
            by_domain["all"].append(i)
        elif sources[key]["domain"] and sources[key]["canonical_sha256"]:
            by_domain[sources[key]["domain"]].append(i)
    lengths = [len(records[key]["body_full"].split()) for key in ids]
    bodies = [hashlib.sha256(records[key]["body_1000"].casefold().encode()).digest()
              for key in ids]
    groups = [manifest[key]["group_id"] for key in ids]
    articles = [sources[key]["canonical_sha256"] for key in ids] if sources is not None else ids
    edges = []
    for indices in by_domain.values():
        similarity = (topics[indices] @ topics[indices].T).tocoo()
        for a, b, cosine in zip(similarity.row, similarity.col, similarity.data):
            if a >= b or not 0.10 <= cosine <= 0.80:
                continue
            i, j = indices[a], indices[b]
            if groups[i] == groups[j] or articles[i] == articles[j] or bodies[i] == bodies[j]:
                continue
            ratio = min(lengths[i], lengths[j]) / max(lengths[i], lengths[j])
            if ratio >= 2 / 3:
                edges.append((-float(cosine), -ratio, i, j))

    pairs, used_groups, used_articles = [], set(), set()
    for neg_cosine, neg_ratio, i, j in sorted(edges):
        if {groups[i], groups[j]} & used_groups or {articles[i], articles[j]} & used_articles:
            continue
        used_groups.update((groups[i], groups[j]))
        used_articles.update((articles[i], articles[j]))
        pairs.append((i, j, -neg_cosine, -neg_ratio))
    if not pairs:
        raise ValueError("No eligible same-host topic/length matched pairs")
    return pairs, len(edges), lengths


def bootstrap_gap(y, original, swapped):
    # Each adjacent pair of rows exchanges its two bodies. Sample those units intact.
    contributions = []
    for prediction in (original, swapped):
        cell = (2 * y.astype(int) + prediction.astype(int)).reshape(-1, 2)
        contributions.append(np.stack([(cell == k).sum(axis=1) for k in range(4)], axis=-1))
    counts = np.stack(contributions)
    n = counts.shape[1]
    rng = np.random.default_rng(SEED)
    gaps = []
    for _ in range(2000):
        draw = rng.integers(n, size=n)
        tn, fp, fn, tp = counts[:, draw].sum(axis=1).T
        errors = fp + fn
        f1 = (np.divide(2 * tn, 2 * tn + errors, out=np.zeros(2), where=2 * tn + errors > 0)
              + np.divide(2 * tp, 2 * tp + errors, out=np.zeros(2), where=2 * tp + errors > 0)) / 2
        gaps.append(f1[0] - f1[1])
    return np.quantile(gaps, [0.025, 0.975]).tolist()


def distribution(values):
    return dict(zip(("min", "q25", "median", "q75", "max"),
                    np.quantile(values, [0, .25, .5, .75, 1]).tolist()))


def run(topic_only=False):
    processed = ROOT / "data/processed"
    train_ids, texts, train_labels, _, hashes = load_training(processed)
    primary = json.loads((OUTPUT / "primary_evaluation.json").read_text(encoding="utf-8"))
    if hashes != primary["inputs_sha256"]:
        raise ValueError("Data or split differs from primary evaluation")
    records = read_jsonl(processed / "records.jsonl")
    manifest = read_jsonl(processed / "split_manifest.jsonl")
    ids = sorted(key for key, row in manifest.items() if row["split"] == "validation")
    sources = None
    source_rows = []
    if not topic_only:
        with SOURCES.open(encoding="utf-8", newline="") as stream:
            source_rows = list(csv.DictReader(stream))
        sources = {row["id"]: row for row in source_rows}
        if set(ids) != set(sources) or len(ids) != len(source_rows):
            raise ValueError("Source table must contain exactly one audit per validation ID")

    pairs, edge_count, lengths = match_articles(train_ids, ids, records, manifest, sources)
    recipients = np.array([idx for i, j, _, _ in pairs for idx in (i, j)])
    donors = recipients.reshape(-1, 2)[:, ::-1].ravel()
    audit = []
    for number, (i, j, cosine, ratio) in enumerate(pairs, 1):
        for a, b in ((i, j), (j, i)):
            key, other = ids[a], ids[b]
            audit.append(dict(pair=number, recipient=key, donor=other, domain="" if topic_only else sources[key]["domain"],
                              group=manifest[key]["group_id"], donor_group=manifest[other]["group_id"],
                              words=lengths[a], donor_words=lengths[b], cosine=round(cosine, 6),
                              length_ratio=round(ratio, 6), label=manifest[key]["truthClass"]))

    settings = primary["settings"]
    for field in FIELDS:
        texts[field].extend(records[key][field] for key in ids)
    n = len(train_ids)
    vectors, _, _ = vectorize_fold(texts, np.arange(n), np.arange(n, n + len(ids)),
                                    settings["limits"], tuple(settings["ngram"]))
    valid_y = np.array([manifest[key]["truthClass"] == "clickbait" for key in ids])
    y = valid_y[recipients]
    used_domains = Counter(row["domain"] for row in audit if row["domain"])
    result = {
        "split": "validation", "match": "topic_and_length" if topic_only else "same_host_topic_and_length",
        "source_window_utc": None if topic_only else "2016-12",
        "inputs_sha256": hashes, "sources_sha256": None if topic_only else sha256(SOURCES),
        "sklearn": sklearn.__version__, "primary_sklearn": primary["runtime"]["sklearn"],
        "python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__,
        "validation_rows": len(ids), "verified_source_rows": None if topic_only else sum(bool(v["domain"]) for v in source_rows),
        "source_evidence": dict(Counter(v["evidence"] for v in source_rows)),
        "eligible_edges": edge_count, "pairs": len(pairs), "matched_rows": len(audit),
        "matched_clickbait": int(y.sum()), "matched_nonclickbait": int((~y).sum()),
        "host_counts": dict(sorted(used_domains.items())),
        "topic_cosine": distribution([p[2] for p in pairs]),
        "shorter_longer_ratio": distribution([p[3] for p in pairs]),
        "bootstrap_unit": "disjoint reciprocal pair", "bootstrap_draws": 2000,
        "seed": SEED, "models": {},
    }
    weights = settings["weights"]
    correct_inputs = {condition: combine_blocks(vectors, CONDITIONS[condition], weights)
                      for condition in ("B", "C")}
    # Reindex the already-encoded validation body block only.
    wrong_blocks = {field: (train[:0], val[donors if field == "body_1000" else recipients])
                    for field, (train, val) in vectors.items()}
    _, wrong_x = combine_blocks(wrong_blocks, FIELDS, weights)
    for name in MODELS:
        selected = {condition: primary["models"][f"{condition}/{name}"] for condition in ("B", "C")}
        scores = {}
        for condition, (xt, xv) in correct_inputs.items():
            config = selected[condition]
            model = pilot_model(name).set_params(C=config["C"]).fit(xt, train_labels == "clickbait")
            scores[condition] = model.decision_function(xv)
            if condition == "C":
                scores["swapped"] = model.decision_function(wrong_x)
        c_cutoff = selected["C"]["threshold"]
        predictions = {key: (value if key == "swapped" else value[recipients])
                       >= selected["B" if key == "B" else "C"]["threshold"]
                       for key, value in scores.items()}
        subset = {key: metrics(y, predictions[key], value if key == "swapped" else value[recipients])
                  for key, value in scores.items()}
        orig, wrong = predictions["C"], predictions["swapped"]
        # Donor labels are examined only after matching and scoring.
        same_label = np.repeat(y[::2] == y[1::2], 2)
        changes_by_label = {}
        for kind, mask in (("same", same_label), ("different", ~same_label)):
            changes_by_label[kind] = {
                "pairs": int(mask.sum() // 2),
                "correct_only_with_original": int((mask & (orig == y) & (wrong != y)).sum()),
                "correct_only_with_swap": int((mask & (orig != y) & (wrong == y)).sum()),
            }
        result["models"][name] = {
            "selected": {key: {"C": val["C"], "threshold": val["threshold"]} for key, val in selected.items()},
            "full_validation_correct_macro_f1": metrics(valid_y, scores["C"] >= c_cutoff, scores["C"])["macro_f1"],
            "full_validation_B_macro_f1": metrics(valid_y, scores["B"] >= selected["B"]["threshold"], scores["B"])["macro_f1"],
            "original_validation_macro_f1": {key: val["validation_macro_f1"] for key, val in selected.items()},
            "matched_B": subset["B"], "matched_correct_C": subset["C"], "matched_swapped_C": subset["swapped"],
            "correct_minus_swapped": subset["C"]["macro_f1"] - subset["swapped"]["macro_f1"],
            "paired_ci95": bootstrap_gap(y, orig, wrong),
            "correct_only_with_original": int(((orig == y) & (wrong != y)).sum()),
            "correct_only_with_swap": int(((orig != y) & (wrong == y)).sum()),
            "changes_by_pair_label_agreement": changes_by_label,
        }
        for i, row in enumerate(audit):
            for key in ("B", "C", "swapped"):
                row[f"{name}_{key}"] = int(predictions[key][i])
    prefix = "matched_body_topic_only" if topic_only else "matched_body"
    pair_file = OUTPUT / f"{prefix}_pairs.csv"
    with pair_file.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=audit[0], lineterminator="\n")
        writer.writeheader()
        writer.writerows(audit)
    result["pairs_sha256"] = sha256(pair_file)
    (OUTPUT / f"{prefix}_validation.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n")
    print(f"Scored {len(pairs)} reciprocal pairs on validation.")
    for name, row in result["models"].items():
        print(name, row["correct_minus_swapped"], row["paired_ci95"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--topic-only", action="store_true",
                        help="Secondary control: match topic and length without a host requirement.")
    run(parser.parse_args().topic_only)
