"""Recover validation article domains from archived HTML in Webis' official ZIPs.

Optional provenance step. The evaluation can use the checked-in source table offline.
Only ZIP indexes and bounded prefixes of validation HTML members are requested.
"""
import csv
import hashlib
import io
import json
import re
import struct
import threading
import zipfile
import zlib
from concurrent.futures import ThreadPoolExecutor, as_completed
from email.utils import parsedate_to_datetime
from html.parser import HTMLParser
from urllib.parse import quote, urlsplit

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from tfidf_preexperiment import ROOT, read_jsonl

RECORD = "https://zenodo.org/api/records/5530410"
FILES = "https://zenodo.org/records/5530410/files/"
TABLE = ROOT / "output" / "validation_article_sources.csv"
CACHE = ROOT / "data" / "source_recovery.jsonl"
FIELDS = ("id", "domain", "canonical_sha256", "evidence")
SOURCE_MONTH = "2016-12"
LOCAL = threading.local()


def session():
    if not hasattr(LOCAL, "session"):
        client = requests.Session()
        retry = Retry(total=4, backoff_factor=1, status_forcelist=(429, 500, 502, 503, 504))
        client.mount("https://", HTTPAdapter(max_retries=retry))
        client.headers["User-Agent"] = "Webis-clickbait-course-study/1.0 (archive provenance)"
        LOCAL.session = client
    return LOCAL.session


def byte_range(url, start, end):
    # A range-specific query also separates partial responses in intermediary caches.
    with session().get(f"{url}?range={start}-{end}", headers={"Range": f"bytes={start}-{end}",
                       "Accept-Encoding": "identity"}, stream=True, timeout=(12, 35)) as res:
        res.raise_for_status()
        if res.status_code != 206 or not res.headers.get("Content-Range", "").startswith(
                f"bytes {start}-{end}/"):
            raise ValueError("Archive server did not return the requested byte range")
        data = res.raw.read(end - start + 2)
        if len(data) != end - start + 1:
            raise ValueError("Truncated archive range")
        return data


def html_index(wanted):
    res = session().get(RECORD, timeout=30)
    res.raise_for_status()
    entries = {}
    for item in sorted(res.json()["files"], key=lambda f: f["key"]):
        if not item["key"].startswith("archives-clickbait17-train-170630-part"):
            continue
        url, size = FILES + quote(item["key"]), item["size"]
        start = max(0, size - 4 * 1024**2)
        # ZipFile rebases member offsets relative to this tail. Restore the file origin.
        cache_dir = ROOT / "data/archive_indexes"
        cache_dir.mkdir(exist_ok=True)
        tail_path = cache_dir / (item["key"] + ".tail")
        if not tail_path.exists():
            tail_path.write_bytes(byte_range(url, start, size - 1))
        with zipfile.ZipFile(io.BytesIO(tail_path.read_bytes())) as archive:
            for member in archive.infolist():
                match = re.fullmatch(r"\d+/(\d+)/url_\1\.html", member.filename)
                if match and match[1] in wanted:
                    if match[1] in entries:
                        raise ValueError(f"Duplicate archived HTML for {match[1]}")
                    entries[match[1]] = (url, start + member.header_offset, member)
    return entries


class HeadLinks(HTMLParser):
    def __init__(self):
        super().__init__()
        self.links = {}
        self.done = False

    def handle_endtag(self, tag):
        if tag == "head":
            self.done = True

    def handle_starttag(self, tag, attrs):
        if self.done:
            return
        attrs = {key: value or "" for key, value in attrs}
        if tag == "link" and "canonical" in attrs.get("rel", "").lower().split():
            self.links.setdefault("canonical", attrs.get("href", ""))
        if tag == "meta" and (attrs.get("property") or attrs.get("name", "")).lower() == "og:url":
            self.links.setdefault("og:url", attrs.get("content", ""))


def recover(item):
    key, entry = item
    row = dict(id=key, domain="", canonical_sha256="", evidence="missing_html")
    if entry is None:
        return row
    url, offset, member = entry
    prefix = byte_range(url, offset, offset + min(member.compress_size, 49152) + 256)
    if prefix[:4] != b"PK\x03\x04":
        raise ValueError(f"Invalid local header at {key}")
    n, extra = struct.unpack_from("<HH", prefix, 26)
    if prefix[30:30+n].decode("ascii") != member.filename:
        raise ValueError(f"Archive member mismatch at {key}")
    data = prefix[30+n+extra:][:member.compress_size]
    if member.compress_type == zipfile.ZIP_DEFLATED:
        data = zlib.decompressobj(-15).decompress(data, 512 * 1024)
    elif member.compress_type != zipfile.ZIP_STORED:
        raise ValueError(f"Unsupported ZIP compression at {key}")
    parser = HeadLinks()
    parser.feed(data.decode("utf-8", errors="replace"))
    row["evidence"] = "no_article_url"
    for source in ("canonical", "og:url"):
        raw = parser.links.get(source, "").strip()
        parsed = urlsplit("https:" + raw if raw.startswith("//") else raw)
        host = (parsed.hostname or "").lower().removeprefix("www.")
        if parsed.scheme in ("http", "https") and "." in host and parsed.path.strip("/"):
            # The path identifies the story without tracking query strings or fragments.
            fingerprint = host + "/" + parsed.path.strip("/")
            row.update(domain=host, canonical_sha256=hashlib.sha256(fingerprint.encode()).hexdigest(),
                       evidence=source)
            break
    return row


def main():
    manifest = read_jsonl(ROOT / "data/processed/split_manifest.jsonl")
    ids = sorted(key for key, row in manifest.items() if row["split"] == "validation")
    instances = read_jsonl(ROOT / "data/instances.jsonl")
    window = {key for key in ids if parsedate_to_datetime(
        instances[key]["postTimestamp"]).strftime("%Y-%m") == SOURCE_MONTH}
    cached = read_jsonl(CACHE) if CACHE.exists() else {}
    missing = window - cached.keys()
    errors = []
    if missing:
        index = html_index(missing)
        work = [(key, index.get(key)) for key in sorted(missing)]
        # Preserve successes on interrupted runs and retry transient failures on the next run.
        def attempt(item):
            try:
                return recover(item), None
            except (requests.RequestException, OSError, ValueError) as exc:
                return None, (item[0], str(exc))

        with CACHE.open("a", encoding="utf-8") as stream, ThreadPoolExecutor(max_workers=6) as pool:
            futures = [pool.submit(attempt, item) for item in work]
            for idx, future in enumerate(as_completed(futures), 1):
                row, error = future.result()
                if error:
                    errors.append(error)
                    if len(errors) <= 3:
                        print(f"Unresolved {error[0]}: {error[1]}", flush=True)
                else:
                    cached[row["id"]] = row
                    stream.write(json.dumps(row) + "\n")
                    stream.flush()
                if idx % 200 == 0:
                    print(f"Checked {idx}/{len(work)}; transient failures: {len(errors)}", flush=True)
    if errors:
        raise RuntimeError(f"Retry source recovery; {len(errors)} incomplete entries, first: {errors[0]}")
    with TABLE.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS, lineterminator="\n")
        writer.writeheader()
        for key in ids:
            writer.writerow(cached[key] if key in window else
                            dict(id=key, domain="", canonical_sha256="", evidence="outside_source_window"))
    print(f"Saved {len(window)} in-window source audits and {len(ids)-len(window)} out-of-window flags")


if __name__ == "__main__":
    main()
