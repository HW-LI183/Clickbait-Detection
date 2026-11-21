"""Download the labeled Webis-Clickbait-17 archive used by the proposal."""

import argparse
import hashlib
import shutil
import time
import urllib.error
import urllib.request
import zipfile
import zlib
from pathlib import Path


RECORD_URL = "https://zenodo.org/api/records/5530410"
ARCHIVE_NAME = "clickbait17-train-170630.zip"
USER_AGENT = "clickbait-detection-course-project/1.0"
CHUNK_SIZE = 1024 * 1024


def open_url(url, headers=None):
    request = urllib.request.Request(
        url, headers={"User-Agent": USER_AGENT, **(headers or {})}
    )
    return urllib.request.urlopen(request, timeout=60)


def archive_metadata():
    import json

    with open_url(RECORD_URL) as response:
        record = json.load(response)
    matches = [item for item in record["files"] if item["key"] == ARCHIVE_NAME]
    if len(matches) != 1:
        raise ValueError(f"Expected exactly one {ARCHIVE_NAME} in Zenodo record")
    item = matches[0]
    algorithm, digest = item["checksum"].split(":", 1)
    if algorithm.lower() != "md5" or len(digest) != 32:
        raise ValueError(f"Unexpected checksum: {item['checksum']}")
    return item["links"]["self"], int(item["size"]), digest.lower()


def md5sum(path):
    digest = hashlib.md5()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(CHUNK_SIZE), b""):
            digest.update(chunk)
    return digest.hexdigest()


def matches_zip_member(path, member):
    if not path.exists() or path.stat().st_size != member.file_size:
        return False
    checksum = 0
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(CHUNK_SIZE), b""):
            checksum = zlib.crc32(chunk, checksum)
    return checksum == member.CRC


def download(url, destination, size, checksum):
    if destination.exists():
        if destination.stat().st_size == size and md5sum(destination) == checksum:
            print(f"Already verified: {destination}")
            return
        raise ValueError(
            f"{destination} exists but does not match Zenodo's size/MD5; "
            "move it aside before retrying"
        )

    partial = destination.with_name(destination.name + ".part")
    for attempt in range(3):
        offset = partial.stat().st_size if partial.exists() else 0
        if offset > size:
            raise ValueError(f"Partial download is larger than expected: {partial}")
        if offset == size:
            print("Verifying MD5...")
            if md5sum(partial) != checksum:
                raise ValueError(f"MD5 mismatch for {partial}; not using this archive")
            partial.replace(destination)
            print(f"Downloaded and verified: {destination}")
            return
        try:
            headers = {"Range": f"bytes={offset}-"} if offset else {}
            with open_url(url, headers) as response:
                status = response.status
                if offset and status == 206:
                    content_range = response.headers.get("Content-Range", "")
                    if not content_range.startswith(f"bytes {offset}-"):
                        raise ValueError(f"Unexpected Content-Range: {content_range}")
                elif status != 200:
                    raise ValueError(f"Unexpected HTTP status: {status}")
                # If the server ignores Range, start over rather than append duplicates.
                mode = "ab" if offset and status == 206 else "wb"
                with partial.open(mode) as output:
                    shutil.copyfileobj(response, output, CHUNK_SIZE)
            if partial.stat().st_size != size:
                raise IOError(
                    f"Incomplete download: {partial.stat().st_size}/{size} bytes"
                )
            continue
        except (OSError, urllib.error.URLError) as error:
            if attempt == 2 and partial.stat().st_size != size:
                raise
            if partial.stat().st_size != size:
                print(f"Transfer interrupted ({error}); retrying in {2 ** attempt}s...")
                time.sleep(2 ** attempt)
    if partial.stat().st_size == size:
        print("Verifying MD5...")
        if md5sum(partial) != checksum:
            raise ValueError(f"MD5 mismatch for {partial}; not using this archive")
        partial.replace(destination)
        print(f"Downloaded and verified: {destination}")


def extract_jsonl(archive, directory):
    required = ("instances.jsonl", "truth.jsonl")
    with zipfile.ZipFile(archive) as zipped:
        members = {}
        for name in required:
            matches = [
                entry
                for entry in zipped.infolist()
                if not entry.is_dir() and Path(entry.filename).name == name
            ]
            if len(matches) != 1:
                raise ValueError(f"Expected exactly one {name} in {archive}")
            members[name] = matches[0]
        for name, member in members.items():
            target = directory / name
            if target.exists():
                if matches_zip_member(target, member):
                    print(f"Already extracted and verified: {target}")
                    continue
                raise FileExistsError(
                    f"Existing file does not match archive; refusing to overwrite: {target}"
                )
            temporary = directory / (name + ".part")
            with zipped.open(member) as source, temporary.open("wb") as output:
                shutil.copyfileobj(source, output, CHUNK_SIZE)
            temporary.replace(target)
            print(f"Extracted: {target}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "data",
        help="Download directory (default: project-root data/)",
    )
    parser.add_argument(
        "--extract",
        action="store_true",
        help="Extract instances.jsonl and truth.jsonl after verification",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show Zenodo file metadata without downloading",
    )
    args = parser.parse_args()

    url, size, checksum = archive_metadata()
    print(f"{ARCHIVE_NAME}: {size:,} bytes, md5:{checksum}")
    if args.dry_run:
        print(f"Source: {url}")
        return
    args.output_dir.mkdir(parents=True, exist_ok=True)
    archive = args.output_dir / ARCHIVE_NAME
    download(url, archive, size, checksum)
    if args.extract:
        extract_jsonl(archive, args.output_dir)


if __name__ == "__main__":
    main()
