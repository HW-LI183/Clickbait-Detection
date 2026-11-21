import hashlib
import io
import json
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch

import download_data


class FakeResponse(io.BytesIO):
    def __init__(self, body, status=200, headers=None):
        super().__init__(body)
        self.status = status
        self.headers = headers or {}


class DownloadTests(unittest.TestCase):
    def test_metadata_selects_only_proposal_archive(self):
        record = {
            "files": [
                {"key": "other.zip"},
                {
                    "key": download_data.ARCHIVE_NAME,
                    "size": 4,
                    "checksum": "md5:" + "a" * 32,
                    "links": {"self": "https://example.test/archive"},
                },
            ]
        }
        with patch.object(
            download_data,
            "open_url",
            return_value=FakeResponse(json.dumps(record).encode()),
        ):
            self.assertEqual(
                download_data.archive_metadata(),
                ("https://example.test/archive", 4, "a" * 32),
            )

    def test_resumes_partial_download_and_verifies(self):
        contents = b"begin and end"
        with tempfile.TemporaryDirectory() as folder:
            destination = Path(folder) / download_data.ARCHIVE_NAME
            (Path(folder) / (destination.name + ".part")).write_bytes(contents[:6])
            with patch.object(
                download_data,
                "open_url",
                return_value=FakeResponse(
                    contents[6:], 206, {"Content-Range": "bytes 6-12/13"}
                ),
            ) as request:
                download_data.download(
                    "https://example.test/archive",
                    destination,
                    len(contents),
                    hashlib.md5(contents).hexdigest(),
                )
            self.assertEqual(destination.read_bytes(), contents)
            request.assert_called_once_with(
                "https://example.test/archive", {"Range": "bytes=6-"}
            )

    def test_complete_partial_is_verified_without_network(self):
        contents = b"complete"
        with tempfile.TemporaryDirectory() as folder:
            destination = Path(folder) / download_data.ARCHIVE_NAME
            (Path(folder) / (destination.name + ".part")).write_bytes(contents)
            with patch.object(download_data, "open_url") as request:
                download_data.download(
                    "https://example.test/archive",
                    destination,
                    len(contents),
                    hashlib.md5(contents).hexdigest(),
                )
            request.assert_not_called()
            self.assertEqual(destination.read_bytes(), contents)

    def test_extracts_only_required_jsonl(self):
        with tempfile.TemporaryDirectory() as folder:
            directory = Path(folder)
            archive = directory / "data.zip"
            with zipfile.ZipFile(archive, "w") as zipped:
                zipped.writestr("dataset/instances.jsonl", '{"id":"1"}\n')
                zipped.writestr("dataset/truth.jsonl", '{"truthClass":"no-clickbait"}\n')
                zipped.writestr("dataset/unrelated.txt", "ignore")
            download_data.extract_jsonl(archive, directory)
            download_data.extract_jsonl(archive, directory)
            self.assertTrue((directory / "instances.jsonl").exists())
            self.assertTrue((directory / "truth.jsonl").exists())
            self.assertFalse((directory / "unrelated.txt").exists())

            (directory / "truth.jsonl").write_text("changed")
            with self.assertRaises(FileExistsError):
                download_data.extract_jsonl(archive, directory)


if __name__ == "__main__":
    unittest.main()
