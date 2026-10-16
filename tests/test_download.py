import hashlib
import io
import json
import shutil
import subprocess
import sys
import tarfile
import threading
import zipfile
from dataclasses import replace
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from document_vos.data.download import (
    DATASETS,
    MARKER,
    DownloadError,
    dataset_archives,
    download_archive,
    extract_archive,
    verify_archive,
)


def tar_bytes(files):
    stream = io.BytesIO()
    with tarfile.open(fileobj=stream, mode="w") as tar:
        for name, content in files.items():
            member = tarfile.TarInfo(name)
            member.size = len(content)
            tar.addfile(member, io.BytesIO(content))
    return stream.getvalue()


def sample_dataset(payload, **kwargs):
    return replace(
        DATASETS["midv2020"], size=len(payload), checksum=hashlib.md5(payload).hexdigest(), **kwargs
    )


def test_resume_real_curl_and_skip_verified_archive(tmp_path):
    if not shutil.which("curl"):
        pytest.skip("curl is not installed")
    payload = tar_bytes(
        {"images/doc/00/000001.jpg": b"synthetic frame", "annotations/doc/00.json": b"{}"}
    )
    requests = []

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            header = self.headers.get("Range")
            requests.append(header)
            offset = int(header.removeprefix("bytes=").removesuffix("-")) if header else 0
            self.send_response(206 if header else 200)
            if header:
                self.send_header(
                    "Content-Range", f"bytes {offset}-{len(payload) - 1}/{len(payload)}"
                )
            self.send_header("Content-Length", str(len(payload) - offset))
            self.end_headers()
            self.wfile.write(payload[offset:])

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        dataset = sample_dataset(payload, url=f"http://127.0.0.1:{server.server_port}/clips.tar")
        archives = tmp_path / "archives"
        archives.mkdir()
        # Имитируем прерванную прежнюю ручную загрузку, ещё без суффикса .part.
        (archives / dataset.archive).write_bytes(payload[:137])
        archive, fingerprint = download_archive(dataset, archives)
        assert requests == ["bytes=137-"]
        assert archive.read_bytes() == payload
        assert fingerprint == hashlib.sha256(payload).hexdigest()
        assert not archive.with_name(archive.name + ".part").exists()
        assert download_archive(dataset, archives) == (archive, fingerprint)
        assert len(requests) == 1  # Повторный запуск не делает HTTP-запроса.
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


def test_corrupt_archive_preserved_and_never_downloaded(tmp_path, monkeypatch):
    payload = b"correct contents"
    dataset = sample_dataset(payload)
    archive = tmp_path / dataset.archive
    archive.write_bytes(b"damaged contents")
    assert archive.stat().st_size == len(payload)
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: pytest.fail("Must not download"))
    with pytest.raises(DownloadError, match="сумма НЕ совпадает"):
        download_archive(dataset, tmp_path)
    assert archive.read_bytes() == b"damaged contents"


def test_failed_transfer_keeps_partial_file(tmp_path, monkeypatch):
    dataset = sample_dataset(b"abcdef")
    partial = tmp_path / (dataset.archive + ".part")

    def interrupted(command, **kwargs):
        partial.write_bytes(b"abc")
        return subprocess.CompletedProcess(command, 18)

    monkeypatch.setattr(subprocess, "run", interrupted)
    with pytest.raises(DownloadError, match="Частичный файл сохранён"):
        download_archive(dataset, tmp_path)
    assert partial.read_bytes() == b"abc"
    assert not (tmp_path / dataset.archive).exists()


def test_extract_normalizes_wrapper_preserves_extras_and_skips_repeat(tmp_path):
    payload = tar_bytes(
        {
            "clips/images/doc/00/000001.jpg": b"frame",
            "clips/annotations/doc/00.json": b"{}",
            "LICENSE": b"license",
        }
    )
    dataset = sample_dataset(payload)
    archive = tmp_path / dataset.archive
    archive.write_bytes(payload)
    data_dir = tmp_path / "data"
    (data_dir / dataset.target).mkdir(parents=True)  # Каталог, созданный прежним mkdir.
    fingerprint = hashlib.sha256(payload).hexdigest()
    target = extract_archive(dataset, archive, data_dir, fingerprint)
    assert (target / "images/doc/00/000001.jpg").read_bytes() == b"frame"
    assert (target / ".archive-extras/LICENSE").read_bytes() == b"license"
    assert json.loads((target / MARKER).read_text())["archive_sha256"] == fingerprint
    assert extract_archive(dataset, archive, data_dir, fingerprint) == target


def test_existing_user_data_not_overwritten(tmp_path):
    dataset = sample_dataset(b"not even a tar")
    target = tmp_path / dataset.target
    target.mkdir(parents=True)
    existing = target / "keep.txt"
    existing.write_text("user data")
    with pytest.raises(DownloadError, match="не будут перезаписаны"):
        extract_archive(dataset, tmp_path / "absent.tar", tmp_path, "hash")
    assert existing.read_text() == "user data"


@pytest.mark.parametrize("name", ["../outside", "/tmp/outside"])
def test_rejects_unsafe_archive_paths(tmp_path, name):
    payload = tar_bytes({name: b"must not be written"})
    dataset = sample_dataset(payload)
    archive = tmp_path / dataset.archive
    archive.write_bytes(payload)
    with pytest.raises(DownloadError, match="Небезопасный"):
        extract_archive(dataset, archive, tmp_path / "data", "hash")
    assert not (tmp_path / "data" / dataset.target).exists()


def test_plan_works_from_any_directory_and_writes_nothing(tmp_path):
    result = subprocess.run(
        [sys.executable, "-m", "document_vos", "download", "--plan", "--dataset", "midv2020"],
        cwd=tmp_path,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert str(tmp_path / "data/archives/midv2020-clips.tar") in result.stdout
    assert list(tmp_path.iterdir()) == []
    data_dir = tmp_path / "other-disk"
    result = subprocess.run(
        [sys.executable, "-m", "document_vos", "download", "--plan", "--archives", str(data_dir)],
        cwd=tmp_path,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0
    assert not data_dir.exists()


def test_midv_zip_layout_and_repeat(tmp_path):
    dataset = dataset_archives("midv2019")[0]
    archive = tmp_path / dataset.archive
    with zipfile.ZipFile(archive, "w") as zipped:
        zipped.writestr("01_alb_id/images/clip/000001.jpg", b"frame")
        zipped.writestr("01_alb_id/ground_truth/clip/000001.json", b'{"quad": []}')
        zipped.writestr("README.txt", b"license information")
    target = extract_archive(dataset, archive, tmp_path / "data", "hash")
    assert target == tmp_path / "data/raw/midv2019/01_alb_id"
    assert (target / "images/clip/000001.jpg").read_bytes() == b"frame"
    assert (target / ".archive-extras/README.txt").is_file()
    assert extract_archive(dataset, archive, tmp_path / "data", "hash") == target


@pytest.mark.parametrize("name", ["../outside", "/tmp/outside", "..\\outside"])
def test_rejects_unsafe_zip_paths(tmp_path, name):
    dataset = dataset_archives("midv2019")[0]
    archive = tmp_path / dataset.archive
    with zipfile.ZipFile(archive, "w") as zipped:
        zipped.writestr(name, b"must not be written")
    with pytest.raises(DownloadError, match="Небезопасный"):
        extract_archive(dataset, archive, tmp_path / "data", "hash")
    assert not (tmp_path / "data" / dataset.target).exists()


def test_holo_separate_archives_share_adapter_root(tmp_path):
    for dataset in dataset_archives("midvholo"):
        part = dataset.root_directory
        payload = tar_bytes({f"{part}/example/file.json": b"{}"})
        dataset = replace(dataset, size=len(payload))
        archive = tmp_path / dataset.archive
        archive.write_bytes(payload)
        fingerprint = verify_archive(archive, dataset)
        target = extract_archive(dataset, archive, tmp_path / "data", fingerprint)
        receipt = json.loads((target / MARKER).read_text())
        assert receipt["source_checksum_verified"] is False
        assert receipt["archive_sha256"] == hashlib.sha256(payload).hexdigest()
        assert (target / "example/file.json").is_file()
        assert extract_archive(dataset, archive, tmp_path / "data", fingerprint) == target
    root = tmp_path / "data/raw/midvholo"
    assert all((root / part).is_dir() for part in ["images", "markup", "metadata"])


def test_extension_plan_does_not_download(tmp_path):
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "document_vos",
            "download",
            "--plan",
            "--dataset",
            "midv2019",
            "midvholo",
            "--archives",
            str(tmp_path / "data"),
        ],
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    plan = json.loads(result.stdout)
    assert plan["download_bytes"] == 53427326141
    assert len(plan["archives"]) == 53
    assert plan["archives"][0]["dataset"] == "midv2019"
    assert plan["archives"][-1]["path"].endswith("holo-metadata.tar")
    assert list(tmp_path.iterdir()) == []
