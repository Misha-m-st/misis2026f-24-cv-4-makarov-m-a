import hashlib
import io
import tarfile
import zipfile
from dataclasses import replace

import numpy as np
import pytest
from PIL import Image

from document_vos.archive_prepare import RECEIPT, normalize_archive
from document_vos.download import Dataset
from document_vos.io import PipelineError, read_json
from document_vos.runtime import Runtime


def archive_fixture(tmp_path):
    archive = tmp_path / "sample.zip"
    pixels = np.arange(24 * 32 * 3, dtype=np.uint8).reshape(24, 32, 3)
    buffer = io.BytesIO()
    Image.fromarray(pixels).save(buffer, format="TIFF")
    with zipfile.ZipFile(archive, "w") as stream:
        stream.writestr("01_alb_id/images/CA/CA01_01.tif", buffer.getvalue())
        stream.writestr(
            "01_alb_id/ground_truth/CA/CA01_01.json", '{"quad": [[0,0],[1,0],[1,1],[0,1]]}'
        )
        stream.writestr("01_alb_id/videos/CA01.mp4", b"retain-original-video")
    spec = Dataset(
        "midv500",
        "fixture",
        archive.name,
        archive.stat().st_size,
        "md5",
        hashlib.md5(archive.read_bytes()).hexdigest(),
        "raw/midv500/01_alb_id",
    )
    return archive, spec, pixels


def test_lossless_parallel_normalization_and_opt_in_deletion(tmp_path):
    archive, spec, pixels = archive_fixture(tmp_path)
    outputs = [tmp_path / "serial", tmp_path / "parallel"]
    for workers, target in zip([1, 2], outputs):
        result = normalize_archive(archive, spec, target, Runtime(workers=workers))
        assert result["status"] == "VERIFIED" and archive.exists()
        with Image.open(target / "images/CA/CA01_01.png") as image:
            np.testing.assert_array_equal(pixels, np.asarray(image))
        assert (target / "videos/CA01.mp4").read_bytes() == b"retain-original-video"
    assert read_json(outputs[0] / RECEIPT)["files"] == read_json(outputs[1] / RECEIPT)["files"]
    assert normalize_archive(
        archive, spec, outputs[1], Runtime(workers=1), delete_verified_zip=True
    )["deleted"]
    assert not archive.exists()
    assert normalize_archive(archive, spec, outputs[1], Runtime(workers=1))["status"] == "VERIFIED"


def test_modified_output_blocks_deletion(tmp_path):
    archive, spec, _ = archive_fixture(tmp_path)
    target = tmp_path / "output"
    normalize_archive(archive, spec, target, Runtime(workers=1))
    (target / "videos/CA01.mp4").write_bytes(b"corrupt")
    with pytest.raises(PipelineError, match="изменён"):
        normalize_archive(archive, spec, target, Runtime(workers=1), delete_verified_zip=True)
    assert archive.exists()


def test_bad_checksum_and_disk_reserve_never_delete(tmp_path):
    archive, spec, _ = archive_fixture(tmp_path)
    with pytest.raises(Exception, match="сумма НЕ совпадает"):
        normalize_archive(
            archive,
            replace(spec, checksum="0" * 32),
            tmp_path / "bad",
            Runtime(workers=1),
            delete_verified_zip=True,
        )
    with pytest.raises(PipelineError, match="Недостаточно места"):
        normalize_archive(
            archive,
            spec,
            tmp_path / "disk",
            Runtime(workers=1, min_free_bytes=10**18),
            delete_verified_zip=True,
        )
    assert archive.exists()


@pytest.mark.parametrize("names", [["../escape"], ["images/a.tif", "images/a.png"]])
def test_traversal_and_conversion_collision_fail_closed(tmp_path, names):
    archive = tmp_path / "sample.zip"
    with zipfile.ZipFile(archive, "w") as stream:
        for name in names:
            stream.writestr(name, b"x")
    spec = Dataset(
        "midv2019",
        "fixture",
        archive.name,
        archive.stat().st_size,
        "sha256",
        hashlib.sha256(archive.read_bytes()).hexdigest(),
        "raw/midv2019/doc",
    )
    with pytest.raises(PipelineError):
        normalize_archive(
            archive, spec, tmp_path / "out", Runtime(workers=1), delete_verified_zip=True
        )
    assert archive.exists()


def test_resume_after_worker_failure(tmp_path, monkeypatch):
    import document_vos.archive_prepare as module

    archive, spec, _ = archive_fixture(tmp_path)
    original = module._materialize
    calls = 0

    def interrupted(task):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("interrupted")
        return original(task)

    monkeypatch.setattr(module, "_materialize", interrupted)
    with pytest.raises(RuntimeError, match="interrupted"):
        normalize_archive(
            archive, spec, tmp_path / "out", Runtime(workers=1), delete_verified_zip=True
        )
    assert archive.exists()
    monkeypatch.setattr(module, "_materialize", original)
    result = normalize_archive(
        archive, spec, tmp_path / "out", Runtime(workers=1), delete_verified_zip=True
    )
    assert result["deleted"] and result["files"] == 3


def test_tar_offsets_copy_every_file_and_never_delete_by_zip_flag(tmp_path):
    archive = tmp_path / "holo-markup.tar"
    with tarfile.open(archive, "w") as stream:
        for name, data in [
            ("markup/origins/doc/frame.jpg.json", b'{"quad":[]}'),
            ("markup/README", b"preserve this too"),
        ]:
            member = tarfile.TarInfo(name)
            member.size = len(data)
            stream.addfile(member, io.BytesIO(data))
    spec = Dataset(
        "midvholo",
        "fixture",
        archive.name,
        archive.stat().st_size,
        "sha256",
        hashlib.sha256(archive.read_bytes()).hexdigest(),
        "raw/midvholo/markup",
        root_directory="markup",
    )
    target = tmp_path / "markup"
    result = normalize_archive(archive, spec, target, Runtime(workers=2), delete_verified_zip=True)
    assert result["files"] == 2 and not result["deleted"] and archive.exists()
    assert (target / "README").read_bytes() == b"preserve this too"
