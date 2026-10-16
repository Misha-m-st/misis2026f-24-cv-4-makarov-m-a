"""Bounded, resumable HTTP assets. No fallback to dataset archives."""

import re
import urllib.error
import urllib.request
from pathlib import Path

from document_vos.io import PipelineError, sha256
from document_vos.locking import exclusive_lock


def check_asset(asset: dict) -> None:
    if (
        type(asset.get("bytes")) is not int
        or asset["bytes"] < 1
        or not isinstance(asset.get("sha256"), str)
        or not re.fullmatch("[0-9a-f]{64}", asset["sha256"])
        or not isinstance(asset.get("url"), str)
        or not asset["url"].startswith(("https://", "http://", "file://"))
    ):
        raise PipelineError("Каталог: необходимы URL, точный размер и SHA-256")


def fetch(asset: dict, target: Path) -> Path:
    check_asset(asset)
    target.parent.mkdir(parents=True, exist_ok=True)
    with exclusive_lock(target.with_name(target.name + ".lock")):
        if target.exists():
            if target.stat().st_size != asset["bytes"] or sha256(target) != asset["sha256"]:
                raise PipelineError(f"Повреждён файл: {target}; выберите новый каталог")
            return target
        partial = target.with_name(target.name + ".part")
        offset = partial.stat().st_size if partial.exists() else 0
        if offset > asset["bytes"]:
            raise PipelineError(f"Частичный файл больше ожидаемого: {partial}")
        if offset < asset["bytes"]:
            headers = {"Range": f"bytes={offset}-"} if offset else {}
            request = urllib.request.Request(asset["url"], headers=headers)
            try:
                response = urllib.request.urlopen(request, timeout=60)
            except (urllib.error.URLError, OSError) as exc:
                raise PipelineError(
                    f"Не удалось скачать {asset['url']}: {exc}. "
                    "Release ещё не опубликован? .part сохранён."
                ) from exc
            with response:
                # file:// is useful for an offline Release dress rehearsal.
                if offset and asset["url"].startswith("file://"):
                    response.read(offset)
                elif offset:
                    expected = f"bytes {offset}-"
                    if response.status != 206 or not response.headers.get(
                        "Content-Range", ""
                    ).startswith(expected):
                        raise PipelineError("Сервер не подтвердил Range; частичный файл сохранён")
                import shutil

                if shutil.disk_usage(target.parent).free < asset["bytes"] - offset:
                    raise PipelineError("Недостаточно места для загрузки")
                with partial.open("ab" if offset else "xb") as stream:
                    total = offset
                    while block := response.read(min(1024 * 1024, asset["bytes"] - total + 1)):
                        if total + len(block) > asset["bytes"]:
                            raise PipelineError("Ответ больше размера из каталога")
                        stream.write(block)
                        total += len(block)
        if partial.stat().st_size != asset["bytes"] or sha256(partial) != asset["sha256"]:
            raise PipelineError(f"Неверный размер/SHA-256: {partial}; файл сохранён")
        partial.rename(target)
    return target
