"""Кеш явной полной проверки; обычные команды не читают содержимое кадров."""

import sqlite3
from pathlib import Path

from document_vos.io import PipelineError, digest

# Менять при изменении правил полной проверки, геометрии или бинарных масок.
VALIDATION_VERSION = 1


def file_signature(path: Path) -> list:
    """ctime замечает и замену файла с сохранёнными размером и mtime."""
    resolved = path.resolve()
    stat = resolved.stat()
    return [
        str(resolved),
        stat.st_dev,
        stat.st_ino,
        stat.st_size,
        stat.st_mtime_ns,
        stat.st_ctime_ns,
    ]


def frame_signature(image: Path, mask: Path | None, polygon: list | None) -> str:
    return digest(
        [
            VALIDATION_VERSION,
            file_signature(image),
            file_signature(mask) if mask else None,
            polygon,
        ]
    )


class ValidationCache:
    """SQLite хранит только успешно проверенные кадры, включая хеш и пустоту GT."""

    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            self.connection = sqlite3.connect(path, timeout=30)
            self.connection.execute(
                "CREATE TABLE IF NOT EXISTS frames "
                "(signature TEXT PRIMARY KEY, image_hash TEXT NOT NULL, empty_gt INTEGER)"
            )
        except sqlite3.Error as error:
            if hasattr(self, "connection"):
                self.connection.close()
            raise PipelineError(f"Не удалось открыть кеш проверки {path}: {error}") from error
        self.pending = 0

    def get(self, signature: str) -> tuple[str, bool | None] | None:
        row = self.connection.execute(
            "SELECT image_hash, empty_gt FROM frames WHERE signature = ?", (signature,)
        ).fetchone()
        if row is None:
            return None
        return row[0], bool(row[1]) if row[1] is not None else None

    def put(self, signature: str, image_hash: str, empty_gt: bool | None) -> None:
        self.connection.execute(
            "INSERT OR REPLACE INTO frames VALUES (?, ?, ?)",
            (signature, image_hash, empty_gt),
        )
        self.pending += 1
        # Уже проверенные кадры сохраняются и при прерывании большого прохода.
        if self.pending >= 100:
            self.connection.commit()
            self.pending = 0

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        try:
            self.connection.commit()
        except sqlite3.Error as error:
            raise PipelineError(f"Не удалось сохранить кеш проверки: {error}") from error
        finally:
            self.connection.close()
