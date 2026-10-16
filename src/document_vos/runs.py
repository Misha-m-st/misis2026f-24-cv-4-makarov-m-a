"""Immutable run artifacts and content-verified, explicitly resumed stages."""

import html
import traceback
import uuid
from contextlib import redirect_stderr, redirect_stdout
from datetime import datetime, timezone
from pathlib import Path

import yaml

from document_vos.io import (
    PipelineError,
    atomic_json,
    digest,
    new_directory,
    read_json,
    sha256,
    write_json,
)


def inventory(directory: Path) -> dict:
    return {
        str(p.relative_to(directory)): sha256(p)
        for p in sorted(directory.rglob("*"))
        if p.is_file()
        and not p.name.endswith(".lock")
        and "__pycache__" not in p.parts
        and not p.name.startswith(".validation-cache.sqlite3")
    }


def source_signature() -> str:
    root = Path(__file__).parent
    return digest(
        {
            str(p.relative_to(root)): sha256(p)
            for p in sorted(root.rglob("*"))
            if p.suffix in {".py", ".json", ".csv", ".yaml"}
        }
    )


def verify_inventory(directory: Path, hashes: dict, *, exact: bool = True) -> None:
    if exact and inventory(directory) != hashes:
        raise PipelineError(f"Артефакты изменились: {directory}")
    for name, expected in hashes.items():
        path = directory / name
        if not path.is_file() or sha256(path) != expected:
            raise PipelineError(f"Артефакт изменился: {path}")


def index(root: Path) -> None:
    root.mkdir(parents=True, exist_ok=True)
    items = []
    for path in sorted(root.glob("*/status.json"), reverse=True):
        try:
            state = read_json(path)
        except (OSError, ValueError):
            continue
        name = path.parent.name
        items.append(
            f'<li><a href="{html.escape(name)}/index.html">{html.escape(name)}</a> '
            f"{html.escape(state.get('status', 'UNKNOWN'))}</li>"
        )
    temporary = root / f".index-{uuid.uuid4().hex}.tmp"
    temporary.write_text(
        '<!doctype html><meta charset="utf-8"><title>document-vos</title>'
        "<h1>Результаты</h1><ul>" + "".join(items) + "</ul>",
        encoding="utf-8",
    )
    temporary.replace(root / "index.html")


class Run:
    def __init__(
        self,
        root: Path,
        task: str,
        cfg: dict,
        inputs: dict,
        *,
        output: Path | None = None,
        resume: Path | None = None,
    ):
        self.root = root
        self.signature = {"config": cfg, "inputs": inputs, "source_sha256": source_signature()}
        if resume:
            if output:
                raise PipelineError("--resume и --output несовместимы")
            self.output = resume.resolve()
            self.state = read_json(self.output / "status.json")
            if self.state["signature"] != self.signature:
                raise PipelineError(
                    "Входы, конфигурация или исходники изменились; нужен новый запуск"
                )
            for stage in self.state["stages"].values():
                if stage["status"] == "COMPLETE":
                    verify_inventory(self.output / stage["directory"], stage["outputs"])
                    log = self.output / stage["log"]
                    if not log.is_file() or sha256(log) != stage["log_sha256"]:
                        raise PipelineError(f"Журнал этапа изменился: {log}")
        else:
            name = f"{datetime.now(timezone.utc):%Y%m%d-%H%M%S}-{task}-{uuid.uuid4().hex[:8]}"
            self.output = new_directory(output or root / name)
            self.state = {
                "schema": "document-vos/run-v1",
                "task": task,
                "status": "RUNNING",
                "signature": self.signature,
                "stages": {},
            }
            (self.output / "config.yaml").write_text(
                yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False), encoding="utf-8"
            )
            write_json(self.output / "provenance.json", inputs)
        self.state["status"] = "RUNNING"
        self.save()

    def save(self):
        atomic_json(self.output / "status.json", self.state)
        text = f"# {self.state['task']}\n\nСтатус: {self.state['status']}\n\n"
        for name, stage in self.state["stages"].items():
            text += f"- {name}: {stage['status']} ({stage['directory']})\n"
        (self.output / "summary.md").write_text(text, encoding="utf-8")
        links = "".join(
            f'<li><a href="{html.escape(str(p.relative_to(self.output)))}">'
            f"{html.escape(str(p.relative_to(self.output)))}</a></li>"
            for p in sorted(self.output.rglob("*.mp4"))
        )
        (self.output / "index.html").write_text(
            '<!doctype html><meta charset="utf-8"><title>document-vos</title><pre>'
            + html.escape(text)
            + "</pre><ul>"
            + links
            + "</ul>",
            encoding="utf-8",
        )
        index(self.root)

    def stage(self, name: str, action, *, live: bool = False) -> Path:
        previous = self.state["stages"].get(name)
        if previous and previous["status"] == "COMPLETE":
            path = self.output / previous["directory"]
            verify_inventory(path, previous["outputs"])
            return path
        attempt = previous.get("attempt", 0) + 1 if previous else 1
        path = self.output / f"{name}-{attempt}"
        entry = {"status": "RUNNING", "directory": path.name, "attempt": attempt}
        if previous:
            entry["history"] = previous.get("history", []) + [
                {k: v for k, v in previous.items() if k != "history"}
            ]
        self.state["stages"][name] = entry
        self.save()
        print(f"[{name}] start", flush=True)
        try:
            log = self.output / f"{name}-{attempt}.log"
            with log.open("x", encoding="utf-8") as stream:
                if live:
                    action(path)
                else:
                    with redirect_stdout(stream), redirect_stderr(stream):
                        action(path)
            if not path.is_dir() or not any(path.iterdir()):
                raise PipelineError(f"Этап {name} не создал артефактов")
            entry.update(
                status="COMPLETE", outputs=inventory(path), log=log.name, log_sha256=sha256(log)
            )
        except BaseException as exc:
            entry.update(status="FAILED", error=f"{type(exc).__name__}: {exc}")
            self.state["status"] = "FAILED"
            (self.output / f"{name}-{attempt}.error.log").write_text(
                traceback.format_exc(), encoding="utf-8"
            )
            self.save()
            raise
        self.save()
        print(f"[{name}] COMPLETE", flush=True)
        return path

    def finish(self, *, status: str = "COMPLETE") -> dict:
        if not self.state["stages"] or any(
            s["status"] != "COMPLETE" for s in self.state["stages"].values()
        ):
            raise PipelineError("Нельзя завершить pipeline с незавершёнными этапами")
        self.state["status"] = status
        self.save()
        videos = [str(p) for p in sorted(self.output.rglob("*.mp4"))]
        return {
            "status": status,
            "output": str(self.output),
            "videos": videos,
            "summary": str(self.output / "summary.md"),
            "index": str(self.output / "index.html"),
        }
