"""Разбиение компонент связанных документов и шаблонов, включая разные датасеты."""

import random
from pathlib import Path

from document_vos.data import Sequence, load_manifest, save_manifest, validate_data
from document_vos.io import PipelineError, digest, new_directory, read_json, sha256, write_json


def components(sequences: list[Sequence]) -> list[list[Sequence]]:
    parent = list(range(len(sequences)))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    owners = {}
    for i, seq in enumerate(sequences):
        for field in ["document", "template", "group"]:
            token = (field, getattr(seq, field))
            if token in owners:
                parent[find(i)] = find(owners[token])
            owners[token] = i
    groups = {}
    for i, seq in enumerate(sequences):
        groups.setdefault(find(i), []).append(seq)
    return sorted(groups.values(), key=lambda group: min(s.key for s in group))


def assert_disjoint(train: list[Sequence], val: list[Sequence]) -> None:
    for field in ["key", "document", "template", "group"]:
        overlap = {getattr(s, field) for s in train} & {getattr(s, field) for s in val}
        if overlap:
            raise PipelineError(f"Утечка train/validation по {field}: {sorted(overlap)[:5]}")
    overlap = {f.path for s in train for f in s.frames} & {f.path for s in val for f in s.frames}
    if overlap:
        raise PipelineError("Утечка файлов кадров train/validation")


def split_manifest(
    manifest: Path,
    output: Path,
    seed: int = 42,
    val_fraction: float = 0.2,
    previous: Path | None = None,
) -> dict:
    if not 0 < val_fraction < 1:
        raise PipelineError("val_fraction должна быть между 0 и 1")
    sequences = load_manifest(manifest)
    validate_data(sequences)
    if any(s.dataset == "custom" for s in sequences):
        raise PipelineError(
            "Независимые custom-видео оцениваются отдельно, не входят в train/validation"
        )
    groups = components(sequences)
    if len(groups) < 2:
        raise PipelineError(
            "После объединения связей осталось менее двух групп; безопасное разбиение невозможно"
        )
    old = {}
    previous_hash = None
    if previous:
        meta = read_json(previous / "split.json")
        for label in ["train", "validation"]:
            path = previous / f"{label}.jsonl"
            if sha256(path) != meta[f"{label}_sha256"]:
                raise PipelineError("Предыдущее разбиение изменено")
            for s in load_manifest(path):
                old[s.key] = (label, digest(s.to_dict()))
        if old.keys() - {s.key for s in sequences}:
            raise PipelineError("Расширение потеряло видео предыдущей версии")
        for s in sequences:
            if s.key in old and digest(s.to_dict()) != old[s.key][1]:
                raise PipelineError(f"Расширение изменило существующее видео {s.key}")
        previous_hash = sha256(previous / "split.json")
    assigned = {"train": [], "validation": []}
    free = []
    for group in groups:
        labels = {old[s.key][0] for s in group if s.key in old}
        if len(labels) > 1:
            raise PipelineError(
                "Новая связь объединила старые train и validation; создайте отдельный эксперимент"
            )
        if labels:
            assigned[labels.pop()].extend(group)
        else:
            free.append(group)
    random.Random(seed).shuffle(free)
    target = max(1, round(len(sequences) * val_fraction))
    for group in free:
        current = len(assigned["validation"])
        label = (
            "validation" if abs(current + len(group) - target) <= abs(current - target) else "train"
        )
        assigned[label].extend(group)
    if not previous:
        for label, other in [("validation", "train"), ("train", "validation")]:
            if not assigned[label]:
                candidates = [
                    g for g in groups if {s.key for s in g} <= {s.key for s in assigned[other]}
                ]
                moved = min(candidates, key=len)
                assigned[label].extend(moved)
                assigned[other] = [s for s in assigned[other] if s not in moved]
    if not all(assigned.values()):
        raise PipelineError("Разбиение содержит пустую часть")
    assert_disjoint(assigned["train"], assigned["validation"])
    output = new_directory(output)
    for label, items in assigned.items():
        save_manifest(output / f"{label}.jsonl", sorted(items, key=lambda s: s.key))
    meta = {
        "seed": seed,
        "target_val_fraction": val_fraction,
        "actual_val_fraction": len(assigned["validation"]) / len(sequences),
        "group_count": len(groups),
        "manifest_sha256": sha256(manifest),
        "previous_split_sha256": previous_hash,
        "train_sha256": sha256(output / "train.jsonl"),
        "validation_sha256": sha256(output / "validation.jsonl"),
        "counts": {k: len(v) for k, v in assigned.items()},
        "policy": "connected document, template, group; approximate 80/20 by video",
    }
    write_json(output / "split.json", meta)
    return meta


def verify_training_split(manifest: Path, split_dir: Path) -> None:
    meta = read_json(split_dir / "split.json")
    if sha256(manifest) != meta["train_sha256"]:
        raise PipelineError("Экспорт разрешён только из train-манифеста данного разбиения")
    if sha256(split_dir / "validation.jsonl") != meta["validation_sha256"]:
        raise PipelineError("Validation-манифест изменён после разбиения")
    assert_disjoint(load_manifest(manifest), load_manifest(split_dir / "validation.jsonl"))
