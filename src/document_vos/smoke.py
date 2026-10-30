"""Real two-iteration CUDA check using synthetic inputs. Never a quality evaluation."""

import argparse
import json
from pathlib import Path

from document_vos.config import load
from document_vos.runs import Run
from document_vos.runtime import Runtime, configure


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--cutie-root", type=Path)
    parser.add_argument("--weights", type=Path)
    args = parser.parse_args(argv)
    cfg = load(args.config)
    if args.cutie_root:
        cfg["paths"]["cutie"] = str(args.cutie_root.resolve())
    cfg["training"]["num_iterations"] = 2
    configure(Runtime.parse(cfg["runtime"]))
    import numpy as np
    from PIL import Image

    from document_vos.cutie import check_checkout
    from document_vos.data import Frame, Sequence, save_manifest
    from document_vos.data.export import export_training
    from document_vos.data.split import split_manifest
    from document_vos.workflows import paired, weight

    root = check_checkout(Path(cfg["paths"]["cutie"]))
    initial = weight(cfg, "baseline", args.weights)
    run = Run(
        Path(cfg["paths"]["results"]),
        "cuda-smoke",
        cfg,
        {"scope": "synthetic execution smoke; NOT quality; NOT recipe reproduction"},
        output=args.output,
    )

    def prepare(output):
        output.mkdir()
        rows = []
        for n in range(5):
            frames = []
            for index in range(8):
                rgb = np.random.default_rng(42 + n * 8 + index).integers(
                    30, 80, (96, 128, 3), dtype=np.uint8
                )
                rgb[20:80, 25:105] = 220
                frame = output / "sources" / f"video-{n}" / f"{index:06d}.png"
                frame.parent.mkdir(parents=True, exist_ok=True)
                Image.fromarray(rgb).save(frame)
                frames.append(
                    Frame(
                        str(index),
                        str(frame.resolve()),
                        polygon=[[25, 20], [104, 20], [104, 79], [25, 79]],
                    )
                )
            rows.append(
                Sequence(
                    "midv2020", f"synthetic-{n}", f"doc-{n}", f"type-{n}", f"group-{n}", frames
                )
            )
        save_manifest(output / "manifest.jsonl", rows)
        split_manifest(output / "manifest.jsonl", output / "split")
        export_training(
            output / "split" / "train.jsonl",
            output / "split",
            output / "training-export",
            8,
            workers=1,
        )

    prepared = run.stage("prepare", prepare)
    print(json.dumps(paired(run, cfg, prepared, root, initial), ensure_ascii=False, indent=2))
