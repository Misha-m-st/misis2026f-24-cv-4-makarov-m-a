"""Read-only SAM diagnostics on all sampled frames of one extracted video."""

import argparse
import html
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from ..io import sha256  # noqa: E402
from .geometry import binary  # noqa: E402
from .sam import INSTANCE_SELECTION, REVISION, SAM  # noqa: E402

COLORS = [(66, 220, 110), (255, 115, 70), (75, 160, 255), (235, 95, 230), (255, 220, 70)]

VIEWER = """<!doctype html>
<html lang="ru"><meta charset="utf-8"><meta name="viewport" content="width=device-width">
<title>SAM 3 — просмотр масок</title>
<style>
body{margin:0;background:#151a22;color:#e8edf4;font:16px system-ui}main{max-width:1000px;margin:auto;padding:20px}
h1{font-size:22px}p{line-height:1.5}button,input{font:inherit}button{padding:8px 14px;margin:4px;background:#263349;color:white;border:1px solid #526078;border-radius:6px}
input{width:100%}img{display:block;max-width:100%;height:70vh;object-fit:contain;margin:14px auto;background:#090d12}a{color:#91bcff}
#status{min-height:24px}#legend span{display:inline-block;padding:5px 10px;margin:4px;border-radius:5px;color:#111}
</style><main><h1>__TITLE__ — SAM 3</h1>
<p>Видимые маски SAM до геометрической обработки. Каждая маска имеет свой цвет и confidence.
Пересекающиеся меньшие маски удалены. Показано не больше __MAX_MASKS__ масок на кадр,
выбранных по убыванию площади после порога confidence.
Разные цвета могут обозначать весь лист, его части или другой объект; номера масок между кадрами не являются track ID.
Показаны кадры с шагом __STEP__, исходные timestamps указаны под изображением.
Просмотр не меняет допуск видео в test set.</p>
<button id="prev">← Назад</button><button id="next">Вперёд →</button>
<button id="play">Проиграть</button><button id="toggle">Убрать наложение</button>
<input id="seek" type="range" min="0" max="__MAX__" value="0" step="1" aria-label="Кадр">
<div id="status"></div><div id="legend"></div><img id="frame" alt="Кадр видео и маски SAM">
<p><a id="original" target="_blank">Исходный кадр</a> · <a href="preview.json">Данные просмотра</a></p></main>
<script>
const frames=__FRAMES__;let i=0,overlay=true,timer=null;
const seek=document.getElementById('seek'),image=document.getElementById('frame');
function show(){const f=frames[i];seek.value=i;image.src=overlay?f.overlay:f.original;
 document.getElementById('status').textContent=`${i+1}/${frames.length} · кадр ${f.index} · ${f.time_s.toFixed(6)} с · ${f.error||('масок: '+f.instances.length)}`;
 document.getElementById('original').href=f.original;
 const legend=document.getElementById('legend');legend.replaceChildren();
 for(const m of f.instances){const tag=document.createElement('span');tag.style.background=`rgb(${m.color.join(',')})`;
 tag.textContent=`#${m.instance}: ${m.score.toFixed(3)}`;legend.append(tag);}}
function stop(){clearTimeout(timer);timer=null;document.getElementById('play').textContent='Проиграть';}
function advance(){if(i>=frames.length-1){stop();return;}const delay=Math.max(20,(frames[i+1].time_s-frames[i].time_s)*1000);
 timer=setTimeout(()=>{i++;show();advance();},delay);}
function move(delta){stop();i=Math.max(0,Math.min(frames.length-1,i+delta));show();}
document.getElementById('prev').onclick=()=>move(-1);document.getElementById('next').onclick=()=>move(1);
seek.oninput=()=>{stop();i=Number(seek.value);show();};
document.getElementById('toggle').onclick=()=>{overlay=!overlay;document.getElementById('toggle').textContent=overlay?'Убрать наложение':'Показать маски';show();};
document.getElementById('play').onclick=()=>{if(timer){stop();return;}if(i===frames.length-1)i=0;show();document.getElementById('play').textContent='Пауза';advance();};
document.addEventListener('keydown',e=>{if(e.key==='ArrowLeft'){e.preventDefault();move(-1);}if(e.key==='ArrowRight'){e.preventDefault();move(1);}});show();
</script></html>"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument(
        "--video", required=True, help="Source filename, absolute source path or manifest video ID"
    )
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--step", type=int, default=10)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    parser.add_argument("--prompt", default="sheet of paper")
    parser.add_argument("--min-score", type=float, default=0.5)
    parser.add_argument(
        "--max-masks", type=int, default=1, help="Keep at most N masks, largest first (default: 1)"
    )
    args = parser.parse_args()
    if args.step < 1:
        parser.error("step must be positive")
    if args.max_masks < 1:
        parser.error("--max-masks must be positive")
    rows = [json.loads(line) for line in (args.dataset / "manifest.jsonl").read_text().splitlines()]
    matches = []
    for row in rows:
        source = row.get("provenance", {}).get("source_video", {}).get("path", "")
        if args.video in {row["video"], source, Path(source).name}:
            matches.append(row)
    if len(matches) != 1:
        parser.error("video must match exactly one dataset sequence")
    row = matches[0]
    output = args.output.resolve()
    if output.exists():
        parser.error("output already exists; choose a new preview directory")
    frames = row["frames"]
    if not frames:
        parser.error("sequence has no extracted frames")
    model = SAM(
        args.weights,
        device=args.device,
        prompt=args.prompt,
        min_score=args.min_score,
        max_masks=args.max_masks,
    )
    output.mkdir(parents=True)
    (output / "overlays").mkdir()
    (output / "originals").mkdir()
    (output / "raw_masks").mkdir()
    font_path = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")
    font = (
        ImageFont.truetype(str(font_path), 25) if font_path.exists() else ImageFont.load_default()
    )
    records = []
    for index in range(0, len(frames), args.step):
        frame = frames[index]
        stem = f"{index:06d}"
        with Image.open(frame["path"]) as image:
            rgb = np.asarray(image.convert("RGB"))
        original = f"originals/{stem}.png"
        Image.fromarray(rgb).save(output / original)
        record = {
            "index": index,
            "time_s": frame["time_s"],
            "original": original,
            "overlay": f"overlays/{stem}.png",
            "instances": [],
        }
        rendered = rgb.copy()
        try:
            masks, scores = model.predict_instances(rgb)
            if masks.ndim == 4 and masks.shape[1] == 1:
                masks = masks[:, 0]
            if masks.ndim != 3 or masks.shape[1:] != rgb.shape[:2] or scores.shape != (len(masks),):
                raise ValueError("invalid raw SAM output shape")
            if not np.isfinite(scores).all() or ((scores < 0) | (scores > 1)).any():
                raise ValueError("invalid raw SAM confidence")
            for number, (mask, score) in enumerate(zip(masks, scores)):
                mask = binary(mask, rgb.shape[:2])
                color = COLORS[number % len(COLORS)]
                selected = mask.astype(bool)
                rendered[selected] = (0.7 * rendered[selected] + 0.3 * np.array(color)).astype(
                    np.uint8
                )
                contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(rendered, contours, -1, color, 3)
                directory = output / "raw_masks" / stem
                directory.mkdir(exist_ok=True)
                mask_path = directory / f"{number:03d}.png"
                Image.fromarray(mask).save(mask_path)
                record["instances"].append(
                    {
                        "instance": number,
                        "score": float(score),
                        "color": list(color),
                        "mask": str(mask_path.relative_to(output)),
                        "mask_sha256": sha256(mask_path),
                        "area": int(mask.sum()),
                        "model_instance": model.last_instance_selection["original_indices"][number],
                    }
                )
        except Exception as exc:
            record["error"] = f"{type(exc).__name__}: {exc}"
            record["instances"] = []
            rendered = rgb.copy()
        image = Image.fromarray(rendered)
        draw = ImageDraw.Draw(image)
        draw.rectangle((8, 8, min(rgb.shape[1] - 1, 810), 50), fill=(16, 20, 28))
        draw.text(
            (16, 14),
            f"SAM 3 RAW | frame {index} | {frame['time_s']:.6f} s",
            font=font,
            fill="white",
        )
        image.save(output / record["overlay"])
        records.append(record)
        print(
            f"frame {index}/{len(frames) - 1}: {record.get('error', str(len(record['instances'])) + ' masks')}",
            flush=True,
        )
    report = {
        "kind": "raw SAM visible-mask diagnostics; not dataset admission",
        "model": "SAM 3 image",
        "revision": REVISION,
        "weights_sha256": sha256(args.weights),
        "source_video": row["provenance"]["source_video"],
        "prompt": args.prompt,
        "min_score": args.min_score,
        "max_masks": args.max_masks,
        "instance_selection": INSTANCE_SELECTION,
        "step": args.step,
        "device": args.device,
        "frames": records,
    }
    (output / "preview.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    page = VIEWER.replace("__TITLE__", html.escape(Path(report["source_video"]["path"]).name))
    page = page.replace("__STEP__", str(args.step)).replace("__MAX__", str(len(records) - 1))
    page = page.replace("__MAX_MASKS__", str(args.max_masks))
    page = page.replace("__FRAMES__", json.dumps(records, ensure_ascii=False).replace("</", "<\\/"))
    (output / "index.html").write_text(page)
    print(f"Open: {output / 'index.html'}", flush=True)


if __name__ == "__main__":
    main()
