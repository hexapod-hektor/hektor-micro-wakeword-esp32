"""Прогнати модель по кліпах і порівняти групи: оригінали з диска проти перезаписів через вуха робота.

    cd record && uv run score.py models/hektor.onnx [--test-only]

Для кожного кліпу — максимальний score моделі. Групуємо за set × source і показуємо медіану,
частку кліпів вище порогу 0.5 (для positive це recall, для negative — хибні спрацьовування).
Різниця між source=cartesia/elevenlabs/gemini і source=robot і є «призма мікрофона».
--test-only — лише відкладені голоси зі split.json (чесна оцінка, а не пам'ять моделі).
"""
import argparse
import collections
import json
import statistics
import sys
from pathlib import Path

from openwakeword.model import Model

WAKE = Path(__file__).resolve().parents[1] / "data"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--test-only", action="store_true")
    ap.add_argument("--limit", type=int, default=0, help="макс. кліпів на групу (0 = всі)")
    a = ap.parse_args()

    files = json.loads((WAKE / "manifest.json").read_text())["files"]
    test_voices = set()
    if a.test_only:
        split = json.loads((WAKE.parent / "train" / "dataset" / "split.json").read_text())
        test_voices = {v for vs in split["test_voices"].values() for v in vs}
    m = Model(wakeword_models=[a.model], inference_framework="onnx")
    key = Path(a.model).stem

    groups = collections.defaultdict(list)
    for rel, v in files.items():
        if v["set"] == "background":
            continue
        voice = v.get("orig_voice") or v.get("voice")
        if a.test_only and voice not in test_voices:
            continue
        g = (v["set"], v["source"])
        if a.limit and len(groups[g]) >= a.limit:
            continue
        groups[g].append(rel)

    print(f"модель {key}, поріг {a.threshold}{', лише тестові голоси' if a.test_only else ''}\n")
    print(f"{'набір':9} {'джерело':11} {'n':>5} {'медіана':>8} {'≥поріг':>7}")
    for (set_, src), rels in sorted(groups.items()):
        scores = []
        for rel in rels:
            preds = m.predict_clip(str(WAKE / rel))
            scores.append(max(p[key] for p in preds) if preds else 0.0)
        hit = sum(s >= a.threshold for s in scores) / len(scores)
        print(f"{set_:9} {src:11} {len(scores):5d} {statistics.median(scores):8.2f} {hit:7.0%}")


if __name__ == "__main__":
    main()
