"""Темп-аугментація позитивів: швидші версії слова без зміни висоти (WSOLA, audiomentations.TimeStretch).

    python data/aug_tempo.py [--sources cartesia,elevenlabs,gemini,firered] [--rates 0.6,0.72,0.85]

Навіщо: v1 сліпне на швидкому слові (TTS стиснуто до 0.8× → 42 %, 0.65× → 0 %), а живий «Гектор» 0.5–0.7 с
проти 0.8–1.2 с у TTS. Для кожного позитива з названих джерел — по кліпу на кожен rate (<1 = швидше).
Вихід: positive/<stem>-t<rate>.wav, маніфест: source "aug-tempo", original = вихідний кліп, tempo = rate,
решта полів (voice, phrase, text, emotion) успадковані. Повторний запуск пропускає готове.
"""
import argparse, json, sys
from pathlib import Path
import numpy as np
from scipy.io import wavfile
from audiomentations import TimeStretch

WAKE = Path(__file__).resolve().parent
ap = argparse.ArgumentParser()
ap.add_argument("--sources", default="cartesia,elevenlabs,gemini,firered")
ap.add_argument("--rates", default="0.6,0.72,0.85")
ap.add_argument("--piper-per-voice", type=int, default=120, help="скільки кліпів Piper на диктора теж прискорити (0 = жодного)")
a = ap.parse_args()
srcs = set(a.sources.split(","))
rates = [float(r) for r in a.rates.split(",")]
m = json.loads((WAKE / "manifest.json").read_text()); fs = m["files"]
todo = [(k, v) for k, v in fs.items() if v["set"] == "positive" and v["source"] in srcs]
if a.piper_per_voice:
    import collections, random
    rng = random.Random(3); per = collections.defaultdict(list)
    for k, v in fs.items():
        if v["set"] == "positive" and v["source"] == "piper": per[v["voice"]].append((k, v))
    for voice, items in per.items():
        rng.shuffle(items); todo += items[: a.piper_per_voice]
print(f"кліпів: {len(todo)} × {len(rates)} темпів", flush=True)
n_new = 0
for rate in rates:
    ts = TimeStretch(min_rate=1 / rate, max_rate=1 / rate, leave_length_unchanged=False, p=1.0)   # rate у TimeStretch = прискорення
    for k, v in todo:
        src = WAKE / k
        out = src.with_name(f"{src.stem}-t{rate}.wav"); rel = str(out.relative_to(WAKE))
        if rel in fs and out.exists():
            continue
        sr, x = wavfile.read(src); x = x.astype(np.float32) / 32768
        y = ts(samples=x, sample_rate=sr)
        wavfile.write(out, sr, (np.clip(y, -1, 1) * 32767).astype(np.int16))
        e = {kk: vv for kk, vv in v.items() if kk in ("set", "phrase", "text", "voice", "voice_name", "emotion", "language", "speaker")}
        e.update({"source": "aug-tempo", "orig_source": v["source"], "model": v.get("model"), "original": k, "tempo": rate,
                  "speed": round(v.get("speed", 1.0) / rate, 2), "sample_rate": sr, "duration_s": round(len(y) / sr, 2),
                  "created": v.get("created")})
        fs[rel] = e; n_new += 1
        if n_new % 500 == 0:
            (WAKE / "manifest.json").write_text(json.dumps(m, ensure_ascii=False, indent=1, sort_keys=True) + "\n")
            print(f"  {n_new}", flush=True)
(WAKE / "manifest.json").write_text(json.dumps(m, ensure_ascii=False, indent=1, sort_keys=True) + "\n")
print("додано", n_new, "усього", len(fs))
