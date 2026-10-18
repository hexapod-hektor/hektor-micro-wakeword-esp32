"""Нарізати довгий запис живої мови на короткі негативи: слова і фрази по 2–3 слова за Whisper word timestamps.

    python data/cut_speech.py <запис.mp3> --speaker martyn [--max 600]

Вихід: negative/neg-speech-<speaker>-<n>.wav (16 кГц, 100 мс запасу з боків), маніфест: source "human-clean"
(чистий мікрофон, не вуха робота), set negative, phrase "speech", text — що сказано, via "mic-clean".
Шматки зі словом, схожим на «Гектор», пропускаються.
"""
import argparse, json, re, subprocess, datetime as dt
from pathlib import Path
import numpy as np, mlx_whisper
from scipy.io import wavfile

WAKE = Path(__file__).resolve().parent
ap = argparse.ArgumentParser()
ap.add_argument("src"); ap.add_argument("--speaker", default="martyn"); ap.add_argument("--max", type=int, default=600)
a = ap.parse_args()
tmp = WAKE / "ref" / "_full16k.wav"
subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-i", a.src, "-ar", "16000", "-ac", "1", "-af", "loudnorm=I=-20:TP=-2", str(tmp)], check=True)
sr, x = wavfile.read(tmp); x = x.astype(np.float32) / 32768
r = mlx_whisper.transcribe(str(tmp), path_or_hf_repo="mlx-community/whisper-large-v3-turbo", language="uk", word_timestamps=True)
words = [w for s in r["segments"] for w in s.get("words", []) if w["end"] > w["start"]]
print("слів:", len(words))
m = json.loads((WAKE / "manifest.json").read_text()); fs = m["files"]
out = WAKE / "negative"; out.mkdir(exist_ok=True)
n = 0; i = 0; rng = np.random.default_rng(1)
while i < len(words) and n < a.max:
    k = int(rng.choice([1, 1, 2, 3]))                      # частіше одне слово, іноді 2–3
    chunk = words[i:i + k]; i += k
    text = "".join(w["word"] for w in chunk).strip()
    if re.search(r"гект|хект|ект[оа]р", text.lower()) or len(re.sub(r"[^\w]", "", text)) < 2:
        continue
    s0, e0 = chunk[0]["start"] - 0.1, chunk[-1]["end"] + 0.1
    if e0 - s0 < 0.3 or e0 - s0 > 2.5:
        continue
    y = x[int(max(0, s0) * sr): int(e0 * sr)]
    if np.abs(y).max() < 0.02:
        continue
    n += 1
    path = out / f"neg-speech-{a.speaker}-{n}.wav"
    wavfile.write(path, sr, (np.clip(y, -1, 1) * 32767).astype(np.int16))
    fs[str(path.relative_to(WAKE))] = {"set": "negative", "phrase": "speech", "text": text, "source": "human-clean", "via": "mic-clean",
        "speaker": a.speaker, "voice": a.speaker, "language": "uk", "emotion": "neutral", "speed": 1.0, "sample_rate": sr,
        "duration_s": round(len(y) / sr, 2), "created": dt.date.today().isoformat(), "original": Path(a.src).name,
        "note": f"нарізка живої мови за Whisper, {s0:.1f}–{e0:.1f} с"}
(WAKE / "manifest.json").write_text(json.dumps(m, ensure_ascii=False, indent=1, sort_keys=True) + "\n")
tmp.unlink(missing_ok=True)
print("негативів з живої мови:", n, "усього у маніфесті", len(fs))
