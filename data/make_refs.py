"""Нарізати референси для клонування з довгого запису: шматки ~10 с у різних місцях, межі по словах
(Whisper word timestamps), 24 кГц, нормалізована гучність, текст від Whisper поруч у .txt.

    python data/make_refs.py <запис.mp3> --speaker martyn --every 60 --len 10
Пише data/ref/<speaker>-r<N>.wav/.txt. Придатність кожного референсу перевіряє check_refs.sh.
"""
import argparse, subprocess, json
from pathlib import Path
import mlx_whisper

WAKE = Path(__file__).resolve().parent
ap = argparse.ArgumentParser()
ap.add_argument("src"); ap.add_argument("--speaker", default="martyn")
ap.add_argument("--every", type=float, default=60); ap.add_argument("--len", type=float, default=10); ap.add_argument("--start", type=float, default=20)
a = ap.parse_args()
out = WAKE / "ref"; out.mkdir(exist_ok=True)
dur = float(subprocess.check_output(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", a.src]).decode())
t = a.start; n = 0
while t + a.len < dur:
    tmp = out / "_tmp.wav"
    subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-i", a.src, "-ss", str(t), "-t", str(a.len + 3), "-ar", "16000", "-ac", "1", str(tmp)], check=True)
    r = mlx_whisper.transcribe(str(tmp), path_or_hf_repo="mlx-community/whisper-large-v3-turbo", language="uk", word_timestamps=True)
    words = [w for s in r["segments"] for w in s.get("words", [])]
    keep = [w for w in words if w["end"] <= a.len]        # слова, що закінчуються в межах ~len с
    if len(keep) >= 8:
        end = keep[-1]["end"] + 0.15
        text = "".join(w["word"] for w in keep).strip()
        n += 1
        dst = out / f"{a.speaker}-r{n}.wav"
        subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-i", a.src, "-ss", str(t), "-t", f"{end:.2f}", "-ar", "24000", "-ac", "1",
                        "-af", "loudnorm=I=-18:TP=-2", "-sample_fmt", "s16", str(dst)], check=True)
        dst.with_suffix(".txt").write_text(text)
        print(f"r{n} @{t:.0f}s {end:.1f}с: {text}")
    t += a.every
tmp.unlink(missing_ok=True)
print("референсів:", n)
