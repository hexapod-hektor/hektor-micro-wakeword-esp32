"""Імпорт файлів, скачаних з плейграунда Cartesia, у data/positive + manifest.json.

Запуск: python data/import_cartesia.py [файли або теки...]
Без аргументів бере все з data/inbox/cartesia/.

З назви cartesia_<час>_<голос>_<модель>_<мова>.wav беремо голос, модель, мову і дату. Фрази в назві
нема — її визначає Whisper (mlx, локально) і підганяє під наші три: hektor / hektore / pryvit.
Якщо не впізнав — файл пропускається, про це пишемо; тоді фразу можна задати вручну: --phrase hektor.
Результат: positive/hektor-<фраза>-cartesia-<голос>.wav (16 кГц, 16 біт, моно, тиша обрізана).
"""
import argparse
import re
import sys
from pathlib import Path

import numpy as np
from scipy.io import wavfile
from scipy.signal import resample_poly

sys.path.insert(0, str(Path(__file__).resolve().parent))
import gen_gemini as g   # PHRASES, load_manifest, save_manifest, RATE

HERE = g.HERE
INBOX = HERE / "inbox" / "cartesia"
NAME = re.compile(r"cartesia_(\d{4}-\d{2}-\d{2})T\d{2}_\d{2}_\d{2}Z_(?P<voice>[^_]+)_(?P<model>[^_]+)_(?P<lang>[a-z]{2})\.wav$")

# Як Whisper може почути кожну фразу (нижній регістр, без розділових)
HINTS = {
    "pryvit": ["привіт"],
    "hektore": ["гекторе", "хекторе"],
    "hektor": ["гектор", "хектор", "гектар", "ектор"],
}


def guess_phrase(path: Path) -> str | None:
    import mlx_whisper
    text = mlx_whisper.transcribe(str(path), path_or_hf_repo="mlx-community/whisper-large-v3-turbo",
                                  language="uk")["text"]
    t = re.sub(r"[^\w\s]", "", text.lower())
    for key, words in HINTS.items():      # порядок важливий: «привіт» перевіряємо першим
        if any(w in t for w in words):
            return key
    print(f"  {path.name}: Whisper почув «{text.strip()}» — фразу не впізнав", file=sys.stderr)
    return None


def to16k(path: Path) -> np.ndarray:
    rate, x = wavfile.read(path)
    if x.ndim > 1:
        x = x[:, 0]
    x = x.astype(np.float32) / 32768
    if rate != g.RATE:
        from math import gcd
        d = gcd(rate, g.RATE)
        x = resample_poly(x, g.RATE // d, rate // d)
    thr = np.abs(x).max() * 0.01
    idx = np.where(np.abs(x) > thr)[0]
    if len(idx):
        pad = g.RATE // 10
        x = x[max(0, idx[0] - pad): idx[-1] + pad]
    return (x * 32767).astype(np.int16)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="*", type=Path)
    ap.add_argument("--phrase", choices=list(g.PHRASES), help="не вгадувати Whisper'ом, а взяти цю фразу")
    a = ap.parse_args()
    files = []
    for p in a.paths or [INBOX]:
        files += sorted(p.glob("cartesia_*.wav")) if p.is_dir() else [p]

    m = g.load_manifest()
    done = 0
    for f in files:
        mt = NAME.search(f.name)
        if not mt:
            print(f"  {f.name}: назва не схожа на cartesia_<час>_<голос>_<модель>_<мова>.wav — пропускаю",
                  file=sys.stderr)
            continue
        voice, model, lang, created = mt["voice"], mt["model"], mt["lang"], mt.group(1)
        key = a.phrase or guess_phrase(f)
        if not key:
            continue
        out = g.OUT / f"hektor-{key}-cartesia-{voice.lower()}.wav"
        n = 2
        while out.exists():   # той самий голос і фраза ще раз — нумеруємо
            out = g.OUT / f"hektor-{key}-cartesia-{voice.lower()}-{n}.wav"
            n += 1
        x = to16k(f)
        wavfile.write(out, g.RATE, x)
        m["files"][str(out.relative_to(HERE))] = {
            "set": "positive", "phrase": key, "text": g.PHRASES[key],
            "source": "cartesia", "model": model, "voice": voice, "language": lang,
            "sample_rate": g.RATE, "duration_s": round(len(x) / g.RATE, 2), "created": created,
            "original": f.name,
        }
        g.save_manifest(m)
        print(f"{out.name}: {len(x) / g.RATE:.2f} с, голос {voice}, {model}, фраза «{g.PHRASES[key]}»")
        done += 1
    print(f"імпортовано {done} з {len(files)}")


if __name__ == "__main__":
    main()
