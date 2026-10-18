"""Влити партію з inbox/ (архів від стороннього генератора) у набір і manifest.json.

Запуск: python data/merge_inbox.py data/inbox/<партія>/data/wake/manifest.<x>.json

Партія — тека зі структурою data/wake/{positive,negative}/*.wav і manifest.<джерело>.json у нашому форматі
(поле "files", ключ — шлях відносно data/wake). Скрипт: перевіряє формат (16 кГц, моно, int16), обрізає
тишу по краях, копіює у positive/ чи negative/, вирівнює поля (speed число, emotion не null, whispers →
whisper, нерозривні пробіли в voice_name) і дописує записи в manifest.json з полем original = назва партії.
НЕ запускати, поки працює gen_gemini.py: той тримає маніфест у пам'яті й перезапише.
"""
import collections
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import gen_gemini as g
import import_cartesia as ic
from scipy.io import wavfile


def main():
    src = Path(sys.argv[1]).resolve()
    if not src.is_absolute() or not src.exists():
        sys.exit(f"нема {src}")
    batch = src.relative_to(g.HERE / "inbox").parts[0]
    root = src.parent
    new = json.loads(src.read_text())["files"]
    m = g.load_manifest()
    n = trimmed = 0
    bad = []
    for rel, e in new.items():
        f = root / rel
        if not f.exists():
            bad.append(f"{rel}: нема файлу"); continue
        rate, raw = wavfile.read(f)
        if rate != 16000 or raw.ndim != 1 or raw.dtype.name != "int16":
            bad.append(f"{rel}: {rate} Гц, {raw.dtype}, {raw.ndim}D"); continue
        x = ic.to16k(f)
        if len(x) < len(raw) - 1600:
            trimmed += 1
        out = g.HERE / rel
        out.parent.mkdir(exist_ok=True)
        wavfile.write(out, g.RATE, x)
        e = dict(e)
        e["speed"] = float(e.get("speed") or 1.0)
        emo = e.get("emotion") or "neutral"
        e["emotion"] = {"whispers": "whisper"}.get(emo, emo)
        if e.get("voice_name"):
            e["voice_name"] = e["voice_name"].replace("\xa0", " ")
        e["duration_s"] = round(len(x) / g.RATE, 2)
        e["original"] = batch
        m["files"][rel] = e
        n += 1
    g.save_manifest(m)
    fs = m["files"]
    sets = collections.Counter(v["set"] for v in fs.values())
    print(f"{batch}: додано {n}, пропущено {len(bad)}, обрізано тишу {trimmed}")
    for b in bad[:10]:
        print("  ", b)
    print(f"усього {len(fs)}: позитивів {sets['positive']}, негативів {sets['negative']}")
    print("позитиви", dict(collections.Counter(v["source"] for v in fs.values() if v["set"] == "positive")))
    print("негативи", dict(collections.Counter(v["source"] for v in fs.values() if v["set"] == "negative")))


if __name__ == "__main__":
    main()
