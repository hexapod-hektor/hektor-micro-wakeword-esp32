"""Перезапис кліпів через вуха робота: Mac грає кліп у динамік, record.py зберігає почуте.

Запуск (бот-записувач уже працює: cd record && uv run record.py -t webrtc --esp32 --host …):
    cd record && uv run rerecord.py --positive 300 --negative 500 --distance 1 --position front
    uv run rerecord.py --background 10 --label servo --count 6   # фон: шість по 10 с

Кліпи обираються з manifest.json рівномірно по джерелах і голосах (seed фіксований, тому один і
той самий набір можна прогнати з кількох позицій динаміка — --distance / --position у маніфесті).
Мітки дає порядок, не розпізнавання: перед кліпом /rec/start з назвою, після — /rec/stop.
Між кліпами пауза 0.5 с: на хвіст кімнати й затримку WebRTC. Прогнані кліпи пропускаються.
"""
import argparse
import collections
import re
import json
import random
import subprocess
import sys
import time
from pathlib import Path

import requests
import numpy as np
import sounddevice as sd
from scipy.io import wavfile

WAKE = Path(__file__).resolve().parents[1] / "data"
MANIFEST = WAKE / "manifest.json"
REC = "http://127.0.0.1:8091/rec"
TRANS = {ord(c): l for c, l in zip("абвгґдеєжзиіїйклмнопрстуфхцчшщьюя",
                                   ["a","b","v","h","g","d","e","ie","zh","z","y","i","i","i","k","l","m","n","o","p",
                                    "r","s","t","u","f","kh","ts","ch","sh","shch","","iu","ia"])}


def pick(files, set_, n, rng):
    """n кліпів набору set_, рівномірно по (джерело, голос): по колу беремо з кожної групи по одному."""
    groups = collections.defaultdict(list)
    for rel, v in files.items():
        if v["set"] != set_ or v["source"] in ("robot", "aug-tempo"):   # темпові — похідні, їх не граємо
            continue
        g = (v["source"], v["voice"])
        if v["source"] == "firered":                                     # клони голосу Мартина — один голос, але
            g = ("firered", f"{v['voice']}#{(v.get('seed') or 0) % 5}")   # 5 груп, щоб у прогін ішло 5–8, а не 1
        groups[g].append(rel)
    for g in groups.values():
        rng.shuffle(g)
    keys = sorted(groups)
    rng.shuffle(keys)
    out, i = [], 0
    while len(out) < n and any(groups.values()):
        k = keys[i % len(keys)]
        if groups[k]:
            out.append(groups[k].pop())
        i += 1
    return out


def status():
    try:
        return requests.get(f"{REC}/status", timeout=3).json()
    except Exception as e:
        sys.exit(f"record.py не відповідає на {REC}/status: {e}")


def wait_connected(timeout_s=180):
    """Плата час від часу рве WebRTC і підключається знову за ~15–20 с (бачили 05.10 на 55-му кліпі)."""
    t0 = time.monotonic()
    while time.monotonic() - t0 < timeout_s:
        if status().get("connected"):
            return True
        time.sleep(2)
    return False


PLAY_DEVICE = "Динаміки MacBook Pro"   # грати саме в динаміки Mac, а не в системний вихід (Audient iD14 без програмної гучності)
PLAY_GAIN = 0.3


def play(path: Path):
    """Програти WAV у динаміки Mac з гучністю PLAY_GAIN (0..1), блокує до кінця."""
    r, x = wavfile.read(path)
    x = x.astype(np.float32) / 32768 if x.dtype != np.float32 else x
    dev = next((i for i, d in enumerate(sd.query_devices()) if d["name"] == PLAY_DEVICE and d["max_output_channels"] > 0), None)
    sd.play(np.clip(x * PLAY_GAIN, -1, 1), r, device=dev, blocking=True)


def record(name, set_, meta, play_path: Path | None, seconds: float):
    for attempt in range(3):
        r = requests.post(f"{REC}/start", json={"name": name, "set": set_, "meta": meta}, timeout=5)
        if r.status_code != 503:
            r.raise_for_status()
            break
        print(f"\n  плата відключилась — чекаю, поки підключиться знову …", end=" ", flush=True)
        if not wait_connected():
            sys.exit("плата не повернулась за 3 хв")
        time.sleep(3)                                        # дати звуку піти після з'єднання
        print("є, повторюю", flush=True)
    else:
        sys.exit("плата весь час відключена")
    if play_path:
        play(play_path)                                      # блокує до кінця кліпу
        time.sleep(0.4)                                      # хвіст: кімната + WebRTC
    else:
        time.sleep(seconds)
    r = requests.post(f"{REC}/stop", timeout=5).json()
    if not r.get("ok"):                                      # обрив посеред кліпу: запису нема, повторити
        print("\n  обрив під час кліпу — повторюю", flush=True)
        return record(name, set_, meta, play_path, seconds)
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--positive", type=int, default=0)
    ap.add_argument("--negative", type=int, default=0)
    ap.add_argument("--background", type=float, default=0, help="секунд на один фоновий запис (без програвання)")
    ap.add_argument("--count", type=int, default=1, help="скільки фонових записів")
    ap.add_argument("--label", default="quiet", help="фон: quiet | servo | speaking | tv | …")
    ap.add_argument("--live", type=int, default=0, help="живий голос: скільки разів сказати --text по сигналу")
    ap.add_argument("--text", default="Гектор", help="що казати в --live; негатив — --set negative")
    ap.add_argument("--set", default="positive", choices=["positive", "negative"], help="для --live")
    ap.add_argument("--speaker", default="martyn", help="хто говорить у --live")
    ap.add_argument("--seconds", type=float, default=2.5, help="вікно запису одного --live, с")
    ap.add_argument("--distance", type=float, default=1.0, help="м від динаміка до робота")
    ap.add_argument("--position", default="front", help="front | side | back | far …")
    ap.add_argument("--run", default="", help="суфікс прогону; порожньо = distance+position")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--gap", type=float, default=0.5)
    ap.add_argument("--volume", type=float, default=0.3, help="гучність програвання в динаміки Mac, 0..1")
    a = ap.parse_args()

    global PLAY_GAIN
    PLAY_GAIN = a.volume
    st = status()
    if not st.get("connected"):
        sys.exit("плата не підключена до record.py — зачекай «Плата підключилась»")
    if st.get("frames", 0) == 0:
        print("увага: від плати ще не було жодного аудіокадру", file=sys.stderr)

    files = json.loads(MANIFEST.read_text())["files"]
    run = a.run or f"{a.distance:g}m-{a.position}"
    done = 0

    if a.background:
        for i in range(a.count):
            name = f"{a.label}-{time.strftime('%Y%m%d-%H%M%S')}-{i+1}"
            print(f"фон {name}: {a.background:g} с …", end=" ", flush=True)
            r = record(name, "background", {"label": a.label, "note": f"{a.distance:g} м, {a.position}"},
                       None, a.background)
            print(r.get("duration_s"), "с")
            time.sleep(a.gap)
        return

    if a.live:
        key = {"гектор": "hektor", "гекторе": "hektore", "привіт, гектор": "pryvit"}.get(a.text.lower().rstrip("!"))
        key = key or re.sub(r"[^a-z0-9]+", "-", a.text.lower().translate(TRANS)).strip("-")
        print(f"Живий запис: {a.live}× «{a.text}», {a.speaker}, {a.distance:g} м {a.position}. Говори після «ГОВОРИ».")
        for i in range(1, a.live + 1):
            for t in ("3", "2", "1"):
                print(f"  {t}…", end=" ", flush=True); time.sleep(0.7)
            name = f"live-{key}-{a.speaker}-{run}-{time.strftime('%H%M%S')}"
            print("ГОВОРИ", end=" ", flush=True)
            r = record(name, a.set, {"phrase": key, "text": a.text, "via": "human", "speaker": a.speaker,
                                     "distance_m": a.distance, "position": a.position, "run": run,
                                     "emotion": "neutral", "speed": 1.0}, None, a.seconds)
            print(f"→ {i}/{a.live}: {r.get('duration_s')} с" + ("  (порожньо? повтори)" if (r.get("duration_s") or 0) < 0.3 else ""))
            time.sleep(0.8)
        return

    rng = random.Random(a.seed)
    todo = [(rel, "positive") for rel in pick(files, "positive", a.positive, rng)]
    todo += [(rel, "negative") for rel in pick(files, "negative", a.negative, rng)]
    rng.shuffle(todo)    # позитиви й негативи впереміш, щоб умови були однакові
    existing = set(files)
    for n, (rel, set_) in enumerate(todo, 1):
        src = files[rel]
        name = f"{Path(rel).stem}-robot-{run}"
        if f"robot/{name}.wav" in existing:
            continue
        meta = {k: src[k] for k in ("phrase", "text", "emotion", "speed", "group") if k in src}
        meta.update({"original": rel, "orig_source": src["source"], "orig_voice": src.get("voice"),
                     "via": "mac-speaker", "distance_m": a.distance, "position": a.position, "run": run})
        r = record(name, set_, meta, WAKE / rel, 0)
        done += 1
        print(f"{n}/{len(todo)} {set_:8} {Path(rel).name[:48]:50} → {r.get('duration_s')} с")
        time.sleep(a.gap)
    print(f"записано {done} з {len(todo)}")


if __name__ == "__main__":
    main()
