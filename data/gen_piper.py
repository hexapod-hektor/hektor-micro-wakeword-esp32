"""Масова синтетика через Piper (локально, безкоштовно): усі українські голоси + трохи російських.

    python data/gen_piper.py --per 400 [--neg-per 25] [--workers 4] [--dry 5]

Голоси Piper (*.onnx + *.onnx.json) — у data/voices/ (не комітяться).

Голоси: uk mykyta/oleksa/tetiana (high), lada (x_low), ukrainian_tts (medium, 3 диктори) = 7 дикторів;
ru denis/dmitri/irina/ruslan (medium) — «Гэк-тор» / «Гэкторэ», щоб espeak дав ɡˈɛktˈor, а не ɡˈɛktʌr
(«гектар»): [ɡ] замість [ɦ], тому їх лише частина. Наголос espeak для uk не міняє (hˈɛkt̪or завжди),
різноманітність — length_scale (темп), noise_scale (просодія), noise_w (тривалості фонем), випадково
на кожен кліп. Вихід 16 кГц, пік −3 дБ, тиша по краях обрізана (−40 дБ від піку, як у gen_gemini).
Файли: positive/hektor-<фраза>-piper-<голос>[-<диктор>]-<n>.wav, negative/neg-<ключ>-piper-….wav;
маніфест — source piper, model (файл голосу), voice (голос/диктор), language, length_scale, noise_scale,
noise_w, speed = 1/length_scale. Воркери пишуть частини маніфесту в inbox/piper-*.json, main зливає.
"""
import argparse
import datetime as dt
import json
import random
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from scipy.io import wavfile
from scipy.signal import resample_poly
from math import gcd

WAKE = Path(__file__).resolve().parent
VOICES = WAKE / "voices"
RATE = 16000

UK = {"uk_UA-mykyta-high": [None], "uk_UA-oleksa-high": [None], "uk_UA-tetiana-high": [None],
      "uk_UA-lada-x_low": [None], "uk_UA-ukrainian_tts-medium": [0, 1, 2]}
RU = {"ru_RU-denis-medium": [None], "ru_RU-dmitri-medium": [None], "ru_RU-irina-medium": [None], "ru_RU-ruslan-medium": [None]}

POS_UK = {"hektor": "Гектор", "hektore": "Гекторе!", "pryvit": "Привіт, Гектор", "hey": "Гей, Гектор",
          "cmd": ["Гектор, встань", "Гекторе, йди сюди", "Гектор, стоп", "Гекторе, сядь", "Гектор, дивись на мене"]}
POS_RU = {"hektor": "Гэк-тор", "hektore": "Гэкторэ!"}
NEG_UK = {"vektor": "вектор", "sektor": "сектор", "hektar": "гектар", "nektar": "нектар", "lektor": "лектор",
          "rektor": "ректор", "detektor": "детектор", "dyrektor": "директор", "inspektor": "інспектор",
          "proiektor": "проєктор", "traktor": "трактор", "aktor": "актор", "doktor": "доктор", "viktor": "Віктор",
          "hera": "Гера", "hetman": "гетьман", "spektr": "спектр", "efekt": "ефект", "nektar2": "нектар!",
          "pryvit-viktore": "Привіт, Вікторе!", "doktore": "Докторе!", "vstan": "встань", "siad": "сядь",
          "stop": "стоп", "idy-siudy": "йди сюди", "dyvys": "дивись на мене", "tykho": "тихо", "diakuiu": "дякую",
          "de-kliuchi": "де мої ключі", "kotra-hodyna": "котра година", "svitlo": "увімкни світло",
          "vecheria": "що на вечерю", "khlib": "треба купити хліб", "chuiesh": "ти чуєш мене"}


def trim(x, pad=RATE // 10):
    thr = np.abs(x).max() * 0.01
    idx = np.where(np.abs(x) > thr)[0]
    if not len(idx):
        return x
    return x[max(0, idx[0] - pad): idx[-1] + pad]


def work(job):
    """Один голос: усі фрази × per варіантів. Повертає записи маніфесту."""
    name, speakers, phrases, set_, per, seed, dry = job
    from piper import PiperVoice, SynthesisConfig
    v = PiperVoice.load(str(VOICES / f"{name}.onnx"))
    sr = v.config.sample_rate
    d = gcd(sr, RATE)
    rng = random.Random(seed)
    lang = name[:2]
    out_dir = WAKE / ("positive" if set_ == "positive" else "negative")
    out_dir.mkdir(exist_ok=True)
    part = WAKE / "inbox" / f"piper-{set_}-{name}.json"
    part.parent.mkdir(exist_ok=True)
    entries = json.loads(part.read_text()) if part.exists() else {}   # продовжити після перерви
    short = name.replace("uk_UA-", "").replace("ru_RU-", "").replace("-high", "").replace("-medium", "").replace("-x_low", "")
    for spk in speakers:
        vname = short if spk is None else f"{short}-{v.config.speaker_id_map and list(v.config.speaker_id_map)[spk]}"
        for key, text in phrases.items():
            texts = text if isinstance(text, list) else [text]
            n_items = min(per, dry) if dry else per
            for i in range(n_items):
                t = texts[i % len(texts)]
                ls, ns, nw = rng.uniform(0.75, 1.35), rng.uniform(0.3, 1.0), rng.uniform(0.4, 1.2)
                stem0 = (f"hektor-{key}-piper-{vname}-{i+1}" if set_ == "positive" else f"neg-{key}-piper-{vname}-{i+1}")
                if str((out_dir / f"{stem0}.wav").relative_to(WAKE)) in entries and (out_dir / f"{stem0}.wav").exists():
                    continue
                cfg = SynthesisConfig(length_scale=ls, noise_scale=ns, noise_w_scale=nw, speaker_id=spk)
                x = np.concatenate([np.frombuffer(c.audio_int16_bytes, np.int16) for c in v.synthesize(t.lower(), cfg)]).astype(np.float32) / 32768
                if sr != RATE:
                    x = resample_poly(x, RATE // d, sr // d)
                x = trim(x)
                pk = np.abs(x).max()
                if pk > 0:
                    x = x / pk * 0.7                               # пік −3 дБ
                stem = (f"hektor-{key}-piper-{vname}-{i+1}" if set_ == "positive" else f"neg-{key}-piper-{vname}-{i+1}")
                path = out_dir / f"{stem}.wav"
                rel = str(path.relative_to(WAKE))
                if rel in entries and path.exists():
                    continue
                wavfile.write(path, RATE, (x * 32767).astype(np.int16))
                entries[str(path.relative_to(WAKE))] = {
                    "set": set_, "phrase": key, "text": t, "source": "piper", "model": name, "voice": vname,
                    "language": lang, "speed": round(1 / ls, 2), "emotion": "neutral",
                    "length_scale": round(ls, 3), "noise_scale": round(ns, 3), "noise_w": round(nw, 3),
                    "sample_rate": RATE, "duration_s": round(len(x) / RATE, 2), "created": dt.date.today().isoformat(),
                    "original": "gen_piper.py",
                }
                if len(entries) % 50 == 0:                         # зберігати частину маніфесту по ходу
                    part.write_text(json.dumps(entries, ensure_ascii=False))
    part.write_text(json.dumps(entries, ensure_ascii=False))
    return name, set_, len(entries)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per", type=int, default=400, help="варіантів на фразу на диктора (uk позитиви)")
    ap.add_argument("--ru-per", type=int, default=60)
    ap.add_argument("--neg-per", type=int, default=25)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--dry", type=int, default=0, help="тест: не більше N на фразу")
    ap.add_argument("--merge-only", action="store_true", help="лише влити частини з inbox/piper-*.json у маніфест")
    a = ap.parse_args()
    if a.merge_only:
        merge(); return
    jobs = []
    for i, (name, spk) in enumerate(UK.items()):
        jobs.append((name, spk, POS_UK, "positive", a.per, 100 + i, a.dry))
        jobs.append((name, spk, NEG_UK, "negative", a.neg_per, 200 + i, a.dry))
    for i, (name, spk) in enumerate(RU.items()):
        jobs.append((name, spk, POS_RU, "positive", a.ru_per, 300 + i, a.dry))
    with Pool(a.workers) as p:
        for name, set_, n in p.imap_unordered(work, jobs):
            print(f"{name:28} {set_:8} {n}", flush=True)
    merge()


def merge():
    """Частини з inbox/piper-*.json → manifest.json (лише файли, що існують). Безпечно повторювати:
    якщо паралельно хтось інший переписав маніфест (whisper_check.py), запустити ще раз --merge-only."""
    m = json.loads((WAKE / "manifest.json").read_text())
    added = 0
    for part in (WAKE / "inbox").glob("piper-*.json"):
        for k, v in json.loads(part.read_text()).items():
            if (WAKE / k).exists():
                m["files"][k] = v; added += 1
    (WAKE / "manifest.json").write_text(json.dumps(m, ensure_ascii=False, indent=1, sort_keys=True) + "\n")
    print("у маніфест:", added, "усього", len(m["files"]))


if __name__ == "__main__":
    main()
