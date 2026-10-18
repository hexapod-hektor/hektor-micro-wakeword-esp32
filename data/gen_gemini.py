"""Синтетичні зразки wake-фрази «Гектор» голосами Gemini TTS, усіма TTS-моделями по черзі.

Запуск (з кореня репо):
    python data/gen_gemini.py [--voices N] [--repeats 5] [--limit N]

Ключ: GOOGLE_API_KEY у середовищі або в data/.env (не комітити).

Ліміти Gemini TTS — не гроші, а запити: ~10/хв і ~100/день НА КОЖНУ модель. Тому:
  • в одному запиті фраза повторюється --repeats разів через паузи, відповідь ріжеться по тиші
    на окремі кліпи (один запит → до 5 файлів);
  • запити йдуть по колу через усі TTS-моделі (MODELS); модель, що віддала 429 двічі поспіль,
    вважається вичерпаною на сьогодні й випадає з кола.
Емоції — короткою міткою в дужках на початку тексту ([angry] …): flash-lite її виконує, а не
читає (перевірено 04.10.2026 для [whispering]); для інших моделей — перевірити на слух.

Результат: positive/hektor-<фраза>-gemini-<модель>-<голос>-<емоція>-<n>.wav — 16 кГц, 16 біт,
моно, тиша по краях обрізана. Кожен файл — у manifest.json (див. README.md): source gemini, model,
voice, emotion, phrase, batch (який запит). Файли без запису скрипт дописує при старті (старі назви
hektor-<фраза>-<голос>.wav теж).
"""
import argparse
import datetime as dt
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
from dotenv import load_dotenv
from google import genai
from google.genai import types
from scipy.io import wavfile
from scipy.signal import resample_poly

HERE = Path(__file__).resolve().parent
OUT = HERE / "positive"
MANIFEST = HERE / "manifest.json"
SRC_RATE = 24000                         # Gemini віддає PCM 24 кГц 16 біт моно
RATE = 16000

# Усі TTS-моделі акаунта (список 05.10.2026) і короткі імена для назв файлів
MODELS = {
    "gemini-3.8-flash-lite-tts": "38lite",
    "gemini-3.8-flash-tts": "38flash",
    "gemini-3.1-flash-tts-preview": "31flash",
    "gemini-2.5-flash-preview-tts": "25flash",
    "gemini-2.5-pro-preview-tts": "25pro",
}
MODEL = "gemini-2.5-flash-preview-tts"   # для старих записів без моделі в назві (import_cartesia теж бере)

# Фрази: як до робота звертаються насправді. Ключ — частина назви файлу.
PHRASES = {
    "hektor": "Гектор",
    "hektore": "Гекторе!",
    "pryvit": "Привіт, Гектор",
    "cmd": "Гектор, встань",     # ім'я + команда: у житті ім'я майже завжди йде з командою
}

# Емоція → мітка стилю для Gemini (порожня = нейтрально). Назви як у Cartesia/ElevenLabs
EMOTIONS = {"neutral": "", "angry": "[angry]", "excited": "[excited]", "calm": "[calmly]",
            "whisper": "[whispering]"}

# 30 вбудованих голосів Gemini; порядок — від різноманітніших
VOICES = ["Puck", "Charon", "Kore", "Fenrir", "Aoede", "Leda", "Orus", "Zephyr", "Despina", "Iapetus",
          "Achird", "Algenib", "Algieba", "Alnilam", "Autonoe", "Callirrhoe", "Enceladus", "Erinome",
          "Gacrux", "Laomedeia", "Pulcherrima", "Rasalgethi", "Sadachbia", "Sadaltager", "Schedar",
          "Sulafat", "Umbriel", "Vindemiatrix", "Zubenelgenubi", "Achernar"]


def load_manifest() -> dict:
    if MANIFEST.exists():
        return json.loads(MANIFEST.read_text())
    return {"note": "Звідки кожен файл. Ключ — шлях відносно data/. Поля: set (positive/negative), "
                    "phrase (ключ фрази), text (що сказано), source (gemini/cartesia/elevenlabs/piper/robot), "
                    "model, voice, speed, emotion, sample_rate, duration_s, created.", "files": {}}


def save_manifest(m: dict):
    MANIFEST.write_text(json.dumps(m, ensure_ascii=False, indent=1, sort_keys=True) + "\n")


def entry(path: Path, key: str, model: str, voice: str, emotion: str, batch: str | None) -> dict:
    rate, x = wavfile.read(path)
    e = {
        "set": "positive", "phrase": key, "text": PHRASES[key],
        "source": "gemini", "model": model, "voice": voice,
        "speed": 1.0, "emotion": emotion,
        "sample_rate": rate, "duration_s": round(len(x) / rate, 2),
        "created": dt.datetime.fromtimestamp(path.stat().st_mtime).date().isoformat(),
        "original": "gen_gemini.py",
    }
    if batch:
        e["batch"] = batch   # з одного запиту: <модель>-<голос>-<фраза>-<емоція>
    return e


def reindex(m: dict) -> int:
    """Файли цього скрипта без запису в маніфесті: старі hektor-<фраза>-<голос>.wav (по одному на запит,
    модель MODEL) і нові hektor-<фраза>-gemini-<модель>-<голос>-<емоція>-<n>.wav."""
    by_lower = {v.lower(): v for v in VOICES}
    by_short = {s: mname for mname, s in MODELS.items()}
    n = 0
    for path in sorted(OUT.glob("hektor-*.wav")):
        rel = str(path.relative_to(HERE))
        if rel in m["files"]:
            continue
        p = path.stem.split("-")
        if len(p) == 3 and p[1] in PHRASES and p[2] in by_lower:
            m["files"][rel] = entry(path, p[1], MODEL, by_lower[p[2]], "neutral", None)
        elif len(p) == 7 and p[2] == "gemini" and p[1] in PHRASES and p[3] in by_short and p[4] in by_lower:
            m["files"][rel] = entry(path, p[1], by_short[p[3]], by_lower[p[4]], p[5],
                                    f"{p[3]}-{p[4]}-{p[1]}-{p[5]}")
        else:
            continue
        n += 1
    return n


def tts(client, model, text, voice):
    """PCM 24 кГц або None. Кидає QuotaError на 429, щоб викликач вирішив, що з моделлю."""
    r = client.models.generate_content(
        model=model,
        contents=text,
        config=types.GenerateContentConfig(
            response_modalities=["AUDIO"],
            speech_config=types.SpeechConfig(
                voice_config=types.VoiceConfig(
                    prebuilt_voice_config=types.PrebuiltVoiceConfig(voice_name=voice))),
        ),
    )
    parts = r.candidates[0].content.parts if r.candidates else []
    for p in parts:
        if p.inline_data and p.inline_data.data:
            return p.inline_data.data
    return None


def split_by_silence(pcm: bytes, min_gap_ms=250, min_len_ms=250, pad_ms=100):
    """24 кГц PCM → список кліпів 16 кГц int16, порізаних по тиші (поріг −40 дБ від піку)."""
    x = np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768
    x = resample_poly(x, 2, 3)
    f = RATE // 100                                   # кадр 10 мс
    n = len(x) // f
    if n == 0:
        return []
    rms = np.sqrt((x[:n * f].reshape(n, f) ** 2).mean(axis=1))
    loud = rms > rms.max() * 0.01                     # −40 дБ від піку
    gap, out, start, quiet = min_gap_ms // 10, [], None, 0
    for i, l in enumerate(list(loud) + [False]):
        if l:
            if start is None:
                start = i
            quiet = 0
        elif start is not None:
            quiet += 1
            if quiet >= gap or i == n:
                out.append((start, i - quiet + 1))
                start, quiet = None, 0
    pad = RATE * pad_ms // 1000
    clips = []
    for s, e in out:
        if (e - s) * 10 < min_len_ms:
            continue
        seg = x[max(0, s * f - pad): min(len(x), e * f + pad)]
        clips.append((seg * 32767).astype(np.int16))
    return clips


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--voices", type=int, default=len(VOICES), help="скільки голосів з початку списку")
    ap.add_argument("--repeats", type=int, default=5, help="повторів фрази в одному запиті")
    ap.add_argument("--emotions", default=",".join(EMOTIONS), help="які емоції, через кому")
    ap.add_argument("--models", default=",".join(MODELS), help="які моделі, через кому")
    ap.add_argument("--limit", type=int, default=0, help="зупинитись після N вдалих запитів (0 = без межі)")
    ap.add_argument("--gap", type=float, default=1.5, help="пауза між запитами, с (по колу моделей)")
    a = ap.parse_args()

    load_dotenv(HERE / ".env")
    client = genai.Client(api_key=os.environ["GOOGLE_API_KEY"],
                          http_options=types.HttpOptions(timeout=90_000))   # мс; без цього запит може повиснути назавжди
    OUT.mkdir(parents=True, exist_ok=True)
    m = load_manifest()
    if n := reindex(m):
        print(f"маніфест: дописано {n} наявних файлів")
        save_manifest(m)

    models = [x for x in a.models.split(",") if x in MODELS]
    emotions = [x for x in a.emotions.split(",") if x in EMOTIONS]
    # Порядок: спершу нейтральні всі фрази всіма голосами, потім емоції — щоб при обриві ліміту
    # набір був рівний. Моделі — по колу на кожен запит.
    jobs = [(emo, key, voice) for emo in emotions for key in PHRASES for voice in VOICES[: a.voices]]
    alive = {mod: 0 for mod in models}   # модель → 429 поспіль
    done = failed = 0
    mi = 0
    for emo, key, voice in jobs:
        # пропустити, якщо ця комбінація вже є від будь-якої моделі
        if any(OUT.glob(f"hektor-{key}-gemini-*-{voice.lower()}-{emo}-1.wav")):
            continue
        while alive:
            model = list(alive)[mi % len(alive)]
            mi += 1
            short = MODELS[model]
            text = f"{EMOTIONS[emo]} " if EMOTIONS[emo] else ""
            text += "\n\n".join([PHRASES[key] + ("" if PHRASES[key].endswith("!") else ".")] * a.repeats)
            try:
                pcm = tts(client, model, text, voice)
                alive[model] = 0
            except Exception as e:
                s = str(e)
                if "429" in s or "RESOURCE_EXHAUSTED" in s:
                    alive[model] += 1
                    if alive[model] >= 2:
                        print(f"{model}: ліміт на сьогодні — виключаю", file=sys.stderr)
                        del alive[model]
                    else:
                        time.sleep(20)
                    continue
                print(f"{model} {voice}: {s[:100]}", file=sys.stderr)
                failed += 1
                break
            clips = split_by_silence(pcm) if pcm else []
            if not clips:
                print(f"{model} {voice} {key} {emo}: порожня відповідь", file=sys.stderr)
                failed += 1
                break
            batch = f"{short}-{voice.lower()}-{key}-{emo}"
            for i, x in enumerate(clips, 1):
                path = OUT / f"hektor-{key}-gemini-{short}-{voice.lower()}-{emo}-{i}.wav"
                wavfile.write(path, RATE, x)
                m["files"][str(path.relative_to(HERE))] = entry(path, key, model, voice, emo, batch)
            save_manifest(m)
            done += 1
            print(f"{batch}: {len(clips)} кліпів ({', '.join(f'{len(x)/RATE:.1f}' for x in clips)} с)")
            time.sleep(a.gap)
            break
        if not alive:
            print("усі моделі вичерпані на сьогодні", file=sys.stderr)
            break
        if a.limit and done >= a.limit:
            break
    print(f"вдалих запитів {done}, невдалих {failed}, файлів у маніфесті {len(m['files'])}")


if __name__ == "__main__":
    main()
