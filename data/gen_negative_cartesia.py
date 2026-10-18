#!/usr/bin/env python3
"""
Негативні зразки для wake-слова «Гектор» через Cartesia TTS (sonic-3, uk).

Запуск з кореня проєкту (id 15 голосів вшиті в скрипт):
    pip install numpy requests
    export CARTESIA_API_KEY=...
    python gen_negative_cartesia.py
Опції:
    --root .            корінь репо (де data/)
    --workers 4         паралельні запити
    --d-voices id1,id2,id3,id4   інші 4 голоси для групи D (за замовч. Oleh, Sergei, Tatiana, Daria)
    --mock              без API: генерує тон замість мови (перевірка пайплайну)
Повторний запуск докачує лише відсутні файли.
"""
import argparse, io, json, os, sys, time, wave, zipfile, threading
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np

API = "https://api.cartesia.ai"
API_VERSION = "2025-04-16"
MODEL = "sonic-3"
SR = 16000
CREATED = "2026-10-05"

# 15 голосів з позитивної партії (manifest.cartesia.json); стать — для групи D
VOICES = {
    "05ffab9c-d380-4909-8375-cd12f59238c3": ("Oleh - Professional Guy", "m"),
    "1e4176b1-3db9-44d6-a601-4fe68b041942": ("Sergei - Steady Supporter", "m"),
    "064b17af-d36b-4bfb-b003-be07dba1b649": ("Tatiana - Friendly Storyteller", "f"),
    "25b7aaa6-1670-42dc-b791-419322400803": ("Daria - Decisive Dispatcher", "f"),
    "642014de-c0e3-4133-adc0-36b5309c23e6": ("Irina - Poetic Voice", "f"),
    "d358377a-cd1d-45c5-abd0-701314e36cbe": ("Marcin - Charismatic Presenter", "m"),
    "887149a8-4616-42ad-b2ce-c3819176f45d": ("Andrzej - Elder Voice", "m"),
    "ea7b5eee-39d9-40b0-b241-1910cbca9c62": ("Kasia - Natural Conversationalist", "f"),
    "dcf62f33-7cff-4f20-85b2-2efaa68cbc32": ("Zofia - Audiobook Muse", "f"),
    "b66b3a19-31b3-401f-8b08-40d46a61e4f1": ("Milena - Composed Clarifier", "f"),
    "89266bab-6e15-455d-8654-e18c440b0656": ("Petr - Pastor", "m"),
    "abf68668-6549-462c-8426-1fa7b466b91d": ("Katarina - Friendly Sales", "f"),
    "a1a16724-b1f3-4b27-9e47-8a175115e93c": ("Ivan - Bar Companion", "m"),
    "2a2624ad-bd06-4563-81fd-0519742e25d2": ("Petra - Strict Lecturer", "f"),
    "ca590fdc-df56-4d2e-94a4-ef5b423c7ddf": ("Peter - Narrator Man", "m"),
}
# група D: 2 чоловічі + 2 жіночі
D_VOICES = ["05ffab9c-d380-4909-8375-cd12f59238c3", "1e4176b1-3db9-44d6-a601-4fe68b041942",
            "064b17af-d36b-4bfb-b003-be07dba1b649", "25b7aaa6-1670-42dc-b791-419322400803"]

# ---------- корпус: (ключ, текст) ----------
A = [("vektor", "вектор"), ("sektor", "сектор"), ("hektar", "гектар"), ("nektar", "нектар"),
     ("lektor", "лектор"), ("rektor", "ректор"), ("detektor", "детектор"), ("dyrektor", "директор"),
     ("inspektor", "інспектор"), ("korektor", "коректор"), ("proiektor", "проєктор"),
     ("traktor", "трактор"), ("reflektor", "рефлектор"), ("aktor", "актор"), ("doktor", "доктор"),
     ("viktor", "Віктор"), ("hera", "Гера"), ("hetman", "гетьман"),
     ("spektr", "спектр"), ("efekt", "ефект")]
B = [("pryvit-viktore", "Привіт, Вікторе!"), ("pryvit-doktore", "Привіт, докторе!"),
     ("pryvit-lektore", "Привіт, лекторе!"), ("rektore", "Ректоре!"), ("viktore", "Вікторе!"),
     ("doktore", "Докторе!")]
C = [("vstan", "встань"), ("siad", "сядь"), ("stop", "стоп"), ("stii", "стій"),
     ("idy-siudy", "йди сюди"), ("povernys", "повернись"), ("dyvys-na-mene", "дивись на мене"),
     ("tykho", "тихо"), ("vpered", "вперед"), ("nazad", "назад"), ("molodets", "молодець"),
     ("dobre", "добре"), ("diakuiu", "дякую"), ("ni", "ні"), ("tak", "так")]
D = [("de-moi-kliuchi", "де мої ключі"), ("kotra-hodyna", "котра година"),
     ("uvimkny-svitlo", "увімкни світло"), ("shcho-na-vecheriu", "що на вечерю"),
     ("ya-pishov", "я пішов"), ("podzvony-mami", "подзвони мамі"), ("zakryi-dveri", "закрий двері"),
     ("kholodno-sohodni", "холодно сьогодні"), ("treba-kupyty-khlib", "треба купити хліб"),
     ("ty-chuiesh-mene", "ти чуєш мене")]
GROUPS = {"A": (A, ["neutral", "angry", "excited"]),
          "B": (B, ["neutral", "angry"]),
          "C": (C, ["neutral"]),
          "D": (D, ["neutral"])}


# ---------- голоси ----------
def load_voices(manifest_path):
    m = json.load(open(manifest_path, encoding="utf-8"))
    files = m.get("files", m)
    voices = {}
    if isinstance(files, dict):
        recs = files.values()
    else:
        recs = files
    for r in recs:
        if isinstance(r, dict) and r.get("voice"):
            voices.setdefault(r["voice"], {"name": r.get("voice_name", ""),
                                           "gender": r.get("gender") or r.get("voice_gender")})
    if not voices and isinstance(m.get("voices"), list):
        for v in m["voices"]:
            voices[v["id"]] = {"name": v.get("name", ""), "gender": v.get("gender")}
    return voices


def fetch_gender(session, vid):
    try:
        r = session.get(f"{API}/voices/{vid}", timeout=30)
        if r.ok:
            return (r.json().get("gender") or "").lower()
    except Exception:
        pass
    return ""


def pick_d_voices(voices, session, explicit):
    if explicit:
        return explicit
    male, female = [], []
    for vid, v in voices.items():
        g = (v.get("gender") or "").lower()
        if not g and session is not None:
            g = fetch_gender(session, vid)
            v["gender"] = g
        if g.startswith(("masc", "male", "m", "чол")) and not g.startswith("fem"):
            male.append(vid)
        elif g.startswith(("fem", "f", "жін")):
            female.append(vid)
    if len(male) < 2 or len(female) < 2:
        sys.exit(f"Не вдалося визначити стать голосів (чол={len(male)}, жін={len(female)}). "
                 f"Задайте --d-voices id1,id2,id3,id4")
    return male[:2] + female[:2]


# ---------- аудіо ----------
def wav_to_int16(data):
    with wave.open(io.BytesIO(data)) as w:
        assert w.getsampwidth() == 2, "очікувався pcm_s16le"
        sr, ch = w.getframerate(), w.getnchannels()
        x = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)
    if ch > 1:
        x = x.reshape(-1, ch).mean(axis=1).astype(np.int16)
    assert sr == SR, f"sample_rate {sr} != {SR}"
    return x


def trim(x, pad_ms=100, thr_db=-45.0, frame_ms=10):
    f = SR * frame_ms // 1000
    n = len(x) // f
    if n == 0:
        return x
    fr = x[:n * f].astype(np.float32).reshape(n, f) / 32768.0
    rms_db = 20 * np.log10(np.sqrt((fr ** 2).mean(axis=1)) + 1e-10)
    # поріг відносно піку, але не нижче абсолютного
    thr = max(thr_db, rms_db.max() - 40.0)
    idx = np.where(rms_db > thr)[0]
    if len(idx) == 0:
        return x
    pad = SR * pad_ms // 1000
    s = max(0, idx[0] * f - pad)
    e = min(len(x), (idx[-1] + 1) * f + pad)
    return x[s:e]


def write_wav(path, x):
    with wave.open(path, "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(SR)
        w.writeframes(x.astype(np.int16).tobytes())


def wav_duration(path):
    with wave.open(path) as w:
        return w.getnframes() / w.getframerate()


# ---------- TTS ----------
_local = threading.local()


def session_for(key):
    import requests
    s = getattr(_local, "s", None)
    if s is None:
        s = requests.Session()
        s.headers.update({"X-API-Key": key, "Cartesia-Version": API_VERSION,
                          "Content-Type": "application/json"})
        _local.s = s
    return s


def tts(key, text, vid, emotion, mock=False):
    if mock:
        t = np.arange(int(SR * 0.6)) / SR
        tone = (0.3 * np.sin(2 * np.pi * 220 * t) * 32767).astype(np.int16)
        x = np.concatenate([np.zeros(SR // 2, np.int16), tone, np.zeros(SR // 2, np.int16)])
        b = io.BytesIO(); write_wav_obj(b, x); return b.getvalue()
    gen = {"speed": 1.0}
    if emotion != "neutral":
        gen["emotion"] = emotion
    body = {"model_id": MODEL, "transcript": text, "voice": {"mode": "id", "id": vid},
            "language": "uk", "generation_config": gen,
            "output_format": {"container": "wav", "encoding": "pcm_s16le", "sample_rate": SR}}
    s = session_for(key)
    last = None
    for attempt in range(6):
        try:
            r = s.post(f"{API}/tts/bytes", json=body, timeout=60)
            if r.status_code == 200:
                return r.content
            last = f"HTTP {r.status_code}: {r.text[:200]}"
            if r.status_code not in (408, 429, 500, 502, 503, 504):
                break
        except Exception as e:
            last = repr(e)
        time.sleep(min(30, 2 ** attempt))
    raise RuntimeError(last)


def write_wav_obj(fobj, x):
    with wave.open(fobj, "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(SR)
        w.writeframes(x.tobytes())


# ---------- main ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--d-voices", default="")
    ap.add_argument("--mock", action="store_true")
    a = ap.parse_args()

    wake = os.path.join(a.root, "data")
    neg_dir = os.path.join(wake, "negative")
    os.makedirs(neg_dir, exist_ok=True)
    key = os.environ.get("CARTESIA_API_KEY", "")
    if not key and not a.mock:
        sys.exit("Задайте CARTESIA_API_KEY")

    voices = {vid: {"name": n, "gender": g} for vid, (n, g) in VOICES.items()}
    all_v = list(voices)
    d_v = [v for v in a.d_voices.split(",") if v] or D_VOICES

    jobs = []
    for g, (items, emos) in GROUPS.items():
        vs = d_v if g == "D" else all_v
        for k, text in items:
            for vid in vs:
                for emo in emos:
                    suf = "" if emo == "neutral" else f"-{emo}"
                    rel = f"negative/neg-{k}-cartesia-{vid}{suf}.wav"
                    jobs.append((rel, g, k, text, vid, emo))
    print(f"Усього завдань: {len(jobs)}")

    man_path = os.path.join(wake, "manifest.cartesia-negative.json")
    files = {}
    if os.path.exists(man_path):
        files = json.load(open(man_path, encoding="utf-8")).get("files", {})

    def rec(rel, g, k, text, vid, emo, dur):
        return {"set": "negative", "group": g, "phrase": k, "text": text, "source": "cartesia",
                "model": MODEL, "voice": vid, "voice_name": voices[vid]["name"], "language": "uk",
                "speed": 1.0, "emotion": emo, "sample_rate": SR, "duration_s": round(dur, 2),
                "created": CREATED}

    def run(job):
        rel, g, k, text, vid, emo = job
        path = os.path.join(wake, rel)
        if not os.path.exists(path):
            x = trim(wav_to_int16(tts(key, text, vid, emo, a.mock)))
            tmp = path + ".tmp"
            write_wav(tmp, x); os.replace(tmp, path)
        return job, wav_duration(path)

    failed = []
    done = 0
    with ThreadPoolExecutor(a.workers) as ex:
        futs = {ex.submit(run, j): j for j in jobs}
        for f in as_completed(futs):
            j = futs[f]
            try:
                job, dur = f.result()
                files[job[0]] = rec(*job, dur)
            except Exception as e:
                failed.append((j[0], str(e)))
            done += 1
            if done % 50 == 0:
                print(f"  {done}/{len(jobs)}  помилок: {len(failed)}")

    expected = {j[0] for j in jobs}
    files = {k: files[k] for k in sorted(files) if k in expected}
    json.dump({"files": files}, open(man_path, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    if failed:
        json.dump(failed, open(os.path.join(wake, "failed.cartesia-negative.json"), "w",
                               encoding="utf-8"), ensure_ascii=False, indent=2)

    zpath = os.path.join(a.root, "hektor-wake-cartesia-negative.zip")
    with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED) as z:
        for rel in files:
            z.write(os.path.join(wake, rel), f"data/wake/{rel}")
        z.write(man_path, "data/wake/manifest.cartesia-negative.json")

    lst = os.path.join(wake, "cartesia_negative_files.txt")
    with open(lst, "w", encoding="utf-8") as fh:
        for rel, r in files.items():
            emo = "" if r["emotion"] == "neutral" else f" ({r['emotion']})"
            fh.write(f"{os.path.basename(rel)} | {r['voice_name']} | {r['text']}{emo}\n")
    with zipfile.ZipFile(zpath, "a", zipfile.ZIP_DEFLATED) as z:
        z.write(lst, "data/wake/cartesia_negative_files.txt")

    cnt = Counter(r["group"] for r in files.values())
    exp = Counter(j[1] for j in jobs)
    print("\nФайлів по групах:")
    for g in "ABCD":
        print(f"  {g}: {cnt.get(g, 0)} / {exp[g]}")
    print(f"  Разом: {len(files)} / {len(jobs)}")
    print(f"Не вдалось запитів: {len(failed)}")
    for rel, err in failed[:20]:
        print(f"  {rel}: {err}")
    print(f"Архів: {zpath}")


if __name__ == "__main__":
    main()
