"""Клони голосу через FireRedTTS3 (zero-shot, локально на Mac, ~x4 від реального часу).

    cd <тека з FireRedTTS3 і pretrained_models/> && .venv/bin/python <цей репо>/data/gen_firered.py \
        --ref <цей репо>/data/ref/martyn.wav --speaker martyn --per 30

Референс: 10–15 с чистого запису + поруч <ref>.txt із точним текстом (record_ref.py). Для кожної фрази
--per варіантів з різними seed (seed = інша подача). Вихід 16 кГц, пік −3 дБ, тиша обрізана,
positive/hektor-<фраза>-firered-<speaker>-<n>.wav + маніфест (source firered, voice <speaker>-clone, seed).
Кліпи довші за 2.5 с або коротші за 0.3 с відкидаються (модель іноді «заговорюється»).
"""
import argparse, datetime as dt, json, sys, time
from pathlib import Path
import numpy as np, torch, torchaudio, soundfile as sf

sys.path.insert(0, str(Path.cwd()))
from fireredtts3.core import FireRedTTS3

WAKE = Path(__file__).resolve().parent
RATE = 16000
NEG = {"vektor": "вектор", "sektor": "сектор", "hektar": "гектар", "nektar": "нектар", "lektor": "лектор", "rektor": "ректор",
       "detektor": "детектор", "dyrektor": "директор", "inspektor": "інспектор", "traktor": "трактор", "aktor": "актор",
       "doktor": "доктор", "viktor": "Віктор", "hera": "Гера", "spektr": "спектр", "efekt": "ефект",
       "pryvit-viktore": "Привіт, Вікторе!", "doktore": "Докторе!", "vstan": "встань", "siad": "сядь", "stop": "стоп",
       "idy-siudy": "йди сюди", "dyvys": "дивись на мене", "tykho": "тихо", "diakuiu": "дякую", "de-kliuchi": "де мої ключі",
       "kotra-hodyna": "котра година", "svitlo": "увімкни світло", "vecheria": "що на вечерю", "chuiesh": "ти чуєш мене",
       "zakryi": "закрий двері", "kholodno": "холодно сьогодні", "khlib": "треба купити хліб"}
PHRASES = {"hektor": "Гектор.", "hektore": "Гекторе!", "pryvit": "Привіт, Гектор.", "hey": "Гей, Гектор!",
           "question": "Гектор?", "chuiesh": "Гекторе, чуєш?",
           "cmd": ["Гектор, встань.", "Гекторе, йди сюди.", "Гектор, стоп!", "Гекторе, сядь.", "Гектор, дивись на мене.",
                   "Гекторе, повернись.", "Гектор, тихо.", "Гекторе, до мене."]}


def trim(x, pad=RATE // 10):
    thr = np.abs(x).max() * 0.01
    idx = np.where(np.abs(x) > thr)[0]
    return x[max(0, idx[0] - pad): idx[-1] + pad] if len(idx) else x


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", required=True); ap.add_argument("--speaker", required=True)
    ap.add_argument("--per", type=int, default=30); ap.add_argument("--seed0", type=int, default=1)
    ap.add_argument("--merge-only", action="store_true")
    ap.add_argument("--negatives", action="store_true", help="генерувати негативи (NEG) замість позитивів")
    a = ap.parse_args()
    if a.merge_only:
        merge(); return
    ref = Path(a.ref); ref_text = ref.with_suffix(".txt").read_text().strip()
    tts = FireRedTTS3("pretrained_models", use_fasttext=False, use_wetext=False)
    prompt, psr = torchaudio.load(str(ref))
    phrases = NEG if a.negatives else PHRASES
    set_ = "negative" if a.negatives else "positive"
    out_dir = WAKE / set_; out_dir.mkdir(exist_ok=True)
    main = json.loads((WAKE / "manifest.json").read_text())["files"]
    part = WAKE / "inbox" / f"firered-{set_}-{a.speaker}.json"; part.parent.mkdir(exist_ok=True)
    fs = json.loads(part.read_text()) if part.exists() else {}      # своя частина; main — лише щоб пропускати готове
    n_ok = n_bad = 0; t0 = time.time()
    for key, text in phrases.items():
        texts = text if isinstance(text, list) else [text]
        for i in range(a.per):
            seed = a.seed0 + i
            t = texts[i % len(texts)]
            path = out_dir / (f"neg-{key}-firered-{a.speaker}-{seed}.wav" if a.negatives else f"hektor-{key}-firered-{a.speaker}-{seed}.wav")
            rel = str(path.relative_to(WAKE))
            if (rel in fs or rel in main) and path.exists():
                continue
            audio, sr = tts.generate(text=t, language="Ukrainian", prompt_text=ref_text, prompt_audio=prompt, prompt_audio_sr=psr, seed=seed)
            a16 = audio.cpu().float()
            if sr != RATE:
                a16 = torchaudio.functional.resample(a16, sr, RATE)
            x = a16.numpy().reshape(-1)
            if np.abs(x).max() < 0.01:                          # тиша (fp16 на Metal іноді віддає нулі)
                n_bad += 1; continue
            x = trim(x); dur = len(x) / RATE
            if not 0.3 <= dur <= 2.5:
                n_bad += 1; continue
            x = x / (np.abs(x).max() + 1e-9) * 0.7
            sf.write(str(path), (x * 32767).astype(np.int16), RATE, subtype="PCM_16")
            fs[rel] = {"set": set_, "phrase": key, "text": t, "source": "firered", "model": "FireRedTTS3-Base",
                       "voice": f"{a.speaker}-clone", "speaker": a.speaker.split("-")[0], "language": "uk", "speed": 1.0, "emotion": "neutral", "seed": seed,
                       "reference": str(ref.name), "sample_rate": RATE, "duration_s": round(dur, 2),
                       "created": dt.date.today().isoformat(), "original": "gen_firered.py"}
            n_ok += 1
            if n_ok % 10 == 0:
                part.write_text(json.dumps(fs, ensure_ascii=False))
                print(f"  {n_ok} ок, {n_bad} відкинуто, {(time.time()-t0)/60:.0f} хв", flush=True)
    part.write_text(json.dumps(fs, ensure_ascii=False))
    merge()
    print(f"готово: {n_ok} кліпів, відкинуто {n_bad}, {(time.time()-t0)/60:.0f} хв")


def merge():
    """inbox/firered-*.json → manifest.json (лише файли, що існують). Безпечно повторювати: --merge-only."""
    m = json.loads((WAKE / "manifest.json").read_text()); added = 0
    for part in (WAKE / "inbox").glob("firered-*.json"):
        for k, v in json.loads(part.read_text()).items():
            if (WAKE / k).exists():
                m["files"][k] = v; added += 1
    (WAKE / "manifest.json").write_text(json.dumps(m, ensure_ascii=False, indent=1, sort_keys=True) + "\n")
    print("у маніфест:", added, "усього", len(m["files"]))


if __name__ == "__main__":
    main()
