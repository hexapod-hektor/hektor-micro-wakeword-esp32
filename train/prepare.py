"""Розкласти набір з manifest.json у теки, які читає openwakeword/train.py.

Запуск (з кореня репо): python train/prepare.py [--out train/dataset] [--zip]

Поділ train/test — ПО ГОЛОСАХ, не по файлах: тестові голоси модель під час навчання не чує, тож
positive_test_clips_recall у логах train.py означає «впізнає нових людей», а не «вивчила ці самі».
Відкладені голоси фіксуються в split.json (поруч із dataset/), щоб результат можна було повторити.
Негативи тих самих відкладених голосів ідуть у negative_test.
Беруться лише файли з записом у manifest.json (README: безрідні не тренуємо).
"""
import argparse
import collections
import json
import random
import shutil
import sys
import zipfile
from pathlib import Path

WAKE = Path(__file__).resolve().parents[1] / "data"   # кліпи й manifest.json — у data/
MANIFEST = WAKE / "manifest.json"
MODEL = "hektor"
HOLDOUT = {"cartesia": 2, "elevenlabs": 2, "gemini": 4, "piper": 1, "aug-tempo": 0}   # скільки голосів кожного джерела відкласти на тест


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(Path(__file__).resolve().parent / "dataset"))
    ap.add_argument("--seed", type=int, default=5)
    ap.add_argument("--zip", action="store_true", help="ще й запакувати dataset.zip для Colab")
    ap.add_argument("--tts-only", action="store_true", help="лише хмарні TTS (cartesia, elevenlabs, gemini), як у v1")
    ap.add_argument("--exclude-via", default="", help="викинути перезаписи з таким via: mac-speaker | human")
    ap.add_argument("--only-run", default="", help="з перезаписів лишити лише цей run, напр. 1.5m-front")
    ap.add_argument("--cap", default="piper=4000",
                    help="ліміт позитивів train на джерело, напр. piper=4000,aug-tempo=6000; решта — без ліміту")
    ap.add_argument("--drop-neg", default="vektor,rektor,viktor,viktore,pryvit-viktore",
                    help="негативи, які НЕ брати: найближчі сусіди «Гектор» (06.10: живий «Гектор» Мартина з м'яким Г "
                         "Whisper чує як «Вектор»/«Віктор» — такі негативи вчать модель відкидати саме його вимову)")
    ap.add_argument("--phrases", default="", help="лишити лише ці фрази позитивів, напр. hektor,hektore (microWakeWord: одне слово)")
    ap.add_argument("--live-test-every", type=int, default=2,
                    help="живі кліпи через вуха: кожен N-й у тест (2 = навпіл, як було; 4 = чверть у тест)")
    ap.add_argument("--ears-only", action="store_true",
                    help="лише звук через вуха робота (перезаписи TTS + живі + фон), без чистих TTS — v3a")
    a = ap.parse_args()
    files = json.loads(MANIFEST.read_text())["files"]

    rng = random.Random(a.seed)
    voices = collections.defaultdict(set)
    for v in files.values():
        if v["set"] == "positive" and v["source"] != "robot":
            voices[v["source"]].add(v["voice"])
    test_voices = set()
    for src, n in HOLDOUT.items():
        vs = sorted(voices.get(src, []))
        rng.shuffle(vs)
        test_voices |= set(vs[:n])

    if a.phrases:
        keep = set(a.phrases.split(","))
        files = {k: v for k, v in files.items() if v["set"] != "positive" or v.get("phrase") in keep}
    drop_neg = set(a.drop_neg.split(",")) if a.drop_neg else set()
    files = {k: v for k, v in files.items() if not (v["set"] == "negative" and v.get("phrase") in drop_neg)}
    if a.ears_only:
        files = {k: v for k, v in files.items() if v.get("source") == "robot"}
    if a.tts_only:
        files = {k: v for k, v in files.items() if v.get("source") in ("cartesia", "elevenlabs", "gemini")}
    if a.exclude_via:
        files = {k: v for k, v in files.items() if v.get("via") != a.exclude_via}
    if a.only_run:
        files = {k: v for k, v in files.items() if v.get("via") != "mac-speaker" or v.get("run") == a.only_run}
    out = Path(a.out) / MODEL
    if out.exists():
        shutil.rmtree(out)
    counts = collections.Counter()
    missing = 0
    live_n = collections.Counter()
    skipped = collections.Counter()
    # cap: "piper=4000" — позитиви і негативи; "neg:piper=8000" — окремий ліміт для негативів цього джерела
    caps_raw = {kv.split("=")[0]: int(kv.split("=")[1]) for kv in a.cap.split(",") if "=" in kv}
    caps = {k: v for k, v in caps_raw.items() if not k.startswith("neg:")}
    caps_neg = {k[4:]: v for k, v in caps_raw.items() if k.startswith("neg:")}
    def cap_for(set_, src):
        return caps_neg.get(src, caps.get(src)) if set_ == "negative" else caps.get(src)
    rng_cap = random.Random(a.seed + 1)
    # скільки позитивів навчання має кожне обмежене джерело (до вибірки), щоб вибрати рівно ~cap
    eligible = collections.Counter()
    for k, v in files.items():
        if cap_for(v["set"], v["source"]) is None: continue
        if v.get("via") == "human": continue
        vv = v.get("orig_voice") or v.get("voice")
        is_fr = v["source"] == "firered" or (v["source"] == "aug-tempo" and v.get("orig_source") == "firered")
        in_test = (((v.get("seed") or files.get(v.get("original", ""), {}).get("seed") or 0) % 5 == 0) if is_fr else (vv in test_voices))
        if not in_test: eligible[(v["set"], v["source"])] += 1
    for rel, v in sorted(files.items()):
        f = WAKE / rel
        if not f.exists():
            missing += 1
            continue
        if v["set"] == "background":        # фон → dataset/robot_bg для аугментації (background_paths у yaml)
            if v.get("whisper_wake"):       # у фоні прозвучало слово — не брати
                skipped["background_with_wake"] += 1
                continue
            d = Path(a.out) / "robot_bg"
            d.mkdir(parents=True, exist_ok=True)
            shutil.copy2(f, d / f.name)
            counts["robot_bg"] += 1
            continue
        if v["set"] == "positive" and v.get("whisper_ok") is False:   # шум з міткою «Гектор» (whisper_check.py)
            skipped["positive_whisper_bad"] += 1
            continue
        if v.get("via") == "human":         # живий голос через вуха: навпіл, парні — навчання, непарні — тест
            k = (v.get("speaker"), v["set"], v.get("phrase"))
            part = "test" if live_n[k] % a.live_test_every == a.live_test_every - 1 else "train"
            live_n[k] += 1
        elif v.get("source") == "firered" or (v.get("source") == "aug-tempo" and v.get("orig_source") == "firered"):
            # клони голосу Мартина: кожен 5-й seed у тест, щоб його тембр мірявся окремо
            seed = v.get("seed") or files.get(v.get("original", ""), {}).get("seed") or 0
            part = "test" if seed % 5 == 0 else "train"
        else:
            voice = v.get("orig_voice") or v.get("voice")   # перезапис через робота тримає голос оригіналу
            part = "test" if voice in test_voices else "train"
        if part == "train" and cap_for(v["set"], v["source"]) is not None:
            # ліміт на джерело (Piper 89 % позитивів у v4 → модель вивчила «голос Piper»): рівномірна вибірка cap з N,
            # окремо для позитивів і негативів
            if rng_cap.random() > cap_for(v["set"], v["source"]) / max(eligible[(v["set"], v["source"])], 1):
                skipped[f"cap_{v['source']}"] += 1
                continue
        d = out / f"{v['set']}_{part}"
        d.mkdir(parents=True, exist_ok=True)
        shutil.copy2(f, d / f.name)
        counts[d.name] += 1

    split = {
        "seed": a.seed, "holdout": HOLDOUT, "live_test_every": a.live_test_every, "ears_only": a.ears_only, "tts_only": a.tts_only,
        "exclude_via": a.exclude_via, "only_run": a.only_run, "drop_neg": sorted(drop_neg),
        "test_voices": {src: sorted(v for v in test_voices if v in voices[src]) for src in voices},
        "counts": dict(counts), "manifest_files": len(files), "missing": missing, "skipped": dict(skipped), "caps": caps_raw, "eligible": {f"{a_}/{b_}": n for (a_, b_), n in eligible.items()},
        "live_clips": {f"{k[0]}/{k[1]}/{k[2]}": n for k, n in live_n.items()},
    }
    (Path(a.out) / "split.json").write_text(json.dumps(split, ensure_ascii=False, indent=1) + "\n")
    print(json.dumps(split, ensure_ascii=False, indent=1))
    if a.zip:
        z = Path(a.out).with_suffix(".zip")
        with zipfile.ZipFile(z, "w", zipfile.ZIP_DEFLATED) as zf:
            for p in sorted(Path(a.out).rglob("*")):
                # лише кліпи й split.json: без ознак *.npy і моделей *.onnx з локального тренування в цій же теці
                if p.is_file() and (p.suffix == ".wav" or p.name == "split.json"):
                    zf.write(p, p.relative_to(Path(a.out).parent))
        print(f"{z}: {z.stat().st_size / 1e6:.0f} МБ")


if __name__ == "__main__":
    main()
