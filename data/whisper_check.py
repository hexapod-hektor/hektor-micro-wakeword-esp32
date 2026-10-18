"""Перевірка кліпів Whisper'ом (mlx, локально): чи чути слово там, де воно має бути, і чи нема його там, де не має.

    python data/whisper_check.py [--via mac-speaker,human] [--background]

Позитиви з вух робота (via mac-speaker / human): пише в маніфест whisper_ok true/false і whisper_text;
prepare.py не бере позитиви з whisper_ok false (шум з міткою «Гектор» отруює навчання).
--background: фонові записи — шукає «гектор» у тексті; якщо є, ставить whisper_wake true (такий фон
у навчання не брати). Whisper на одиночних словах помиляється («Гектар», «Діхтар»), тому збіг м'який.
"""
import argparse, json, re, sys
from pathlib import Path
import mlx_whisper

WAKE = Path(__file__).resolve().parent
MANIFEST = WAKE / "manifest.json"
PAT = re.compile(r"гект|хект|ект[оа]р|гектар|хактор|ректор")   # м'яко: що завгодно схоже на «Гектор»
MODEL = "mlx-community/whisper-large-v3-turbo"

def hear(rel):
    return mlx_whisper.transcribe(str(WAKE / rel), path_or_hf_repo=MODEL, language="uk")["text"].strip()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--via", default="mac-speaker,human")
    ap.add_argument("--background", action="store_true")
    ap.add_argument("--redo", action="store_true", help="перевірити й ті, що вже мають whisper_ok")
    a = ap.parse_args()
    m = json.loads(MANIFEST.read_text()); fs = m["files"]
    vias = set(a.via.split(","))
    todo = [(k, v) for k, v in fs.items() if v["set"] == "positive" and v.get("via") in vias and (a.redo or "whisper_ok" not in v)]
    if a.background:
        todo += [(k, v) for k, v in fs.items() if v["set"] == "background" and (a.redo or "whisper_wake" not in v)]
    print(f"перевіряю {len(todo)} кліпів", flush=True)
    ok = bad = wake_bg = 0
    for i, (k, v) in enumerate(todo, 1):
        t = hear(k)
        if v["set"] == "background":
            v["whisper_wake"] = bool(PAT.search(t.lower())); v["whisper_text"] = t[:200]
            wake_bg += v["whisper_wake"]
        else:
            v["whisper_ok"] = bool(PAT.search(t.lower())); v["whisper_text"] = t[:80]
            ok += v["whisper_ok"]; bad += not v["whisper_ok"]
        if i % 100 == 0:
            MANIFEST.write_text(json.dumps(m, ensure_ascii=False, indent=1, sort_keys=True) + "\n")
            print(f"  {i}/{len(todo)}: ok {ok}, bad {bad}, фон зі словом {wake_bg}", flush=True)
    MANIFEST.write_text(json.dumps(m, ensure_ascii=False, indent=1, sort_keys=True) + "\n")
    print(f"готово: позитивів чує {ok}, не чує {bad}; фонових зі словом {wake_bg}")

if __name__ == "__main__":
    main()
