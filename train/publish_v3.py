"""v3 step 5: package trained/<variant> -> MartynU/hektor-wakeword/mww/<variant>/ (like publish.py),
comparison table v1/v2/v3 (all evaluated on the v3 test set), README-mww.md / TRAINING-mww.md additions."""
import os, sys, json, glob, shutil, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from hk_common import W, hf_upload
from publish import manifest

ORDER = ["ALL", "live", "live-cmd", "live+cmd", "robot", "firered", "aug-tempo", "tts", "piper", "other",
         "kind:hektor", "kind:hektore", "kind:cmd"]

def table(ev, w, cuts=("0.5", "0.6", "0.7", "0.8", "0.9", "0.95", "0.97")):
    L = [f"| вікно {w[1:]} | n | " + " | ".join(f"@{c}" for c in cuts) + " |", "|---|---|" + "---|" * len(cuts)]
    pos, neg = ev["recall_positive_test"], ev["false_accept_rate_negative_test"]
    for g in ORDER:
        if g in pos:
            L.append(f"| recall {g} | {pos[g]['n']} | " + " | ".join(f"{pos[g][w][c]*100:.1f}%" for c in cuts) + " |")
    for g, v in neg.items():
        L.append(f"| FA negative_test {g} | {v['n']} | " + " | ".join(f"{v[w][c]*100:.2f}%" for c in cuts) + " |")
    L.append(f"| FA/год DiPCo ({ev['dipco_hours']} год) | - | " + " | ".join(str(ev['dipco_fa_per_hour'][w][c]) for c in cuts) + " |")
    L.append(f"| FA/год наш фон (testing_ambient, {ev['room_hours']} год) | - | " + " | ".join(str(ev['room_fa_per_hour'][w][c]) for c in cuts) + " |")
    return "\n".join(L)

def row(name, ev, w, c, note=""):
    p, n = ev["recall_positive_test"], ev["false_accept_rate_negative_test"]
    f = lambda g: f"{p[g][w][c]*100:.1f}% ({round(p[g][w][c]*p[g]['n'])}/{p[g]['n']})" if g in p else "-"
    return (f"| {name} | {w[1:]} | {c} | {f('live')} | {f('live-cmd')} | {p['ALL'][w][c]*100:.1f}% | "
            f"{n['ALL'][w][c]*100:.2f}% | {ev['dipco_fa_per_hour'][w][c]} | {ev['room_fa_per_hour'][w][c]} | {note} |")

def compare(evs):
    L = ["| модель | вікно | поріг | recall live | recall live-cmd | recall all | FA neg-test | FA/год DiPCo | FA/год наш фон | примітка |",
         "|---|---|---|---|---|---|---|---|---|---|"]
    for name, ev, c, note in evs:
        L.append(row(name, ev, "w5", c, note))
    return "\n".join(L)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="v3")
    ap.add_argument("--notes", default="")
    ap.add_argument("--upload", type=int, default=1)
    a = ap.parse_args()
    td = f"{W}/trained/{a.variant}"
    ev = json.load(open(f"{td}/eval.json"))
    rec = ev["recommended"]
    out = f"{W}/publish/{a.variant}"
    shutil.rmtree(out, ignore_errors=True); os.makedirs(f"{out}/logs", exist_ok=True)
    shutil.copy(f"{td}/tflite_stream_state_internal_quant/stream_state_internal_quant.tflite", f"{out}/model.tflite")
    man = manifest(ev, rec["probability_cutoff"], 5)
    json.dump(man, open(f"{out}/manifest.json", "w"), ensure_ascii=False, indent=2)
    shutil.copy(f"{W}/trained/training_parameters_{a.variant}.yaml", f"{out}/training_parameters.yaml")
    shutil.copy(f"{td}/training_config.yaml", f"{out}/training_config_resolved.yaml")
    shutil.copy(f"{td}/model_summary.txt", f"{out}/model_summary.txt")
    json.dump({k: v for k, v in ev.items() if k != "per_file"}, open(f"{out}/eval.json", "w"), ensure_ascii=False, indent=1)
    json.dump(ev["per_file"], open(f"{out}/logs/eval_per_file.json", "w"), ensure_ascii=False)
    for p in [f"{td}/train.log", f"{W}/logs/train_{a.variant}.log", f"{W}/logs/eval_{a.variant}.log",
              f"{W}/logs/prep_v3.log", f"{W}/logs/gen_v3.log", f"{W}/data/prep_summary.json",
              f"{W}/logs/eval_v1_on_v3.log", f"{W}/logs/eval_v2_on_v3.log"] + glob.glob(f"{td}/tflite_stream_state_internal_quant/*.txt"):
        if os.path.exists(p):
            shutil.copy(p, f"{out}/logs/")
    for n in (5, 10):
        open(f"{out}/logs/table_w{n}.md", "w").write(table(ev, f"w{n}"))
    evs = []
    for v, c, note in (("v1", "0.6", "manifest v1: вікно 5 @0.6"), ("v2", "0.5", "manifest v2: вікно 10 @0.5 (тут вікно 5)")):
        p = f"{W}/trained/eval_{v}_on_v3.json"
        if os.path.exists(p):
            evs.append((v, json.load(open(p)), c, note))
    for v in ("v3", "v3b"):
        p = f"{W}/trained/{v}/eval.json"
        if os.path.exists(p):
            e = json.load(open(p))
            evs.append((v, e, str(e["recommended"]["probability_cutoff"]), "рекомендований поріг"))
    cmp_md = compare(evs)
    open(f"{out}/logs/compare_v1_v2_v3.md", "w").write(cmp_md + "\n")
    print(json.dumps(man, ensure_ascii=False)); print(cmp_md); print(table(ev, "w5")); print(table(ev, "w10"))
    # README-mww.md / TRAINING-mww.md: add the v3 section before the conclusions
    readme = open(f"{W}/mww/README-mww.md").read()
    sec = [f"## {a.variant}: набір micro-wake-hector-v3 (Гектор / Гекторе / з командою, 93 живих), live ×25 (10 % позитивів у батчі, penalty 3), neg_piper повернуто (1.5), ваги за FA/год ≤ 0.5 на (наш фон + DiPCo 1 год), 10 000 кроків", "",
           f"manifest: probability_cutoff **{man['micro']['probability_cutoff']}**, sliding_window_size **5**, tensor_arena_size {man['micro']['tensor_arena_size']}", "",
           "Порівняння (усі три моделі оцінені на тестовому наборі v3, вікно 5):", "", cmp_md, "",
           "### ковзне вікно 5", "", table(ev, "w5"), "", "### ковзне вікно 10", "", table(ev, "w10"), ""]
    if a.notes and os.path.exists(a.notes):
        sec += open(a.notes).read().splitlines() + [""]
    marker = "## Висновки й застереження"
    import re
    readme = re.sub(rf"\n## {a.variant}:.*?(?=\n## |\Z)", "\n", readme, flags=re.S)
    readme = readme.replace(marker, "\n".join(sec) + "\n" + marker) if marker in readme else readme + "\n" + "\n".join(sec)
    readme = readme.replace("Файли: mww/v1/, mww/v2/ ", "Файли: mww/v1/, mww/v2/, mww/v3/ ") if "mww/v3/" not in readme else readme
    os.makedirs(f"{W}/publish", exist_ok=True)
    open(f"{W}/publish/README-mww.md", "w").write(readme)
    if a.upload:
        hf_upload(out, a.variant, f"mww: publish {a.variant}")
        hf_upload(f"{W}/publish/README-mww.md", "README-mww.md", f"mww: README + {a.variant}")
        for f in glob.glob(f"{W}/v3s/*.py"):
            if os.path.basename(f) not in ("dl.py", "insp.py"):
                hf_upload(f, f"scripts/{os.path.basename(f)}", "mww: v3 scripts")
        if os.path.exists(f"{W}/publish/TRAINING-mww.md"):
            hf_upload(f"{W}/publish/TRAINING-mww.md", "TRAINING-mww.md", "mww: TRAINING v3")
            hf_upload(f"{W}/publish/TRAINING-mww.md", "scripts/TRAINING-mww.md", "mww: TRAINING v3")
    print("PUBLISH_DONE")

if __name__ == "__main__":
    main()
