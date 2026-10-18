"""Step 5: package a trained variant -> MartynU/hektor-wakeword/mww/<variant>/ (+ optional copy at mww/)."""
import os, sys, json, glob, shutil, argparse, math

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from hk_common import W, hf_upload

GROUP_ORDER = ["ALL", "live", "robot", "firered", "aug-tempo", "tts", "piper", "other"]


def manifest(ev, cutoff, window):
    arena = ev["tech"]["tflm_arena_min_bytes"]
    arena = int(math.ceil((arena or 30000) * 1.1 / 1024) * 1024)  # +10 % headroom, KiB-aligned
    return {"type": "micro", "wake_word": "Гектор", "author": "MartynU",
            "website": "https://huggingface.co/MartynU/hektor-wakeword", "model": "model.tflite",
            "trained_languages": ["uk"], "version": 2,
            "micro": {"probability_cutoff": cutoff, "sliding_window_size": window, "feature_step_size": 10,
                      "tensor_arena_size": arena, "minimum_esphome_version": "2024.7.0"}}


def md_table(ev, window, cutoffs=("0.5", "0.8", "0.9", "0.95")):
    w = f"w{window}"
    L = ["| група | n | " + " | ".join(f"@{c}" for c in cutoffs) + " |", "|---|---|" + "---|" * len(cutoffs)]
    pos = ev["recall_positive_test"]
    for g in GROUP_ORDER:
        if g in pos:
            L.append(f"| recall {g} | {pos[g]['n']} | " + " | ".join(f"{pos[g][w][c]*100:.1f}%" for c in cutoffs) + " |")
    for g, v in ev["false_accept_rate_negative_test"].items():
        L.append(f"| FA negative_test {g} | {v['n']} | " + " | ".join(f"{v[w][c]*100:.2f}%" for c in cutoffs) + " |")
    L.append(f"| **FA/год ambient (DiPCo {ev['ambient_hours']} год)** | - | " +
             " | ".join(f"{ev['ambient_fa_per_hour'][w][c]}" for c in cutoffs) + " |")
    return "\n".join(L)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", required=True)
    ap.add_argument("--also-root", action="store_true")
    a = ap.parse_args()
    td = f"{W}/trained/{a.variant}"
    ev = json.load(open(f"{td}/eval.json"))
    rec = ev["recommended"]
    out = f"{W}/publish/{a.variant}"
    shutil.rmtree(out, ignore_errors=True); os.makedirs(f"{out}/logs", exist_ok=True)
    shutil.copy(f"{td}/tflite_stream_state_internal_quant/stream_state_internal_quant.tflite", f"{out}/model.tflite")
    json.dump(manifest(ev, rec["probability_cutoff"], rec["sliding_window_size"]), open(f"{out}/manifest.json", "w"),
              ensure_ascii=False, indent=2)
    shutil.copy(f"{W}/trained/training_parameters_{a.variant}.yaml", f"{out}/training_parameters.yaml")
    shutil.copy(f"{td}/training_config.yaml", f"{out}/training_config_resolved.yaml")
    shutil.copy(f"{td}/model_summary.txt", f"{out}/model_summary.txt")
    ev_small = {k: v for k, v in ev.items() if k != "per_file"}
    json.dump(ev_small, open(f"{out}/eval.json", "w"), ensure_ascii=False, indent=1)
    json.dump(ev["per_file"], open(f"{out}/logs/eval_per_file.json", "w"), ensure_ascii=False)
    for p in [f"{td}/train.log", f"{W}/logs/train_{a.variant}.log", f"{W}/logs/eval_{a.variant}.log",
              f"{W}/logs/prep_data.log", f"{W}/logs/gen_features.log", f"{W}/data/prep_summary.json"] + \
             glob.glob(f"{td}/tflite_stream_state_internal_quant/*.txt"):
        if os.path.exists(p):
            shutil.copy(p, f"{out}/logs/")
    for n in (5, 10):
        open(f"{out}/logs/table_w{n}.md", "w").write(md_table(ev, n))
    hf_upload(out, a.variant, f"mww: publish {a.variant}")
    if a.also_root:
        for f in ["model.tflite", "manifest.json", "training_parameters.yaml", "eval.json"]:
            hf_upload(f"{out}/{f}", f, f"mww: {a.variant} as default")
    print(json.dumps(manifest(ev, rec["probability_cutoff"], rec["sliding_window_size"]), ensure_ascii=False))
    print(md_table(ev, rec["sliding_window_size"]))


if __name__ == "__main__":
    main()
