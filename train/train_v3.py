"""v3 step 3: v1 config with the owner's v3 changes; runner identical to train_run.py."""
import os, sys, subprocess, yaml, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from hk_common import W, PeriodicSync, hf_upload
from train_run import ARCH, count, mm

F, NEG = f"{W}/features", f"{W}/negative_datasets"

def build(variant, piper_w, neg_cw, steps):
    w_main = 2.0
    w_live = round(w_main / 9, 5)  # live = 10 % of positive draws
    n_other, n_piper = count("neg_other_v3"), count("neg_piper")
    n_add_o, n_add_p = count("neg_add_other"), count("neg_add_piper")
    # new v3 negatives: same per-spectrogram probability as the cached set they extend
    wo, wp = 4.0, piper_w
    feats = [mm(f"{F}/pos_main_v3", w_main, True, "truncate_start"),
             dict(mm(f"{F}/pos_live_v3", w_live, True, "truncate_start"), penalty_weight=3.0)]
    if n_add_o:
        feats += [mm(f"{F}/neg_other_v3", round(wo * n_other / (n_other + n_add_o), 4), False, "random"),
                  mm(f"{F}/neg_add_other", round(wo * n_add_o / (n_other + n_add_o), 4), False, "random")]
    else:
        feats += [mm(f"{F}/neg_other_v3", wo, False, "random")]
    if n_add_p:
        feats += [mm(f"{F}/neg_piper", round(wp * n_piper / (n_piper + n_add_p), 4), False, "random"),
                  mm(f"{F}/neg_add_piper", round(wp * n_add_p / (n_piper + n_add_p), 4), False, "random")]
    else:
        feats += [mm(f"{F}/neg_piper", wp, False, "random")]
    feats += [mm(f"{NEG}/speech", 8.0, False, "random"), mm(f"{NEG}/dinner_party", 6.0, False, "random"),
              mm(f"{NEG}/no_speech", 4.0, False, "random"),
              mm(f"{F}/room_ambient", 0.0, False, "split"), mm(f"{NEG}/dinner_party_eval_1h", 0.0, False, "split")]
    cfg = {"window_step_ms": 10, "train_dir": f"{W}/trained/{variant}", "features": feats,
           "training_steps": [steps], "positive_class_weight": [1.5], "negative_class_weight": [neg_cw],
           "learning_rates": [0.001], "batch_size": 128,
           "time_mask_max_size": [5], "time_mask_count": [2], "freq_mask_max_size": [5], "freq_mask_count": [2],
           "eval_step_interval": 500, "clip_duration_ms": 1500,
           "target_minimization": 0.5, "minimization_metric": "ambient_false_positives_per_hour",
           "maximization_metric": "average_viable_recall"}
    print(f"counts: pos_main_v3={count('pos_main_v3')} pos_live_v3={count('pos_live_v3')} neg_other_v3={n_other} "
          f"neg_add_other={n_add_o} neg_piper={n_piper} neg_add_piper={n_add_p}", flush=True)
    return cfg

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="v3")
    ap.add_argument("--piper-w", type=float, default=1.5)
    ap.add_argument("--neg-cw", type=float, default=1.0)
    ap.add_argument("--steps", type=int, default=10000)
    a = ap.parse_args()
    os.makedirs(f"{W}/trained", exist_ok=True)
    cfg = build(a.variant, a.piper_w, a.neg_cw, a.steps)
    cfg_path = f"{W}/trained/training_parameters_{a.variant}.yaml"
    yaml.dump(cfg, open(cfg_path, "w"), sort_keys=False)
    print(open(cfg_path).read(), flush=True)
    os.makedirs(cfg["train_dir"], exist_ok=True)
    hf_upload(cfg_path, f"{a.variant}/training_parameters.yaml")
    sync = PeriodicSync(cfg["train_dir"], f"work/train_{a.variant}", period=600)
    sync.start()
    cmd = [sys.executable, "-m", "microwakeword.model_train_eval", f"--training_config={cfg_path}",
           "--train=1", "--restore_checkpoint=1", "--test_tf_nonstreaming=0",
           "--test_tflite_nonstreaming=0", "--test_tflite_nonstreaming_quantized=0",
           "--test_tflite_streaming=0", "--test_tflite_streaming_quantized=1", "--use_weights=best_weights"] + ARCH
    print(" ".join(cmd), flush=True)
    with open(f"{cfg['train_dir']}/train.log", "a") as lf:
        p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, cwd=W)
        for line in p.stdout:
            lf.write(line); lf.flush()
            if any(k in line for k in ("Step ", "So far", "Error", "Traceback", "Cutoff", "AUC", "Converting")):
                print(line.rstrip()[:400], flush=True)
        rc = p.wait()
    sync.stop.set()
    sync.push()
    print("exit code", rc, flush=True)
    sys.exit(rc)

if __name__ == "__main__":
    main()
