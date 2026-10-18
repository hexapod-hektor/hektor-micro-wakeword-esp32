"""Step 3: write training_parameters.yaml and run microwakeword.model_train_eval.

basic_training_notebook, overridden with interkelstar's ru_jarvis.yaml:
  training_steps 20000 (nb 10000) | class weights pos 1.5 / neg 1 (nb 1 / 20)
  SpecAugment time 2x5, freq 2x5 (nb 0) | lr 0.001, batch 128 (same)
  mixednet: pointwise 64x4, kernels [5],[7,11],[9,15],[23], first conv 32 k5, stride 3 (same as nb),
            residual_connection 1,1,1,1 (nb 0,0,0,0) -> spectrogram_length 204 (2.04 s)
Checkpoint selection = notebook default (max average_viable_recall = recall integrated over
0..2 FA/h on validation_ambient), so the low negative weight cannot silently pick an FA-heavy step.
Training dir is pushed to HF every 10 min and at the end.
validation_ambient: a 1 h subset of dinner_party_eval (full 3.2 h set OOM-kills the 12 GB Colab VM).
"""
import os, sys, argparse, subprocess, yaml

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from hk_common import W, PeriodicSync, hf_upload

F, NEG = f"{W}/features", f"{W}/negative_datasets"
ARCH = ["mixednet", "--pointwise_filters", "64,64,64,64", "--repeat_in_block", "1,1,1,1",
        "--mixconv_kernel_sizes", "[5], [7,11], [9,15], [23]", "--residual_connection", "1,1,1,1",
        "--first_conv_filters", "32", "--first_conv_kernel_size", "5", "--stride", "3"]


def count(name, split="training"):
    p = f"{F}/{name}/{split}/wakeword_mmap/.done"
    return int(open(p).read()) if os.path.exists(p) else 0


def mm(path, w, truth, trunc):
    return dict(features_dir=path, sampling_weight=w, penalty_weight=1.0, truth=truth,
                truncation_strategy=trunc, type="mmap")


def build_config(variant, live_name, steps, piper_neg, out, ambient="dinner_party_eval_1h", select_faph=False):
    n_main, n_live = count("pos_main"), count(live_name)
    w_main = 2.0  # notebook positive sampling weight
    # mWW picks a feature set by sampling_weight, then a uniform spectrogram inside it. Weight
    # proportional to count => every positive spectrogram equally likely; live clips have 3x (v1)
    # or 5x (v2) more augmented repetitions => 3x / 5x oversampling.
    w_live = round(w_main * n_live / max(1, n_main), 5)
    feats = [mm(f"{F}/pos_main", w_main, True, "truncate_start"),
             mm(f"{F}/{live_name}", w_live, True, "truncate_start"),
             mm(f"{F}/neg_other", 4.0, False, "random")]
    if piper_neg:
        feats.append(mm(f"{F}/neg_piper", 2.0, False, "random"))
    feats += [mm(f"{NEG}/speech", 10.0, False, "random"), mm(f"{NEG}/dinner_party", 10.0, False, "random"),
              mm(f"{NEG}/no_speech", 5.0, False, "random"), (mm(f"{F}/{ambient}", 0.0, False, "split") if ambient == "room_ambient" else mm(f"{NEG}/{ambient}", 0.0, False, "split"))]
    cfg = {"window_step_ms": 10, "train_dir": f"{W}/trained/{variant}", "features": feats,
           "training_steps": [steps], "positive_class_weight": [1.5], "negative_class_weight": [1.0],
           "learning_rates": [0.001], "batch_size": 128,
           "time_mask_max_size": [5], "time_mask_count": [2], "freq_mask_max_size": [5], "freq_mask_count": [2],
           "eval_step_interval": int(os.environ.get("HK_EVAL", 500)), "clip_duration_ms": 1500,
           "target_minimization": 0.0 if select_faph else 0.9, "minimization_metric": "ambient_false_positives_per_hour" if select_faph else None, "maximization_metric": "average_viable_recall"}
    yaml.dump(cfg, open(out, "w"), sort_keys=False)
    print(f"config -> {out}; n_main={n_main} n_live={n_live} w_live={w_live} piper_neg={piper_neg}", flush=True)
    return cfg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="v1")
    ap.add_argument("--live-name", default="pos_live")
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--no-piper-neg", action="store_true")
    ap.add_argument("--ambient-set", default="dinner_party_eval_1h")
    ap.add_argument("--select-faph", action="store_true")
    ap.add_argument("--train", type=int, default=1)
    ap.add_argument("--restore", type=int, default=1)
    a = ap.parse_args()
    os.makedirs(f"{W}/trained", exist_ok=True)
    cfg_path = f"{W}/trained/training_parameters_{a.variant}.yaml"
    cfg = build_config(a.variant, a.live_name, a.steps, not a.no_piper_neg, cfg_path, a.ambient_set, a.select_faph)
    os.makedirs(cfg["train_dir"], exist_ok=True)
    hf_upload(cfg_path, f"{a.variant}/training_parameters.yaml")
    sync = PeriodicSync(cfg["train_dir"], f"work/train_{a.variant}", period=600)
    sync.start()
    cmd = [sys.executable, "-m", "microwakeword.model_train_eval", f"--training_config={cfg_path}",
           f"--train={a.train}", f"--restore_checkpoint={a.restore}", "--test_tf_nonstreaming=0",
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
