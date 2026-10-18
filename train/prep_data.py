"""Step 1: bring our WAVs to what the microWakeWord pipeline expects.

mWW (microwakeword/audio/augmentation.py) does not need fixed-length clips: augment_clip()
appends jitter (0.195-0.205 s of silence) at the END and LEFT-pads to augmentation_duration_s,
training keeps the newest frames (truncate_start). So the word must END at the end of the
file - the model learns to fire ~0.2 s after the word. Our clips are silence-trimmed already,
so no re-alignment is needed. We only: check/convert format (int16 mono 16 kHz), remove DC,
add 5 ms fades (trimmed edges click), and report durations vs clip_duration_ms.
Layout under $HK_WORK/data:
  pos_main/  positive_train minus live minus val      pos_live/ live-hektor-martyn-* (train)
  pos_val/   5 % of non-live positive_train (checkpoint selection only)
  neg_piper/ neg_other/  negative_train minus val      neg_val/ 5 % of negative_train
  test/positive_test, test/negative_test + test_index.json (group of every file)
"""
import os, sys, json, random, zipfile, shutil, collections
import numpy as np
from scipy.io import wavfile
from scipy.signal import resample_poly

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from hk_common import W, pos_group, neg_group

RAW = f"{W}/raw/micro-wake-hector"
D = f"{W}/data"
random.seed(42)


def load_manifest():
    m = json.load(open(f"{RAW}/manifest.json"))["files"]
    return {os.path.basename(k): v for k, v in m.items()}


def clean(src, dst):
    sr, a = wavfile.read(src)
    if a.ndim > 1:
        a = a[:, 0]
    if a.dtype != np.int16:
        a = a.astype(np.float32) * (32767 if np.abs(a).max() <= 1.0 else 1.0)
    a = a.astype(np.float32)
    if sr != 16000:
        a = resample_poly(a, 16000, sr)
    a -= a.mean()
    f = min(80, len(a) // 4)
    ramp = np.linspace(0, 1, f, dtype=np.float32)
    a[:f] *= ramp
    a[-f:] *= ramp[::-1]
    wavfile.write(dst, 16000, np.clip(a, -32768, 32767).astype(np.int16))
    return len(a) / 16000, sr


def main():
    if not os.path.exists(f"{W}/dataset-mww"):
        with zipfile.ZipFile(f"{RAW}/dataset.zip") as z:
            z.extractall(W)
    root = f"{W}/dataset-mww/hektor"
    man = load_manifest()
    shutil.rmtree(D, ignore_errors=True)
    for d in ["pos_main", "pos_live", "pos_val", "neg_piper", "neg_other", "neg_val",
              "test/positive_test", "test/negative_test"]:
        os.makedirs(f"{D}/{d}", exist_ok=True)
    stats, counts, srs, test_index = collections.defaultdict(list), collections.Counter(), collections.Counter(), []

    for fn in sorted(os.listdir(f"{root}/positive_train")):
        g = pos_group(fn, man.get(fn, {}))
        dst = "pos_live" if g == "live" else ("pos_val" if random.random() < 0.05 else "pos_main")
        dur, sr = clean(f"{root}/positive_train/{fn}", f"{D}/{dst}/{fn}")
        stats["positive_train/" + g].append(dur); counts[f"{dst}/{g}"] += 1; srs[sr] += 1

    for fn in sorted(os.listdir(f"{root}/negative_train")):
        g = neg_group(fn, man.get(fn, {}))
        dst = "neg_val" if random.random() < 0.05 else ("neg_piper" if g == "piper" else "neg_other")
        dur, sr = clean(f"{root}/negative_train/{fn}", f"{D}/{dst}/{fn}")
        stats["negative_train/" + g].append(dur); counts[f"{dst}/{g}"] += 1; srs[sr] += 1

    for split, label, grp in [("positive_test", 1, pos_group), ("negative_test", 0, neg_group)]:
        for fn in sorted(os.listdir(f"{root}/{split}")):
            g = grp(fn, man.get(fn, {}))
            dur, sr = clean(f"{root}/{split}/{fn}", f"{D}/test/{split}/{fn}")
            test_index.append({"path": f"{D}/test/{split}/{fn}", "label": label, "group": g, "dur": round(dur, 3)})
            stats[f"{split}/{g}"].append(dur); counts[f"{split}/{g}"] += 1; srs[sr] += 1
    json.dump(test_index, open(f"{D}/test_index.json", "w"), ensure_ascii=False)

    print("sample rates:", dict(srs))
    print("== counts ==")
    for k in sorted(counts):
        print(f"{k:32s} {counts[k]}")
    print("== durations s: n / min / p50 / p95 / p99 / max ==")
    summary = {}
    for k in sorted(stats):
        v = np.array(stats[k])
        summary[k] = [len(v)] + [round(float(x), 2) for x in (v.min(), np.median(v), np.percentile(v, 95), np.percentile(v, 99), v.max())]
        print(f"{k:32s} {summary[k]}")
    json.dump({"counts": counts, "durations": summary}, open(f"{D}/prep_summary.json", "w"), indent=1)


if __name__ == "__main__":
    main()
