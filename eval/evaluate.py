"""Step 4: evaluate the quantised streaming model the way the device runs it.

Per test clip: [1.0 s -60 dBFS noise] + clip + [0.5 s noise] -> micro-frontend (pymicro_features:
40 mel bins, 30 ms window, 10 ms step) -> streaming tflite, 3 new frames per call (= one inference
every 30 ms), fresh state per clip -> sliding average of the last N probabilities (zero-initialised
ring buffer like ESPHome) -> detected if max(avg) >= cutoff.
Ambient FA/h: kahrendt dinner_party_eval/testing_ambient (DiPCo; the set mWW's own test uses),
streamed track by track; one false accept per rising edge of avg>=cutoff, 1 s refractory.
"""
import os, sys, json, glob, argparse, time, collections
import numpy as np
from scipy.io import wavfile

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "train"))
from hk_common import W

CUTOFFS = [0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 0.97]
WINDOWS = [5, 10]


def new_model(path):
    from microwakeword.inference import Model
    return Model(path, stride=3)


def avg_curve(p, n):
    p = np.asarray(p, dtype=np.float32)
    if len(p) == 0:
        return np.zeros(1, np.float32)
    c = np.cumsum(np.insert(p, 0, 0))
    idx = np.arange(len(p))
    lo = np.maximum(0, idx + 1 - n)
    return (c[idx + 1] - c[lo]) / n


def clip_probs(path, tflite, rng):
    from microwakeword.audio.audio_utils import generate_features_for_clip
    sr, a = wavfile.read(path)
    a = a.astype(np.float32)
    floor = lambda k: rng.normal(0, 32768 * 10 ** (-60 / 20), k).astype(np.float32)
    x = np.clip(np.concatenate([floor(16000), a + floor(len(a)), floor(8000)]), -32768, 32767).astype(np.int16)
    return new_model(tflite).predict_spectrogram(generate_features_for_clip(x, step_ms=10))


def ambient_triggers(tflite, pattern=None):
    from mmap_ninja.ragged import RaggedMmap
    dirs = sorted(glob.glob(pattern or f"{W}/negative_datasets/dinner_party_eval/testing_ambient/*mmap"))
    hours, trig = 0.0, {(n, c): 0 for n in WINDOWS for c in CUTOFFS}
    for d in dirs:
        for spec in RaggedMmap(d):
            p = new_model(tflite).predict_spectrogram(np.asarray(spec))
            hours += len(spec) * 0.01 / 3600
            for n in WINDOWS:
                av = avg_curve(p, n)
                for c in CUTOFFS:
                    above = av >= c
                    last = -10 ** 9
                    for i in np.flatnonzero(above & ~np.r_[False, above[:-1]]):
                        if i - last >= 34:
                            trig[(n, c)] += 1
                            last = i
    return hours, trig


def tech_info(tflite):
    from ai_edge_litert.interpreter import Interpreter, OpResolverType
    it = Interpreter(model_path=tflite, experimental_op_resolver_type=OpResolverType.BUILTIN_WITHOUT_DEFAULT_DELEGATES)
    it.allocate_tensors()
    T = {t["index"]: t for t in it.get_tensor_details()}
    macs, ops = 0, collections.Counter()
    for op in it._get_ops_details():
        ops[op["op_name"]] += 1
        ins, outs = op["inputs"], op["outputs"]
        if not len(outs) or outs[0] < 0:
            continue
        out = int(np.prod(T[outs[0]]["shape"]))
        if op["op_name"] == "CONV_2D":
            macs += out * int(np.prod(T[ins[1]]["shape"][1:]))
        elif op["op_name"] == "DEPTHWISE_CONV_2D":
            w = T[ins[1]]["shape"]; macs += out * int(w[1] * w[2])
        elif op["op_name"] == "FULLY_CONNECTED":
            macs += out * int(T[ins[1]]["shape"][-1])
    arena = None
    try:
        from tflite_micro.python.tflite_micro import runtime
        lo, hi = 1024, 200000
        while hi - lo > 64:
            mid = (lo + hi) // 2
            try:
                runtime.Interpreter.from_file(tflite, arena_size=mid); hi = mid
            except Exception:
                lo = mid
        arena = hi
    except Exception as e:
        print("tflite-micro arena probe unavailable:", e)
    inp, outp = it.get_input_details()[0], it.get_output_details()[0]
    t0 = time.perf_counter()
    for _ in range(2000):
        it.set_tensor(inp["index"], np.zeros(inp["shape"], inp["dtype"])); it.invoke()
    us = (time.perf_counter() - t0) / 2000 * 1e6
    q = lambda d: dict(shape=[int(s) for s in d["shape"]], dtype=str(np.dtype(d["dtype"])),
                       scale=float(d["quantization"][0]), zero_point=int(d["quantization"][1]))
    return dict(macs_per_inference=int(macs), ops=dict(ops), tflm_arena_min_bytes=arena, input=q(inp), output=q(outp),
                host_cpu_us_per_inference=round(us, 1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tflite", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--room", action="store_true", help="choose cutoff by FA/h on room testing_ambient")
    a = ap.parse_args()
    rng = np.random.default_rng(0)
    idx = json.load(open(f"{W}/data/test_index.json"))
    rows, t0 = [], time.time()
    for k, r in enumerate(idx):
        p = clip_probs(r["path"], a.tflite, rng)
        rows.append({**r, "max": {str(n): float(avg_curve(p, n).max()) for n in WINDOWS}})
        if k % 1000 == 0:
            print(f"  {k}/{len(idx)} {time.time()-t0:.0f}s", flush=True)

    def table(label):
        g = collections.defaultdict(list)
        for r in rows:
            if r["label"] == label:
                g[r["group"]].append(r); g["ALL"].append(r)
        return {k: {"n": len(v), **{f"w{n}": {str(c): round(float(np.mean([x["max"][str(n)] >= c for x in v])), 4)
                                            for c in CUTOFFS} for n in WINDOWS}} for k, v in sorted(g.items())}

    pos, neg = table(1), table(0)
    dip_hours, dip_trig = ambient_triggers(a.tflite)
    dip_faph = {f"w{n}": {str(c): round(dip_trig[(n, c)] / dip_hours, 3) for c in CUTOFFS} for n in WINDOWS}
    if a.room:
        hours, trig = ambient_triggers(a.tflite, f"{W}/features/room_ambient/testing_ambient/*mmap")
    else:
        hours, trig = dip_hours, dip_trig
    faph = {f"w{n}": {str(c): round(trig[(n, c)] / hours, 3) for c in CUTOFFS} for n in WINDOWS}
    tech = tech_info(a.tflite)
    best = None
    for lim in (0.5, 1.0, 1e9):  # best live recall (tie: overall recall) with ambient FA/h <= lim
        for n in WINDOWS:
            for c in CUTOFFS:
                if trig[(n, c)] / hours <= lim:
                    key = (pos.get("live", pos["ALL"])[f"w{n}"][str(c)], pos["ALL"][f"w{n}"][str(c)], -trig[(n, c)], c)
                    if best is None or key > best[0]:
                        best = (key, n, c, lim)
        if best:
            break
    res = {"model": a.tflite, "size_bytes": os.path.getsize(a.tflite), "ambient_hours": round(hours, 3),
           "recall_positive_test": pos, "false_accept_rate_negative_test": neg, "ambient_fa_per_hour": faph, "ambient_set": "room_testing_ambient" if a.room else "dipco", "dipco_hours": round(dip_hours, 3), "dipco_fa_per_hour": dip_faph,
           "recommended": {"sliding_window_size": best[1], "probability_cutoff": best[2], "fa_limit_per_hour": best[3]},
           "tech": tech, "per_file": rows}
    json.dump(res, open(a.out, "w"), ensure_ascii=False, indent=1)
    print(f"\nmodel {os.path.getsize(a.tflite)/1024:.1f} KB; ambient {hours:.2f} h")
    for n in WINDOWS:
        print(f"\n--- sliding window {n} --- cutoffs {CUTOFFS}")
        print(f"{'ambient FA/h':24s}       ", [faph[f'w{n}'][str(c)] for c in CUTOFFS])
        print(f"{'DiPCo FA/h':24s}       ", [dip_faph[f'w{n}'][str(c)] for c in CUTOFFS])
        for k, v in pos.items():
            print(f"{'recall '+k:24s} n={v['n']:4d}", [v[f'w{n}'][str(c)] for c in CUTOFFS])
        for k, v in neg.items():
            print(f"{'neg-test FA '+k:24s} n={v['n']:4d}", [v[f'w{n}'][str(c)] for c in CUTOFFS])
    print("\nrecommended:", res["recommended"])
    print("tech:", json.dumps(tech))


if __name__ == "__main__":
    main()
