"""v3 step 4: evaluate.py logic (device-like streaming int8, ring-buffer average) on the v3 test set.
Groups: live, live-cmd (separately), tts, piper, firered, aug-tempo, robot; kinds Гектор/Гекторе/команда;
FA on negative_test (+ similar words гектар/вектор, reference only); FA/h on DiPCo (5.33 h) and on the
room testing_ambient (files 2,3,5,9,10). Recommended cutoff (window 5): best live recall with
DiPCo <= 2 FA/h and room <= 0.5 FA/h (tie: overall recall, then fewer DiPCo FA)."""
import os, sys, json, argparse, time, collections
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "train"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from hk_common import W
import evaluate as E

CUT, WIN = E.CUTOFFS, E.WINDOWS

def rate(v, n, c):
    return round(float(np.mean([x["max"][str(n)] >= c for x in v])), 4)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tflite", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    rng = np.random.default_rng(0)
    idx = json.load(open(f"{W}/data/test_index.json"))
    rows, t0 = [], time.time()
    for k, r in enumerate(idx):
        p = E.clip_probs(r["path"], a.tflite, rng)
        rows.append({**r, "max": {str(n): float(E.avg_curve(p, n).max()) for n in WIN}})
        if k % 1000 == 0:
            print(f" {k}/{len(idx)} {time.time()-t0:.0f}s", flush=True)
    pos, neg = collections.defaultdict(list), collections.defaultdict(list)
    for r in rows:
        if r["label"] == 1:
            pos["ALL"].append(r); pos[r["group"]].append(r)
            if r["group"] in ("live", "live-cmd"):
                pos["live+cmd"].append(r)
            else:
                pos["kind:" + r["kind"]].append(r)
        else:
            neg["ALL"].append(r); neg[r["group"]].append(r)
            if r["phrase"] in ("hektar", "vektor"):
                neg["sim:" + r["phrase"]].append(r)
    tab = lambda g: {k: {"n": len(v), **{f"w{n}": {str(c): rate(v, n, c) for c in CUT} for n in WIN}} for k, v in sorted(g.items())}
    pos, neg = tab(pos), tab(neg)
    dh, dt = E.ambient_triggers(a.tflite)
    rh, rt = E.ambient_triggers(a.tflite, f"{W}/features/room_ambient/testing_ambient/*mmap")
    dip = {f"w{n}": {str(c): round(dt[(n, c)] / dh, 3) for c in CUT} for n in WIN}
    room = {f"w{n}": {str(c): round(rt[(n, c)] / rh, 3) for c in CUT} for n in WIN}
    live = pos.get("live", pos["ALL"])
    best = None
    for dlim, rlim in ((2.0, 0.5), (2.0, 1e9), (1e9, 1e9)):
        for c in CUT:
            if dip["w5"][str(c)] <= dlim and room["w5"][str(c)] <= rlim:
                key = (live["w5"][str(c)], pos["ALL"]["w5"][str(c)], -dip["w5"][str(c)], c)
                if best is None or key > best[0]:
                    best = (key, c, dlim, rlim)
        if best:
            break
    c80 = [c for c in CUT if live["w5"][str(c)] >= 0.8]
    chk = {"live>=0.8_max_cutoff_w5": max(c80) if c80 else None,
           "dipco_faph_there": dip["w5"][str(max(c80))] if c80 else None}
    chk["v3b_needed"] = bool(c80) and chk["dipco_faph_there"] > 2.0
    rc = best[1]
    chk["live_cmd_recall_at_recommended"] = pos.get("live-cmd", {}).get("w5", {}).get(str(rc))
    chk["live_cmd_below_50"] = chk["live_cmd_recall_at_recommended"] is not None and chk["live_cmd_recall_at_recommended"] < 0.5
    tech = E.tech_info(a.tflite)
    res = {"model": a.tflite, "size_bytes": os.path.getsize(a.tflite),
           "recall_positive_test": pos, "false_accept_rate_negative_test": neg,
           "dipco_hours": round(dh, 3), "dipco_fa_per_hour": dip, "room_hours": round(rh, 3), "room_fa_per_hour": room,
           "ambient_hours": round(rh, 3), "ambient_fa_per_hour": room, "ambient_set": "room_testing_ambient",
           "recommended": {"sliding_window_size": 5, "probability_cutoff": rc, "dipco_limit": best[2], "room_limit": best[3]},
           "checks": chk, "tech": tech, "per_file": rows}
    json.dump(res, open(a.out, "w"), ensure_ascii=False, indent=1)
    print(f"\nmodel {os.path.getsize(a.tflite)/1024:.1f} KB; DiPCo {dh:.2f} h; room {rh:.3f} h")
    for n in WIN:
        print(f"\n--- sliding window {n} --- cutoffs {CUT}")
        print(f"{'DiPCo FA/h':26s}       ", [dip[f'w{n}'][str(c)] for c in CUT])
        print(f"{'room FA/h':26s}       ", [room[f'w{n}'][str(c)] for c in CUT])
        for k, v in pos.items():
            print(f"{'recall '+k:26s} n={v['n']:4d}", [v[f'w{n}'][str(c)] for c in CUT])
        for k, v in neg.items():
            print(f"{'neg-test FA '+k:26s} n={v['n']:4d}", [v[f'w{n}'][str(c)] for c in CUT])
    print("\nrecommended:", res["recommended"]); print("checks:", chk)
    print("tech:", json.dumps(tech)); print("EVAL_DONE")

if __name__ == "__main__":
    main()
