"""v3 step 1: data from micro-wake-hector-v3 (same cleaning as prep_data.py).
Positives (incl. commands and 93 live) are rebuilt. Negatives: the v1 assignment
(neg_piper / neg_other / neg_val, random.seed(42) after 9065 positive draws) is replayed
from the OLD dataset.zip so the cached features_neg_*.tar can be reused; files that are new
in v3 (name or CRC differ) go to neg_add_piper / neg_add_other (training only)."""
import os, sys, json, random, zipfile, shutil, collections
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from hk_common import W, pos_group, neg_group
from prep_data import clean

RAW3, RAW1 = f"{W}/raw/micro-wake-hector-v3", f"{W}/raw/micro-wake-hector"
D = f"{W}/data"

def man(p):
    m = json.load(open(p))["files"]
    return {os.path.basename(k): v for k, v in m.items()}

def is_cmd(fn, meta):
    t = str(meta.get("text", ""))
    return "-cmd" in fn.lower() or meta.get("phrase") == "cmd" or "," in t

def kind(fn, meta):
    if is_cmd(fn, meta):
        return "cmd"
    ph = str(meta.get("phrase", ""))
    if ph in ("hektor", "hektore"):
        return ph
    t = str(meta.get("text", "")).lower()
    return "hektore" if ("тор" in t and t.rstrip("!.").endswith(("е", "э"))) else "hektor"

def pos_group3(fn, meta):
    g = pos_group(fn, meta)
    return "live-cmd" if g == "live" and is_cmd(fn, meta) else g

def zlist(zpath, split):
    z = zipfile.ZipFile(zpath)
    out = {}
    for i in z.infolist():
        parts = i.filename.split("/")
        if len(parts) >= 2 and parts[-2] == split and parts[-1].endswith(".wav"):
            out[parts[-1]] = i.CRC
    return out

def main():
    m3, m1 = man(f"{RAW3}/manifest.json"), man(f"{RAW1}/manifest.json")
    # ---- replay v1 negative assignment
    old_pos = sorted(zlist(f"{RAW1}/dataset.zip", "positive_train"))
    old_neg = zlist(f"{RAW1}/dataset.zip", "negative_train")
    random.seed(42)
    nd = 0
    for fn in old_pos:
        if pos_group(fn, m1.get(fn, {})) != "live":
            random.random(); nd += 1
    old_assign = {}
    for fn in sorted(old_neg):
        g = neg_group(fn, m1.get(fn, {}))
        old_assign[fn] = "neg_val" if random.random() < 0.05 else ("neg_piper" if g == "piper" else "neg_other")
    oc = collections.Counter(old_assign.values())
    print("replayed v1 negatives:", dict(oc), "after", nd, "positive draws (v1 log: neg_piper 7599, neg_other 2083, neg_val 490)")

    root = f"{W}/dataset-mww3"
    if not os.path.isdir(root):
        with zipfile.ZipFile(f"{RAW3}/dataset.zip") as z:
            z.extractall(W)
    root = f"{root}/hektor"
    new_neg = zlist(f"{RAW3}/dataset.zip", "negative_train")
    same = [f for f in new_neg if f in old_neg and old_neg[f] == new_neg[f]]
    changed = [f for f in new_neg if f in old_neg and old_neg[f] != new_neg[f]]
    added = [f for f in new_neg if f not in old_neg]
    removed = [f for f in old_neg if f not in new_neg]
    print(f"negative_train v1={len(old_neg)} v3={len(new_neg)}: same={len(same)} changed={len(changed)} added={len(added)} removed={len(removed)}")
    print(" added by group:", dict(collections.Counter(neg_group(f, m3.get(f, {})) for f in added + changed)))
    print(" added phrases:", collections.Counter(str(m3.get(f, {}).get("phrase")) for f in added + changed).most_common(12))
    print(" removed by group:", dict(collections.Counter(neg_group(f, m1.get(f, {})) for f in removed)),
          collections.Counter(str(m1.get(f, {}).get("phrase")) for f in removed).most_common(8))

    shutil.rmtree(D, ignore_errors=True)
    for d in ["pos_main", "pos_live", "pos_val", "neg_add_piper", "neg_add_other", "test/positive_test", "test/negative_test"]:
        os.makedirs(f"{D}/{d}", exist_ok=True)
    counts, stats, test_index = collections.Counter(), collections.defaultdict(list), []
    rng = random.Random(42)
    for fn in sorted(os.listdir(f"{root}/positive_train")):
        meta = m3.get(fn, {}); g = pos_group3(fn, meta)
        dst = "pos_live" if g in ("live", "live-cmd") else ("pos_val" if rng.random() < 0.05 else "pos_main")
        dur, _ = clean(f"{root}/positive_train/{fn}", f"{D}/{dst}/{fn}")
        counts[f"{dst}/{g}/{kind(fn, meta)}"] += 1; stats["positive_train/" + g].append(dur)
    for fn in added + changed:
        g = neg_group(fn, m3.get(fn, {}))
        dst = "neg_add_piper" if g == "piper" else "neg_add_other"
        dur, _ = clean(f"{root}/negative_train/{fn}", f"{D}/{dst}/{fn}")
        counts[f"{dst}/{g}"] += 1
    for fn in same:
        counts[f"reused:{old_assign[fn]}"] += 1
    for split, label in [("positive_test", 1), ("negative_test", 0)]:
        for fn in sorted(os.listdir(f"{root}/{split}")):
            meta = m3.get(fn, {})
            g = pos_group3(fn, meta) if label else neg_group(fn, meta)
            dur, _ = clean(f"{root}/{split}/{fn}", f"{D}/test/{split}/{fn}")
            test_index.append({"path": f"{D}/test/{split}/{fn}", "label": label, "group": g,
                               "kind": kind(fn, meta) if label else "", "phrase": str(meta.get("phrase", "")),
                               "text": str(meta.get("text", "")), "dur": round(dur, 3)})
            counts[f"{split}/{g}" + (f"/{kind(fn, meta)}" if label else "")] += 1; stats[f"{split}/{g}"].append(dur)
    json.dump(test_index, open(f"{D}/test_index.json", "w"), ensure_ascii=False)
    summ = {"replay_v1": dict(oc), "neg_diff": {"same": len(same), "changed": len(changed), "added": len(added),
            "removed": len(removed), "removed_files": removed}, "counts": counts,
            "durations": {k: [len(v)] + [round(float(x), 2) for x in (min(v), np.median(v), np.percentile(v, 99), max(v))] for k, v in stats.items()}}
    json.dump(summ, open(f"{D}/prep_summary.json", "w"), ensure_ascii=False, indent=1)
    for k in sorted(counts):
        print(f"{k:44s} {counts[k]}")
    for k in sorted(summ["durations"]):
        print(f"{k:32s} {summ['durations'][k]}")
    print("PREP_DONE")

if __name__ == "__main__":
    main()
