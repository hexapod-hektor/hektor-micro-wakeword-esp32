"""v3 step 2: features only for what changed (positives, live x25, negative_test, new negatives).
neg_other training/validation and neg_piper are the cached v1 tars (hard-linked)."""
import os, sys, shutil, subprocess
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gen_features as g
from gen_features import write_set, F, D

def main():
    P, N = (0.195, 0.205), (0.0, 0.6)
    os.makedirs(f"{F}/neg_other_v3", exist_ok=True)
    for sp in ("training", "validation"):
        if not os.path.exists(f"{F}/neg_other_v3/{sp}"):
            subprocess.run(["cp", "-al", f"{F}/neg_other/{sp}", f"{F}/neg_other_v3/{sp}"], check=True)
    sets = [("pos_live_v3", f"{D}/pos_live", "training", 25, 10, 210, P),
            ("pos_main_v3", f"{D}/pos_val", "validation", 1, 10, 210, P),
            ("neg_other_v3", f"{D}/test/negative_test", "testing", 1, None, None, N),
            ("pos_main_v3", f"{D}/test/positive_test", "testing", 1, 1, None, P)]
    for nm in ("neg_add_piper", "neg_add_other"):
        if os.listdir(f"{D}/{nm}"):
            sets.append((nm, f"{D}/{nm}", "training", 2, None, None, N))
    sets.append(("pos_main_v3", f"{D}/pos_main", "training", 2, 10, 210, P))
    for name, wd, split, rep, slide, keep, jit in sets:
        write_set(name, wd, split, rep, slide, keep, jit)
    print("GEN_DONE", flush=True)

if __name__ == "__main__":
    main()
