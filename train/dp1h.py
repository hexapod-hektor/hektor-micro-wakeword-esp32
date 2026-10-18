"""dinner_party_eval_1h as in v1 (OOM workaround): validation_ambient = one track cut to 1 h,
testing_ambient = full DiPCo (hard links)."""
import os, sys, glob, subprocess, shutil
import numpy as np
from mmap_ninja.ragged import RaggedMmap
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from hk_common import W
NEG = f"{W}/negative_datasets"
src, dst = f"{NEG}/dinner_party_eval", f"{NEG}/dinner_party_eval_1h"
for d in sorted(glob.glob(f"{src}/*/*")):
    r = RaggedMmap(d); print(d, len(r), [r[i].shape for i in range(min(len(r), 6))], r[0].dtype)
if not os.path.exists(f"{dst}/.done"):
    shutil.rmtree(dst, ignore_errors=True)
    os.makedirs(f"{dst}/validation_ambient")
    subprocess.run(["cp", "-al", f"{src}/testing_ambient", f"{dst}/testing_ambient"], check=True)
    vd = sorted(glob.glob(f"{src}/validation_ambient/*mmap"))[0]
    r = RaggedMmap(vd)
    spec = np.asarray(r[int(np.argmax([r[i].shape[0] for i in range(len(r))]))])[:360000]
    RaggedMmap.from_generator(out_dir=f"{dst}/validation_ambient/dp1h_mmap", sample_generator=iter([spec]), batch_size=1, verbose=False)
    open(f"{dst}/.done", "w").write(str(spec.shape))
r = RaggedMmap(f"{dst}/validation_ambient/dp1h_mmap"); print("1h:", r[0].shape, r[0].dtype, float(r[0].shape[0]) / 360000, "h")
for sp in ("validation_ambient", "testing_ambient"):
    for d in glob.glob(f"{W}/features/room_ambient/{sp}/*mmap"):
        r = RaggedMmap(d); print("room", sp, len(r), [r[i].shape[0] for i in range(len(r))], r[0].dtype, round(sum(r[i].shape[0] for i in range(len(r))) / 6000, 2), "min")
print("DP1H_DONE")
