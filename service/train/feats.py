"""Run one DFD-FCG checkpoint on cached crops (cache_builds.py): head logits d and features y_s, y_t per clip and
condition, fp16 on the GPU (as the research pipeline). The output has the keys fit_readout.py reads.
    python service/train/feats.py <checkpoint> <builds dir> <out.npz>
"""
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CK, BUILDS, OUT = sys.argv[1], Path(sys.argv[2]), Path(sys.argv[3])
sys.path[:0] = [str(ROOT)]
import numpy as np  # noqa: E402
import torch  # noqa: E402

import qa.engine as E  # noqa: E402

E.CKPT = Path(CK)
model = E.load_model(device="cuda")
rec = E.Recorder(model)
T = {k: [] for k in ("vid", "method", "clip", "cond", "label", "d", "y_s", "y_t")}
files = sorted(f for f in BUILDS.glob("*.npz") if not f.name.endswith(".tmp.npz"))
CLIPS = [int(c) for c in os.environ["CLIPS"].split(",")] if os.environ.get("CLIPS") else None   # time limit: a subset of the clips
for k, f in enumerate(files):
    z = np.load(f)
    for c in z["conds"]:
        crops = z[f"c:{c}"]
        x = torch.stack([model.transform(torch.from_numpy(np.ascontiguousarray(crops[ci])).permute(0, 3, 1, 2).contiguous())
                         for ci in (z["clip_idx"][CLIPS] if CLIPS else z["clip_idx"])])
        with torch.inference_mode():
            res = E.analyse(model, rec, x, fp16=True)
        for ci, q in enumerate(res):
            for key, v in (("vid", str(z["vid"])), ("method", str(z["method"])), ("clip", ci), ("cond", str(c)),
                           ("label", int(z["label"])), ("d", q["d"]), ("y_s", q["y_s"]), ("y_t", q["y_t"])):
                T[key].append(v)
    if (k + 1) % 25 == 0:
        print(f"{k + 1}/{len(files)}", flush=True)
np.savez_compressed(OUT, **{k: np.array(v) for k, v in T.items()})
print("wrote", OUT, len(T["vid"]), "rows", flush=True)
