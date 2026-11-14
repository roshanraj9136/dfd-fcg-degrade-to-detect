"""The live checker's GenD (service/app/gend_live.py, built from the architecture only) must give the same frame logits
as the evaluation code (service/ens/gend.py, the authors' modelling code). CPU, fp32, a few frames of one video.
    python tests/check_gend_live.py [clip|pe]
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "service" / "app"), str(ROOT / "service" / "ens")]
import numpy as np  # noqa: E402
import torch  # noqa: E402

import gend  # noqa: E402
import gend_live  # noqa: E402

kind = sys.argv[1] if len(sys.argv) > 1 else "clip"
torch.set_num_threads(4)
_load = gend.load
gend.load = lambda name, device="cuda", dtype=None: _load(name, device, torch.float32)   # full-precision reference
z = np.load(r"D:\dfd-data\crops\FF_DF__325_492.npz")
crops, clip_idx = z["face0"], z["clip_idx"][:1]
ref = gend.Scorer(kind, device="cpu")
ref.model.float()
fr = crops[np.unique(clip_idx.ravel())]
_, lg_ref = ref.features(fr)
del ref
live = gend_live.GenDMember(dict(backbone=kind, frames=24), device="cpu")
lg_live, _ = live.frame_scores(crops, clip_idx)
print("ref ", np.round(lg_ref, 4))
print("live", np.round(lg_live, 4))
d = float(np.abs(lg_ref - lg_live).max())
print("max abs diff", d)
assert d < 1e-3, "live GenD differs from the evaluation code"
print("OK")
