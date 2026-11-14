"""Smoke test: one GenD model on a few prepared videos; prints per-video mean P(fake) and frames/s."""
import csv
import sys
import time
from pathlib import Path

sys.path[:0] = [str(Path(__file__).resolve().parent)]
import numpy as np  # noqa: E402
import torch  # noqa: E402

from gend import Scorer  # noqa: E402

name = sys.argv[1]
t = time.time()
sc = Scorer(name)
print(f"loaded {name} in {time.time() - t:.0f}s, GPU mem {torch.cuda.memory_allocated() / 1e9:.2f} GB", flush=True)
rows = [r for r in csv.DictReader(open(r"D:\dfd-data\new\sets.csv")) if r["set"] == "val"]
by = {}
for r in rows:
    by.setdefault((r["source"], r["label"]), []).append(r)
for key in sorted(by):
    for r in by[key][:3]:
        p = Path(r"D:\dfd-data\crops") / (r["vid"].replace("/", "__") + ".npz")
        if not p.exists():
            continue
        z = np.load(p)
        fr = np.unique(z["clip_idx"].ravel())
        t = time.time()
        _, lg = sc.features(z["face0"][fr])
        torch.cuda.synchronize()
        dt = time.time() - t
        print(f"{key[0]:8s} label={key[1]} p_fake={np.mean(1 / (1 + np.exp(-lg))):.3f}  {len(fr) / dt:5.1f} fr/s  {r['vid']}",
              flush=True)
print(f"peak GPU mem {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")
