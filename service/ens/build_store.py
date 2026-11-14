"""Pack the aligned face frames of the train / val videos (prep_crops.py, main face, 40 frames = 4 clips x 10) into one
uint8 memory-mapped array, so training reads single frames at random without opening a 2.7 MB file each time.

DFDC is left out (its parts can share actors with the reported DFDC test part), as in service/train/train.py.
    python service/ens/build_store.py <out dir>
Output: frames.u8 (N_videos, 40, 150, 150, 3), index.csv (row, vid, set, source, method, label, n_frames)
"""
import csv
import os
import sys
from pathlib import Path

import numpy as np

OUT = Path(sys.argv[1])
OUT.mkdir(parents=True, exist_ok=True)
CROPS = Path(r"D:\dfd-data\crops")
SETS = [Path(p) for p in os.environ.get("STORE_SETS", r"D:\dfd-data\new\sets.csv;D:\dfd-data\new\sets_gated.csv").split(";")]
F = 40

rows, seen = [], set()
for f in SETS:
    for r in csv.DictReader(open(f)):
        p = CROPS / (r["vid"].replace("/", "__") + ".npz")
        if r["set"] in ("train", "val") and r["source"] != "dfdc" and r["vid"] not in seen and p.exists():
            seen.add(r["vid"])
            rows.append(dict(r, npz=p))
mm = np.lib.format.open_memmap(OUT / "frames.npy", mode="w+", dtype=np.uint8, shape=(len(rows), F, 150, 150, 3))
with open(OUT / "index.csv", "w", newline="") as fh:
    w = csv.writer(fh)
    w.writerow(["row", "vid", "set", "source", "method", "label", "n_frames"])
    for i, r in enumerate(rows):
        z = np.load(r["npz"])
        fr = np.unique(z["clip_idx"].ravel())
        x = z["face0"][fr][:F]
        mm[i, :len(x)] = x
        w.writerow([i, r["vid"], r["set"], r["source"], r["method"], r["label"], len(x)])
        if (i + 1) % 200 == 0:
            print(f"{i + 1}/{len(rows)}", flush=True)
mm.flush()
print("finished", len(rows), "videos", flush=True)
