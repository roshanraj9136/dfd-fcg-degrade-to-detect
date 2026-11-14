"""Run a GenD model on every prepared face clip and keep its per-frame class tokens and logits, so ensembles, probes
and thresholds can be fitted and tested offline without the GPU.

  new:  D:\\dfd-data\\crops\\*.npz      (fine-tuning sets: train / val / test / test_open; every face; clean)
  rep:  D:\\dfd-data\\builds\\test\\*.npz (the 225 reported test videos; clean / GB_5 / VC_5; research crops)

Frames: the distinct frames the DFD-FCG clips use, at most MAX_FRAMES of them, evenly spaced.
    python service/ens/score_gend.py <clip|pe|dino> <new|rep> <out.npz>
Output: keys ("vid|cond|face"), start (offset of each key's frames), feats (N,1024 fp16), logit (N,) fake-minus-real.
"""
import csv
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "service" / "ens")]
import numpy as np  # noqa: E402

from gend import Scorer  # noqa: E402

NAME, WHICH, OUT = sys.argv[1], sys.argv[2], Path(sys.argv[3])
MAX_FRAMES = int(sys.argv[4]) if len(sys.argv) > 4 else 40
CROPS, BUILDS = Path(r"D:\dfd-data\crops"), Path(r"D:\dfd-data\builds\test")
SETS = [Path(r"D:\dfd-data\new\sets.csv"), Path(r"D:\dfd-data\new\sets_gated.csv")]
SCORE_SPLITS = {s.strip() for s in os.environ.get("SCORE_SPLITS", "").split(",") if s.strip()}


def pick(clip_idx, n=MAX_FRAMES):
    u = np.unique(np.asarray(clip_idx).ravel())
    if len(u) <= n:
        return u
    return u[np.linspace(0, len(u) - 1, n).round().astype(int)]


def items():
    if WHICH == "new":
        seen = set()
        for f in SETS:
            for r in csv.DictReader(open(f)):
                if SCORE_SPLITS and r["set"] not in SCORE_SPLITS:
                    continue
                p = CROPS / (r["vid"].replace("/", "__") + ".npz")
                if r["vid"] in seen or not p.exists():
                    continue
                seen.add(r["vid"])
                z = np.load(p)
                fr = pick(z["clip_idx"])
                for face in sorted(k for k in z.files if k.startswith("face")):
                    yield f"{r['vid']}|clean|{face[4:]}", z[face][fr]
    else:
        builds = BUILDS if WHICH == "rep" else BUILDS.parent / "fit"
        for p in sorted(builds.glob("*.npz")):
            z = np.load(p)
            vid = str(z["vid"])
            fr = pick(z["clip_idx"])
            for c in z["conds"]:
                yield f"{vid}|{c}|0", z[f"c:{c}"][fr]


def main():
    sc = Scorer(NAME)
    if os.environ.get("GEND_LN"):                     # fine-tuned LayerNorm + head weights (train_gend.py)
        import torch
        sd = torch.load(os.environ["GEND_LN"], map_location="cpu")
        own = sc.model.state_dict()
        sc.model.load_state_dict({**own, **{k: v.to(own[k].dtype) for k, v in sd.items()}})
        print("loaded fine-tuned weights", os.environ["GEND_LN"], len(sd), "tensors", flush=True)
    keys, start, feats, logits = [], [], [], []
    n, t0 = 0, time.time()
    for k, (key, frames) in enumerate(items()):
        f, lg = sc.features(frames)
        keys.append(key)
        start.append(n)
        n += len(lg)
        feats.append(f.astype(np.float16))
        logits.append(lg.astype(np.float32))
        if (k + 1) % 100 == 0:
            print(f"{NAME}/{WHICH}: {k + 1} items, {n} frames, {n / (time.time() - t0):.1f} frames/s", flush=True)
    np.savez(OUT, keys=np.array(keys), start=np.array(start + [n]), feats=np.concatenate(feats),
             logit=np.concatenate(logits))
    print(f"finished: {len(keys)} items, {n} frames, {(time.time() - t0) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
