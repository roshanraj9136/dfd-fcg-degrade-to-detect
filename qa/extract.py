"""
Run DFD-FCG with exact attribution on every (video, condition) and cache the results.

One process owns the GPU and alternates between face alignment (qa.prep) and DFD-FCG
inference, so the two models never compete for the 4 GB card. CPU workers build the
perturbed / re-encoded face crops in parallel. Results are stored per video in
data/feats/<vid>.pkl; the run is resumable and only missing work is done.

    python -m qa.extract                         # whole manifest, priority order
    python -m qa.extract --splits test --methods REAL DF
"""
import argparse
import pickle
import time
from collections import deque
from multiprocessing import Pool

import numpy as np
import torch

from qa.conditions import build
from qa.config import DATA, conditions, p
from qa.manifest import local_video, read_manifest
from qa.prep import safe
from qa.quality import background_features, face_mask, face_scale_of, frame_features, landmarks_in_roi


def feats_path(vid):
    return DATA / "feats" / (safe(vid) + ".pkl")


def load_feats(vid):
    f = feats_path(vid)
    if f.exists():
        with open(f, "rb") as fh:
            return pickle.load(fh)
    return {}


PROBES_RUN = ["pB2", "pJ50"]       # test-time probes evaluated for every input condition


def wanted_conditions(row, clean_only_methods=("FSh", "DFDC_R", "DFDC_F")):
    base = ["clean"] if row["method"] in clean_only_methods else \
        conditions("fit" if row["split"] == "fit" else "test")
    return base + [f"{c}+{p}" for c in base for p in PROBES_RUN]


def _work(args):
    row, conds = args
    prep = dict(np.load(DATA / "prep" / (safe(row["vid"]) + ".npz")))
    built = build(local_video(row), prep, row["vid"], conds)
    return row["vid"], summarise_built(prep, built)


def summarise_built(prep, built):
    """Crops + per-clip label-free quality descriptors (face crop and background-referenced)."""
    needed = list(prep["needed"])
    pos = {f: i for i, f in enumerate(needed)}
    scales = [face_scale_of(M) for M in prep["M"]]
    masks = None
    out = {}
    for cond, (crops, grays, rois) in built.items():
        if masks is None:
            masks = [~face_mask(rois[i].shape, landmarks_in_roi(prep["crop_lm"][i], prep["M"][i],
                                                                prep["off"][i], prep["roi"]))
                     for i in range(len(needed))]
        fq = np.stack([frame_features(crops[i], grays[i], scales[i]) for i in range(len(needed))])
        fb = np.stack([background_features(rois[i], masks[i], scales[i]) for i in range(len(needed))])
        clip_idx = np.array([[pos[f] for f in clip] for clip in prep["layout"]])   # (K,10)
        out[cond] = (crops, clip_idx, fq[clip_idx].mean(1), fb[clip_idx].mean(1))
    return out


def _low_priority():
    """Pool initializer: CPU workers yield to the process that drives the GPU."""
    try:
        import ctypes
        ctypes.windll.kernel32.SetPriorityClass(ctypes.windll.kernel32.GetCurrentProcess(), 0x4000)
    except Exception:  # noqa: BLE001  (non-Windows)
        import os
        os.nice(5)
    import cv2
    cv2.setNumThreads(1)


def priority_order(rows):
    """Interleave the groups so that a balanced subset of every group is finished early
    (partial results stay usable while the run continues). FaceShifter goes last."""
    groups = {}
    for r in rows:
        groups.setdefault((r["split"], r["method"]), []).append(r)
    keyed = []
    for g, rs in groups.items():
        for i, r in enumerate(rs):
            late = r["method"] in ("FSh", "DFDC_R", "DFDC_F")
            keyed.append(((late, r["split"] != "demo", (i + 0.5) / len(rs)), r))
    return [r for _, r in sorted(keyed, key=lambda x: x[0])]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", nargs="+", default=["demo", "test", "fit", "cross"])
    ap.add_argument("--methods", nargs="+", default=None)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--fp32", action="store_true")
    ap.add_argument("--no-prep", action="store_true",
                    help="only run inference on already-aligned videos (face model not loaded)")
    args = ap.parse_args()
    try:                                     # the GPU-feeding process gets CPU priority
        import ctypes
        ctypes.windll.kernel32.SetPriorityClass(ctypes.windll.kernel32.GetCurrentProcess(), 0x8000)
    except Exception:  # noqa: BLE001
        pass

    rows = [r for r in read_manifest() if r["split"] in args.splits
            and (args.methods is None or r["method"] in args.methods)]
    rows = priority_order(rows)
    todo = []
    for r in rows:
        if (DATA / "prep" / (safe(r["vid"]) + ".fail")).exists():
            continue
        if args.no_prep and not (DATA / "prep" / (safe(r["vid"]) + ".npz")).exists():
            continue
        miss = [c for c in wanted_conditions(r) if c not in load_feats(r["vid"])]
        if miss:
            todo.append((r, miss))
    if args.limit:
        todo = todo[:args.limit]
    print(f"{len(todo)} videos to process", flush=True)
    if not todo:
        return

    from qa.engine import Recorder, analyse, load_model
    from qa.prep import prep_one

    torch.backends.cudnn.benchmark = True
    fa = None
    if not args.no_prep:
        # NOTE: on a 4 GB WDDM card, sharing the GPU with the face model pushes DFD-FCG's
        # allocations into system memory (3x slower). qa.run_pipeline alternates processes instead.
        import face_alignment
        fa = face_alignment.FaceAlignment(face_alignment.LandmarksType.TWO_D, face_detector="sfd",
                                          dtype=torch.float16, flip_input=False, device="cuda")
        fa.max_batch_size = 32
    model = load_model()
    rec = Recorder(model)
    tf = model.transform

    def infer(vid, out):
        store = load_feats(vid)
        t_tf = t_an = 0.0
        tim = {}
        for cond, (crops, clip_idx, q, qb) in out.items():
            a = time.time()
            fr = torch.from_numpy(crops).permute(0, 3, 1, 2).contiguous()
            clips = torch.stack([tf(fr[ci]) for ci in clip_idx])
            b = time.time()
            store[cond] = dict(clips=analyse(model, rec, clips, fp16=not args.fp32, timing=tim), q=q, q_bg=qb)
            t_tf += b - a
            t_an += time.time() - b
        with open(p("feats", safe(vid) + ".pkl"), "wb") as fh:
            pickle.dump(store, fh)
        print(f"      transform {t_tf:.1f}s  analyse {t_an:.1f}s (forward {tim.get('forward', 0):.1f}s)",
              flush=True)
        return store

    t0, done = time.time(), 0
    pending = deque()
    with Pool(args.workers, initializer=_low_priority) as pool:
        def drain(block):
            nonlocal done
            while pending and (block or pending[0].ready() or len(pending) > 4):
                vid, out = pending.popleft().get()
                ti = time.time()
                store = infer(vid, out)
                done += 1
                pf = {c: round(float(np.mean([r["p_final"] for r in store[c]["clips"]])), 3)
                      for c in ["clean", "RE_0", "GB_3", "VC_3", "JPEG_3", "GB_4", "VC_4"] if c in store}
                print(f"[{done}/{len(todo)}] {vid}: {len(out)} conds, infer {time.time() - ti:.0f}s, "
                      f"{(time.time() - t0) / done:.0f}s/video  {pf}", flush=True)

        for row, miss in todo:
            if not (DATA / "prep" / (safe(row["vid"]) + ".npz")).exists():
                s = prep_one(fa, row)
                print(f"   prep {row['vid']}: {s}", flush=True)
                if not s.startswith("ok"):
                    continue
            pending.append(pool.apply_async(_work, ((row, miss),)))
            drain(block=False)
        drain(block=True)
    print(f"finished in {(time.time() - t0) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
