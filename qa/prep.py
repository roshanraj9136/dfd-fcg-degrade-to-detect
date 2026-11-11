"""
Face alignment, reproducing the upstream DFD-FCG preprocessing
(src/preprocess/fetch_landmark_bbox.py + crop_main_face.py) on the first
MAX_FRAMES frames of each video.

For every video we store the per-frame affine transform that maps the
full-resolution frame onto the 150x150 aligned face crop the model sees.
Perturbed versions of the video are then cropped with exactly the same
geometry (the authors' robustness protocol also re-uses the clean landmarks).

    python -m qa.prep
"""
import json
import math
import os
import sys
import time

import cv2
import numpy as np
import torch

from qa.config import (CLIPS_PER_VIDEO, CROP_SIZE, DATA, MAX_FRAMES, NUM_FRAMES,
                       SMOOTH_MARGIN, STRIDE_SEC, TARGET_SIZE, UPSTREAM, p)
from qa.manifest import local_video, read_manifest

sys.path.insert(0, str(UPSTREAM))
from src.preprocess.crop_main_face import get_main_face_data  # noqa: E402

STABLE_POINTS = (28, 33, 36, 39, 42, 45, 48, 54)
MEAN_FACE = np.load(UPSTREAM / "misc" / "20words_mean_face.npy")


def safe(vid):
    return vid.replace("/", "__")


def read_frames(path, max_frames=MAX_FRAMES):
    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    frames = []
    while len(frames) < max_frames:
        ok, fr = cap.read()
        if not ok:
            break
        frames.append(fr)
    cap.release()
    return frames, fps


def clip_layout(fps_int, n_frames):
    """Frame indices of the evaluated clips (same offsets as the upstream demo)."""
    offsets = np.array([int(math.floor(i * STRIDE_SEC * fps_int)) for i in range(NUM_FRAMES)])
    usable = min(n_frames, MAX_FRAMES) - SMOOTH_MARGIN
    n_avail = usable - offsets[-1]
    if n_avail <= 0:
        raise RuntimeError("video too short")
    starts = np.linspace(0, n_avail - 1, CLIPS_PER_VIDEO).round().astype(int)
    return np.stack([s + offsets for s in starts])          # (K, 10)


@torch.inference_mode()
def detect(fa, frames, batch=None, max_res=800):
    batch = batch or int(os.environ.get("PREP_BATCH", "2"))
    h, w = frames[0].shape[:2]
    scale = max_res / max(h, w) if max(h, w) > max_res else 1.0
    out = [None] * len(frames)
    for i in range(0, len(frames), batch):
        chunk = [cv2.resize(cv2.cvtColor(f, cv2.COLOR_BGR2RGB), None, fx=scale, fy=scale)
                 for f in frames[i:i + batch]]
        x = torch.tensor(np.stack(chunk).transpose(0, 3, 1, 2))
        lms, _, bbs = fa.get_landmarks_from_batch(x, return_bboxes=True)
        for j, (lm, bb) in enumerate(zip(lms, bbs)):
            if len(lm) > 0:
                lm = lm.reshape(-1, 68, 2) / scale
                out[i + j] = dict(landmarks=[l for l in lm],
                                  bboxes=[np.asarray(b[:-1]).reshape(2, 2) / scale for b in bb])
    return out


def transforms_for(frame_data, needed, fps_int, n_frames):
    """Upstream main-face tracking + smoothed affine alignment, for the needed frames only."""
    f_lm = [[] if d is None else d["landmarks"] for d in frame_data]
    f_bb = [[] if d is None else d["bboxes"] for d in frame_data]
    lms, bbs, idx = get_main_face_data(f_lm, f_bb, d_rate=0.65, max_paddings=fps_int * 3)
    if len(lms) < n_frames * 0.9:
        raise RuntimeError(f"face tracked in only {len(lms)}/{n_frames} frames")
    _lm = [None] * n_frames
    _bb = [None] * n_frames
    for i, k in enumerate(idx):
        _lm[k], _bb[k] = lms[i], bbs[i]

    ref = np.vstack([MEAN_FACE[x] for x in STABLE_POINTS]) * (TARGET_SIZE / 256)
    Ms, offs, crop_lms = [], [], []
    for fi in needed:
        if _lm[fi] is None:
            raise RuntimeError(f"no tracked face at needed frame {fi}")
        margin = min(12 // 2, fi, n_frames - 1 - fi)
        win = [_lm[i] for i in range(fi - margin, fi + margin + 1) if _lm[i] is not None]
        sm = np.mean(win, axis=0)
        sm += _lm[fi].mean(axis=0) - sm.mean(axis=0)
        M = cv2.estimateAffinePartial2D(np.vstack([sm[x] for x in STABLE_POINTS]), ref,
                                        method=cv2.LMEDS)[0]
        tl = sm @ M[:, :2].T + M[:, 2]
        # crop_driver(): centre on landmarks 15..67, clamp inside the 256 canvas
        size = CROP_SIZE // 2
        cx, cy = np.mean(tl[15:68], axis=0)
        if cy - size < 0:
            cy = size + 1
        elif cy + size > TARGET_SIZE:
            cy = TARGET_SIZE - size - 1
        if cx - size < 0:
            cx = size + 1
        elif cx + size > TARGET_SIZE:
            cx = TARGET_SIZE - size - 1
        uy, lx = int(cy - size), int(cx - size)
        Ms.append(M)
        offs.append((lx, uy))
        crop_lms.append(tl - [lx, uy])
    return np.array(Ms), np.array(offs), np.array(crop_lms)


def roi_for(Ms, offs, h, w, pad=40):
    """Bounding box (aligned to the 16-px macroblock grid) of all source pixels the crops read."""
    pts = []
    for M, (lx, uy) in zip(Ms, offs):
        Minv = cv2.invertAffineTransform(M)
        c = np.array([[lx, uy], [lx + CROP_SIZE, uy], [lx, uy + CROP_SIZE],
                      [lx + CROP_SIZE, uy + CROP_SIZE]], float)
        pts.append(c @ Minv[:, :2].T + Minv[:, 2])
    pts = np.concatenate(pts)
    x0, y0 = pts.min(0) - pad
    x1, y1 = pts.max(0) + pad
    x0 = max(0, int(x0) // 16 * 16)
    y0 = max(0, int(y0) // 16 * 16)
    x1 = min(w, int(math.ceil(x1 / 16) * 16))
    y1 = min(h, int(math.ceil(y1 / 16) * 16))
    x1 -= (x1 - x0) % 2
    y1 -= (y1 - y0) % 2
    return np.array([x0, y0, x1, y1])


def crop(frame, M, off):
    """Exactly the upstream affine_transform + crop_driver pixel path."""
    warped = cv2.warpAffine(frame, M, dsize=(TARGET_SIZE, TARGET_SIZE), flags=cv2.INTER_LINEAR,
                            borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    lx, uy = off
    return warped[uy:uy + CROP_SIZE, lx:lx + CROP_SIZE]


def prep_one(fa, row):
    out = p("prep", safe(row["vid"]) + ".npz")
    if out.exists() or out.with_suffix(".fail").exists():
        return "skip"
    t0 = time.time()
    frames, fps = read_frames(local_video(row))
    fps_int = int(round(fps))
    n = len(frames)
    t1 = time.time()
    try:
        layout = clip_layout(fps_int, n)
        needed = np.unique(layout)
        fd = detect(fa, frames)
        t2 = time.time()
        Ms, offs, clms = transforms_for(fd, needed, fps_int, n)
    except Exception as e:  # noqa: BLE001
        out.with_suffix(".fail").write_text(json.dumps(dict(vid=row["vid"], error=str(e))))
        return f"fail: {e}"
    h, w = frames[0].shape[:2]
    np.savez_compressed(out, fps=fps_int, n_frames=n, size=np.array([h, w]), layout=layout,
                        needed=needed, M=Ms, off=offs, crop_lm=clms, roi=roi_for(Ms, offs, h, w))
    return f"ok {w}x{h} read {t1 - t0:.1f}s detect {t2 - t1:.1f}s rest {time.time() - t2:.1f}s"


def main():
    import face_alignment
    torch.backends.cudnn.benchmark = True       # fixed frame size per video -> autotuned convs
    fa = face_alignment.FaceAlignment(face_alignment.LandmarksType.TWO_D, face_detector="sfd",
                                      dtype=torch.float16, flip_input=False, device="cuda")
    fa.max_batch_size = 32      # run FAN on all faces of a frame at once (identical outputs, ~5x faster)
    from qa.extract import priority_order
    rows = priority_order(read_manifest())
    rows = [r for r in rows if not (DATA / "prep" / (safe(r["vid"]) + ".npz")).exists()
            and not (DATA / "prep" / (safe(r["vid"]) + ".fail")).exists()]
    limit = int(os.environ.get("PREP_LIMIT", "0")) or None
    if limit:
        rows = rows[:limit]
    if len(sys.argv) > 1:
        rows = [r for r in rows if any(s in r["vid"] for s in sys.argv[1:])]
    shard = os.environ.get("PREP_SHARD")            # e.g. "0/3": run several processes in parallel
    if shard:
        i, n = map(int, shard.split("/"))
        rows = rows[i::n]
    t0 = time.time()
    for i, r in enumerate(rows):
        s = prep_one(fa, r)
        if s != "skip":
            el = time.time() - t0
            print(f"[{i + 1}/{len(rows)}] {r['vid']}: {s}  ({el:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
