"""
Run DFD-FCG + exact attribution + Evidence Nulling on any video.

    python -m qa.run_video path/to/video.mp4 [--degrade GB_5]

Prints the released verdict, the corrected verdict and how much of each head's logit is
quality evidence. Needs a CUDA GPU (about 2.5 GB), the upstream DFD-FCG checkout and its
checkpoint (see README). The fitted correction is read from web/frontend/public/data/projection.json.
"""
import argparse
import json
import math

import numpy as np
import torch

from qa.conditions import build
from qa.config import ROOT
from qa.engine import Recorder, analyse, load_model
from qa.extract import summarise_built
from qa.prep import clip_layout, detect, read_frames, roi_for, transforms_for

PARTS = ["lips", "skin", "eyes", "nose"]


def term_name(h, k):
    if h == "s":
        return f"spatial layer {k // 4 + 1} / {PARTS[k % 4]}"
    if h == "t":
        return f"temporal layer {k + 1}"
    return f"fusion: spatial layer {k // 4 + 1} / {PARTS[k % 4]}" if k < 96 else f"fusion: temporal layer {k - 95}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("--degrade", default=None, help="optionally apply one test condition first, e.g. GB_5")
    a = ap.parse_args()

    import face_alignment
    fa = face_alignment.FaceAlignment(face_alignment.LandmarksType.TWO_D, face_detector="sfd",
                                      dtype=torch.float16, flip_input=False, device="cuda")
    fa.max_batch_size = 32
    frames, fps = read_frames(a.video)
    fps_int = int(round(fps))
    layout = clip_layout(fps_int, len(frames))
    needed = np.unique(layout)
    Ms, offs, clms = transforms_for(detect(fa, frames), needed, fps_int, len(frames))
    h, w = frames[0].shape[:2]
    prep = dict(fps=fps_int, n_frames=len(frames), size=np.array([h, w]), layout=layout, needed=needed,
                M=Ms, off=offs, crop_lm=clms, roi=roi_for(Ms, offs, h, w))
    del fa
    torch.cuda.empty_cache()

    cond = a.degrade or "clean"
    out = summarise_built(prep, build(a.video, prep, "user/video", [cond]))
    crops, clip_idx, _, q_bg = out[cond]
    model = load_model()
    rec = Recorder(model)
    fr = torch.from_numpy(crops).permute(0, 3, 1, 2).contiguous()
    clips = torch.stack([model.transform(fr[ci]) for ci in clip_idx])
    res = analyse(model, rec, clips)

    m = json.load(open(ROOT / "web" / "frontend" / "public" / "data" / "projection.json"))
    phi0, scale = np.array(m["phi0"]), np.array(m["scale"])
    W = {k: np.array(v) for k, v in m["W"].items()}
    sig = lambda x: 1 / (1 + math.exp(-x))  # noqa: E731
    p0s, p1s = [], []
    for i, r in enumerate(res):
        q = q_bg[i][:7].astype(float)
        z = (np.concatenate([q, q ** 2]) - phi0) / scale
        per = {k: W[k] @ z for k in "sta"}
        nu = [per[k].sum() for k in "sta"]
        p0 = np.mean([sig(x) for x in r["d"]])
        p1 = np.mean([sig(x - n) for x, n in zip(r["d"], nu)])
        p0s.append(p0)
        p1s.append(p1)
        top = sorted(((v, k, j) for k in "sta" for j, v in enumerate(per[k])), key=lambda t: -abs(t[0]))[:3]
        print(f"clip {i}: released P(FAKE)={p0:.3f}  corrected={p1:.3f}  "
              f"quality logit (s,t,a)=({nu[0]:+.2f}, {nu[1]:+.2f}, {nu[2]:+.2f})")
        print("        most quality-affected terms: " + "; ".join(f"{term_name(k, j)} {v:+.3f}" for v, k, j in top))
    p0, p1 = float(np.mean(p0s)), float(np.mean(p1s))
    print(f"\nvideo ({cond}): released {'FAKE' if p0 >= .5 else 'REAL'} {p0:.3f}  ->  "
          f"Evidence Nulling {'FAKE' if p1 >= .5 else 'REAL'} {p1:.3f}")


if __name__ == "__main__":
    main()
