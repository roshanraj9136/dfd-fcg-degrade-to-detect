"""
Degrade-to-Detect on any video: DFD-FCG verdict, probe response and corrected verdict.

    python -m qa.run_video path/to/video.mp4 [--degrade GB_5]

Needs a CUDA GPU (about 2.5 GB), the upstream DFD-FCG checkout and its checkpoint (see README).
The fitted lambda is read from web/frontend/public/data/model.json.
"""
import argparse
import json
import math

import numpy as np
import torch

from qa import probes
from qa.conditions import build
from qa.config import ROOT
from qa.engine import FACE_PARTS, Recorder, analyse, load_model
from qa.prep import clip_layout, detect, read_frames, roi_for, transforms_for


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
    crops = build(a.video, prep, "user/video", [cond])[cond][0]
    pos = {f: i for i, f in enumerate(needed)}
    clip_idx = np.array([[pos[f] for f in c] for c in layout])
    model = load_model()
    rec = Recorder(model)

    def run(x):
        fr = torch.from_numpy(np.ascontiguousarray(x)).permute(0, 3, 1, 2).contiguous()
        return analyse(model, rec, torch.stack([model.transform(fr[ci]) for ci in clip_idx]))

    base, probed = run(crops), run(probes.apply("pB2", crops))
    lam = json.load(open(ROOT / "web" / "frontend" / "public" / "data" / "model.json"))["lam"]
    sig = lambda x: 1 / (1 + math.exp(-x))  # noqa: E731
    p0s, p1s = [], []
    for i, (b, q) in enumerate(zip(base, probed)):
        r = q["d"] - b["d"]
        p0 = np.mean([sig(x) for x in b["d"]])
        p1 = np.mean([sig(x - lam * ri) for x, ri in zip(b["d"], r)])
        p0s.append(p0)
        p1s.append(p1)
        dq = (q["cs_lq"] - b["cs_lq"]).sum(0)
        print(f"clip {i}: released P(FAKE)={p0:.3f}  Degrade-to-Detect={p1:.3f}  "
              f"response (s,t,a)=({r[0]:+.2f}, {r[1]:+.2f}, {r[2]:+.2f})  "
              f"spatial change by query: " + ", ".join(f"{n} {v:+.2f}" for n, v in zip(FACE_PARTS, dq)))
    p0, p1 = float(np.mean(p0s)), float(np.mean(p1s))
    print(f"\nvideo ({cond}): released {'FAKE' if p0 >= .5 else 'REAL'} {p0:.3f}  ->  "
          f"Degrade-to-Detect {'FAKE' if p1 >= .5 else 'REAL'} {p1:.3f}  (lambda = {lam})")


if __name__ == "__main__":
    main()
