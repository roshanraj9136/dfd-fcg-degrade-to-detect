"""
Sanity checks for the pipeline:
  1. our clean face crops match the upstream crop_main_face output (REAL/000);
  2. the exact decomposition sums back to the model's own head logits;
  3. fp16 inference changes P(FAKE) only negligibly versus fp32.

    python tests/check_pipeline.py <path to upstream cropped 000.avi>
"""
import sys

import cv2
import numpy as np
import torch

sys.path.insert(0, ".")
from qa.conditions import build                       # noqa: E402
from qa.config import DATA                            # noqa: E402
from qa.engine import Recorder, analyse, load_model   # noqa: E402
from qa.manifest import local_video, read_manifest    # noqa: E402

rows = {r["vid"]: r for r in read_manifest()}
row = rows["REAL/000"]
prep = dict(np.load(DATA / "prep" / "REAL__000.npz"))
built = build(local_video(row), prep, row["vid"], ["clean", "GB_3"])
crops = built["clean"][0]

# ---- 1. crops vs upstream ----
if len(sys.argv) > 1:
    cap = cv2.VideoCapture(sys.argv[1])
    up = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        up.append(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))
    diffs = [np.abs(crops[i].astype(int) - up[f].astype(int)).mean()
             for i, f in enumerate(prep["needed"])]
    print(f"[1] mean |ours - upstream| per needed frame: max {max(diffs):.3f}, mean {np.mean(diffs):.3f} (0-255 scale)")

# ---- 2/3. exactness and fp16 ----
model = load_model()
rec = Recorder(model)
tf = model.transform
pos = {f: i for i, f in enumerate(prep["needed"])}
ci = np.array([[pos[f] for f in c] for c in prep["layout"]])
fr = torch.from_numpy(crops).permute(0, 3, 1, 2).contiguous()
clips = torch.stack([tf(fr[c]) for c in ci])

r32 = analyse(model, rec, clips, fp16=False)
r16 = analyse(model, rec, clips, fp16=True)
for a, b in zip(r32, r16):
    print(f"[2] fp32: decomposition P={a['p_final']:.6f}  model P={a['model_p']:.6f}  "
          f"|diff|={abs(a['p_final'] - a['model_p']):.2e}")
    s_sum = a["cs_lq"].sum() + a["const"][0]
    p_sum = a["cs_patch"].sum() + a["const"][0]
    print(f"    d_s={a['d'][0]:.6f}  sum(layer x part)={s_sum:.6f}  sum(patches)={p_sum:.6f}")
    print(f"[3] fp16 P={b['p_final']:.6f}  |fp16-fp32|={abs(a['p_final'] - b['p_final']):.2e}  "
          f"d fp32={np.round(a['d'], 4)} fp16={np.round(b['d'], 4)}")

with torch.no_grad():
    out = model.model(clips.cuda())
    ls = out["logits_s"].double().cpu()
    print("[2] model logit_s diff:", (ls[:, 1] - ls[:, 0]).numpy(), " ours:", [x["d"][0] for x in r32])
