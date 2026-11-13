"""Old vs fine-tuned DFD-FCG on the held-out test sets of the newer datasets (Celeb-DF v2 official test list, 2026
social-media test part, 2026 AI-generated talking heads groups 31-36, MAVOS-DD test + open-set-model test,
SynthForensics test), on the aligned face clips from prep_crops.py (8 clips from the first 10 s, up to 2 faces, as the
live checker). Each model uses its own self-calibration read-out; the most suspicious face decides, as on the website.
    python service/train/eval_new.py <name> <checkpoint> <model.json> <out.jsonl>
"""
import csv
import json
import math
import os
import sys
from pathlib import Path

NAME, CK, MJ, OUT = sys.argv[1], sys.argv[2], sys.argv[3], Path(sys.argv[4])
os.environ["DFD_FCG_CKPT"] = CK
ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT)]
import numpy as np  # noqa: E402
import torch  # noqa: E402

from qa.engine import Recorder, analyse, load_model  # noqa: E402

CROPS = Path(r"D:\dfd-data\crops")
SETS = [Path(r"D:\dfd-data\new\sets.csv"), Path(r"D:\dfd-data\new\sets_gated.csv")]
sig = lambda x: 1 / (1 + np.exp(-x))  # noqa: E731

m = json.load(open(MJ))["selfcal"]
mu, sd, W, b = (np.array(m[k]) for k in ("mean", "scale", "coef", "intercept"))
model = load_model(device="cuda")
rec = Recorder(model)
done = {json.loads(l)["vid"] for l in open(OUT, encoding="utf-8")} if OUT.exists() else set()
rows = [r for f in SETS for r in csv.DictReader(open(f)) if r["set"].startswith("test")]
for k, r in enumerate(rows):
    p = CROPS / (r["vid"].replace("/", "__") + ".npz")
    if r["vid"] in done or not p.exists():
        continue
    z = np.load(p)
    faces = []
    for f in [k2 for k2 in z.files if k2.startswith("face")]:
        D, Y = [], []
        sel = [int(c) for c in os.environ.get("CLIPS", "").split(",") if c] or list(range(len(z["clip_idx"])))
        cidx = z["clip_idx"][[c for c in sel if c < len(z["clip_idx"])]]
        for c in range(0, len(cidx), 2):
            batch = [model.transform(torch.from_numpy(np.ascontiguousarray(z[f][ci])).permute(0, 3, 1, 2).contiguous())
                     for ci in cidx[c:c + 2]]
            with torch.inference_mode():
                for res in analyse(model, rec, torch.stack(batch), fp16=True):
                    D.append(res["d"])
                    Y.append(np.r_[res["y_s"], res["y_t"]])
        D, Y = np.array(D), np.array(Y)
        G = ((Y - mu) / sd) @ W.T + b
        faces.append(dict(p=float(sig(D - G).mean()), p_orig=float(sig(D).mean()),
                          clip_p=np.round(sig(D - G).mean(1), 4).tolist()))
    best = max(faces, key=lambda x: x["p"])
    rec_ = dict(vid=r["vid"], set=r["set"], source=r["source"], method=r["method"], label=int(r["label"]), model=NAME,
                p=best["p"], p_orig=best["p_orig"], faces=[round(x["p"], 4) for x in faces], clip_p=best["clip_p"])
    with open(OUT, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(rec_) + "\n")
    if (k + 1) % 25 == 0:
        print(f"{NAME} {k + 1}/{len(rows)}", flush=True)
print("finished", flush=True)
