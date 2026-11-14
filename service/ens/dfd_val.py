"""Released DFD-FCG (+ the live self-calibration read-out) on the validation videos of the fine-tuning sets
(prep_crops.py, main face, 4 clips of 10 frames). Gives the DFD-FCG scores the ensemble is fitted on.
    python service/ens/dfd_val.py <out.jsonl> [checkpoint] [model.json]
One JSON line per video: vid, set, source, method, label, d (clips x 3 head logits), g (spurious evidence), p, p_orig
"""
import csv
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(sys.argv[1])
if len(sys.argv) > 2:
    os.environ["DFD_FCG_CKPT"] = sys.argv[2]
MJ = sys.argv[3] if len(sys.argv) > 3 else str(ROOT / "service" / "app" / "model.json")
sys.path[:0] = [str(ROOT)]
import numpy as np  # noqa: E402
import torch  # noqa: E402

from qa.engine import Recorder, analyse, load_model  # noqa: E402

CROPS = Path(r"D:\dfd-data\crops")
SETS = [Path(r"D:\dfd-data\new\sets.csv"), Path(r"D:\dfd-data\new\sets_gated.csv")]
SPLITS = os.environ.get("SPLITS", "val").split(",")
sig = lambda x: 1 / (1 + np.exp(-x))  # noqa: E731

m = json.load(open(MJ))["selfcal"]
mu, sd, W, b = (np.array(m[k]) for k in ("mean", "scale", "coef", "intercept"))
model = load_model(device="cuda")
rec = Recorder(model)
done = {json.loads(l)["vid"] for l in open(OUT, encoding="utf-8")} if OUT.exists() else set()
rows = [r for f in SETS for r in csv.DictReader(open(f)) if r["set"] in SPLITS and r["source"] != "dfdc"]
for k, r in enumerate(rows):
    p = CROPS / (r["vid"].replace("/", "__") + ".npz")
    if r["vid"] in done or not p.exists():
        continue
    z = np.load(p)
    D, Y = [], []
    for c in range(0, len(z["clip_idx"]), 2):
        batch = [model.transform(torch.from_numpy(np.ascontiguousarray(z["face0"][ci])).permute(0, 3, 1, 2).contiguous())
                 for ci in z["clip_idx"][c:c + 2]]
        with torch.inference_mode():
            for res in analyse(model, rec, torch.stack(batch), fp16=True):
                D.append(res["d"])
                Y.append(np.r_[res["y_s"], res["y_t"]])
    D, Y = np.array(D), np.array(Y)
    G = ((Y - mu) / sd) @ W.T + b
    out = dict(vid=r["vid"], set=r["set"], source=r["source"], method=r["method"], label=int(r["label"]),
               d=np.round(D, 5).tolist(), g=np.round(G, 5).tolist(), p=float(sig(D - G).mean()), p_orig=float(sig(D).mean()))
    with open(OUT, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(out) + "\n")
    if (k + 1) % 50 == 0:
        print(f"{k + 1}/{len(rows)}", flush=True)
print("finished", flush=True)
