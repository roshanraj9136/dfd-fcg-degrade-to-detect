"""Released vs fine-tuned DFD-FCG on the 225 reported test videos (FF++ test 130, FaceShifter 15, DFDC part 0 80),
with the research protocol: cached face geometry (data/prep), 2 clips, the authors' damage code for the three
website options (clean, GB_5 strong blur, VC_5 heavy compression). Both models are run on the same crops in one
pass; each uses its own self-calibration read-out. One JSON line per video.
    python service/train/eval_reported.py <new checkpoint> <new model.json> <out.jsonl>
"""
import csv
import json
import os
import sys
from pathlib import Path

NEW_CK, NEW_MJ, OUT = sys.argv[1], sys.argv[2], Path(sys.argv[3])
ROOT = Path(__file__).resolve().parents[2]
OLD_CK = os.environ["DFD_FCG_CKPT"]
sys.path[:0] = [str(ROOT)]
import numpy as np  # noqa: E402
import torch  # noqa: E402

import qa.engine as E  # noqa: E402
from qa.conditions import build  # noqa: E402
from qa.config import DATA  # noqa: E402
from qa.manifest import local_video  # noqa: E402
from qa.prep import safe  # noqa: E402

sig = lambda x: 1 / (1 + np.exp(-x))  # noqa: E731
CONDS = ["clean", "GB_5", "VC_5"]


def readout(path):
    m = json.load(open(path))["selfcal"]
    return [np.array(m[k]) for k in ("mean", "scale", "coef", "intercept")]


def load(ck):
    E.CKPT = Path(ck)
    m = E.load_model(device="cuda")
    return m, E.Recorder(m)


models = {"old": (*load(OLD_CK), readout(ROOT / "service" / "app" / "model.json")),
          "new": (*load(NEW_CK), readout(NEW_MJ))}
done = {json.loads(l)["vid"] for l in open(OUT, encoding="utf-8")} if OUT.exists() else set()
rows = [r for r in csv.DictReader(open(DATA / "manifest.csv")) if r["split"] in ("test", "cross") and r["vid"] not in done]
for k, r in enumerate(rows):
    pp = DATA / "prep" / (safe(r["vid"]) + ".npz")
    if not pp.exists():
        continue
    prep = dict(np.load(pp))
    built = build(str(local_video(r)), prep, r["vid"], CONDS)
    pos = {f: i for i, f in enumerate(prep["needed"])}
    rec = dict(vid=r["vid"], method=r["method"], label=int(r["label"]))
    for c in CONDS:
        x = torch.stack([E.analyse.__globals__["torch"].from_numpy(np.ascontiguousarray(built[c][0][[pos[f] for f in clip]]))
                         .permute(0, 3, 1, 2).contiguous() for clip in prep["layout"]])
        for name, (m, recd, (mu, sd, W, b)) in models.items():
            xx = torch.stack([m.transform(v) for v in x])
            with torch.inference_mode():
                res = E.analyse(m, recd, xx, fp16=True)
            D = np.stack([q["d"] for q in res])
            Y = np.stack([np.r_[q["y_s"], q["y_t"]] for q in res])
            G = ((Y - mu) / sd) @ W.T + b
            rec[f"{name}_{c}"] = dict(p=float(sig(D - G).mean()), p_orig=float(sig(D).mean()))
    with open(OUT, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(rec) + "\n")
    print(f"[{k + 1}/{len(rows)}] {r['vid']}", flush=True)
print("finished", flush=True)
