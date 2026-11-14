"""The live checker with fusion.json must reproduce the offline scores the combiner was fitted and tested on.
Runs service/app/pipeline.Analyzer (model part only) on the stored face crops of a few held-out test videos and compares
every member's per-face score with the offline records (eval_new_*.jsonl, gend_*_new.npz). CPU fp32 here vs GPU fp16
offline, so differences of about 0.01 in probability are expected.
    python tests/check_fusion_live.py [device=cpu]
"""
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "service" / "app"), str(ROOT / "service" / "ens"), str(ROOT)]
import numpy as np  # noqa: E402

dev = sys.argv[1] if len(sys.argv) > 1 else "cpu"
os.environ.setdefault("MODEL_DIR", r"D:\dfd-data\ens\models")
import data as Dt  # noqa: E402
import evaluate as Ev  # noqa: E402
from pipeline import Analyzer  # noqa: E402

an = Analyzer(device=dev)
fu = an.fusion
assert fu, "no fusion.json"
new, _ = Ev.load_members(fu["members"])
pool = [k[0] for k in new[fu["members"][0]] if k[2] == 0]
rows = Dt.rows()
pick = []
for pref in ("CDF_fake", "CDF_real", "SOC_fake", "SOC_real", "MAV_inswapper", "MAV_real", "SYN_"):
    c = sorted(v for v in pool if v.startswith(pref) and rows.get(v, {}).get("set", "").startswith("test"))
    if c:
        pick.append(c[0])
worst = 0.0
for vid in pick:
    z = np.load(Dt.D / "crops" / (vid.replace("/", "__") + ".npz"))
    crops = [z[k] for k in sorted(z.files) if k.startswith("face")]
    outs, _ = an.infer(crops, z["clip_idx"])
    for f, fo in enumerate(outs):
        sc = an.face_scores(fo)
        line = [f"{vid[:40]:40s} face{f}"]
        for m in fu["members"]:
            live = sc["members"][m]
            off = float(Dt.sig(new[m][(vid, "clean", f)]))
            worst = max(worst, abs(live - off))
            line.append(f"{m}: live {live:.3f} offline {off:.3f}")
        X = np.array([[new[m][(vid, "clean", f)] for m in fu["members"]]])
        p_off = float(Dt.sig(X @ np.array(fu["coef"]) + fu["intercept"])[0])
        line.append(f"| final live {sc['p']:.3f} offline {p_off:.3f}")
        worst = max(worst, abs(sc["p"] - p_off))
        print("  ".join(line), flush=True)
print("largest difference", round(worst, 4))
assert worst < 0.05, "live fusion differs from the offline scores"
print("OK")
