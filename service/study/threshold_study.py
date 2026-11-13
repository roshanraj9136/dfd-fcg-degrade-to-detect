"""Would a different decision line (instead of 0.5 on the corrected score) be more accurate?
Chosen on the selection set only (balanced accuracy, mean of FF++ and DFDC, over the three website options),
then reported once on the test set. python service/study/threshold_study.py"""
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from study_report import CONDS, video_p  # noqa: E402

R = HERE.parents[1] / "results" / "accuracy_study"
V2 = dict(layout="new", flip=0, faces=2)


def load(name):
    return [r for r in (json.loads(l) for l in open(R / name, encoding="utf-8")) if "error" not in r]


def table(recs):
    out = {}
    for g, sel in (("ffpp", lambda r: not r["method"].startswith("DFDC")), ("dfdc", lambda r: r["method"].startswith("DFDC"))):
        rs = [r for r in recs if sel(r)]
        out[g] = (np.array([r["label"] for r in rs for _ in CONDS]), np.array([video_p(r, c, **V2) for r in rs for c in CONDS]))
    return out


def stats(y, p, t):
    pred = p >= t
    fa, tp = pred[y == 0].mean(), pred[y == 1].mean()
    return dict(acc=float((pred == (y == 1)).mean()), bacc=float((1 - fa + tp) / 2), false_alarm=float(fa), caught=float(tp))


sel, test = table(load("study_select.jsonl")), table(load("study_test.jsonl"))
grid = np.round(np.arange(0.40, 0.561, 0.01), 2)
print("threshold | selection: FF++ bacc, DFDC bacc, mean | test: FF++ bacc (FA, caught), DFDC bacc (FA, caught)")
best = None
for t in grid:
    s = {g: stats(*sel[g], t) for g in sel}
    m = (s["ffpp"]["bacc"] + s["dfdc"]["bacc"]) / 2
    q = {g: stats(*test[g], t) for g in test}
    print(f"{t:.2f} | {s['ffpp']['bacc']:.3f} {s['dfdc']['bacc']:.3f} {m:.3f} | "
          f"{q['ffpp']['bacc']:.3f} ({q['ffpp']['false_alarm']:.3f}, {q['ffpp']['caught']:.3f})  "
          f"{q['dfdc']['bacc']:.3f} ({q['dfdc']['false_alarm']:.3f}, {q['dfdc']['caught']:.3f})")
    if best is None or m > best[1] + 1e-9:
        best = (float(t), m)
print("chosen on the selection set:", best)
