"""Where does the live checker (version 2) go wrong? Accuracy per kind of video and website option on the test set,
how far from the 0.5 line the scores sit, and the original detector's view (AUROC) of the same videos.
python service/study/where_it_fails.py"""
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from study_report import CONDS, auroc, video_p  # noqa: E402

R = HERE.parents[1] / "results" / "accuracy_study"
V2 = dict(layout="new", flip=0, faces=2)
recs = [r for r in (json.loads(l) for l in open(R / "study_test.jsonl", encoding="utf-8")) if "error" not in r]
NAMES = {"REAL": "FF++ real", "DF": "Deepfakes (face swap)", "F2F": "Face2Face (expression)", "FS": "FaceSwap",
         "NT": "NeuralTextures (expression)", "FSh": "FaceShifter (never seen)", "DFDC_R": "DFDC real", "DFDC_F": "DFDC fake"}
print(f"{'kind of video':<30}{'n':>4}" + "".join(f"{c:>18}" for c in CONDS) + "   (right answers; median |p-0.5|)")
for m, nm in NAMES.items():
    rs = [r for r in recs if r["method"] == m]
    row = f"{nm:<30}{len(rs):>4}"
    for c in CONDS:
        p = np.array([video_p(r, c, **V2) for r in rs])
        ok = (p >= .5) == (rs[0]["label"] == 1)
        row += f"{ok.mean() * 100:>11.0f}% ({np.median(np.abs(p - .5)):.2f})"
    print(row)
print()
out = dict(rows=[], auroc={})
for m, nm in NAMES.items():
    rs = [r for r in recs if r["method"] == m]
    row = dict(kind=m, name=nm, n=len(rs), label=rs[0]["label"])
    for c in CONDS:
        p = np.array([video_p(r, c, **V2) for r in rs])
        row[c] = dict(right=round(float(((p >= .5) == (rs[0]["label"] == 1)).mean()), 4),
                      margin=round(float(np.median(np.abs(p - .5))), 3))
    out["rows"].append(row)
for g, sel in (("FF++ (+FaceShifter)", lambda r: not r["method"].startswith("DFDC")), ("DFDC", lambda r: r["method"].startswith("DFDC"))):
    rs = [r for r in recs if sel(r)]
    y = [r["label"] for r in rs]
    a = {c: round(auroc(y, [video_p(r, c, **V2) for r in rs]), 3) for c in CONDS}
    out["auroc"]["ffpp" if g.startswith("FF") else "dfdc"] = a
    print(g, "AUROC (can the score tell real from fake at all; 0.5 = coin flip):", a)
(R / "where_it_fails.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
(HERE.parents[1] / "web" / "viva" / "src" / "fails.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
