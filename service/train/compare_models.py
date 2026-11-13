"""Released vs fine-tuned DFD-FCG, both with their own self-calibration and the live checker's version-2 rules.
  1. the 225 reported test videos (FF++ test, FaceShifter, DFDC part 0), three website options (study.py runs)
  2. the held-out test sets of the newer datasets (eval_new.py runs), as uploaded
Paired bootstrap (videos resampled, same videos for both models) for the change in balanced accuracy.
    python service/train/compare_models.py <old study.jsonl> <new study.jsonl> <old eval_new.jsonl> <new eval_new.jsonl> <out.json>
"""
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "study"))
from study_report import CONDS, auroc, video_p  # noqa: E402

V2 = dict(layout="new", flip=0, faces=2)
old_s, new_s, old_e, new_e, out = sys.argv[1:6]
rng = np.random.default_rng(0)


def load(f):
    return {r["vid"]: r for r in (json.loads(l) for l in open(f, encoding="utf-8")) if "error" not in r}


def stats(y, p):
    y, p = np.asarray(y), np.asarray(p)
    pred = p >= 0.5
    fa = float(pred[y == 0].mean()) if (y == 0).any() else None
    tp = float(pred[y == 1].mean()) if (y == 1).any() else None
    bacc = (1 - fa + tp) / 2 if fa is not None and tp is not None else None
    return dict(n=int(len(y)), acc=float((pred == (y == 1)).mean()), false_alarm=fa, caught=tp, bacc=bacc,
                auroc=auroc(y, p) if fa is not None and tp is not None else None)


def boot(y, pa, pb, n=2000):
    y, pa, pb = map(np.asarray, (y, pa, pb))
    r, f = np.where(y == 0)[0], np.where(y == 1)[0]

    def b(idx, p):
        pr = p[idx] >= .5
        yy = y[idx]
        return ((1 - pr[yy == 0].mean()) + pr[yy == 1].mean()) / 2

    d = []
    for _ in range(n):
        idx = np.r_[rng.choice(r, len(r)), rng.choice(f, len(f))]
        d.append(b(idx, pb) - b(idx, pa))
    return [round(float(np.percentile(d, 2.5)), 4), round(float(np.percentile(d, 97.5)), 4)]


res = dict(reported={}, new={})
A, B = load(old_s), load(new_s)
common = sorted(set(A) & set(B))
GROUPS = {"ffpp": lambda r: r["method"] in ("REAL", "DF", "F2F", "FS", "NT"),
          "fsh": lambda r: r["method"] in ("REAL", "FSh"), "dfdc": lambda r: r["method"].startswith("DFDC")}
for g, sel in GROUPS.items():
    vids = [v for v in common if sel(A[v])]
    y = [A[v]["label"] for v in vids]
    for c in CONDS:
        pa = [video_p(A[v], c, **V2) for v in vids]
        pb = [video_p(B[v], c, **V2) for v in vids]
        res["reported"].setdefault(g, {})[c] = dict(old=stats(y, pa), new=stats(y, pb), ci95=boot(y, pa, pb))
    # all three options pooled (each video counted 3 times; bootstrap over videos)
    ya = [A[v]["label"] for v in vids for _ in CONDS]
    pa = [video_p(A[v], c, **V2) for v in vids for c in CONDS]
    pb = [video_p(B[v], c, **V2) for v in vids for c in CONDS]
    res["reported"][g]["all"] = dict(old=stats(ya, pa), new=stats(ya, pb), ci95=boot(ya, pa, pb))

if Path(old_e).exists() and Path(new_e).exists():
    E1, E2 = load(old_e), load(new_e)
    vids = sorted(set(E1) & set(E2))
    for key, sel in (("celebdf", lambda r: r["source"] == "celebdf"), ("social", lambda r: r["source"] == "social"),
                     ("mavos", lambda r: r["source"] == "mavos" and r["set"] == "test"),
                     ("mavos_open", lambda r: r["source"] == "mavos" and r["set"] == "test_open"),
                     ("synth", lambda r: r["source"] == "synth"), ("sf", lambda r: r["source"] == "sf")):
        vs = [v for v in vids if sel(E1[v])]
        if not vs:
            continue
        y = [E1[v]["label"] for v in vs]
        pa, pb = [E1[v]["p"] for v in vs], [E2[v]["p"] for v in vs]
        if len(set(y)) == 1:                          # fakes-only sets: report fakes caught
            res["new"][key] = dict(old=stats(y, pa), new=stats(y, pb), n_fake=len(y))
        else:
            res["new"][key] = dict(old=stats(y, pa), new=stats(y, pb), ci95=boot(y, pa, pb))
json.dump(res, open(out, "w"), indent=1)
fmt = lambda s: "  ".join(f"{k} {s[k]:.3f}" if isinstance(s[k], float) else f"{k} {s[k]}" for k in ("acc", "bacc", "false_alarm", "caught", "auroc") if s.get(k) is not None)  # noqa: E731
for part in ("reported", "new"):
    for g, v in res[part].items():
        if "old" in v:
            print(f"{part:<8} {g:<11} old: {fmt(v['old'])}\n{'':<20} new: {fmt(v['new'])}  ci {v.get('ci95')}")
        else:
            for c, w in v.items():
                print(f"{part:<8} {g:<5} {c:<6} old: {fmt(w['old'])}\n{'':<20} new: {fmt(w['new'])}  ci {w['ci95']}")
