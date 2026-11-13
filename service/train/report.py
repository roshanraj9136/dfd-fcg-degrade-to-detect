"""Released vs fine-tuned DFD-FCG: the numbers for the viva guide and the decision rule (results/retrain/PLAN.md).
  reported test videos: feats_test_{old,new}.npz (feats.py on cache_builds.py test) + each model's read-out
  newer test sets:      eval_new_{old,new}.jsonl (eval_new.py)
  training:             train_v1/train_log.jsonl
    python service/train/report.py <out.json>
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
D = Path(r"D:\dfd-data")
OUT = Path(sys.argv[1])
sig = lambda x: 1 / (1 + np.exp(-x))  # noqa: E731
rng = np.random.default_rng(0)
CONDS = ["clean", "GB_5", "VC_5"]


def readout(path):
    m = json.load(open(path))["selfcal"]
    return [np.array(m[k]) for k in ("mean", "scale", "coef", "intercept")]


def video_scores(feats, ro):
    T = np.load(feats, allow_pickle=True)
    mu, sd, W, b = ro
    Y = np.concatenate([T["y_s"], T["y_t"]], 1)
    p = sig(T["d"] - (((Y - mu) / sd) @ W.T + b)).mean(1)
    out = {}
    for v, c, lab, m, q in zip(T["vid"], T["cond"], T["label"], T["method"], p):
        out.setdefault((str(v), str(c)), dict(label=int(lab), method=str(m), p=[]))["p"].append(float(q))
    return {k: dict(v, p=float(np.mean(v["p"]))) for k, v in out.items()}


def stats(y, p):
    y, p = np.asarray(y), np.asarray(p)
    pred = p >= 0.5
    fa = float(pred[y == 0].mean()) if (y == 0).any() else None
    tp = float(pred[y == 1].mean()) if (y == 1).any() else None
    out = dict(n=int(len(y)), acc=float((pred == (y == 1)).mean()), false_alarm=fa, caught=tp)
    if fa is not None and tp is not None:
        pos, neg = p[y == 1], p[y == 0]
        out["bacc"] = (1 - fa + tp) / 2
        out["auroc"] = float(((pos[:, None] > neg[None]).sum() + 0.5 * (pos[:, None] == neg[None]).sum()) / (len(pos) * len(neg)))
    return out


def boot(y, pa, pb, groups=None, n=2000):
    """Paired bootstrap of the change in balanced accuracy; resamples whole videos (groups) when given."""
    y, pa, pb = map(np.asarray, (y, pa, pb))
    g = np.asarray(groups) if groups is not None else np.arange(len(y))
    ug = np.unique(g)
    lab = {u: y[g == u][0] for u in ug}
    real, fake = [u for u in ug if lab[u] == 0], [u for u in ug if lab[u] == 1]
    rows = {u: np.where(g == u)[0] for u in ug}

    def bacc(idx, p):
        pr, yy = p[idx] >= .5, y[idx]
        return ((1 - pr[yy == 0].mean()) + pr[yy == 1].mean()) / 2

    d = []
    for _ in range(n):
        pick = list(rng.choice(real, len(real))) + list(rng.choice(fake, len(fake)))
        idx = np.concatenate([rows[u] for u in pick])
        d.append(bacc(idx, pb) - bacc(idx, pa))
    return [round(float(np.percentile(d, 2.5)), 4), round(float(np.percentile(d, 97.5)), 4)]


res = dict(reported={}, new={}, training={})
# ---- reported test videos
A = video_scores(D / "feats_test_old.npz", readout(D / "readout_old.json"))
B = video_scores(D / "feats_test_new.npz", readout(D / "readout_new.json"))
GR = {"ffpp": ("REAL", "DF", "F2F", "FS", "NT"), "fsh": ("REAL", "FSh"), "dfdc": ("DFDC_R", "DFDC_F")}
for g, methods in GR.items():
    keys = sorted(k for k in A if A[k]["method"] in methods and k in B)
    for c in CONDS + ["all"]:
        ks = [k for k in keys if c == "all" or k[1] == c]
        y = [A[k]["label"] for k in ks]
        pa, pb = [A[k]["p"] for k in ks], [B[k]["p"] for k in ks]
        res["reported"].setdefault(g, {})[c] = dict(old=stats(y, pa), new=stats(y, pb), ci95=boot(y, pa, pb, [k[0] for k in ks]))
# ---- newer test sets
E1 = {r["vid"]: r for r in (json.loads(l) for l in open(D / "eval_new_old.jsonl", encoding="utf-8"))}
E2 = {r["vid"]: r for r in (json.loads(l) for l in open(D / "eval_new_new.jsonl", encoding="utf-8"))}
SETS = {"celebdf": ("Celeb-DF v2", lambda r: r["source"] == "celebdf"),
        "social": ("Social-media deepfakes (2026)", lambda r: r["source"] == "social"),
        "mavos": ("MAVOS-DD, methods seen in training (2025)", lambda r: r["source"] == "mavos" and r["set"] == "test"),
        "mavos_open": ("MAVOS-DD, methods never seen (2025)", lambda r: r["source"] == "mavos" and r["set"] == "test_open"),
        "sf": ("SynthForensics, AI-generated people (2026)", lambda r: r["source"] == "sf"),
        "synth": ("AI-generated talking heads (2026)", lambda r: r["source"] == "synth")}
for key, (name, sel) in SETS.items():
    vs = sorted(v for v in E1 if v in E2 and sel(E1[v]))
    if not vs:
        continue
    y = [E1[v]["label"] for v in vs]
    pa, pb = [E1[v]["p"] for v in vs], [E2[v]["p"] for v in vs]
    entry = dict(name=name, old=stats(y, pa), new=stats(y, pb))
    if len(set(y)) == 2:
        entry["ci95"] = boot(y, pa, pb)
    res["new"][key] = entry
# ---- training log
log = [json.loads(l) for l in open(D / "train_v1" / "train_log.jsonl", encoding="utf-8")]
vals = [x for x in log if x["event"] == "val"]
data = next(x for x in log if x["event"] == "data")
res["training"] = dict(videos=data["train"], val_videos=data["val"], by=data["train_by"],
                       val=[dict(step=v["step"], score=v["score"], clean=v["clean"]["mean"], degraded=v["degraded"]["mean"],
                                 per_source=v["clean"]["per_source"]) for v in vals],
                       best_step=max(vals, key=lambda v: v["score"])["step"])
# ---- decision rule
two = [k for k, v in res["new"].items() if "bacc" in v["old"]]
gain_new = float(np.mean([res["new"][k]["new"]["bacc"] - res["new"][k]["old"]["bacc"] for k in two]))
d_ffpp = res["reported"]["ffpp"]["all"]["new"]["bacc"] - res["reported"]["ffpp"]["all"]["old"]["bacc"]
d_dfdc = res["reported"]["dfdc"]["all"]["new"]["bacc"] - res["reported"]["dfdc"]["all"]["old"]["bacc"]
res["decision"] = dict(gain_new_sets=round(gain_new, 4), change_ffpp=round(d_ffpp, 4), change_dfdc=round(d_dfdc, 4),
                       deploy=bool(gain_new >= 0.05 and d_ffpp >= -0.02 and d_dfdc >= -0.02))
OUT.parent.mkdir(parents=True, exist_ok=True)
json.dump(res, open(OUT, "w"), indent=1)
print(json.dumps(res["decision"]), "best step", res["training"]["best_step"])
for part in ("reported", "new"):
    for k, v in res[part].items():
        if part == "reported":
            v = v["all"]
        o, n = v["old"], v["new"]
        print(f"{part:<9}{k:<11} old acc {o['acc']:.3f} bacc {o.get('bacc') or 0:.3f} auroc {o.get('auroc') or 0:.3f} caught {o['caught'] or 0:.3f}"
              f" | new acc {n['acc']:.3f} bacc {n.get('bacc') or 0:.3f} auroc {n.get('auroc') or 0:.3f} caught {n['caught'] or 0:.3f} ci {v.get('ci95')}")
