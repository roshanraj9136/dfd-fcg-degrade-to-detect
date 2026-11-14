"""Compare single detectors and fused detectors on every held-out test set, offline.

Fusion: logistic regression on the members' video logits, fitted ONLY on the validation videos of the fine-tuning
sets (sources weighted equally, classes balanced). A video with several faces is FAKE if any face is (the most
suspicious face decides, as on the website). The verdict is always REAL or FAKE (threshold 0.5).

    python service/ens/evaluate.py [member ...]      members: dfd, clip, clipft, pe, dino, ...  (default: all found)
Writes D:\\dfd-data\\ens\\eval.json and prints one table.
"""
import json
import sys
from pathlib import Path

sys.path[:0] = [str(Path(__file__).resolve().parent)]
import numpy as np  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402

import data as Dt  # noqa: E402

REP_CONDS = ["clean", "GB_5", "VC_5"]
import os  # noqa: E402

# round 2b: the combiner is fitted / selected on clean + damaged validation videos (results/ensemble/PLAN.md)
VAL_CONDS = tuple(os.environ.get("VAL_CONDS", "clean").split(","))


def _valdmg(n):
    """Damaged validation scores of one member: {(vid, cond, 0): video logit} (service/ens/score_valdmg.py)."""
    base = n[:-3] if n.endswith("_sc") else n
    f = Dt.E / f"valdmg_{base.replace('_ft', 'ft')}.npz"
    if not f.exists():
        return {}
    z = np.load(f)
    if "p" in z.files:
        return {tuple(str(k).split("|")[:2]) + (0,): float(Dt.logit(p)) for k, p in zip(z["keys"], z["p"])}
    cal = Dt.GenDSelfCal(base.replace("_ft", "ft")) if n.endswith("_sc") else None
    out = {}
    for i, k in enumerate(z["keys"]):
        lg, ft = z["logit"][z["start"][i]:z["start"][i + 1]], z["feats"][z["start"][i]:z["start"][i + 1]]
        out[tuple(str(k).split("|")[:2]) + (0,)] = Dt.video_score(cal.correct(lg, ft) if cal else lg)
    return out


def load_members(names):
    new, rep = _load_members(names)
    if VAL_CONDS != ("clean",):
        for n in list(new):
            new[n].update(_valdmg(n))
    return new, rep


def _load_members(names):
    """name -> {(vid, cond, face): video logit}; one dict for the fine-tuning sets and one for the reported videos."""
    new, rep = {}, {}
    for n in names:
        if n in ("dfd", "dfdft"):
            w = "old" if n == "dfd" else "new"
            if n == "dfdft" and not (Dt.E / "dfd_val_new.jsonl").exists():
                print("skip dfdft: validation scores missing")
                continue
            new[n] = {k: float(Dt.logit(v)) for k, v in Dt.dfd_new(w).items()}
            r, _ = Dt.dfd_rep(w)
            rep[n] = {k: float(Dt.logit(v)) for k, v in r.items()}
            continue
        base = n[:-3] if n.endswith("_sc") else n
        L, LF = Dt.gend(base.replace("_ft", "ft"), "new")
        R, RF = Dt.gend(base.replace("_ft", "ft"), "rep")
        if L is None or R is None:
            print(f"skip {n}: scores missing")
            continue
        if n.endswith("_sc"):
            try:
                cal = Dt.GenDSelfCal(base.replace("_ft", "ft"))
            except FileNotFoundError as e:
                print(f"skip {n}: {e} missing")
                continue
            new[n] = {k: Dt.video_score(cal.correct(v, LF[k])) for k, v in L.items()}
            rep[n] = {k: Dt.video_score(cal.correct(v, RF[k])) for k, v in R.items()}
            continue
        new[n] = {k: Dt.video_score(v) for k, v in L.items()}
        rep[n] = {k: Dt.video_score(v) for k, v in R.items()}
    return new, rep


def table(new, rep, members, fuse=None):
    """Per evaluation group metrics for one detector (members = [name]) or a fusion (fuse = fitted LR)."""
    meta = Dt.rows()
    _, repmeta = Dt.dfd_rep()

    # index faces per (vid, cond)
    def faces_of(src, vid, cond):
        return [f for f in range(3) if all((vid, cond, f) in src[m] for m in members)]

    def video(src, vid, cond):
        fs = faces_of(src, vid, cond)
        if not fs:
            return None
        X = np.array([[src[m][(vid, cond, f)] for m in members] for f in fs])
        s = fuse.decision_function(X) if fuse is not None else X[:, 0]
        return float(np.max(s))

    out = {}
    groups = {}
    for vid, r in meta.items():
        if r["set"] in ("test", "test_open"):
            s = video(new, vid, "clean")
            if s is not None:
                groups.setdefault(Dt.group(r), []).append((r["label"], s))
    for g, v in groups.items():
        y, s = zip(*v)
        out[g] = Dt.metrics(y, s)
    for cond in REP_CONDS:
        gr = {}
        for vid, (lab, meth) in repmeta.items():
            s = video(rep, vid, cond)
            if s is None:
                continue
            g = Dt.rep_group(vid, meth)
            gr.setdefault(g, []).append((lab, s))
            if g == "ffpp" and lab == 0:
                gr.setdefault("fsh", []).append((lab, s))          # FaceShifter is paired with the FF++ test reals
        for g, v in gr.items():
            y, s = zip(*v)
            out[f"{g}:{cond}"] = Dt.metrics(y, s)
    for g in ("ffpp", "fsh", "dfdc"):
        ks = [f"{g}:{c}" for c in REP_CONDS if f"{g}:{c}" in out]
        if ks:
            out[g] = {m: float(np.mean([out[k][m] for k in ks])) for m in ("bacc", "auroc", "false_alarm", "caught")}
    two = [g for g in ("celebdf", "social", "mavos", "mavos_open") if g in out]
    out["mean_new_bacc"] = float(np.mean([out[g]["bacc"] for g in two]))
    out["mean_new_auroc"] = float(np.mean([out[g]["auroc"] for g in two]))
    out["mean_fakeonly_caught"] = float(np.mean([out[g]["caught"] for g in ("sf", "synth") if g in out]))
    out["mean_all_bacc"] = float(np.mean([out[g]["bacc"] for g in two + ["ffpp", "fsh", "dfdc"] if g in out]))
    return out


def val_data(new, members):
    """Validation rows (main face) for every condition in VAL_CONDS: X, y, source, condition, video."""
    meta = Dt.rows()
    X, y, src, cnd, vids = [], [], [], [], []
    for vid, r in sorted(meta.items()):
        if r["set"] != "val" or r["source"] == "dfdc":
            continue
        for c in VAL_CONDS:
            k = (vid, c, 0)
            if all(k in new[m] for m in members):
                X.append([new[m][k] for m in members])
                y.append(r["label"])
                src.append(r["source"])
                cnd.append(c)
                vids.append(vid)
    return np.array(X), np.array(y), np.array(src), np.array(cnd), np.array(vids)


def weights(y, src, cnd=None):
    """Each condition weighs the same; within it, each source weighs the same and classes are balanced (fake-only
    sources are balanced against all reals of the condition)."""
    cnd = np.zeros(len(y), int) if cnd is None else cnd
    w = np.zeros(len(y))
    for c in set(cnd):
        cm = cnd == c
        for s in set(src[cm]):
            for lab in (0, 1):
                sel = cm & (src == s) & (y == lab)
                if sel.any():
                    w[sel] = 1.0 / sel.sum()
        r, f = cm & (y == 0), cm & (y == 1)
        w[r] *= w[f].sum() / w[r].sum()
        w[cm] /= w[cm].sum()
    return w / w.mean()


def fit_fusion(new, members, C=1.0):
    """LR on validation videos (main face), clean + the conditions in VAL_CONDS."""
    X, y, src, cnd, vids = val_data(new, members)
    lr = LogisticRegression(C=C, max_iter=1000).fit(X, y, sample_weight=weights(y, src, cnd))
    return lr, len(set(vids))


def src_bacc(y, s, src, cnd=None):
    """Mean over conditions and sources of balanced accuracy at 0 (fake-only sources are paired with all reals)."""
    cnd = np.zeros(len(y), int) if cnd is None else cnd
    out = []
    for c in sorted(set(cnd)):
        cm = cnd == c
        for g in sorted(set(src[cm])):
            fk = s[cm & (src == g) & (y == 1)]
            rl = s[cm & (src == g) & (y == 0)]
            if len(rl) == 0:
                rl = s[cm & (y == 0)]
            out.append((np.mean(rl < 0) + np.mean(fk >= 0)) / 2)
    return float(np.mean(out))


def cv_score(new, members, folds=5, C=1.0):
    """5-fold cross-validated balanced accuracy on the validation videos (the selection rule). Folds by video, so the
    clean and damaged versions of a video are always in the same fold."""
    X, y, src, cnd, vids = val_data(new, members)
    uv = np.array(sorted(set(vids)))
    rng = np.random.default_rng(0)
    rng.shuffle(uv)
    fold_of = {v: i % folds for i, v in enumerate(uv)}
    fo = np.array([fold_of[v] for v in vids])
    s = np.zeros(len(y))
    for k in range(folds):
        te, tr = fo == k, fo != k
        lr = LogisticRegression(C=C, max_iter=1000).fit(X[tr], y[tr], sample_weight=weights(y[tr], src[tr], cnd[tr]))
        s[te] = lr.decision_function(X[te])
    return src_bacc(y, s, src, cnd)


def fmt(t):
    keys = ["celebdf", "social", "mavos", "mavos_open", "sf", "synth", "ffpp", "fsh", "dfdc"]
    cells = []
    for k in keys:
        if k not in t:
            cells.append("   -  ")
        elif "bacc" in t[k]:
            cells.append(f"{100 * t[k]['bacc']:5.1f} ")
        else:
            cells.append(f"{100 * t[k]['caught']:5.1f}c")
    return " ".join(cells) + f" | new {100 * t['mean_new_bacc']:5.1f} auc {100 * t['mean_new_auroc']:5.1f}" \
        f" | all {100 * t['mean_all_bacc']:5.1f}"


def main():
    names = sys.argv[1:] or ["dfd", "clip", "clipft", "pe", "dino"]
    new, rep = load_members(names)
    names = [n for n in names if n in new]
    res = {}
    print(" " * 22 + "   cv  |  cdf   soc   mav  mavO    sf  synth  ffpp   fsh  dfdc   (test bacc %, c = fakes caught)")
    for n in names:
        res[n] = table(new, rep, [n])
        print(f"{n:<22}", "  -   |", fmt(res[n]))
    import itertools
    for k in range(1, len(names) + 1):
        for combo in itertools.combinations(names, k):
            lr, nv = fit_fusion(new, list(combo))
            key = "+".join(combo) + ("+cal" if k == 1 else "")
            res[key] = table(new, rep, list(combo), lr)
            res[key]["weights"] = lr.coef_[0].tolist() + [float(lr.intercept_[0])]
            res[key]["cv_val_bacc"] = cv_score(new, list(combo))
            print(f"{key:<22}", f"{100 * res[key]['cv_val_bacc']:5.1f} |", fmt(res[key]))
    fused = {k: v for k, v in res.items() if "cv_val_bacc" in v}
    best = max(fused, key=lambda k: (round(fused[k]["cv_val_bacc"], 3), -k.count("+")))
    res["_chosen"] = best
    print("chosen on validation:", best)
    json.dump(res, open(Dt.E / "eval.json", "w"), indent=1)


if __name__ == "__main__":
    main()
