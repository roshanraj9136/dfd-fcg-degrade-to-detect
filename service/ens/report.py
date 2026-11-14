"""Round-2 report: every number the viva page and the README show, in one JSON file.

    python service/ens/report.py <chosen option, e.g. dfd+pe_ft> [candidate members ...]
Reads the stored scores (service/ens/data.py), refits the fusions on the validation videos exactly as evaluate.py, and
writes results/ensemble/results.json and web/viva/src/ensemble.json:
  options    every single detector and the chosen fusion: per test group bacc / auroc / caught / false alarms
  selfcal    self-calibration carried over to GenD: raw vs corrected, per reported condition
  decision   the pre-registered rule (results/ensemble/PLAN.md) applied to the chosen option vs the live model
  ci95       bootstrap ranges (videos resampled within each group) of the changes in the rule
  fusion     the weights the live checker uses (also written to service/app/fusion.json by deploy step, not here)
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(Path(__file__).resolve().parent)]
import numpy as np  # noqa: E402

import data as Dt  # noqa: E402
import evaluate as Ev  # noqa: E402

CHOSEN = sys.argv[1]
CANDS = sys.argv[2:] or CHOSEN.split("+")
GROUP_NEW = ["celebdf", "social", "mavos", "mavos_open", "sf", "synth"]


def per_video(new, rep, members, fuse):
    """{group: [(label, score)]} for the new sets and {f'{g}:{cond}': [...]} for the reported videos."""
    meta = Dt.rows()
    _, repmeta = Dt.dfd_rep()

    def video(src, vid, cond):
        fs = [f for f in range(3) if all((vid, cond, f) in src[m] for m in members)]
        if not fs:
            return None
        X = np.array([[src[m][(vid, cond, f)] for m in members] for f in fs])
        s = fuse.decision_function(X) if fuse is not None else X[:, 0]
        return float(np.max(s))

    out = {}
    for vid, r in meta.items():
        if r["set"] in ("test", "test_open"):
            s = video(new, vid, "clean")
            if s is not None:
                out.setdefault(Dt.group(r), []).append((vid, r["label"], s))
    for cond in Ev.REP_CONDS:
        for vid, (lab, meth) in repmeta.items():
            s = video(rep, vid, cond)
            if s is None:
                continue
            g = Dt.rep_group(vid, meth)
            out.setdefault(f"{g}:{cond}", []).append((vid, lab, s))
            if g == "ffpp" and lab == 0:
                out.setdefault(f"fsh:{cond}", []).append((vid, lab, s))
    return out


def summarise(pv):
    t = {k: Dt.metrics([x[1] for x in v], [x[2] for x in v]) for k, v in pv.items()}
    for g in ("ffpp", "fsh", "dfdc"):
        ks = [f"{g}:{c}" for c in Ev.REP_CONDS]
        t[g] = {m: float(np.mean([t[k][m] for k in ks])) for m in ("bacc", "auroc", "false_alarm", "caught")}
    two = ["celebdf", "social", "mavos", "mavos_open"]
    t["mean_new_bacc"] = float(np.mean([t[g]["bacc"] for g in two]))
    t["mean_new_auroc"] = float(np.mean([t[g]["auroc"] for g in two]))
    t["fakeonly_caught"] = float(np.mean([t[g]["caught"] for g in ("sf", "synth")]))
    return t


def bacc(v):
    y = np.array([x[1] for x in v])
    s = np.array([x[2] for x in v])
    return (np.mean(s[y == 0] < 0) + np.mean(s[y == 1] >= 0)) / 2


def boot(pa, pb, keys, n=2000, seed=0):
    """95% range of mean_k bacc_b - bacc_a, resampling videos (paired) within each group."""
    rng = np.random.default_rng(seed)
    al = {k: {x[0]: x for x in pa[k]} for k in keys}
    bl = {k: {x[0]: x for x in pb[k]} for k in keys}
    ids = {k: sorted(set(al[k]) & set(bl[k])) for k in keys}
    d = []
    for _ in range(n):
        dd = []
        for k in keys:
            labs = np.array([al[k][i][1] for i in ids[k]])
            pick = np.concatenate([rng.choice(np.where(labs == c)[0], (labs == c).sum()) for c in (0, 1)])
            sel = [ids[k][i] for i in pick]
            dd.append(bacc([bl[k][i] for i in sel]) - bacc([al[k][i] for i in sel]))
        d.append(np.mean(dd))
    return [round(float(np.percentile(d, 2.5)), 4), round(float(np.percentile(d, 97.5)), 4)]


def main():
    names = sorted(set(CANDS) | set(CHOSEN.split("+")) | {"dfd"}, key=lambda n: (n != "dfd", n))
    new, rep = Ev.load_members(names)
    opts = {}
    pvs = {}
    for n in names:
        pvs[n] = per_video(new, rep, [n], None)
        opts[n] = summarise(pvs[n])
    mem = CHOSEN.split("+")
    lr, nval = Ev.fit_fusion(new, mem)
    pvs[CHOSEN] = per_video(new, rep, mem, lr)
    opts[CHOSEN] = summarise(pvs[CHOSEN])
    opts[CHOSEN]["cv_val_bacc"] = Ev.cv_score(new, mem)
    base = "dfd"                                                   # the live checker's detector today
    two = ["celebdf", "social", "mavos", "mavos_open"]
    rule = dict(gain_new_sets=opts[CHOSEN]["mean_new_bacc"] - opts[base]["mean_new_bacc"],
                change_ffpp=opts[CHOSEN]["ffpp"]["bacc"] - opts[base]["ffpp"]["bacc"],
                change_dfdc=opts[CHOSEN]["dfdc"]["bacc"] - opts[base]["dfdc"]["bacc"])
    rule["deploy"] = bool(rule["gain_new_sets"] >= 0.05 and rule["change_ffpp"] >= -0.02 and rule["change_dfdc"] >= -0.02)
    import os
    rule["live"] = os.environ.get("LIVE_DECISION", "rule")          # "rule" = the live checker follows the rule
    rule["live_reason"] = os.environ.get("LIVE_REASON", "")
    ci = dict(new_sets=boot(pvs[base], pvs[CHOSEN], two),
              ffpp=boot(pvs[base], pvs[CHOSEN], [f"ffpp:{c}" for c in Ev.REP_CONDS]),
              dfdc=boot(pvs[base], pvs[CHOSEN], [f"dfdc:{c}" for c in Ev.REP_CONDS]))
    sc = {}
    for n in names:
        if n.endswith("_sc") and n[:-3] in opts:
            sc[n[:-3]] = {k: dict(raw=opts[n[:-3]][k], cal=opts[n][k])
                          for k in [f"{g}:{c}" for g in ("ffpp", "dfdc") for c in Ev.REP_CONDS]}
    out = dict(chosen=CHOSEN, members=mem, n_val=nval, val_conds=list(Ev.VAL_CONDS), options=opts, selfcal=sc,
               decision=rule, ci95=ci,
               fusion=dict(members=mem, coef=lr.coef_[0].tolist(), intercept=float(lr.intercept_[0])),
               groups=GROUP_NEW + ["ffpp", "fsh", "dfdc"])
    for p in (ROOT / "results" / "ensemble" / "results.json", ROOT / "web" / "viva" / "src" / "ensemble.json"):
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(out, indent=1))
    # numbers for the checker website (web/checker/src/accuracy.json)
    pv = pvs[CHOSEN]
    sig = lambda s: 1 / (1 + np.exp(-s))  # noqa: E731

    def pooled(p):
        return [x for g in GROUP_NEW for x in p[g]] + [x for g in ("ffpp", "dfdc") for x in p[f"{g}:clean"]] + \
            [x for x in p["fsh:clean"] if x[1] == 1]
    pool = pooled(pv)
    right_base = np.array([(x[2] >= 0) == (x[1] == 1) for x in pooled(pvs[base])])
    conf = np.array([max(sig(x[2]), 1 - sig(x[2])) for x in pool])
    right = np.array([(x[2] >= 0) == (x[1] == 1) for x in pool])
    bands = []
    for lo, hi in ((0.5, 0.65), (0.65, 0.8), (0.8, 1.01)):
        sel = (conf >= lo) & (conf < hi)
        bands.append(dict(lo=lo, hi=hi, acc=float(right[sel].mean()) if sel.any() else None, share=float(sel.mean()),
                          n=int(sel.sum())))
    acc = lambda k, o=CHOSEN: Dt.metrics(*zip(*[(x[1], x[2]) for x in pvs[o][k]]))["acc"]  # noqa: E731
    checker = dict(videos=225, new_videos=int(sum(len(pv[g]) for g in GROUP_NEW)),
                   new_bacc=opts[CHOSEN]["mean_new_bacc"], new_bacc_original=opts[base]["mean_new_bacc"],
                   ffpp_clean_acc=acc("ffpp:clean"), ffpp_blur_acc=acc("ffpp:GB_5"),
                   ffpp_blur_acc_original=acc("ffpp:GB_5", base), dfdc_acc=acc("dfdc:clean"),
                   overall_acc=float(right.mean()), overall_acc_original=float(right_base.mean()), overall_videos=len(pool),
                   synth_caught=opts[CHOSEN]["synth"]["caught"], synth_n=opts[CHOSEN]["synth"]["n"],
                   datasets=len(GROUP_NEW) + 3, bands=bands, option=CHOSEN)
    (ROOT / "web" / "checker" / "src" / "accuracy.json").write_text(json.dumps(checker, indent=1))
    out["checker"] = checker
    for p in (ROOT / "results" / "ensemble" / "results.json", ROOT / "web" / "viva" / "src" / "ensemble.json"):
        p.write_text(json.dumps(out, indent=1))
    print(json.dumps(checker, indent=1))
    print(json.dumps(dict(chosen=CHOSEN, decision=rule, ci95=ci, fusion=out["fusion"]), indent=1))


if __name__ == "__main__":
    main()
