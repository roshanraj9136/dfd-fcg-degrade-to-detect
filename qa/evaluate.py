"""
Evaluate every decision rule on the FF++ test split under every condition.

Two regimes:
  all   corrections fitted on the fit split under all training conditions
  lopo  leave-one-perturbation-out: when testing perturbation type K, the corrections are
        fitted WITHOUT any condition of type K (the degradation is unseen)

Operating points:
  0.5   the model's native threshold
  tau5  the threshold that gives 5% false alarms on CLEAN fit-set real videos (a deployment
        would fix it once, on clean data)

    python -m qa.evaluate             # writes results/metrics.json
"""
import json

import numpy as np
from sklearn.metrics import roc_auc_score

from qa.config import PERTURB_PARAMS, ROOT, TEST_LEVELS
from qa.methods import all_methods
from qa.table import load

KINDS = list(PERTURB_PARAMS)
CORE = ["DF", "F2F", "FS", "NT"]


def video_level(T, mask, p):
    """Mean clip score per (vid, cond)."""
    keys = list(zip(T["vid"][mask], T["cond"][mask]))
    acc = {}
    for k, s, lbl, m in zip(keys, p, T["label"][mask], T["method"][mask]):
        a = acc.setdefault(k, [0.0, 0, lbl, m])
        a[0] += s
        a[1] += 1
    vids = np.array([k[0] for k in acc])
    conds = np.array([k[1] for k in acc])
    score = np.array([v[0] / v[1] for v in acc.values()])
    label = np.array([v[2] for v in acc.values()])
    meth = np.array([v[3] for v in acc.values()])
    return dict(vid=vids, cond=conds, score=score, label=label, method=meth)


def ece(y, s, bins=10):
    e = 0.0
    edges = np.linspace(0, 1, bins + 1)
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (s >= lo) & (s < hi if hi < 1 else s <= hi)
        if m.any():
            e += m.mean() * abs(s[m].mean() - y[m].mean())
    return float(e)


def metrics(y, s, taus):
    out = dict(n_real=int((y == 0).sum()), n_fake=int((y == 1).sum()),
               auroc=float(roc_auc_score(y, s)) if len(set(y)) == 2 else None,
               ece=ece(y, s), mean_real=float(s[y == 0].mean()), mean_fake=float(s[y == 1].mean()))
    for name, t in taus.items():
        fpr = float((s[y == 0] >= t).mean())
        tpr = float((s[y == 1] >= t).mean())
        out[f"fpr@{name}"], out[f"tpr@{name}"] = fpr, tpr
        out[f"bacc@{name}"] = 0.5 * (tpr + 1 - fpr)
    return out


def fit_tau(method, T, fit_mask):
    m = fit_mask & (T["cond"] == "clean") & (T["label"] == 0)
    v = video_level(T, m, method.predict(T, m))
    return float(np.quantile(v["score"], 0.95))


def conds_of(kind):
    return [f"{kind}_{lv}" for lv in TEST_LEVELS[kind]]


def run(T, methods=None):
    methods = methods or all_methods()
    fit = T["split"] == "fit"
    eval_conds = ["clean", "RE_0"] + [c for k in KINDS for c in conds_of(k)]
    test_core = ((T["split"] == "test") & np.isin(T["method"], ["REAL"] + CORE)
                 & np.isin(T["cond"], eval_conds))
    test_fsh = (T["split"] == "test") & np.isin(T["method"], ["REAL", "FSh"]) & (T["cond"] == "clean")
    res = {}
    for M in methods:
        r = res[M.name] = dict(all={}, lopo={}, fsh=None, rank=None)
        # ---------------- regime: all ----------------
        M.fit(T, fit)
        taus = {"0.5": 0.5, "tau5": fit_tau(M, T, fit)}
        r["tau5"] = taus["tau5"]
        if isinstance(getattr(M, "rank", None), dict):
            r["rank"] = {h: int(v) for h, v in M.rank.items()}
        if hasattr(M, "lam_"):
            r["lam"] = float(M.lam_)
            r["lam_curve"] = {str(k): v for k, v in getattr(M, "curve", {}).items()}
        v = video_level(T, test_core, M.predict(T, test_core))
        for c in sorted(set(v["cond"])):
            k = v["cond"] == c
            r["all"][c] = metrics(v["label"][k], v["score"][k], taus)
        vf = video_level(T, test_fsh, M.predict(T, test_fsh))
        r["fsh"] = metrics(vf["label"], vf["score"], taus)
        r["pooled_all"] = metrics(v["label"], v["score"], taus)
        r["per_method_clean"] = {
            mm: metrics(np.r_[np.zeros((v["label"][(v["cond"] == "clean") & (v["method"] == "REAL")]).size), np.ones(((v["cond"] == "clean") & (v["method"] == mm)).sum())],
                        np.r_[v["score"][(v["cond"] == "clean") & (v["method"] == "REAL")],
                              v["score"][(v["cond"] == "clean") & (v["method"] == mm)]], taus)
            for mm in CORE}
        # ---------------- regime: leave-one-perturbation-out ----------------
        if M.name in ("original",):
            r["lopo"] = r["all"]
            continue
        for kind in KINDS:
            fm = fit & ~np.char.startswith(T["cond"].astype(str), kind + "_")   # type K unseen
            M.fit(T, fm)
            taus_k = {"0.5": 0.5, "tau5": fit_tau(M, T, fm)}
            tm = test_core & np.isin(T["cond"], conds_of(kind))
            vk = video_level(T, tm, M.predict(T, tm))
            for c in conds_of(kind):
                k = vk["cond"] == c
                if k.any():
                    r["lopo"][c] = metrics(vk["label"][k], vk["score"][k], taus_k)
        for c in ("clean", "RE_0"):
            if c in r["all"]:
                r["lopo"][c] = r["all"][c]
    return res


def pair_groups(T, k=5):
    """Fold id per row: FF++ test pairs (a, b) keep a, b, a_b and b_a together."""
    import json as _json
    from qa.config import DATA as _D
    pairs = _json.load(open(_D / "splits" / "test.json"))
    g = {}
    for i, (a, b) in enumerate(pairs):
        for x in (a, b, f"{a}_{b}", f"{b}_{a}"):
            g[x] = i
    gid = np.array([g.get(v.split("/")[1], -1) for v in T["vid"]])
    return np.where(gid >= 0, gid % k, -1)


def run_cv(T, methods=None, k=5):
    """
    Second protocol: 5-fold cross-validation over FF++ TEST-split identity pairs (videos DFD-FCG
    never saw in training), out-of-fold predictions. 'cv_lopo' also hides the tested
    degradation type from the training folds.
    """
    methods = methods or all_methods()
    eval_conds = ["clean", "RE_0"] + [c for kk in KINDS for c in conds_of(kk)]
    base = ((T["split"] == "test") & np.isin(T["method"], ["REAL"] + CORE) & np.isin(T["cond"], eval_conds))
    fold = pair_groups(T, k)
    res = {}
    for M in methods:
        r = res[M.name] = dict(cv_all={}, cv_lopo={})
        for regime in ("cv_all", "cv_lopo"):
            oof = np.full(len(T["vid"]), np.nan)
            for f in range(k):
                kinds = [None] if regime == "cv_all" else KINDS
                for kind in kinds:
                    tr = base & (fold != f)
                    te = base & (fold == f)
                    if kind is not None:
                        tr &= ~np.char.startswith(T["cond"].astype(str), kind + "_")
                        te &= np.char.startswith(T["cond"].astype(str), kind + "_")
                    if not te.any():
                        continue
                    M.fit(T, tr)
                    oof[te] = M.predict(T, te)
            if regime == "cv_lopo":
                m0 = base & np.isin(T["cond"], ["clean", "RE_0"])
                oof[m0] = np.nan
            m = base & np.isfinite(oof)
            v = video_level(T, m, oof[m])
            for c in sorted(set(v["cond"])):
                kk = v["cond"] == c
                r[regime][c] = metrics(v["label"][kk], v["score"][kk], {"0.5": 0.5, "tau5": 0.5})
            if regime == "cv_all":
                r["cv_pooled"] = metrics(v["label"], v["score"], {"0.5": 0.5, "tau5": 0.5})
        for c in ("clean", "RE_0"):
            if c in r["cv_all"]:
                r["cv_lopo"][c] = r["cv_all"][c]
    return res


def summarise(res):
    deg = [c for k in KINDS for c in conds_of(k)]
    rows = []
    for name, r in res.items():
        for regime in ("all", "lopo", "cv_all", "cv_lopo"):
            if regime not in r:
                continue
            ms = [r[regime][c] for c in deg if c in r[regime]]
            if not ms:
                continue
            ref = r["all"] if regime in ("all", "lopo") else r["cv_all"]
            row = dict(method=name, regime=regime,
                       clean_auroc=ref["clean"]["auroc"],
                       clean_fpr05=ref["clean"]["fpr@0.5"],
                       clean_tpr05=ref["clean"]["tpr@0.5"])
            for key in ("auroc", "fpr@0.5", "tpr@0.5", "bacc@0.5", "fpr@tau5", "tpr@tau5", "bacc@tau5", "ece"):
                vals = [m[key] for m in ms]
                row[f"deg_mean_{key}"] = float(np.mean(vals))
                row[f"deg_worst_{key}"] = float(np.max(vals) if key.startswith("fpr") or key == "ece" else np.min(vals))
            row["n_conds_fpr05_over_50pct"] = int(sum(m["fpr@0.5"] > 0.5 for m in ms))
            if regime == "all":
                row["pooled_auroc"] = r["pooled_all"]["auroc"]
                row["pooled_bacc@0.5"] = r["pooled_all"]["bacc@0.5"]
                row["fsh_auroc"] = r["fsh"]["auroc"]
                row["fsh_tpr@0.5"] = r["fsh"]["tpr@0.5"]
            if regime == "cv_all":
                row["pooled_auroc"] = r["cv_pooled"]["auroc"]
                row["pooled_bacc@0.5"] = r["cv_pooled"]["bacc@0.5"]
            rows.append(row)
    return rows


def main():
    T = load()
    res = run(T)
    cv = run_cv(T)
    for name, r in cv.items():
        res.setdefault(name, {}).update(r)
    summary = summarise(res)
    out = ROOT / "results"
    out.mkdir(exist_ok=True)
    json.dump(dict(per_condition=res, summary=summary), open(out / "metrics.json", "w"), indent=1)
    keys = ["method", "regime", "clean_auroc", "clean_fpr05", "clean_tpr05", "deg_mean_auroc", "deg_mean_fpr@0.5",
            "deg_worst_fpr@0.5", "deg_mean_tpr@0.5", "deg_mean_bacc@0.5", "deg_mean_fpr@tau5",
            "deg_mean_tpr@tau5", "pooled_auroc", "pooled_bacc@0.5"]
    print(" | ".join(keys))
    for row in summary:
        print(" | ".join(f"{row.get(k):.3f}" if isinstance(row.get(k), float) else str(row.get(k, "")) for k in keys))


if __name__ == "__main__":
    main()
