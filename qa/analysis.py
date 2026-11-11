"""
Explainability analyses at scale (clean and degraded FF++ test videos).

  A. replication of the earlier two-video findings (github.com/roshanraj9136/dfd-fcg-explainability)
     on 100 real + 100 fake test videos: component shares, branch shares, regions, layer depth
  B. drift anatomy: where in the model degradation adds FAKE evidence to REAL videos
  C. geometry: is the degradation drift low-dimensional, does it transfer to unseen degradations,
     and is it separate from the forgery direction?
  D. quality-counterfactual explanations: how much of each false alarm is nuisance evidence
  E. manipulation-method fingerprints from the explanation alone

    python -m qa.analysis           # writes results/analysis.json and results/arrays.npz
"""
import json

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from qa.config import DATA, LEVELS, PERTURB_PARAMS, ROOT
from qa.engine import FACE_PARTS
from qa.methods import EvidenceNulling, Projection, head_terms, pairs, paired_drift, sig
from qa.prep import safe
from qa.quality import QB_NAMES

REGIONS = ["eyes", "nose", "lips", "skin"]
LM = {"eyes": list(range(36, 48)) + list(range(17, 27)), "nose": list(range(27, 36)),
      "lips": list(range(48, 68))}
CORE = ["DF", "F2F", "FS", "NT"]
KINDS = list(PERTURB_PARAMS)


# ------------------------------------------------------------------ regions
def region_map(landmarks, grid=16, crop=150, near=12.0):
    """Same rule as the earlier work: nearest landmark group within 12 px, else skin inside the hull."""
    from matplotlib.path import Path as MplPath
    from scipy.spatial import ConvexHull
    step = crop / grid
    centres = np.array([[(c + .5) * step, (r + .5) * step] for r in range(grid) for c in range(grid)])
    hull = ConvexHull(landmarks)
    inside = MplPath(landmarks[hull.vertices])
    labels = []
    for p in centres:
        best, bestd = None, 1e9
        for name, idxs in LM.items():
            d = np.linalg.norm(landmarks[idxs] - p, axis=1).min()
            if d < bestd:
                best, bestd = name, d
        labels.append(best if bestd <= near else ("skin" if inside.contains_point(p) else "background"))
    return np.array(labels)


_RMAP = {}


def rmap(vid):
    if vid not in _RMAP:
        z = np.load(DATA / "prep" / (safe(vid) + ".npz"))
        _RMAP[vid] = region_map(z["crop_lm"].mean(0))
    return _RMAP[vid]


def region_sums(T, idx, key):
    out = np.zeros((len(idx), len(REGIONS)))
    for n, i in enumerate(idx):
        m = rmap(T["vid"][i])
        for k, r in enumerate(REGIONS):
            out[n, k] = T[key][i][m == r].sum()
    return out


def shares(x):
    a = np.abs(x)
    return a / np.maximum(a.sum(1, keepdims=True), 1e-12)


def shapley(T, idx):
    d = T["d"][idx].astype(np.float64)
    ps, pt, pf = sig(d[:, 0]), sig(d[:, 1]), T["p_final"][idx]
    phs = 0.5 * ((ps - 0.5) + (pf - pt))
    pht = 0.5 * ((pt - 0.5) + (pf - ps))
    tot = np.abs(phs) + np.abs(pht)
    return np.abs(phs) / tot, ps, pt


def vmean(T, idx, values):
    """Average clip values to video level."""
    acc = {}
    for i, v in zip(idx, values):
        acc.setdefault(T["vid"][i], []).append(v)
    return np.array([np.mean(v, 0) for v in acc.values()]), list(acc)


# ------------------------------------------------------------------ A. replication
def replication(T):
    res = {}
    clean = (T["split"] == "test") & (T["cond"] == "clean")
    for grp, methods in [("REAL", ["REAL"]), ("FAKE", CORE)] + [(m, [m]) for m in CORE + ["FSh"]]:
        idx = np.where(clean & np.isin(T["method"], methods))[0]
        if not len(idx):
            continue
        parts = T["cs_lq"][idx].sum(1)                               # (n,4) spatial head per query
        sreg = region_sums(T, idx, "cs_patch")
        treg = region_sums(T, idx, "ct_patch")
        s_share, ps, pt = shapley(T, idx)
        lay_s = T["cs_lq"][idx].sum(2)                               # (n,24)
        lay_t = T["ct_l"][idx]
        pv, _ = vmean(T, idx, T["p_final"][idx])
        psv, _ = vmean(T, idx, ps)
        ptv, _ = vmean(T, idx, pt)
        lbl = 0 if grp == "REAL" else 1
        res[grp] = dict(
            n_videos=int(len(pv)),
            accuracy=float(((pv >= .5) == lbl).mean()),
            acc_spatial_only=float(((psv >= .5) == lbl).mean()),
            acc_temporal_only=float(((ptv >= .5) == lbl).mean()),
            query_signed=dict(zip(FACE_PARTS, parts.mean(0).round(4).tolist())),
            query_share=dict(zip(FACE_PARTS, shares(parts).mean(0).round(4).tolist())),
            dominant_query_freq=dict(zip(FACE_PARTS, np.bincount(np.abs(parts).argmax(1), minlength=4)
                                         .__truediv__(len(idx)).round(3).tolist())),
            region_spatial_signed=dict(zip(REGIONS, sreg.mean(0).round(4).tolist())),
            region_spatial_share=dict(zip(REGIONS, shares(sreg).mean(0).round(4).tolist())),
            region_temporal_signed=dict(zip(REGIONS, treg.mean(0).round(4).tolist())),
            region_temporal_share=dict(zip(REGIONS, shares(treg).mean(0).round(4).tolist())),
            spatial_share_shapley=float(s_share.mean()),
            top_layers_spatial=(np.argsort(-np.abs(lay_s).mean(0))[:3] + 1).tolist(),
            top_layers_temporal=(np.argsort(-np.abs(lay_t).mean(0))[:3] + 1).tolist(),
            layer_profile_spatial=np.abs(lay_s).mean(0).round(4).tolist(),
            layer_profile_temporal=np.abs(lay_t).mean(0).round(4).tolist(),
        )
    return res


# ------------------------------------------------------------------ B. drift anatomy
def drift_anatomy(T):
    """Paired change (degraded - clean) for test REAL videos."""
    real = (T["split"] == "test") & (T["method"] == "REAL")
    terms = head_terms(T)
    out, arrays = {}, {}
    key = {(T["vid"][i], T["clip"][i]): i for i in np.where(real & (T["cond"] == "clean"))[0]}
    for cond in sorted(set(T["cond"][real]) - {"clean"}):
        idx = np.where(real & (T["cond"] == cond))[0]
        src = np.array([key[(T["vid"][i], T["clip"][i])] for i in idx])
        dd = T["d"][idx] - T["d"][src]
        ds = (T["cs_lq"][idx] - T["cs_lq"][src]).mean(0)             # (24,4)
        dt = (T["ct_l"][idx] - T["ct_l"][src]).mean(0)               # (24,)
        sreg = (region_sums(T, idx, "cs_patch") - region_sums(T, src, "cs_patch")).mean(0)
        treg = (region_sums(T, idx, "ct_patch") - region_sums(T, src, "ct_patch")).mean(0)
        dq = (T["q"][idx] - T["q"][src]).mean(0)
        out[cond] = dict(
            delta_d=dict(zip(["spatial", "temporal", "fusion"], dd.mean(0).round(4).tolist())),
            delta_p=float((T["p_final"][idx] - T["p_final"][src]).mean()),
            delta_spatial_query=dict(zip(FACE_PARTS, ds.sum(0).round(4).tolist())),
            delta_spatial_region=dict(zip(REGIONS, sreg.round(4).tolist())),
            delta_temporal_region=dict(zip(REGIONS, treg.round(4).tolist())),
            shallow_vs_deep_temporal=[float(dt[:12].sum()), float(dt[12:].sum())],
            shallow_vs_deep_spatial=[float(ds[:12].sum()), float(ds[12:].sum())],
            delta_quality=dq.round(4).tolist(),
        )
        arrays[f"drift_s_{cond}"] = ds
        arrays[f"drift_t_{cond}"] = dt
    return out, arrays


# ------------------------------------------------------------------ C. geometry
def geometry(T):
    """
    1. Is the degradation drift low-dimensional, and how aligned is it with the forgery direction?
       (why projecting the attribution alone cannot work)
    2. Evidence Nulling fitted WITHOUT a degradation type: how much of that unseen type's logit
       drift (paired, real test videos) does it predict?
    """
    fit = T["split"] == "fit"
    test_real = (T["split"] == "test") & (T["method"] == "REAL")
    terms = head_terms(T)
    res = {}
    proj = Projection(rank=1).fit(T, fit)
    for h in "sta":
        clean = T["cond"] == "clean"
        fr = fit & (T["label"] == 0) & clean
        ff = fit & (T["label"] == 1) & clean
        drift, _, _ = paired_drift(T, terms, fit & (T["label"] == 0))
        g = drift[h].mean(0)
        f = terms[h][ff].mean(0) - terms[h][fr].mean(0)
        res[h] = dict(
            drift_energy_top=proj.energy[h][:6].round(4).tolist(),
            cos_mean_drift_vs_forgery=float(g @ f / (np.linalg.norm(g) * np.linalg.norm(f) + 1e-12)),
            cos_top_drift_dir_vs_forgery=float(abs(proj.U[h][:, 0] @ f) / (np.linalg.norm(f) + 1e-12)),
        )
    transfer = {}
    for kind in KINDS:
        fm = fit & ~np.char.startswith(T["cond"].astype(str), kind + "_")
        en = EvidenceNulling().fit(T, fm)
        tm = test_real & (np.char.startswith(T["cond"].astype(str), kind + "_") | (T["cond"] == "clean"))
        tm &= np.isfinite(T["q_bg"]).all(1)
        src, dst = pairs(T, tm)
        if not len(src):
            continue
        transfer[kind] = {}
        idx = np.where(tm)[0]
        pos = np.full(len(T["vid"]), -1)
        pos[idx] = np.arange(len(idx))
        nu = en.nuisance(T, tm)
        for k, h in enumerate("sta"):
            dd = T["d"][dst, k] - T["d"][src, k]
            pred = nu[pos[dst], k] - nu[pos[src], k]
            r2 = 1 - np.sum((dd - pred) ** 2) / max(np.sum((dd - dd.mean()) ** 2), 1e-12)
            transfer[kind][h] = dict(logit_drift_explained=float(r2), mean_logit_drift=float(dd.mean()),
                                     mean_logit_drift_after=float((dd - pred).mean()))
    res["transfer_unseen"] = transfer
    return res


# ------------------------------------------------------------------ D. counterfactual explanations
def counterfactual(T, en):
    """Quality share of each verdict: how much of the FAKE logit is predicted quality evidence."""
    test = (T["split"] == "test") & np.isin(T["method"], ["REAL"] + CORE)
    idx = np.where(test)[0]
    nuis = en.nuisance(T, test)                                         # (n,3)
    d = T["d"][idx].astype(np.float64)
    p_orig = T["p_final"][idx]
    p_corr = sig(d - nuis).mean(1)
    rows = {}
    for n, i in enumerate(idx):
        k = (T["vid"][i], T["cond"][i])
        r = rows.setdefault(k, dict(label=int(T["label"][i]), p=[], pc=[], nu=[], d=[]))
        r["p"].append(p_orig[n])
        r["pc"].append(p_corr[n])
        r["nu"].append(nuis[n])
        r["d"].append(d[n])
    fa_share, fa_fixed, n_fa = [], 0, 0
    for (vid, cond), r in rows.items():
        p, pc = np.mean(r["p"]), np.mean(r["pc"])
        if r["label"] == 0 and cond != "clean" and p >= 0.5:
            n_fa += 1
            nu = np.mean(r["nu"], 0)
            dd = np.mean(r["d"], 0)
            # share of the positive (FAKE) logit mass that is nuisance evidence, averaged over heads
            fa_share.append(float(np.mean(np.clip(nu, 0, None) / np.maximum(np.abs(dd) + np.abs(nu), 1e-9))))
            fa_fixed += int(pc < 0.5)
    return dict(n_false_alarms=n_fa, false_alarms_fixed=fa_fixed,
                mean_nuisance_share_in_false_alarms=float(np.mean(fa_share)) if fa_share else None)


# ------------------------------------------------------------------ E. fingerprints
def fingerprint_features(T, idx):
    parts = T["cs_lq"][idx].sum(1)
    sreg = region_sums(T, idx, "cs_patch")
    treg = region_sums(T, idx, "ct_patch")
    s_share, _, _ = shapley(T, idx)
    return np.concatenate([parts, shares(parts), sreg, treg, shares(sreg), shares(treg),
                           T["cs_lq"][idx].sum(2), T["ct_l"][idx], T["d"][idx], s_share[:, None]], 1)


def fingerprints(T):
    m = (T["cond"] == "clean") & np.isin(T["method"], CORE) & np.isin(T["split"], ["test", "fit"])
    idx = np.where(m)[0]
    X, vids = vmean(T, idx, fingerprint_features(T, idx))
    meth = {T["vid"][i]: T["method"][i] for i in idx}
    y = np.array([CORE.index(meth[v]) for v in vids])
    groups = np.array([v.split("/")[1].split("_")[0] for v in vids])     # target identity
    clf = make_pipeline(StandardScaler(), LogisticRegression(C=0.3, max_iter=5000))
    pred = cross_val_predict(clf, X, y, cv=GroupKFold(5), groups=groups)
    # same classifier on the raw decision only (3 head logits) as a control
    Xd = X[:, -4:-1]
    pred_d = cross_val_predict(make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000)),
                               Xd, y, cv=GroupKFold(5), groups=groups)
    return dict(n_videos=int(len(y)), classes=CORE, chance=0.25,
                accuracy_explanation=float((pred == y).mean()),
                accuracy_logits_only=float((pred_d == y).mean()),
                confusion=confusion_matrix(y, pred, labels=range(4)).tolist())


# ------------------------------------------------------------------ F. fragility of the evidence
def fragility(T):
    """
    How the model's evidence responds to a known test-time probe (blur / JPEG in crop space),
    separately for real and fake videos, overall and per exact term / face region.
    """
    from qa.extract import PROBES_RUN
    from sklearn.metrics import roc_auc_score
    out, arrays = {}, {}
    test = (T["split"] == "test") & np.isin(T["method"], ["REAL"] + CORE + ["FSh"])
    for pr in PROBES_RUN:
        ok = np.isfinite(T[f"probe:{pr}:d"]).all(1)
        o = out[pr] = {"by_input": {}}
        for cond in ["clean", "RE_0", "CS_3", "CC_3", "BW_3", "GNC_3", "GB_1", "GB_3", "GB_5",
                     "JPEG_1", "JPEG_3", "JPEG_5", "VC_1", "VC_3", "VC_5"]:
            m = test & ok & (T["cond"] == cond) & np.isin(T["method"], ["REAL"] + CORE)
            if not m.any() or len(set(T["label"][m])) < 2:
                continue
            idx = np.where(m)[0]
            r = (T[f"probe:{pr}:d"][idx] - T["d"][idx]).astype(float)
            rv, vids = vmean(T, idx, r)
            lab = np.array([T["label"][T["vid"] == v][0] for v in vids])
            o["by_input"][cond] = dict(
                real_mean=rv[lab == 0].mean(0).round(4).tolist(), fake_mean=rv[lab == 1].mean(0).round(4).tolist(),
                auroc_minus_response_fusion=float(roc_auc_score(lab, -rv[:, 2])),
                auroc_minus_response_mean=float(roc_auc_score(lab, -rv.mean(1))),
                frac_fakes_drop=float((rv[lab == 1, 2] < 0).mean()), frac_reals_rise=float((rv[lab == 0, 2] >= 0).mean()))
        # anatomy on clean inputs: per term and per region, reals vs fakes (and per manipulation method)
        m = test & ok & (T["cond"] == "clean")
        for grp, methods in [("REAL", ["REAL"]), ("FAKE", CORE)] + [(mm, [mm]) for mm in CORE + ["FSh"]]:
            idx = np.where(m & np.isin(T["method"], methods))[0]
            if not len(idx):
                continue
            ds = (T[f"probe:{pr}:cs_lq"][idx] - T["cs_lq"][idx]).mean(0)      # (24,4)
            dt = (T[f"probe:{pr}:ct_l"][idx] - T["ct_l"][idx]).mean(0)        # (24,)
            ps = np.zeros((len(idx), len(REGIONS)))
            pt = np.zeros((len(idx), len(REGIONS)))
            for n, i in enumerate(idx):
                rm = rmap(T["vid"][i])
                for k, rg in enumerate(REGIONS):
                    ps[n, k] = (T[f"probe:{pr}:cs_patch"][i] - T["cs_patch"][i])[rm == rg].sum()
                    pt[n, k] = (T[f"probe:{pr}:ct_patch"][i] - T["ct_patch"][i])[rm == rg].sum()
            r = (T[f"probe:{pr}:d"][idx] - T["d"][idx]).astype(float)
            pos = np.clip(T["cs_lq"][idx], 0, None).sum((1, 2))
            o[grp] = dict(n_clips=int(len(idx)), mean_response=r.mean(0).round(4).tolist(),
                          query=dict(zip(FACE_PARTS, ds.sum(0).round(4).tolist())),
                          region_spatial=dict(zip(REGIONS, ps.mean(0).round(4).tolist())),
                          region_temporal=dict(zip(REGIONS, pt.mean(0).round(4).tolist())),
                          shallow_deep_spatial=[float(ds[:12].sum()), float(ds[12:].sum())],
                          shallow_deep_temporal=[float(dt[:12].sum()), float(dt[12:].sum())],
                          spatial_positive_evidence=float(pos.mean()))
            arrays[f"frag_s_{pr}_{grp}"] = ds
            arrays[f"frag_t_{pr}_{grp}"] = dt
    return out, arrays


def main():
    from qa.table import load
    T = load()
    out = {}
    out["replication"] = replication(T)
    out["drift"], arrays = drift_anatomy(T)
    out["geometry"] = geometry(T)
    en = EvidenceNulling().fit(T, T["split"] == "fit")
    out["counterfactual"] = counterfactual(T, en)
    out["fingerprints"] = fingerprints(T)
    out["fragility"], frag_arrays = fragility(T)
    arrays.update(frag_arrays)
    out["en"] = dict(n_pairs=en.n_pairs, descriptors=QB_NAMES[:7])
    # predicted vs measured drift per term (all-type fit) for the website / figures
    test_real = (T["split"] == "test") & (T["method"] == "REAL") & np.isfinite(T["q_bg"]).all(1)
    for cond in ["GB_5", "JPEG_5", "VC_5", "GNC_3"]:
        m = test_real & (T["cond"] == cond)
        if m.any():
            arrays[f"pred_s_{cond}"] = (en.term_nuisance(T, m, "s").mean(0)
                                        - en.term_nuisance(T, test_real & (T["cond"] == "clean"), "s").mean(0)).reshape(24, 4)
            arrays[f"pred_t_{cond}"] = (en.term_nuisance(T, m, "t").mean(0)
                                        - en.term_nuisance(T, test_real & (T["cond"] == "clean"), "t").mean(0))
    (ROOT / "results").mkdir(exist_ok=True)
    json.dump(out, open(ROOT / "results" / "analysis.json", "w"), indent=1)
    np.savez_compressed(ROOT / "results" / "arrays.npz", **arrays)
    print(json.dumps({k: out[k] for k in ["counterfactual", "fingerprints", "en"]}, indent=1))
    print(json.dumps({p: v["by_input"] for p, v in out["fragility"].items()}, indent=1)[:4000])


if __name__ == "__main__":
    main()
