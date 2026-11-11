"""
Export results for the website (static JSON) and for the Neon database.

    python -m qa.export          # writes web/frontend/public/data/*.json
"""
import json

import cv2
import numpy as np

from qa.analysis import REGIONS, region_map
from qa.config import DATA, PERTURB_NAMES, PERTURB_PARAMS, ROOT, TEST_LEVELS
from qa.evaluate import CORE, KINDS
from qa.extract import PROBES_RUN
from qa.methods import ProbeContrast, ProbeLR, SelfCal, response, sig
from qa.prep import safe
from qa.table import load

OUT = ROOT / "web" / "frontend" / "public" / "data"
PUBLIC = {"REAL/000", "DF/000_003"}         # bundled with the upstream DFD-FCG repository
RCODE = {r: i for i, r in enumerate(REGIONS + ["background"])}
MAIN = "selfcal"                            # headline correction
STRICT = "selfcal+d2d"                      # optional low-false-alarm operating mode
EVAL_CONDS = ["clean", "RE_0"] + [f"{k}_{lv}" for k in PERTURB_PARAMS for lv in TEST_LEVELS[k]]


def r3(x):
    return np.round(np.asarray(x, float), 3).tolist()


def clean(o):
    """JSON-safe: NaN/inf -> None (browsers and Postgres reject NaN)."""
    if isinstance(o, float):
        return o if np.isfinite(o) else None
    if isinstance(o, dict):
        return {k: clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [clean(v) for v in o]
    if isinstance(o, np.generic):
        return clean(o.item())
    return o


def dump(obj, path, **kw):
    fh = path if hasattr(path, "write") else open(path, "w")
    with fh:
        json.dump(clean(obj), fh, allow_nan=False, **kw)


def main_method():
    return SelfCal()


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    T = load()
    fit = T["split"] == "fit"
    M = main_method().fit(T, fit)
    S = SelfCal(probe_after=True).fit(T, fit)
    D2D = ProbeContrast("pB2").fit(T, fit)
    LR = ProbeLR().fit(T, fit)
    lopo = {k: main_method().fit(T, fit & ~np.char.startswith(T["cond"].astype(str), k + "_")) for k in KINDS}

    test = np.isin(T["split"], ["test", "demo", "cross"]) & np.isin(T["cond"], EVAL_CONDS)
    idx_all = np.where(test)[0]
    p_main = np.full(len(T["vid"]), np.nan)
    p_main[test] = M.predict(T, test)
    p_strict = np.full(len(T["vid"]), np.nan)
    p_strict[test] = S.predict(T, test)
    drift = np.zeros((len(T["vid"]), 3))
    drift[test] = M.drift(T, test)
    p_lopo = p_main.copy()
    for k, mk in lopo.items():
        m = test & np.char.startswith(T["cond"].astype(str), k + "_")
        if m.any():
            p_lopo[m] = mk.predict(T, m)
    resp = {pr: response(T, pr) for pr in PROBES_RUN}

    videos = {}
    for i in idx_all:
        v = videos.setdefault(T["vid"][i], dict(vid=T["vid"][i], split=T["split"][i], method=T["method"][i],
                                                label=int(T["label"][i]), conds={}))
        c = v["conds"].setdefault(T["cond"][i], dict(p=[], pm=[], pl=[], ps=[], g=[], d=[],
                                                     r={pr: [] for pr in PROBES_RUN}, s=[], t=[], sp=[], tp=[]))
        c["p"].append(T["p_final"][i])
        c["pm"].append(p_main[i])
        c["pl"].append(p_lopo[i])
        c["ps"].append(p_strict[i])
        c["g"].append(drift[i])
        c["d"].append(T["d"][i].astype(float))
        for pr in PROBES_RUN:
            c["r"][pr].append(resp[pr][i])
        c["s"].append(T["cs_patch"][i])
        c["t"].append(T["ct_patch"][i])
        c["sp"].append(T["probe:pB2:cs_patch"][i] - T["cs_patch"][i])
        c["tp"].append(T["probe:pB2:ct_patch"][i] - T["ct_patch"][i])

    index = []
    (OUT / "maps").mkdir(exist_ok=True)
    for vid, v in videos.items():
        z = np.load(DATA / "prep" / (safe(vid) + ".npz"))
        lm = z["crop_lm"].mean(0)
        rm = region_map(lm)
        scores, maps = {}, {}
        for cond, c in v["conds"].items():
            mean = lambda x: float(np.nanmean(x))  # noqa: E731
            scores[cond] = dict(p=round(mean(c["p"]), 4), p_fix=round(mean(c["pm"]), 4),
                                p_fix_unseen=round(mean(c["pl"]), 4), p_strict=round(mean(c["ps"]), 4),
                                g=r3(np.mean(c["g"], 0)), d=r3(np.mean(c["d"], 0)),
                                r={pr: r3(np.nanmean(c["r"][pr], 0)) for pr in PROBES_RUN})
            maps[cond] = dict(s=r3(np.mean(c["s"], 0)), t=r3(np.mean(c["t"], 0)),
                              sp=r3(np.nan_to_num(np.nanmean(c["sp"], 0))), tp=r3(np.nan_to_num(np.nanmean(c["tp"], 0))))
        public = vid in PUBLIC
        index.append(dict(vid=vid, split=v["split"], method=v["method"], label=v["label"], public=public,
                          scores=scores))
        dump(dict(landmarks=r3(lm), regions=[RCODE[x] for x in rm], maps=maps),
                  open(OUT / "maps" / (safe(vid) + ".json"), "w"), separators=(",", ":"))
        if public:
            export_thumbnail(vid, z)
    dump(index, open(OUT / "videos.json", "w"), separators=(",", ":"))

    metrics = json.load(open(ROOT / "results" / "metrics.json"))
    analysis = json.load(open(ROOT / "results" / "analysis.json"))
    dump(metrics["per_condition"], open(OUT / "conditions.json", "w"), separators=(",", ":"))
    dump(analysis, open(OUT / "analysis.json", "w"), separators=(",", ":"))
    arr = np.load(ROOT / "results" / "arrays.npz")
    layers = {}
    for k in arr.files:
        if k.startswith("drift_s_"):
            c = k[len("drift_s_"):]
            layers[c] = dict(s=r3(arr[k]), t=r3(arr["drift_t_" + c]))
        if k.startswith("frag_s_"):
            c = "frag:" + k[len("frag_s_"):]
            layers[c] = dict(s=r3(arr[k]), t=r3(arr["frag_t_" + k[len("frag_s_"):]]))
    dump(layers, open(OUT / "drift_layers.json", "w"), separators=(",", ":"))

    summary = dict(
        table=metrics["summary"], main=MAIN, strict=STRICT, d2d_lam=float(D2D.lam_),
        d2d_lam_curve={str(k): v for k, v in D2D.curve.items()}, strict_lam=float(S.lam), alpha=float(M.alpha),
        drift_targets=M.targets,
        probes={"pB2": "Gaussian blur, sigma = 2 px on the 150x150 aligned face",
                "pJ50": "JPEG re-compression at quality 50 of the aligned face"},
        n=dict(test_real=int(sum(1 for v in index if v["split"] == "test" and v["label"] == 0)),
               test_fake=int(sum(1 for v in index if v["split"] == "test" and v["method"] in CORE)),
               fit_real=int(len(set(T["vid"][fit & (T["label"] == 0)]))),
               fit_fake=int(len(set(T["vid"][fit & (T["label"] == 1)]))),
               fsh=int(sum(1 for v in index if v["method"] == "FSh")),
               dfdc_real=int(sum(1 for v in index if v["method"] == "DFDC_R")),
               dfdc_fake=int(sum(1 for v in index if v["method"] == "DFDC_F"))),
        conditions={c: PERTURB_NAMES.get(c.split("_")[0], c) for c in EVAL_CONDS},
        params={k: v for k, v in PERTURB_PARAMS.items()},
        regions=REGIONS + ["background"],
    )
    dump(summary, open(OUT / "summary.json", "w"), indent=1)
    # the fitted decision rules for the API
    model = dict(main=MAIN, selfcal=dict(mean=M.sc.mean_.tolist(), scale=M.sc.scale_.tolist(),
                                         coef=M.reg.coef_.tolist(), intercept=M.reg.intercept_.tolist()),
                 strict_lam=float(S.lam), d2d_lam=float(D2D.lam_), probe="pB2",
                 lr=dict(probes=LR.probes, mean=LR.sc.mean_.tolist(), scale=LR.sc.scale_.tolist(),
                         coef=LR.lr.coef_[0].tolist(), intercept=float(LR.lr.intercept_[0])))
    dump(model, open(OUT / "model.json", "w"), separators=(",", ":"))
    # one real example for the API playground: a degraded real video that the released model flags as FAKE
    # (preferably one the self-calibration corrects; the aggregate numbers are reported separately)
    base = test & (T["label"] == 0) & (T["cond"] != "clean") & (T["p_final"] >= 0.5) & np.isfinite(T["probe:pB2:d"]).all(1)
    cand = np.where(base & (p_main < 0.5))[0]
    if not len(cand):
        cand = np.where(base)[0]
    if not len(cand):
        cand = np.where(test & (T["label"] == 0) & (T["cond"] != "clean") & np.isfinite(T["probe:pB2:d"]).all(1))[0]
    i = int(cand[np.argmax(T["p_final"][cand])])
    example = dict(vid=T["vid"][i], cond=T["cond"][i], clip=int(T["clip"][i]),
                   d=np.round(T["d"][i].astype(float), 6).tolist(),
                   y_s=np.round(T["y_s"][i].astype(float), 5).tolist(), y_t=np.round(T["y_t"][i].astype(float), 5).tolist(),
                   d_probe={pr: np.round(T[f"probe:{pr}:d"][i].astype(float), 6).tolist() for pr in PROBES_RUN})
    dump(example, open(OUT / "example.json", "w"))
    print(f"exported {len(index)} videos to {OUT}")


def export_thumbnail(vid, z):
    """For the two samples bundled with the upstream repo: the face crop under every test condition."""
    from qa.conditions import build
    from qa.manifest import local_video, read_manifest
    row = [r for r in read_manifest() if r["vid"] == vid][0]
    prep = {k: z[k] for k in z.files}
    conds = EVAL_CONDS + ["clean+pB2", "clean+pJ50"]
    built = build(local_video(row), prep, vid, conds)
    dst = ROOT / "web" / "frontend" / "public" / "samples"
    dst.mkdir(parents=True, exist_ok=True)
    for c, v in built.items():
        name = safe(vid) + ("" if c == "clean" else f"__{c.replace('+', '_')}") + ".png"
        cv2.imwrite(str(dst / name), cv2.cvtColor(v[0][0], cv2.COLOR_RGB2BGR))


if __name__ == "__main__":
    main()
