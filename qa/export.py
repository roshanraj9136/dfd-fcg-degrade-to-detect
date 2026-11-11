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
from qa.methods import ProbeContrast, ProbeLR, response, sig
from qa.prep import safe
from qa.table import load

OUT = ROOT / "web" / "frontend" / "public" / "data"
PUBLIC = {"REAL/000", "DF/000_003"}         # bundled with the upstream DFD-FCG repository
RCODE = {r: i for i, r in enumerate(REGIONS + ["background"])}
MAIN = "d2d-pB2"                            # headline method (training-free, one extra pass)
EVAL_CONDS = ["clean", "RE_0"] + [f"{k}_{lv}" for k in PERTURB_PARAMS for lv in TEST_LEVELS[k]]


def r3(x):
    return np.round(np.asarray(x, float), 3).tolist()


def main_method():
    return ProbeContrast("pB2")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    T = load()
    fit = T["split"] == "fit"
    M = main_method().fit(T, fit)
    LR = ProbeLR().fit(T, fit)
    lopo = {k: main_method().fit(T, fit & ~np.char.startswith(T["cond"].astype(str), k + "_")) for k in KINDS}

    test = np.isin(T["split"], ["test", "demo"]) & np.isin(T["cond"], EVAL_CONDS)
    idx_all = np.where(test)[0]
    p_main = np.full(len(T["vid"]), np.nan)
    p_main[test] = M.predict(T, test)
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
        c = v["conds"].setdefault(T["cond"][i], dict(p=[], pm=[], pl=[], d=[], r={pr: [] for pr in PROBES_RUN},
                                                     s=[], t=[], sp=[], tp=[]))
        c["p"].append(T["p_final"][i])
        c["pm"].append(p_main[i])
        c["pl"].append(p_lopo[i])
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
            scores[cond] = dict(p=round(mean(c["p"]), 4), p_d2d=round(mean(c["pm"]), 4),
                                p_d2d_unseen=round(mean(c["pl"]), 4), d=r3(np.mean(c["d"], 0)),
                                r={pr: r3(np.nanmean(c["r"][pr], 0)) for pr in PROBES_RUN})
            maps[cond] = dict(s=r3(np.mean(c["s"], 0)), t=r3(np.mean(c["t"], 0)),
                              sp=r3(np.nan_to_num(np.nanmean(c["sp"], 0))), tp=r3(np.nan_to_num(np.nanmean(c["tp"], 0))))
        public = vid in PUBLIC
        index.append(dict(vid=vid, split=v["split"], method=v["method"], label=v["label"], public=public,
                          scores=scores))
        json.dump(dict(landmarks=r3(lm), regions=[RCODE[x] for x in rm], maps=maps),
                  open(OUT / "maps" / (safe(vid) + ".json"), "w"), separators=(",", ":"))
        if public:
            export_thumbnail(vid, z)
    json.dump(index, open(OUT / "videos.json", "w"), separators=(",", ":"))

    metrics = json.load(open(ROOT / "results" / "metrics.json"))
    analysis = json.load(open(ROOT / "results" / "analysis.json"))
    json.dump(metrics["per_condition"], open(OUT / "conditions.json", "w"), separators=(",", ":"))
    json.dump(analysis, open(OUT / "analysis.json", "w"), separators=(",", ":"))
    arr = np.load(ROOT / "results" / "arrays.npz")
    layers = {}
    for k in arr.files:
        if k.startswith("drift_s_"):
            c = k[len("drift_s_"):]
            layers[c] = dict(s=r3(arr[k]), t=r3(arr["drift_t_" + c]))
        if k.startswith("frag_s_"):
            c = "frag:" + k[len("frag_s_"):]
            layers[c] = dict(s=r3(arr[k]), t=r3(arr["frag_t_" + k[len("frag_s_"):]]))
    json.dump(layers, open(OUT / "drift_layers.json", "w"), separators=(",", ":"))

    summary = dict(
        table=metrics["summary"], main=MAIN, lam=float(M.lam_), lam_curve={str(k): v for k, v in M.curve.items()},
        probes={"pB2": "Gaussian blur, sigma = 2 px on the 150x150 aligned face",
                "pJ50": "JPEG re-compression at quality 50 of the aligned face"},
        n=dict(test_real=int(sum(1 for v in index if v["split"] == "test" and v["label"] == 0)),
               test_fake=int(sum(1 for v in index if v["split"] == "test" and v["method"] in CORE)),
               fit_real=int(len(set(T["vid"][fit & (T["label"] == 0)]))),
               fit_fake=int(len(set(T["vid"][fit & (T["label"] == 1)]))),
               fsh=int(sum(1 for v in index if v["method"] == "FSh"))),
        conditions={c: PERTURB_NAMES.get(c.split("_")[0], c) for c in EVAL_CONDS},
        params={k: v for k, v in PERTURB_PARAMS.items()},
        regions=REGIONS + ["background"],
    )
    json.dump(summary, open(OUT / "summary.json", "w"), indent=1)
    # the fitted decision rules for the API
    model = dict(main=MAIN, lam=float(M.lam_), probe="pB2",
                 lr=dict(probes=LR.probes, mean=LR.sc.mean_.tolist(), scale=LR.sc.scale_.tolist(),
                         coef=LR.lr.coef_[0].tolist(), intercept=float(LR.lr.intercept_[0])))
    json.dump(model, open(OUT / "model.json", "w"), indent=1)
    # one real example for the API playground: a degraded real video that the released model flags
    cand = np.where(test & (T["label"] == 0) & (T["cond"] == "GB_5") & (T["p_final"] >= 0.5)
                    & np.isfinite(T["probe:pB2:d"]).all(1))[0]
    if not len(cand):
        cand = np.where(test & (T["label"] == 0) & (T["cond"] != "clean") & np.isfinite(T["probe:pB2:d"]).all(1))[0]
    i = int(cand[np.argmax(T["p_final"][cand])])
    example = dict(vid=T["vid"][i], cond=T["cond"][i], clip=int(T["clip"][i]),
                   d=np.round(T["d"][i].astype(float), 6).tolist(),
                   d_probe={pr: np.round(T[f"probe:{pr}:d"][i].astype(float), 6).tolist() for pr in PROBES_RUN})
    json.dump(example, open(OUT / "example.json", "w"))
    print(f"exported {len(index)} videos to {OUT}; main lambda = {M.lam_}")


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
