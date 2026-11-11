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
from qa.methods import EvidenceNulling, head_terms, sig
from qa.prep import safe
from qa.quality import QB_NAMES
from qa.table import load

OUT = ROOT / "web" / "frontend" / "public" / "data"
PUBLIC = {"REAL/000", "DF/000_003"}         # bundled with the upstream DFD-FCG repository
RCODE = {r: i for i, r in enumerate(REGIONS + ["background"])}


def r4(x):
    return np.round(np.asarray(x, float), 4).tolist()


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    T = load()
    fit = T["split"] == "fit"
    nul = EvidenceNulling().fit(T, fit)
    lopo = {}
    for kind in KINDS:
        lopo[kind] = EvidenceNulling().fit(T, fit & ~np.char.startswith(T["cond"].astype(str), kind + "_"))

    test = np.isin(T["split"], ["test", "demo"])
    idx_all = np.where(test)[0]
    nu_all = np.zeros((len(T["vid"]), 3))
    nu_all[test] = nul.nuisance(T, test)
    nu_lopo = np.zeros((len(T["vid"]), 3))
    for kind, nk in lopo.items():
        m = test & np.char.startswith(T["cond"].astype(str), kind + "_")
        if m.any():
            nu_lopo[m] = nk.nuisance(T, m)
    clean_or_re = test & np.isin(T["cond"], ["clean", "RE_0"])
    nu_lopo[clean_or_re] = nu_all[clean_or_re]

    videos = {}
    for i in idx_all:
        v = videos.setdefault(T["vid"][i], dict(vid=T["vid"][i], split=T["split"][i], method=T["method"][i],
                                                label=int(T["label"][i]), conds={}))
        c = v["conds"].setdefault(T["cond"][i], dict(p=[], pn=[], pl=[], nu=[], d=[], s=[], t=[]))
        d = T["d"][i].astype(float)
        c["p"].append(T["p_final"][i])
        c["pn"].append(sig(d - nu_all[i]).mean())
        c["pl"].append(sig(d - nu_lopo[i]).mean())
        c["nu"].append(nu_all[i])
        c["d"].append(d)
        c["s"].append(T["cs_patch"][i])
        c["t"].append(T["ct_patch"][i])

    index, details = [], {}
    for vid, v in videos.items():
        z = np.load(DATA / "prep" / (safe(vid) + ".npz"))
        lm = z["crop_lm"].mean(0)
        rm = region_map(lm)
        scores, maps = {}, {}
        for cond, c in v["conds"].items():
            scores[cond] = dict(p=round(float(np.mean(c["p"])), 4), p_null=round(float(np.mean(c["pn"])), 4),
                                p_null_unseen=round(float(np.mean(c["pl"])), 4),
                                nuisance=r4(np.mean(c["nu"], 0)), d=r4(np.mean(c["d"], 0)))
            maps[cond] = dict(s=r4(np.mean(c["s"], 0)), t=r4(np.mean(c["t"], 0)))
        public = vid in PUBLIC
        index.append(dict(vid=vid, split=v["split"], method=v["method"], label=v["label"], public=public,
                          scores=scores))
        details[vid] = dict(landmarks=r4(lm), regions=[RCODE[x] for x in rm], maps=maps)
        if public:
            export_thumbnail(vid, z)

    json.dump(index, open(OUT / "videos.json", "w"), separators=(",", ":"))
    (OUT / "maps").mkdir(exist_ok=True)
    for vid, d in details.items():
        json.dump(d, open(OUT / "maps" / (safe(vid) + ".json"), "w"), separators=(",", ":"))

    metrics = json.load(open(ROOT / "results" / "metrics.json"))
    analysis = json.load(open(ROOT / "results" / "analysis.json"))
    json.dump(metrics["per_condition"], open(OUT / "conditions.json", "w"), separators=(",", ":"))
    json.dump(analysis, open(OUT / "analysis.json", "w"), separators=(",", ":"))
    arr = np.load(ROOT / "results" / "arrays.npz")
    layers = {}
    for k in arr.files:
        if k.startswith("drift_s_"):
            c = k[len("drift_s_"):]
            layers[c] = dict(s=r4(arr[k]), t=r4(arr["drift_t_" + c]))
    json.dump(layers, open(OUT / "drift_layers.json", "w"), separators=(",", ":"))
    summary = dict(
        table=metrics["summary"],
        n=dict(test_real=int(sum(1 for v in index if v["split"] == "test" and v["label"] == 0)),
               test_fake=int(sum(1 for v in index if v["split"] == "test" and v["method"] in CORE)),
               fit_real=int(len(set(T["vid"][fit & (T["label"] == 0)]))),
               fit_fake=int(len(set(T["vid"][fit & (T["label"] == 1)]))),
               fsh=int(sum(1 for v in index if v["method"] == "FSh"))),
        conditions={c: PERTURB_NAMES.get(c.split("_")[0], c) for c in
                    ["clean", "RE_0"] + [f"{k}_{lv}" for k in PERTURB_PARAMS for lv in TEST_LEVELS[k]]},
        params={k: v for k, v in PERTURB_PARAMS.items()},
        en_pairs=int(nul.n_pairs),
        regions=REGIONS + ["background"],
    )
    json.dump(summary, open(OUT / "summary.json", "w"), indent=1)
    # the fitted correction, for the API: nuisance_h = W_h . ((phi(q_bg) - phi0) / scale), summed over terms
    model = dict(descriptors=QB_NAMES[:7], phi="[q[:7], q[:7]**2]", phi0=np.round(nul.phi0, 6).tolist(),
                 scale=np.round(nul.scale, 6).tolist(),
                 W={h: np.round(nul.W[h].coef_, 6).tolist() for h in "sta"})
    json.dump(model, open(OUT / "projection.json", "w"), separators=(",", ":"))
    # one real example for the API playground: a degraded real video that the original model flags
    terms = head_terms(T)
    ok = np.isfinite(T["q_bg"]).all(1)
    cand = np.where(test & ok & (T["label"] == 0) & (T["cond"] == "GB_5") & (T["p_final"] >= 0.5))[0]
    if not len(cand):
        cand = np.where(test & ok & (T["label"] == 0) & (T["cond"] != "clean"))[0]
    i = int(cand[np.argmax(T["p_final"][cand])])
    example = dict(vid=T["vid"][i], cond=T["cond"][i], clip=int(T["clip"][i]),
                   s=np.round(terms["s"][i], 6).tolist(), t=np.round(terms["t"][i], 6).tolist(),
                   a=np.round(terms["a"][i], 6).tolist(), d=np.round(T["d"][i].astype(float), 6).tolist(),
                   q_bg=np.round(T["q_bg"][i].astype(float), 6).tolist())
    json.dump(example, open(OUT / "example.json", "w"))
    print(f"exported {len(index)} videos to {OUT}")


def export_thumbnail(vid, z):
    """For the two samples bundled with the upstream repo: the face crop under every test condition."""
    from qa.conditions import build
    from qa.manifest import local_video, read_manifest
    row = [r for r in read_manifest() if r["vid"] == vid][0]
    prep = {k: z[k] for k in z.files}
    conds = ["clean", "RE_0"] + [f"{k}_{lv}" for k in PERTURB_PARAMS for lv in TEST_LEVELS[k]]
    built = build(local_video(row), prep, vid, conds)
    dst = ROOT / "web" / "frontend" / "public" / "samples"
    dst.mkdir(parents=True, exist_ok=True)
    for c, v in built.items():
        name = safe(vid) + ("" if c == "clean" else f"__{c}") + ".png"
        cv2.imwrite(str(dst / name), cv2.cvtColor(v[0][0], cv2.COLOR_RGB2BGR))


if __name__ == "__main__":
    main()
