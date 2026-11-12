"""Numbers for the viva guide's "Version 2" section, from the test-set study (study_test.jsonl), the selection-set
study (study_select.jsonl) and the settings chosen on the selection set.
python upgrade_summary.py <study_test.jsonl> <study_select.jsonl> <layout> <flip> <faces> <margin> <agree> <out.json>"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from study_report import CONDS, SETTINGS, metrics, name, video_p, video_views  # noqa: E402

test_f, sel_f, lay, fl, fc, margin, agree, out = sys.argv[1:9]
V2 = dict(layout=lay, flip=int(fl), faces=int(fc))
V1 = dict(layout="old", flip=0, faces=1)
margin, agree = float(margin), float(agree)


def load(f):
    rs = [json.loads(l) for l in open(f, encoding="utf-8")]
    return [r for r in rs if "error" not in r], [r for r in rs if "error" in r]


test, test_err = load(test_f)
sel, sel_err = load(sel_f)
GROUPS = {"ffpp": lambda r: r["method"] in ("REAL", "DF", "F2F", "FS", "NT"),
          "fsh": lambda r: r["method"] in ("REAL", "FSh"),
          "dfdc": lambda r: r["method"].startswith("DFDC")}


def m(x):
    return {k: (None if (isinstance(v, float) and np.isnan(v)) else round(v, 4)) for k, v in x.items()}


res = dict(videos=len(test) + len(test_err), failed=len(test_err), errors=[e["vid"] + ": " + e["error"] for e in test_err],
           settings=dict(clips=8 if lay == "new" else 2, window_s=10 if lay == "new" else 4.8, mirror=bool(int(fl)),
                         faces=int(fc), cert_margin=margin, cert_agree=agree),
           select=dict(videos=len(sel) + len(sel_err), used=len(sel), fake=sum(r["label"] for r in sel),
                       dfdc=sum(r["method"].startswith("DFDC") for r in sel)),
           groups={}, ablation={})
for g, f in GROUPS.items():
    rs = [r for r in test if f(r)]
    y = [r["label"] for r in rs]
    res["groups"][g] = dict(n=len(rs), n_fake=int(sum(y)))
    for c in CONDS:
        if g == "fsh" and c != "clean":
            pass
        res["groups"][g][c] = dict(
            orig=m(metrics(y, [video_p(r, c, "old", 0, 1, "orig") for r in rs])),
            v1=m(metrics(y, [video_p(r, c, **V1) for r in rs])),
            v2=m(metrics(y, [video_p(r, c, **V2) for r in rs])),
            v2_orig=m(metrics(y, [video_p(r, c, V2["layout"], V2["flip"], V2["faces"], "orig") for r in rs])),
            v2_strict=m(metrics(y, [video_p(r, c, V2["layout"], V2["flip"], V2["faces"], "strict") for r in rs])),
            flips=dict(fixed=int(sum((video_p(r, c, **V1) >= .5) != (r["label"] == 1) and (video_p(r, c, **V2) >= .5) == (r["label"] == 1) for r in rs)),
                       broken=int(sum((video_p(r, c, **V1) >= .5) == (r["label"] == 1) and (video_p(r, c, **V2) >= .5) != (r["label"] == 1) for r in rs))))

# ablation: every setting, mean balanced accuracy / AUROC over the 3 options, on both sets
for tag, rs_all in (("select", sel), ("test", test)):
    for s in SETTINGS:
        row = {}
        for g in ("ffpp", "dfdc"):
            rs = [r for r in rs_all if GROUPS[g](r)]
            if not rs:
                continue
            y = [r["label"] for r in rs]
            ms = [metrics(y, [video_p(r, c, **s) for r in rs]) for c in CONDS]
            row[g] = dict(bacc=round(float(np.mean([x["bacc"] for x in ms])), 4),
                          auroc=round(float(np.mean([x["auroc"] for x in ms])), 4))
        res["ablation"].setdefault(tag, []).append(dict(setting=name(s).replace("  ", " ").strip(), **s, **row))

# "sure" answers on the test set
sure_rows = []
for r in test:
    g = "dfdc" if r["method"].startswith("DFDC") else ("fsh" if r["method"] == "FSh" else "ffpp")
    for c in CONDS:
        p, v = video_views(r, c, V2["layout"], V2["flip"], V2["faces"])
        sure = abs(p - .5) >= margin and float(np.mean((v >= .5) == (p >= .5))) >= agree
        sure_rows.append(dict(g=g, c=c, ok=(p >= .5) == (r["label"] == 1), sure=sure))


def sure_stats(rows):
    s = [x for x in rows if x["sure"]]
    u = [x for x in rows if not x["sure"]]
    return dict(n=len(rows), share=round(len(s) / max(1, len(rows)), 4),
                acc_all=round(float(np.mean([x["ok"] for x in rows])), 4) if rows else None,
                acc_sure=round(float(np.mean([x["ok"] for x in s])), 4) if s else None,
                acc_unsure=round(float(np.mean([x["ok"] for x in u])), 4) if u else None,
                errors_sure=int(sum(not x["ok"] for x in s)), errors_unsure=int(sum(not x["ok"] for x in u)))


res["sure"] = dict(all=sure_stats(sure_rows), **{g: sure_stats([x for x in sure_rows if x["g"] == g]) for g in ("ffpp", "fsh", "dfdc")},
                   **{c: sure_stats([x for x in sure_rows if x["c"] == c and x["g"] != "dfdc"]) for c in CONDS})

# risk-coverage curve on the test set: share of answers marked "sure" vs their accuracy, for every margin
# (the website's margin was chosen on the selection set; the curve only shows what other choices would give)
pv = []
for r in test:
    for c in CONDS:
        p, _ = video_views(r, c, V2["layout"], V2["flip"], V2["faces"])
        pv.append((abs(p - .5), (p >= .5) == (r["label"] == 1), r["method"].startswith("DFDC")))
curve = []
for mg in np.round(np.arange(0, 0.31, 0.02), 2):
    row = dict(margin=float(mg))
    for tag, sel_ in (("all", lambda x: True), ("ffpp", lambda x: not x[2]), ("dfdc", lambda x: x[2])):
        xs = [x for x in pv if sel_(x)]
        s = [x for x in xs if x[0] >= mg]
        row[tag] = dict(share=round(len(s) / len(xs), 4) if xs else None,
                        acc=round(float(np.mean([x[1] for x in s])), 4) if s else None)
    curve.append(row)
res["curve"] = curve

# is the v1 -> v2 gain more than noise? paired bootstrap over videos (same videos in both versions), statistic =
# mean balanced accuracy over the three website options; also the exact McNemar test on the pooled answers
rng = np.random.default_rng(0)


def mean_bacc(rs, s):
    y = [r["label"] for r in rs]
    return float(np.mean([metrics(y, [video_p(r, c, **s) for r in rs])["bacc"] for c in CONDS]))


def mcnemar(b, c):
    from math import comb
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    return float(min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n))


res["significance"] = {}
for g in ("ffpp", "dfdc", "fsh"):
    rs = [r for r in test if GROUPS[g](r)]
    if len(rs) < 10:
        continue
    cache = {}
    for r in rs:
        for c in CONDS:
            cache[(r["vid"], c, 1)] = video_p(r, c, **V1)
            cache[(r["vid"], c, 2)] = video_p(r, c, **V2)
    reals = [r for r in rs if r["label"] == 0]
    fakes = [r for r in rs if r["label"] == 1]

    def bacc_of(sample, ver):
        out = []
        for c in CONDS:
            fa = np.mean([cache[(r["vid"], c, ver)] >= .5 for r in sample if r["label"] == 0])
            tp = np.mean([cache[(r["vid"], c, ver)] >= .5 for r in sample if r["label"] == 1])
            out.append((1 - fa + tp) / 2)
        return float(np.mean(out))

    diffs = []
    for _ in range(2000):                      # stratified: resample real and fake videos separately
        smp = [reals[i] for i in rng.integers(0, len(reals), len(reals))] + \
              [fakes[i] for i in rng.integers(0, len(fakes), len(fakes))]
        diffs.append(bacc_of(smp, 2) - bacc_of(smp, 1))
    fixed = sum(res["groups"][g][c]["flips"]["fixed"] for c in CONDS)
    broken = sum(res["groups"][g][c]["flips"]["broken"] for c in CONDS)
    res["significance"][g] = dict(delta=round(bacc_of(rs, 2) - bacc_of(rs, 1), 4),
                                  ci95=[round(float(np.percentile(diffs, 2.5)), 4), round(float(np.percentile(diffs, 97.5)), 4)],
                                  p_gain_below_0=round(float(np.mean(np.array(diffs) <= 0)), 4),
                                  fixed=fixed, broken=broken, mcnemar_p=round(mcnemar(fixed, broken), 4))

# two-person videos: per option, v1 (main face, 2 clips) vs v2 (most suspicious face, 8 clips), and the "sure" flag
two = []
for r in test:
    if r["faces"] < 2:
        continue
    row = dict(vid=r["vid"], label=r["label"], v1=round(video_p(r, "clean", **V1), 3), v2=round(video_p(r, "clean", **V2), 3), by={})
    for c in CONDS:
        p1, p2 = video_p(r, c, **V1), video_p(r, c, **V2)
        row["by"][c] = dict(v1=round(p1, 3), v2=round(p2, 3), v2_sure=bool(abs(p2 - .5) >= margin))
    two.append(row)
res["two_face"] = two


def count(rows, lab, ver, conds):
    return int(sum(t["by"][c][ver] >= .5 for t in rows if t["label"] == lab for c in conds))


nf, nr = sum(t["label"] for t in two), sum(1 - t["label"] for t in two)
res["two_face_summary"] = dict(
    n_fake=nf, n_real=nr,
    clean=dict(caught_v1=count(two, 1, "v1", ["clean"]), caught_v2=count(two, 1, "v2", ["clean"]),
               alarms_v1=count(two, 0, "v1", ["clean"]), alarms_v2=count(two, 0, "v2", ["clean"])),
    all=dict(n_fake=nf * 3, n_real=nr * 3, caught_v1=count(two, 1, "v1", CONDS), caught_v2=count(two, 1, "v2", CONDS),
             alarms_v1=count(two, 0, "v1", CONDS), alarms_v2=count(two, 0, "v2", CONDS),
             alarms_v2_sure=int(sum(t["by"][c]["v2"] >= .5 and t["by"][c]["v2_sure"] for t in two if t["label"] == 0 for c in CONDS))))
def clean(o):
    """JSON for the website: NaN (e.g. no fakes in a group) becomes null."""
    if isinstance(o, dict):
        return {k: clean(v) for k, v in o.items()}
    if isinstance(o, list):
        return [clean(v) for v in o]
    if isinstance(o, float) and np.isnan(o):
        return None
    return o


res["face_s_median"] = float(np.median([r["face_s"] for r in test]))
res = clean(res)
json.dump(res, open(out, "w"), indent=1)
print(json.dumps({k: res[k] for k in ("videos", "failed", "settings", "sure", "two_face", "significance")}, indent=1))
for g in GROUPS:
    for c in CONDS:
        x = res["groups"][g][c]
        print(g, c, "orig", x["orig"]["acc"], "v1", x["v1"]["acc"], x["v1"]["bacc"], x["v1"]["auroc"], "v2", x["v2"]["acc"],
              x["v2"]["bacc"], x["v2"]["auroc"], "flips", x["flips"])
