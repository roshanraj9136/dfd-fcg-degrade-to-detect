"""Score every website setting from a study jsonl (same formulas as pipeline.Analyzer.finish).
python study_report.py <study.jsonl> [--json out.json]
Settings: clips (old = 2 clips from the first 144 frames, new = 8 clips from the first 10 s), mirrored view (0/1),
faces (1 = main face only, 2 = most suspicious of up to 2 faces)."""
import itertools
import json
import sys

import numpy as np

sig = lambda x: 1 / (1 + np.exp(-x))  # noqa: E731
CONDS = ["clean", "GB_5", "VC_5"]
LAM = 0.5


def face_p(f, clips, flip, kind):
    d, g = np.array(f["d"])[clips], np.array(f["g"])[clips]
    D, G = [d], [g]
    if flip:
        D.append(np.array(f["df"])[clips])
        G.append(np.array(f["gf"])[clips])
    d, g = np.concatenate(D), np.concatenate(G)
    if kind == "orig":
        return float(sig(d).mean())
    if kind == "strict":
        if f.get("r") is None:
            return float("nan")
        r = np.array(f["r"])[clips]
        r = np.concatenate([r] * (2 if flip else 1))
        return float(sig(d - g - LAM * r).mean())
    return float(sig(d - g).mean())


def video_p(rec, cond, layout, flip, faces, kind="cal"):
    clips = rec[layout]
    fs = rec[cond][:faces]
    cal = [face_p(f, clips, flip, "cal") for f in fs]
    k = int(np.argmax(cal))                     # the most suspicious face (by the verdict score) decides
    return face_p(fs[k], clips, flip, kind)


def auroc(y, s):
    y, s = np.asarray(y), np.asarray(s)
    pos, neg = s[y == 1], s[y == 0]
    if not len(pos) or not len(neg):
        return float("nan")
    return float(((pos[:, None] > neg[None]).sum() + 0.5 * (pos[:, None] == neg[None]).sum()) / (len(pos) * len(neg)))


def metrics(y, p):
    y, p = np.asarray(y), np.asarray(p)
    pred = p >= 0.5
    real, fake = y == 0, y == 1
    fa = float(pred[real].mean()) if real.any() else float("nan")
    caught = float(pred[fake].mean()) if fake.any() else float("nan")
    eps = 1e-6
    ll = float(-np.mean(y * np.log(np.clip(p, eps, 1)) + (1 - y) * np.log(np.clip(1 - p, eps, 1))))
    return dict(n=int(len(y)), acc=float((pred == (y == 1)).mean()), bacc=float(np.nanmean([1 - fa, caught])),
                false_alarm=fa, caught=caught, auroc=auroc(y, p), logloss=ll)


GROUPS = {
    "FF++": lambda r: r["method"] in ("REAL", "DF", "F2F", "FS", "NT"),
    "FF++ + FSh": lambda r: r["method"] in ("REAL", "DF", "F2F", "FS", "NT", "FSh"),
    "DFDC": lambda r: r["method"].startswith("DFDC"),
    "all": lambda r: True,
}
SETTINGS = [dict(layout=l, flip=f, faces=k) for l, f, k in itertools.product(["old", "new"], [0, 1], [1, 2])]


def name(s):
    return f"{'8 clips/10 s' if s['layout'] == 'new' else '2 clips/4.8 s'} {'+mirror' if s['flip'] else '       '} " \
           f"{'2 faces' if s['faces'] == 2 else '1 face '}"


def video_views(rec, cond, layout, flip, faces):
    """Video score p and the per clip-view calibrated scores of the deciding face (for the certainty rule)."""
    clips = rec[layout]
    fs = rec[cond][:faces]
    cal = [face_p(f, clips, flip, "cal") for f in fs]
    k = int(np.argmax(cal))
    f = fs[k]
    v = [sig(np.array(f["d"])[clips] - np.array(f["g"])[clips]).mean(1)]
    if flip:
        v.append(sig(np.array(f["df"])[clips] - np.array(f["gf"])[clips]).mean(1))
    return cal[k], np.concatenate(v)


def certainty_study(recs, s, margins=(0, .05, .1, .15, .2, .25), agrees=(0, .6, .75, .9, 1.0), target=0.95):
    """Selective answers: 'sure' if |p - 0.5| >= margin and >= agree of the clip checks are on the verdict's side.
    Prints coverage (share of answers that are 'sure') and accuracy of sure / not-sure answers, pooled over the
    three website options, and returns the rule with the largest coverage whose sure-accuracy >= target."""
    rows = []
    for r in recs:
        for c in CONDS:
            p, v = video_views(r, c, s["layout"], s["flip"], s["faces"])
            rows.append((r["label"], p, float(np.mean((v >= .5) == (p >= .5))), r["method"].startswith("DFDC"), c))
    y = np.array([x[0] for x in rows])
    p = np.array([x[1] for x in rows])
    ag = np.array([x[2] for x in rows])
    ok = (p >= .5) == (y == 1)
    print(f"\n--- certainty rule ({name(s)}), {len(rows)} answers (videos x 3 options); all answers acc {ok.mean():.3f}")
    print(f"{'margin':>7}{'agree':>7}{'sure share':>12}{'sure acc':>10}{'unsure acc':>12}{'sure acc FF++':>15}{'sure acc DFDC':>15}")
    best = None
    dfdc = np.array([x[3] for x in rows])
    for m in margins:
        for a in agrees:
            sure = (np.abs(p - .5) >= m) & (ag >= a)
            if not sure.any():
                continue
            acc_s = ok[sure].mean()
            acc_u = ok[~sure].mean() if (~sure).any() else float("nan")
            f_s = ok[sure & ~dfdc].mean() if (sure & ~dfdc).any() else float("nan")
            d_s = ok[sure & dfdc].mean() if (sure & dfdc).any() else float("nan")
            print(f"{m:>7.2f}{a:>7.2f}{sure.mean():>12.3f}{acc_s:>10.3f}{acc_u:>12.3f}{f_s:>15.3f}{d_s:>15.3f}")
            if acc_s >= target and (best is None or sure.mean() > best[2]):
                best = (m, a, float(sure.mean()), float(acc_s))
    print(f"chosen (largest sure share with sure accuracy >= {target}): {best}")
    return best


def choose_margin(recs, s, target=0.95, grid=np.round(np.arange(0.10, 0.205, 0.01), 2)):
    """The website's 'sure' rule: the smallest margin |p - 0.5| at which the sure answers (all videos x 3 options,
    pooled) are at least `target` accurate. Run on the selection set only."""
    rows = []
    for r in recs:
        for c in CONDS:
            p, _ = video_views(r, c, s["layout"], s["flip"], s["faces"])
            rows.append((r["label"], p))
    y = np.array([x[0] for x in rows])
    p = np.array([x[1] for x in rows])
    ok = (p >= .5) == (y == 1)
    chosen = None
    print(f"\n--- margin for 'sure' ({name(s)}), {len(rows)} answers")
    for m in grid:
        sure = np.abs(p - .5) >= m
        acc = ok[sure].mean() if sure.any() else float("nan")
        print(f"margin {m:.2f}: sure share {sure.mean():.3f}, sure acc {acc:.3f}, not-sure acc {ok[~sure].mean():.3f}")
        if chosen is None and acc >= target:
            chosen = float(m)
    print(f"chosen margin (smallest with sure accuracy >= {target}): {chosen}")
    return chosen


def main():
    recs = [json.loads(l) for l in open(sys.argv[1], encoding="utf-8")]
    errs = [r for r in recs if "error" in r]
    recs = [r for r in recs if "error" not in r]
    print(f"{len(recs)} videos ({len(errs)} errors: {[e['vid'] + ' ' + e['error'][:40] for e in errs]})")
    print(f"two-face videos: {[r['vid'] for r in recs if r['faces'] > 1]}")
    out = {}
    for g, sel in GROUPS.items():
        rs = [r for r in recs if sel(r)]
        if not rs or (g == "FF++ + FSh" and not any(r["method"] == "FSh" for r in rs)):
            continue
        if g == "all" and len({k for k in GROUPS if k != 'all' and any(GROUPS[k](r) for r in rs)}) < 2:
            continue
        y = [r["label"] for r in rs]
        print(f"\n=== {g}: {sum(y)} fake / {len(y) - sum(y)} real")
        print(f"{'setting':<36}" + "".join(f"{c:>34}" for c in CONDS) + f"{'mean bacc':>11}{'mean auroc':>11}")
        print(" " * 36 + "".join(f"{'acc  bacc  FA   caught AUROC':>34}" for _ in CONDS))
        for s in SETTINGS:
            line, b, a = f"{name(s):<36}", [], []
            for c in CONDS:
                m = metrics(y, [video_p(r, c, s["layout"], s["flip"], s["faces"]) for r in rs])
                out.setdefault(g, {}).setdefault(name(s), {})[c] = m
                line += f"{m['acc']:>11.3f}{m['bacc']:6.3f}{m['false_alarm']:6.3f}{m['caught']:6.3f}{m['auroc']:6.3f}"
                b.append(m["bacc"])
                a.append(m["auroc"])
            print(line + f"{np.mean(b):>11.3f}{np.mean(a):>11.3f}")
        s0 = SETTINGS[0]
        line = f"{'(original DFD-FCG, ' + name(s0)[:13] + ')':<36}"
        for c in CONDS:
            m = metrics(y, [video_p(r, c, "old", 0, 1, "orig") for r in rs])
            line += f"{m['acc']:>11.3f}{m['bacc']:6.3f}{m['false_alarm']:6.3f}{m['caught']:6.3f}{m['auroc']:6.3f}"
        print(line)
    if "--cert" in sys.argv:
        lay, fl, fc = sys.argv[sys.argv.index("--cert") + 1].split(",")
        certainty_study(recs, dict(layout=lay, flip=int(fl), faces=int(fc)))
        choose_margin(recs, dict(layout=lay, flip=int(fl), faces=int(fc)))
    if "--json" in sys.argv:
        json.dump(out, open(sys.argv[sys.argv.index("--json") + 1], "w"), indent=1)


if __name__ == "__main__":
    main()
