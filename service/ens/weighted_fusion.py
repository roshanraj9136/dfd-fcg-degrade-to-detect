"""Validation-only fusion reweighting using saved scores and NumPy.

This selection tool never evaluates test metrics. Freeze its choice before testing.
See results/ensemble/ROUND3_PLAN.md. No model weights or videos are modified.
"""
import argparse
import copy
import csv
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "service" / "app"))
from reliability import finite_array, logit, sigmoid, validate_fusion

CONDS = ("clean", "blur", "comp")
MEMBERS = ("dfd", "pe_ft", "pe_ft_sc", "pe", "pe_sc")


def fit_logistic(X, y, weights, C=1.0):
    """Newton/Armijo minimizer of sklearn's weighted binary L2 logistic objective.

    Sum of weighted log losses + ||coef||^2 / (2*C); intercept unpenalized.
    Small dense fusion matrices only. Reject failure to converge.
    """
    X = finite_array(X, "features", ndim=2)
    y, w = finite_array(y, "labels", ndim=1), finite_array(weights, "weights", ndim=1)
    if len(X) != len(y) or len(y) != len(w) or set(y) != {0, 1} or (w <= 0).any() or C <= 0:
        raise ValueError("invalid binary training data, weights or C")
    A = np.column_stack([X, np.ones(len(X))])
    reg = np.r_[np.full(X.shape[1], 1 / C), 0.0]
    beta = np.zeros(A.shape[1])

    def loss(b):
        z = A @ b
        return float(np.dot(w, np.logaddexp(0, z) - y * z) + 0.5 * np.dot(reg, b * b))

    for _ in range(100):
        p = sigmoid(A @ beta)
        grad = A.T @ (w * (p - y)) + reg * beta
        if np.max(np.abs(grad)) < 1e-7:
            return beta[:-1], float(beta[-1])
        hess = (A.T * (w * p * (1 - p))) @ A + np.diag(reg)
        direction = np.linalg.solve(hess, grad)
        old = loss(beta)
        step = 1.0
        for _ in range(40):
            trial = beta - step * direction
            if loss(trial) <= old - 1e-4 * step * float(grad @ direction):
                break
            step *= 0.5
        else:
            raise RuntimeError("logistic line search failed")
        beta = trial
    raise RuntimeError("logistic fit did not converge")


def sample_weights(y, source, condition, real_weight=1.0):
    w = np.zeros(len(y))
    for cond in np.unique(condition):
        cm = condition == cond
        for src in np.unique(source[cm]):
            for label in (0, 1):
                mask = cm & (source == src) & (y == label)
                if mask.any():
                    w[mask] = 1.0 / mask.sum()
        real, fake = cm & (y == 0), cm & (y == 1)
        if not real.any() or not fake.any():
            raise ValueError("each training condition needs both classes")
        w[real] *= w[fake].sum() / w[real].sum()
        w[cm] /= w[cm].sum()
    w /= w.mean()
    w[(source == "ffpp") & (y == 0)] *= real_weight
    return w


def statistics(y, scores, source, condition):
    details = {}
    for cond in np.unique(condition):
        cm = condition == cond
        for src in np.unique(source[cm]):
            fake = scores[cm & (source == src) & (y == 1)]
            real = scores[cm & (source == src) & (y == 0)]
            if not len(real):
                real = scores[cm & (y == 0)]
            if not len(fake) or not len(real):
                raise ValueError(f"cannot evaluate {src}/{cond} with missing classes")
            caught, alarm = float(np.mean(fake >= 0)), float(np.mean(real >= 0))
            details[f"{src}:{cond}"] = dict(bacc=(caught + 1 - alarm) / 2, caught=caught, false_alarm=alarm)
    return dict(bacc=float(np.mean([d["bacc"] for d in details.values()])),
                caught=float(np.mean([d["caught"] for d in details.values()])),
                ffpp_clean_false_alarm=details["ffpp:clean"]["false_alarm"], per_source=details)


def metadata(data):
    out = {}
    for name in ("sets.csv", "sets_gated.csv", "sets_more.csv"):
        path = data / "new" / name
        if not path.exists():
            continue
        with path.open(encoding="utf-8-sig", newline="") as fh:
            for row in csv.DictReader(fh):
                row["label"] = int(row["label"])
                if row["vid"] in out:
                    old = out[row["vid"]]
                    if any(old[k] != row[k] for k in ("set", "label", "source")):
                        raise ValueError(f"conflicting manifest rows for {row['vid']}")
                out.setdefault(row["vid"], row)
    return out


def gend_scores(path, allowed, raw_name, fusion):
    """Use frozen production readouts; do not fit a new correction on test data."""
    sc = fusion["detectors"][raw_name + "_sc"]["selfcal"]
    mu, scale, coef, intercept = (np.asarray(sc[k]) for k in ("mean", "scale", "coef", "intercept"))
    out = {raw_name: {}, raw_name + "_sc": {}}
    with np.load(path) as z:
        keys, starts, logits, feats = z["keys"], z["start"], z["logit"], z["feats"]
        for i, key in enumerate(keys):
            vid, cond, face = str(key).split("|")
            if vid not in allowed:
                continue
            k = (vid, cond, int(face))
            sl = slice(starts[i], starts[i + 1])
            lg = logits[sl].astype(float)
            correction = ((feats[sl].astype(float) - mu) / scale) @ coef.T + intercept
            out[raw_name][k] = logit(float(sigmoid(lg).mean()))
            out[raw_name + "_sc"][k] = logit(float(sigmoid(lg - correction.ravel()).mean()))
    return out


def load_new(data, allowed, fusion, validation=False):
    ens = data / "ens"
    out = {m: {} for m in MEMBERS}
    paths = [ens / "dfd_val_old.jsonl"] if validation else [data / "eval_new_old.jsonl"]
    for path in paths:
        with path.open(encoding="utf-8") as fh:
            for line in fh:
                r = json.loads(line)
                if r["vid"] in allowed:
                    for face, p in enumerate(r.get("faces") or [r["p"]]):
                        out["dfd"][(r["vid"], "clean", face)] = logit(p)
    for raw, tag in (("pe_ft", "peft"), ("pe", "pe")):
        scores = gend_scores(ens / f"gend_{tag}_new.npz", allowed, raw, fusion)
        if validation:
            dmg = gend_scores(ens / f"valdmg_{tag}.npz", allowed, raw, fusion)
            for m in scores:
                scores[m].update(dmg[m])
        out.update(scores)
    if validation:
        with np.load(ens / "valdmg_dfd.npz") as z:
            for key, p in zip(z["keys"], z["p"]):
                vid, cond, face = str(key).split("|")
                if vid in allowed:
                    out["dfd"][(vid, cond, int(face))] = logit(p)
    return out


def validation_rows(data, fusion):
    meta = metadata(data)
    wanted = {v: r for v, r in meta.items() if r["set"] == "val" and r["source"] != "dfdc"}
    scores = load_new(data, wanted, fusion, validation=True)
    complete = [v for v in sorted(wanted) if all((v, c, 0) in scores[m] for c in CONDS for m in MEMBERS)]
    keys = [(v, c, 0) for v in complete for c in CONDS]
    X = np.array([[scores[m][k] for m in MEMBERS] for k in keys])
    y = np.array([wanted[k[0]]["label"] for k in keys])
    src = np.array([wanted[k[0]]["source"] for k in keys])
    cond = np.array([k[1] for k in keys])
    vids = np.array([k[0] for k in keys])
    return X, y, src, cond, vids, sorted(set(wanted) - set(complete))


def folds_by_video(vids):
    videos = np.array(sorted(set(vids)))
    np.random.default_rng(0).shuffle(videos)
    lookup = {v: i % 5 for i, v in enumerate(videos)}
    return np.array([lookup[v] for v in vids])


def choose_candidate(records):
    base = next(r for r in records if r["C"] == 1 and r["real_weight"] == 1)
    eligible = [r for r in records if r["bacc"] >= base["bacc"] + 0.005
                and r["ffpp_clean_false_alarm"] < base["ffpp_clean_false_alarm"]
                and r["caught"] >= base["caught"] - 0.01]
    return max(eligible, key=lambda r: (r["bacc"], -abs(r["real_weight"] - 1), -abs(r["C"] - 1))) if eligible else None


def select(data, output, fusion_path):
    fusion = validate_fusion(json.loads(fusion_path.read_text(encoding="utf-8")))
    if tuple(fusion["members"]) != MEMBERS:
        raise ValueError("this preregistered experiment requires the five original live members")
    X, y, src, cond, vids, missing = validation_rows(data, fusion)
    folds = folds_by_video(vids)
    records = []
    for C in (0.3, 1.0, 3.0):
        for rw in (1.0, 1.5, 2.0, 3.0, 4.0):
            oof = np.zeros(len(y))
            for fold in range(5):
                tr, te = folds != fold, folds == fold
                coef, intercept = fit_logistic(X[tr], y[tr], sample_weights(y[tr], src[tr], cond[tr], rw), C)
                oof[te] = X[te] @ coef + intercept
            record = dict(C=C, real_weight=rw, **statistics(y, oof, src, cond))
            records.append(record)
            print(json.dumps({k: v for k, v in record.items() if k != "per_source"}), flush=True)
    base = next(r for r in records if r["C"] == 1 and r["real_weight"] == 1)
    best = choose_candidate(records)
    bc, bi = fit_logistic(X, y, sample_weights(y, src, cond), 1.0)
    old = sigmoid(X @ np.array(fusion["coef"]) + fusion["intercept"])
    parity = float(np.max(np.abs(sigmoid(X @ bc + bi) - old)))
    if parity > 0.01:
        raise RuntimeError(f"baseline solver/readout parity failed: max probability difference {parity}")
    result = dict(protocol="ROUND3_PLAN.md", members=list(MEMBERS), validation_videos=len(set(vids)),
                  validation_rows=len(y), excluded_videos=missing, candidates=records, baseline=base,
                  selected=best, baseline_max_probability_difference=parity,
                  fusion_sha256=hashlib.sha256(fusion_path.read_bytes()).hexdigest())
    candidate = copy.deepcopy(fusion)
    if best is not None:
        coef, intercept = fit_logistic(X, y, sample_weights(y, src, cond, best["real_weight"]), best["C"])
        candidate.update(coef=coef.tolist(), intercept=intercept)
    result["coef"], result["intercept"] = candidate["coef"], candidate["intercept"]
    output.mkdir(parents=True, exist_ok=True)
    (output / "selection.json").write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
    if best is not None:
        candidate["note"] = "Round 3 candidate: selected on grouped validation CV only; not deployed."
        (output / "fusion_candidate.json").write_text(json.dumps(candidate, separators=(",", ":"), allow_nan=False), encoding="utf-8")
    print("selected:", None if best is None else {k: best[k] for k in ("C", "real_weight", "bacc")}, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["select"])
    parser.add_argument("--data-root", type=Path, default=Path(r"D:\dfd-data"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fusion", type=Path, default=ROOT / "service" / "app" / "fusion.json")
    args = parser.parse_args()
    select(args.data_root, args.output, args.fusion)


if __name__ == "__main__":
    main()
