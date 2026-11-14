"""Select CLIP-enhanced fusions on validation, then evaluate frozen choices.

  python service/ens/select_clip_fusion.py select --output results/ensemble/round3_clip
  python service/ens/select_clip_fusion.py test --output results/ensemble/round3_clip

Set DFD_ENS_DIR to the candidate score directory. Existing caches are read via
data.cache_file fallback. The selection command never computes test metrics.
"""
import argparse
import copy
import hashlib
import itertools
import json
import os
from pathlib import Path
import shutil
import sys

import numpy as np
from sklearn.linear_model import LogisticRegression

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "service" / "ens"))
os.environ["VAL_CONDS"] = "clean,blur,comp"
import data as Dt
import evaluate as Ev
from weighted_fusion import folds_by_video, metadata, statistics

MEMBERS = ["dfd", "pe_ft", "pe_ft_sc", "pe", "pe_sc", "clip_ft", "clip_ft_sc"]
BASELINE = MEMBERS[:5]
EXCLUDED = {"dfdc", "synth"}
FUSION = ROOT / "service" / "app" / "fusion.json"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source_snapshot():
    paths = [FUSION, FUSION.parent / "model.json", FUSION.parent / "gend_pe_ft.pt",
             Dt.cache_file("ft_clip/best.pt"),
             Dt.D / "new" / "sets.csv", Dt.D / "new" / "sets_gated.csv",
             Dt.cache_file("dfd_val_old.jsonl"), Dt.cache_file("valdmg_dfd.npz"),
             Dt.D / "eval_new_old.jsonl", Dt.D / "feats_test_old.npz", Dt.D / "feats_fit_old.npz"]
    if (Dt.D / "new" / "sets_more.csv").exists():
        paths.append(Dt.D / "new" / "sets_more.csv")
    for tag in ("pe", "peft", "clipft"):
        paths.extend(Dt.cache_file(f"gend_{tag}_{split}.npz") for split in ("new", "fit", "rep"))
        paths.append(Dt.cache_file(f"valdmg_{tag}.npz"))
    return {str(p): digest(p) for p in paths}


def load_complete():
    new, rep = Ev.load_members(MEMBERS)
    missing = set(MEMBERS) - new.keys()
    if missing:
        raise ValueError(f"missing candidate scores: {sorted(missing)}")
    for source in (new, rep):
        for member, rows in source.items():
            if not rows or not np.isfinite(list(rows.values())).all():
                raise ValueError(f"empty or non-finite scores for {member}")
    meta = metadata(Dt.D)
    wanted = {v for v, r in meta.items() if r["set"] == "val" and r["source"] not in EXCLUDED}
    complete = {v for v in wanted if all((v, c, 0) in new[m] for m in MEMBERS for c in Ev.VAL_CONDS)}
    if not complete:
        raise ValueError("no complete validation videos")
    filtered = {m: {k: x for k, x in rows.items()
                    if meta.get(k[0], {}).get("set") != "val" or k[0] in complete}
                for m, rows in new.items()}
    coverage = dict(videos=len(complete), rows=len(complete) * len(Ev.VAL_CONDS),
                    excluded_sources=sorted(EXCLUDED), incomplete_videos=sorted(wanted - complete),
                    video_ids=sorted(complete))
    return filtered, rep, coverage


def cv(new, members):
    X, y, src, cond, vids = Ev.val_data(new, members)
    folds = folds_by_video(vids)
    oof = np.zeros(len(y))
    for fold in range(5):
        tr, te = folds != fold, folds == fold
        model = LogisticRegression(C=1, max_iter=1000).fit(
            X[tr], y[tr], sample_weight=Ev.weights(y[tr], src[tr], cond[tr]))
        oof[te] = model.decision_function(X[te])
    model = LogisticRegression(C=1, max_iter=1000).fit(X, y, sample_weight=Ev.weights(y, src, cond))
    return dict(members=members, coef=model.coef_[0].tolist(), intercept=float(model.intercept_[0]),
                cv=statistics(y, oof, src, cond),
                networks=len({m.removesuffix("_sc") for m in members}))


def qualifies(record, baseline):
    a, b = record["cv"], baseline["cv"]
    return (a["bacc"] >= b["bacc"] + 0.005 and a["caught"] >= b["caught"] - 0.01
            and a["ffpp_clean_false_alarm"] <= b["ffpp_clean_false_alarm"] + 1e-12)


def export_candidate(record, output):
    cfg = copy.deepcopy(json.loads(FUSION.read_text(encoding="utf-8")))
    cfg.update(members=record["members"], coef=record["coef"], intercept=record["intercept"])
    cfg["note"] = "Frozen round-3 CLIP candidate; validation excludes synth and DFDC; not automatically deployed."
    if "clip_ft_sc" in cfg["members"]:
        cal = Dt.GenDSelfCal("clipft")
        selfcal = dict(mean=cal.sc.mean_.tolist(), scale=cal.sc.scale_.tolist(),
                       coef=[cal.reg.coef_.tolist()], intercept=[float(cal.reg.intercept_)])
    for member in cfg["members"]:
        if member.startswith("clip_ft"):
            spec = dict(type="gend", backbone="clip", frames=24, ln="gend_clip_ft.pt")
            if member.endswith("_sc"):
                spec["selfcal"] = selfcal
            cfg["detectors"][member] = spec
            cfg["names"][member] = "GenD-CLIP (retrained)" + (", quality-checked" if member.endswith("_sc") else "")
    dest = output / "candidate"
    dest.mkdir(parents=True, exist_ok=True)
    for member in cfg["members"]:
        spec = cfg["detectors"].get(member, {})
        if spec.get("ln"):
            src = Dt.cache_file("ft_clip/best.pt") if member.startswith("clip_ft") else FUSION.parent / spec["ln"]
            shutil.copy2(src, dest / spec["ln"])
    (dest / "fusion.json").write_text(json.dumps(cfg, separators=(",", ":"), allow_nan=False), encoding="utf-8")


def select(output):
    if (output / "selection.json").exists():
        raise FileExistsError("selection already frozen; use a new experiment directory")
    new, _, coverage = load_complete()
    options = {}
    for n in range(1, len(MEMBERS) + 1):
        for combo in itertools.combinations(MEMBERS, n):
            key = "+".join(combo)
            record = cv(new, list(combo))
            options[key] = record
            print(key, json.dumps({k: v for k, v in record["cv"].items() if k != "per_source"}), flush=True)
    baseline = options["+".join(BASELINE)]
    eligible = []
    for key, record in options.items():
        record["eligible"] = qualifies(record, baseline)
        if record["eligible"]:
            eligible.append(key)
    chosen = max(eligible, key=lambda k: (round(options[k]["cv"]["bacc"], 3),
                                        -options[k]["networks"], -len(options[k]["members"]))) if eligible else None
    result = dict(protocol="results/ensemble/PLAN.md: Round 3 CLIP", coverage=coverage,
                  options=options, chosen=chosen, snapshot=source_snapshot())
    output.mkdir(parents=True, exist_ok=True)
    (output / "selection.json").write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
    if chosen:
        export_candidate(options[chosen], output)
    print("FROZEN CHOICE", chosen, flush=True)


class FrozenFusion:
    def __init__(self, record):
        self.coef = np.asarray(record["coef"])
        self.intercept = float(record["intercept"])

    def decision_function(self, X):
        return np.asarray(X) @ self.coef + self.intercept


def release_rule(candidate, baseline):
    checks = dict(
        newer_gain=candidate["mean_new_bacc"] > baseline["mean_new_bacc"],
        ffpp_no_loss=candidate["ffpp"]["bacc"] >= baseline["ffpp"]["bacc"],
        dfdc_loss_at_most_2pt=candidate["dfdc"]["bacc"] >= baseline["dfdc"]["bacc"] - 0.02,
        faceshifter_loss_at_most_2pt=candidate["fsh"]["bacc"] >= baseline["fsh"]["bacc"] - 0.02,
        synth_recall_loss_at_most_2pt=candidate["synth"]["caught"] >= baseline["synth"]["caught"] - 0.02,
        sf_recall_loss_at_most_2pt=candidate["sf"]["caught"] >= baseline["sf"]["caught"] - 0.02)
    return dict(checks=checks, metrics_pass=all(checks.values()), deployment_requires_parity=True)


def test(output):
    if (output / "test.json").exists():
        raise FileExistsError("test results already recorded; do not overwrite a frozen experiment")
    selection = json.loads((output / "selection.json").read_text(encoding="utf-8"))
    if selection["snapshot"] != source_snapshot():
        raise ValueError("weights, fusion or manifests changed after selection")
    if not selection["chosen"]:
        print("No qualifying validation candidate; no test evaluation.")
        return
    new, rep, coverage = load_complete()
    if coverage != selection["coverage"]:
        raise ValueError("score coverage changed after selection")
    # Cache the immutable metadata read by every table, avoiding repeated NPZ reads.
    rep_info, meta = Dt.dfd_rep(), Dt.rows()
    Dt.dfd_rep = lambda: rep_info
    Dt.rows = lambda: meta
    live = json.loads(FUSION.read_text(encoding="utf-8"))
    baseline = Ev.table(new, rep, live["members"], FrozenFusion(live))
    all_results = {key: Ev.table(new, rep, r["members"], FrozenFusion(r))
                   for key, r in selection["options"].items()}
    candidate = all_results[selection["chosen"]]
    result = dict(chosen=selection["chosen"], exact_live_baseline=baseline, candidate=candidate,
                  options=all_results, release=release_rule(candidate, baseline))
    (output / "test.json").write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
    print("baseline", Ev.fmt(baseline))
    print("candidate", Ev.fmt(candidate))
    print(json.dumps(result["release"], indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("select", "test"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    (select if args.command == "select" else test)(args.output)


if __name__ == "__main__":
    main()
