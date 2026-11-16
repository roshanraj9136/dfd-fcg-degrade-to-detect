"""Export the viva's compact evidence snapshot from committed results, without ML dependencies.

Run from any directory. --check fails if the website no longer matches the sources.
No weights, video IDs, machine paths or training data are exported.
"""
import argparse
import ast
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / "web/viva/src/evidence.json"


def read(path):
    return json.loads((ROOT / path).read_text(encoding="utf-8"))


def snapshot():
    fusion = read("service/app/fusion.json")
    published = read("results/ensemble/results.json")
    website = read("web/viva/src/ensemble.json")
    if published != website:
        raise ValueError("ensemble.json is stale; regenerate with service/ens/report.py")
    if fusion["members"] != website["members"]:
        raise ValueError("Published results and live fusion members differ")
    if fusion["members"] != ["dfd", "pe_ft", "pe_ft_sc", "pe", "pe_sc"]:
        raise ValueError("Update the architecture paths for the new fusion members")
    defaults = {}
    for node in ast.parse((ROOT / "service/app/pipeline.py").read_text()).body:
        if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name):
            key = node.targets[0].id
            if key in {"CLIPS", "WINDOW_SEC", "MAX_FACES"}:
                defaults[key] = float(ast.literal_eval(node.value.args[0].args[1]))
    if defaults != {"CLIPS": 8, "WINDOW_SEC": 10, "MAX_FACES": 2}:
        raise ValueError("Update the architecture for changed pipeline defaults")
    if any(v["frames"] != 24 for v in fusion["detectors"].values()):
        raise ValueError("Update the GenD frame counts in the architecture")
    reweight = read("results/ensemble/round3/selection.json")
    experiments = {}
    for name in ("round3_clip", "paired_v1"):
        selection = read(f"results/ensemble/{name}/selection.json")
        test = read(f"results/ensemble/{name}/test.json")
        experiments[name] = {
            "options": len(selection["options"]),
            "validation_videos": selection["coverage"]["videos"],
            "chosen": selection["chosen"],
            "live_new_bacc": test["exact_live_baseline"]["mean_new_bacc"],
            "candidate_new_bacc": test["candidate"]["mean_new_bacc"],
            "release_passed": test["release"]["metrics_pass"],
        }
    training = [json.loads(line) for line in
                (ROOT / "results/ensemble/paired_v1/training.jsonl").read_text().splitlines()]
    data = next(row for row in training if row["event"] == "data")
    pairs = next(row for row in training if row["event"] == "pairing")
    transfer = read("results/ensemble/selfcal_transfer.json")
    return {
        "fusion": {k: fusion[k] for k in
                   ("members", "coef", "intercept", "fitted_on_validation_videos", "val_conds")},
        "reweight": {"options": len(reweight["candidates"]), "selected": reweight["selected"]},
        "blur_transfer": {name: {"raw": transfer[name]["ffpp:GB_5"],
                                  "corrected": transfer[name + "_sc"]["ffpp:GB_5"]}
                          for name in ("pe", "pe_ft", "clip")},
        "experiments": experiments,
        "paired_training": {
            "train_videos": data["train"], "validation_videos": data["val"],
            "pairs": pairs["available"],
            "validation": [{"step": row["step"], "auroc": row["score"]}
                           for row in training if row["event"] == "val"],
        },
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    content = json.dumps(snapshot(), indent=2, ensure_ascii=False) + "\n"
    if args.check:
        if not DEST.exists() or DEST.read_text(encoding="utf-8") != content:
            raise SystemExit("Viva evidence is stale. Run python scripts/export_viva_evidence.py")
        print("Viva evidence matches the live configuration and committed results.")
    else:
        DEST.write_text(content, encoding="utf-8")
        print(f"Wrote {DEST.relative_to(ROOT)}")
