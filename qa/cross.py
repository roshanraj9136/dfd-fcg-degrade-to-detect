"""
Cross-dataset set: DFDC (Deepfake Detection Challenge, Meta), never seen by DFD-FCG in training.

From a public mirror of DFDC train part 0: 40 real videos, and for each one fake video made
from it (paired, like FF++). Appended to data/manifest.csv as split "cross", methods DFDC_R / DFDC_F.

    python -m qa.cross
"""
import csv
import json
import random
from concurrent.futures import ThreadPoolExecutor

from huggingface_hub import hf_hub_download

from qa.config import DATA

REPO = "gonnerthetooner/DFDC-extracted-full"
PART = "dfdc_train_part_00/dfdc_train_part_0/"
N = 40


def main():
    meta = json.load(open(hf_hub_download(REPO, PART + "metadata.json", repo_type="dataset",
                                          local_dir=DATA / "hf_dfdc")))
    fakes_of = {}
    for k, v in meta.items():
        if v["label"] == "FAKE" and v.get("original") in meta:
            fakes_of.setdefault(v["original"], []).append(k)
    reals = sorted(k for k, v in meta.items() if v["label"] == "REAL" and k in fakes_of)
    rng = random.Random(0)
    pick = sorted(rng.sample(reals, min(N, len(reals))))
    rows = []
    for r in pick:
        f = sorted(fakes_of[r])[rng.randrange(len(fakes_of[r]))]
        for name, lab in [(r, 0), (f, 1)]:
            rows.append(dict(vid=f"{'DFDC_R' if lab == 0 else 'DFDC_F'}/{name[:-4]}", split="cross",
                             method="DFDC_R" if lab == 0 else "DFDC_F", label=lab, hf_path=PART + name,
                             repo=REPO))

    def dl(r):
        hf_hub_download(REPO, r["hf_path"], repo_type="dataset", local_dir=DATA / "hf_dfdc")

    with ThreadPoolExecutor(6) as ex:
        list(ex.map(dl, rows))
    with open(DATA / "manifest.csv", newline="") as fh:
        old = [r for r in csv.DictReader(fh) if r["split"] != "cross"]
    for r in old:
        r.setdefault("repo", "")
    allrows = old + rows
    with open(DATA / "manifest.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["vid", "split", "method", "label", "hf_path", "repo"])
        w.writeheader()
        for r in allrows:
            w.writerow({k: r.get(k, "") for k in w.fieldnames})
    print(f"added {len(rows)} DFDC videos ({len(pick)} real + {len(pick)} paired fakes)")


if __name__ == "__main__":
    main()
