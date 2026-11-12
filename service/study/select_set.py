"""Selection set for choosing the website's version-2 settings. None of these videos is in any reported result:
  * FF++ official test videos that are on disk (data/manifest_full.csv) but not in the 145-video reported test set:
    40 real + 40 fake (10 per manipulation), and
  * 20 extra DFDC part-0 pairs (a real video + one of its fakes), different from the 40 reported pairs.
Writes <DFD_DATA>/manifest_select.csv (same columns as manifest.csv, split "select") and downloads the DFDC videos.

    python service/study/select_set.py
"""
import csv
import json
import random
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from huggingface_hub import hf_hub_download  # noqa: E402

from qa.config import DATA  # noqa: E402

REPO = "gonnerthetooner/DFDC-extracted-full"
PART = "dfdc_train_part_00/dfdc_train_part_0/"


def main():
    used = list(csv.DictReader(open(DATA / "manifest.csv")))
    used_vids = {r["vid"] for r in used}
    full = [r for r in csv.DictReader(open(DATA / "manifest_full.csv")) if r["split"] == "test" and r["vid"] not in used_vids]
    rng = random.Random(2026)
    rows = []
    for m, k in [("REAL", 40), ("DF", 10), ("F2F", 10), ("FS", 10), ("NT", 10)]:
        pool = sorted([r for r in full if r["method"] == m], key=lambda r: r["vid"])
        for r in sorted(rng.sample(pool, k), key=lambda r: r["vid"]):
            rows.append(dict(vid=r["vid"], split="select", method=m, label=r["label"], hf_path=r["hf_path"], repo=""))

    meta = json.load(open(hf_hub_download(REPO, PART + "metadata.json", repo_type="dataset", local_dir=DATA / "hf_dfdc")))
    used_dfdc = {r["hf_path"].split("/")[-1] for r in used if r["split"] == "cross"}
    fakes_of = {}
    for k, v in meta.items():
        if v["label"] == "FAKE" and v.get("original") in meta:
            fakes_of.setdefault(v["original"], []).append(k)
    reals = sorted(k for k, v in meta.items() if v["label"] == "REAL" and k in fakes_of and k not in used_dfdc
                   and not any(f in used_dfdc for f in fakes_of[k]))
    for r in sorted(rng.sample(reals, 20)):
        f = sorted(fakes_of[r])[rng.randrange(len(fakes_of[r]))]
        for name, lab in [(r, 0), (f, 1)]:
            m = "DFDC_R" if lab == 0 else "DFDC_F"
            rows.append(dict(vid=f"{m}/{name[:-4]}", split="select", method=m, label=lab, hf_path=PART + name, repo=REPO))

    def dl(r):
        if r["repo"]:
            hf_hub_download(REPO, r["hf_path"], repo_type="dataset", local_dir=DATA / "hf_dfdc")

    with ThreadPoolExecutor(6) as ex:
        list(ex.map(dl, rows))
    with open(DATA / "manifest_select.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["vid", "split", "method", "label", "hf_path", "repo"])
        w.writeheader()
        w.writerows(rows)
    print(len(rows), "videos;", sum(r["repo"] != "" for r in rows), "DFDC")


if __name__ == "__main__":
    main()
