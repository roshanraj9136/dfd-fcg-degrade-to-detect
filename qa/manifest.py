"""
Select the evaluation videos from the official FaceForensics++ splits and download them.

  fit  : FF++ *train* split  -> used only to fit the corrections (60 real, 15 per method)
  test : FF++ *test*  split  -> 140 real (all), 35 per method, 70 FaceShifter (unseen method)

Identities in the two sets are disjoint because the official splits are disjoint.

    python -m qa.manifest            # writes data/manifest.csv and downloads the videos
"""
import csv
import json
import random
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from huggingface_hub import hf_hub_download

from qa.config import DATA, HF_PREFIX, HF_REPO, METHODS, p

SEED = 0
FAKE_METHODS = ["Deepfakes", "Face2Face", "FaceSwap", "NeuralTextures"]


def load_split(name):
    with open(DATA / "splits" / f"{name}.json") as f:
        return json.load(f)


def select():
    rng = random.Random(SEED)
    rows = []

    def add(split, folder, name):
        code = METHODS[folder]
        rows.append(dict(vid=f"{code}/{name}", split=split, method=code,
                         label=0 if code == "REAL" else 1,
                         hf_path=f"{HF_PREFIX}{folder}/{name}.mp4"))

    # ---------------- test split ----------------
    pairs = load_split("test")
    ids = sorted({i for pr in pairs for i in pr})
    fakes = sorted([f"{a}_{b}" for a, b in pairs] + [f"{b}_{a}" for a, b in pairs])
    for i in ids:
        add("test", "original", i)
    for m in FAKE_METHODS:
        for n in sorted(rng.sample(fakes, 35)):
            add("test", m, n)
    for n in sorted(rng.sample(fakes, 70)):
        add("test", "FaceShifter", n)

    # ---------------- train split (fit set) ----------------
    pairs = load_split("train")
    ids = sorted({i for pr in pairs for i in pr})
    fakes = sorted([f"{a}_{b}" for a, b in pairs] + [f"{b}_{a}" for a, b in pairs])
    for i in sorted(rng.sample(ids, 60)):
        add("fit", "original", i)
    for m in FAKE_METHODS:
        for n in sorted(rng.sample(fakes, 15)):
            add("fit", m, n)

    # the sample pair bundled with the upstream repository (REAL/000 is already in test)
    if not any(r["vid"] == "DF/000_003" for r in rows):
        add("demo", "Deepfakes", "000_003")
    return rows


def local_video(row):
    if row.get("repo"):
        return DATA / "hf_dfdc" / row["hf_path"]
    return DATA / "hf" / row["hf_path"]


def download(rows, workers=8):
    def one(r):
        dst = local_video(r)
        if dst.exists():
            return 0
        hf_hub_download(HF_REPO, r["hf_path"], repo_type="dataset", local_dir=DATA / "hf")
        return 1

    with ThreadPoolExecutor(workers) as ex:
        n = sum(ex.map(one, rows))
    print(f"downloaded {n} new videos ({len(rows)} total)")


def read_manifest():
    with open(DATA / "manifest.csv", newline="") as f:
        return list(csv.DictReader(f))


def main():
    rows = select()
    with open(p("manifest.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    by = {}
    for r in rows:
        by.setdefault((r["split"], r["method"]), 0)
        by[(r["split"], r["method"])] += 1
    for k, v in sorted(by.items()):
        print(k, v)
    if "--no-download" not in sys.argv:
        download(rows)


if __name__ == "__main__":
    main()
