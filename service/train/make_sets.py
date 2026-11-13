"""Training / validation / test sets for fine-tuning DFD-FCG on newer, open datasets, and their download.

Sources (all public on the Hugging Face Hub, research / non-commercial use):
  ffpp     FaceForensics++ c23 (gonnerthetooner/FaceForensics): official train split for training, official val split
           for validation. The FF++ test split, our fit split and our selection videos are never used here.
  dfdc     DFDC train parts 1-8 (training) and part 9 (validation) (gonnerthetooner/DFDC-extracted-full).
           Part 0, where all our DFDC test and selection videos come from, is never used here.
  celebdf  Celeb-DF v2 (gonnerthetooner/deepfake_video_data): videos outside the official test list for training and
           validation; a sample of the official test list for testing.
  synth    Counterfactual DeepFake Bench, Oct 2026 (machine-intelligence-lab-wvu/counterfactual-deepfake-bench): fully
           AI-generated talking heads from Hunyuan 1.5, LTX 2.3, LTX 2.5 and Wan 2.2 (fakes only). Split by
           counterfactual group (scene/person design): groups 1-24 train, 25-30 validation, 31-36 test.
  social   Social-media deepfakes, 2026 (acroitoru/social_media_deepfakes): its "social_media" part for training and
           validation, its own "social_media_test" part for testing.

Writes <OUT>/sets.csv (vid, set, source, method, label, repo, path) and downloads every video to <OUT>/videos/.
    python service/train/make_sets.py [OUT=D:/dfd-data/new]
"""
import csv
import json
import random
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from huggingface_hub import hf_hub_download

OUT = Path(sys.argv[1] if len(sys.argv) > 1 else r"D:\dfd-data\new")
DATA = Path(r"C:\Users\hp\dfd-work\data")
FF, DFDC, VID = "gonnerthetooner/FaceForensics", "gonnerthetooner/DFDC-extracted-full", "gonnerthetooner/deepfake_video_data"
SYN, SOC = "machine-intelligence-lab-wvu/counterfactual-deepfake-bench", "acroitoru/social_media_deepfakes"
rng = random.Random(2026)
rows = []


def add(vid, set_, source, method, label, repo, path):
    rows.append(dict(vid=vid, set=set_, source=source, method=method, label=label, repo=repo, path=path))


# ---- FF++: official train / val splits, minus anything we already use
used = {r["vid"] for f in ("manifest_full.csv", "manifest.csv", "manifest_select.csv") for r in csv.DictReader(open(DATA / f))}
P = "faceforensics_c23/FaceForensics++_C23/"
FOLD = {"DF": "Deepfakes", "F2F": "Face2Face", "FS": "FaceSwap", "NT": "NeuralTextures"}
for split, n_real, n_fake in (("train", 120, 30), ("val", 40, 10)):
    pairs = json.load(open(DATA / "splits" / f"{split}.json"))
    reals = sorted({i for p in pairs for i in p} - {v.split("/")[1] for v in used if v.startswith("REAL/")})
    for i in sorted(rng.sample(reals, n_real)):
        add(f"FF_REAL/{i}", split, "ffpp", "REAL", 0, FF, f"{P}original/{i}.mp4")
    fakes = [f"{a}_{b}" for a, b in pairs] + [f"{b}_{a}" for a, b in pairs]
    for m, folder in FOLD.items():
        pool = sorted(f for f in fakes if f"{m}/{f}" not in used)
        for f in sorted(rng.sample(pool, n_fake)):
            add(f"FF_{m}/{f}", split, "ffpp", m, 1, FF, f"{P}{folder}/{f}.mp4")

# ---- DFDC parts 1-8 train, 9 val: real + one of its fakes
for k in range(1, 10):
    part = f"dfdc_train_part_0{k}/dfdc_train_part_{k}/"
    meta = json.load(open(hf_hub_download(DFDC, part + "metadata.json", repo_type="dataset", local_dir=OUT / "meta" / "dfdc")))
    fakes_of = {}
    for n, v in meta.items():
        if v["label"] == "FAKE" and v.get("original") in meta:
            fakes_of.setdefault(v["original"], []).append(n)
    reals = sorted(n for n, v in meta.items() if v["label"] == "REAL" and n in fakes_of)
    split = "val" if k == 9 else "train"
    for r in sorted(rng.sample(reals, 30 if k == 9 else 19)):
        f = sorted(fakes_of[r])[rng.randrange(len(fakes_of[r]))]
        add(f"DFDC{k}_R/{r[:-4]}", split, "dfdc", "DFDC_R", 0, DFDC, part + r)
        add(f"DFDC{k}_F/{f[:-4]}", split, "dfdc", "DFDC_F", 1, DFDC, part + f)

# ---- Celeb-DF v2
test_list = open(hf_hub_download(VID, "celebdf_v2/List_of_testing_videos.txt", repo_type="dataset", local_dir=OUT / "meta"))
test_set = {ln.split(" ", 1)[1].strip() for ln in test_list if ln.strip()}
from huggingface_hub import HfApi  # noqa: E402
files = [f for f in HfApi().list_repo_files(VID, repo_type="dataset") if f.startswith("celebdf_v2/") and f.endswith(".mp4")]
by = {"real": [], "fake": [], "real_test": [], "fake_test": []}
for f in sorted(files):
    rel = f.split("celebdf_v2/", 1)[1]
    kind = "fake" if rel.startswith("Celeb-synthesis/") else "real"
    by[kind + ("_test" if rel in test_set else "")].append(f)
for kind, lab in (("real", 0), ("fake", 1)):
    pool = rng.sample(by[kind], 180)
    for i, f in enumerate(sorted(pool)):
        add(f"CDF_{kind}/{Path(f).parent.name}_{Path(f).stem}", "val" if i % 6 == 0 else "train", "celebdf",
            "CDF_R" if lab == 0 else "CDF_F", lab, VID, f)
    for f in sorted(rng.sample(by[kind + "_test"], 60)):
        add(f"CDF_{kind}/{Path(f).parent.name}_{Path(f).stem}", "test", "celebdf", "CDF_R" if lab == 0 else "CDF_F", lab, VID, f)

# ---- Counterfactual DeepFake Bench (fully AI-generated, fakes only)
meta = list(csv.DictReader(open(hf_hub_download(SYN, "metadata.csv", repo_type="dataset", local_dir=OUT / "meta" / "synth"),
                                encoding="utf-8")))
meta = [r for r in meta if r.get("primary") == "1" and r.get("qc_pass") == "1" and r.get("qc_face") == "1"]
for split, groups, per_gen in (("train", range(1, 25), 50), ("val", range(25, 31), 10), ("test", range(31, 37), 20)):
    for gen in sorted({r["generator"] for r in meta}):
        pool = sorted((r for r in meta if r["generator"] == gen and int(r["counterfactual_group"]) in groups),
                      key=lambda r: r["file_name"])
        for r in sorted(rng.sample(pool, min(per_gen, len(pool))), key=lambda r: r["file_name"]):
            add(f"SYN_{gen}/{Path(r['file_name']).stem}", split, "synth", f"SYN_{gen}", 1, SYN, r["file_name"])

# ---- Social-media deepfakes
files = [f for f in HfApi().list_repo_files(SOC, repo_type="dataset") if f.endswith(".mp4")]
for f in sorted(files):
    part, kind = f.split("/")[0], f.split("/")[1]
    lab = 1 if kind == "fake" else 0
    split = "test" if part == "social_media_test" else ("val" if rng.random() < 0.2 else "train")
    add(f"SOC_{kind}/{Path(f).stem[:13]}", split, "social", "SOC_F" if lab else "SOC_R", lab, SOC, f)

OUT.mkdir(parents=True, exist_ok=True)
with open(OUT / "sets.csv", "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0]))
    w.writeheader()
    w.writerows(rows)
import collections  # noqa: E402
print(collections.Counter((r["set"], r["source"], r["label"]) for r in rows), flush=True)
print(len(rows), "videos", flush=True)


def dl(r):
    try:
        hf_hub_download(r["repo"], r["path"], repo_type="dataset", local_dir=OUT / "videos" / r["source"])
        return None
    except Exception as e:  # noqa: BLE001
        return f"{r['vid']}: {type(e).__name__} {str(e)[:100]}"


with ThreadPoolExecutor(8) as ex:
    for k, err in enumerate(ex.map(dl, rows)):
        if err:
            print("FAILED", err, flush=True)
        if (k + 1) % 100 == 0:
            print(f"downloaded {k + 1}/{len(rows)}", flush=True)
print("download finished", flush=True)
