"""More TRAINING videos from the same open datasets (train splits only; validation and test sets are unchanged), aimed
at the weakest areas of round 2: modern face swaps / talking-head animation (MAVOS-DD), fully AI-generated people
(SynthForensics, Counterfactual DeepFake Bench) and more real faces (MAVOS-DD, FF++, Celeb-DF v2) so the fake-only
AI-generated sources are balanced by reals.

Never used: any test / test_open / val video of sets.csv / sets_gated.csv, the FF++ test split, our fit and selection
videos, the Celeb-DF v2 official test list, Counterfactual groups 25-36, DFDC.
Writes <OUT>/sets_more.csv (same columns) and downloads the videos.
    python service/train/make_sets_more.py [OUT=D:/dfd-data/new]
"""
import collections
import csv
import json
import random
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pyarrow as pa
from huggingface_hub import HfApi, hf_hub_download

OUT = Path(sys.argv[1] if len(sys.argv) > 1 else r"D:\dfd-data\new")
DATA = Path(r"C:\Users\hp\dfd-work\data")
rng = random.Random(2028)
old = [r for f in ("sets.csv", "sets_gated.csv") for r in csv.DictReader(open(OUT / f))]
used_path = {(r["source"], r["path"]) for r in old}
used_vid = {r["vid"] for r in old}
rows = []


def add(vid, source, method, label, repo, path):
    if (source, path) in used_path or vid in used_vid:
        return
    used_path.add((source, path))
    used_vid.add(vid)
    rows.append(dict(vid=vid, set="train", source=source, method=method, label=label, repo=repo, path=path))


# ---- MAVOS-DD train split (in-domain methods, closed-set languages): +100 fakes per method, +400 reals
MAV = "unibuc-cs/MAVOS-DD"
arrow = next((OUT / "meta" / "mavos").rglob("*.arrow"))
with pa.memory_map(str(arrow)) as src:
    try:
        meta = pa.ipc.open_stream(src).read_all().to_pylist()
    except pa.ArrowInvalid:
        meta = pa.ipc.open_file(src).read_all().to_pylist()
sel = [r for r in meta if r["split"] == "train" and not r["open_set_model"] and not r["open_set_language"]
       and ("mavos", r["video_path"]) not in used_path]
for m, k in (("inswapper", 100), ("echomimic", 100), ("liveportrait", 100), ("memo", 100), ("real", 400)):
    pool = sorted((r for r in sel if (r["generative_method"] if r["label"] != "real" else "real") == m),
                  key=lambda r: r["video_path"])
    for r in rng.sample(pool, min(k, len(pool))):
        lab = 0 if m == "real" else 1
        add(f"MAV_{m}/{Path(r['video_path']).stem[:40]}", "mavos", "MAV_R" if lab == 0 else f"MAV_{m}", lab, MAV,
            r["video_path"])

# ---- SynthForensics train split, CRF-23: +8 per (mode, generator)
SF = "SynthForensics/SynthForensics"
paths = json.load(open(next((OUT / "meta" / "sf").rglob("train.json"))))
by = collections.defaultdict(list)
for p in paths:
    q = p.split("/")
    if q[2] == "crf23" and ("sf", p) not in used_path:
        by[(q[0], q[3])].append(p)
for (mode, gen), ps in sorted(by.items()):
    for p in sorted(rng.sample(sorted(ps), min(8, len(ps)))):
        add(f"SF_{gen}_{mode}/{Path(p).stem}", "sf", f"SF_{mode}", 1, SF, p)

# ---- Counterfactual DeepFake Bench, groups 1-24 only: +40 per generator
SYN = "machine-intelligence-lab-wvu/counterfactual-deepfake-bench"
cf = list(csv.DictReader(open(next((OUT / "meta" / "synth").rglob("metadata.csv")), encoding="utf-8")))
cf = [r for r in cf if r.get("primary") == "1" and r.get("qc_pass") == "1" and r.get("qc_face") == "1"
      and int(r["counterfactual_group"]) <= 24]
for gen in sorted({r["generator"] for r in cf}):
    pool = sorted((r for r in cf if r["generator"] == gen and ("synth", r["file_name"]) not in used_path),
                  key=lambda r: r["file_name"])
    for r in sorted(rng.sample(pool, min(40, len(pool))), key=lambda r: r["file_name"]):
        add(f"SYN_{gen}/{Path(r['file_name']).stem}", "synth", f"SYN_{gen}", 1, SYN, r["file_name"])

# ---- FF++ official train split: +120 reals, +20 fakes per method (never test / fit / selection videos)
FF, P = "gonnerthetooner/FaceForensics", "faceforensics_c23/FaceForensics++_C23/"
FOLD = {"DF": "Deepfakes", "F2F": "Face2Face", "FS": "FaceSwap", "NT": "NeuralTextures"}
ours = {r["vid"] for f in ("manifest_full.csv", "manifest.csv", "manifest_select.csv") for r in csv.DictReader(open(DATA / f))}
pairs = json.load(open(DATA / "splits" / "train.json"))
reals = sorted({i for p in pairs for i in p} - {v.split("/")[1] for v in ours if v.startswith("REAL/")})
reals = [i for i in reals if f"FF_REAL/{i}" not in used_vid]
for i in sorted(rng.sample(reals, min(120, len(reals)))):
    add(f"FF_REAL/{i}", "ffpp", "REAL", 0, FF, f"{P}original/{i}.mp4")
fakes = [f"{a}_{b}" for a, b in pairs] + [f"{b}_{a}" for a, b in pairs]
for m, folder in FOLD.items():
    pool = sorted(f for f in fakes if f"{m}/{f}" not in ours and f"FF_{m}/{f}" not in used_vid)
    for f in sorted(rng.sample(pool, 20)):
        add(f"FF_{m}/{f}", "ffpp", m, 1, FF, f"{P}{folder}/{f}.mp4")

# ---- Celeb-DF v2 outside the official test list: +100 real, +100 fake
VID = "gonnerthetooner/deepfake_video_data"
test_set = {ln.split(" ", 1)[1].strip() for ln in open(next((OUT / "meta").rglob("List_of_testing_videos.txt")))
            if ln.strip()}
files = [f for f in HfApi().list_repo_files(VID, repo_type="dataset") if f.startswith("celebdf_v2/") and f.endswith(".mp4")]
for kind, lab in (("real", 0), ("fake", 1)):
    pool = []
    for f in sorted(files):
        rel = f.split("celebdf_v2/", 1)[1]
        if rel in test_set or (("fake" if rel.startswith("Celeb-synthesis/") else "real") != kind):
            continue
        if ("celebdf", f) not in used_path:
            pool.append(f)
    for f in sorted(rng.sample(pool, min(100, len(pool)))):
        add(f"CDF_{kind}/{Path(f).parent.name}_{Path(f).stem}", "celebdf", "CDF_R" if lab == 0 else "CDF_F", lab, VID, f)

with open(OUT / "sets_more.csv", "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0]))
    w.writeheader()
    w.writerows(rows)
print(collections.Counter((r["source"], r["label"]) for r in rows), len(rows), "videos", flush=True)


def dl(r):
    try:
        hf_hub_download(r["repo"], r["path"], repo_type="dataset", local_dir=OUT / "videos" / r["source"])
    except Exception as e:  # noqa: BLE001
        return f"{r['vid']}: {type(e).__name__} {str(e)[:100]}"


# interleave sources so every source gets prepared early
order = sorted(range(len(rows)), key=lambda i: (i % 7, i))
with ThreadPoolExecutor(8) as ex:
    for k, err in enumerate(ex.map(dl, [rows[i] for i in order])):
        if err:
            print("FAILED", err, flush=True)
        if (k + 1) % 100 == 0:
            print(f"downloaded {k + 1}/{len(rows)}", flush=True)
print("download finished", flush=True)
