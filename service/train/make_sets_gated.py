"""Adds the two gated 2025-2026 datasets to the fine-tuning sets (accept their terms on Hugging Face first).

  mavos  MAVOS-DD (unibuc-cs/MAVOS-DD, CC BY-NC-SA 4.0): YouTube videos in 8 languages with modern face swaps and
         talking-head animation. Its official splits are used: train -> train, validation -> val, test (in-domain
         methods inswapper / echomimic / liveportrait / memo) -> test, and test "open-set model" (hififace, sonic, roop:
         never in its training split) -> test_open.
  sf     SynthForensics (SynthForensics/SynthForensics, CC BY-NC 4.0, NeurIPS 2026): fully AI-generated people from 8
         text/image-to-video generators, CRF-23 version. Its official splits follow the FF++ split by source video, so
         nothing derives from our FF++ test identities in training.

Writes <OUT>/sets_gated.csv (same columns as sets.csv) and downloads the videos to <OUT>/videos/<source>/.
    python service/train/make_sets_gated.py [OUT=D:/dfd-data/new]
"""
import collections
import csv
import json
import random
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pyarrow as pa
from huggingface_hub import hf_hub_download

OUT = Path(sys.argv[1] if len(sys.argv) > 1 else r"D:\dfd-data\new")
MAV, SF = "unibuc-cs/MAVOS-DD", "SynthForensics/SynthForensics"
rng = random.Random(2027)
rows = []


def add(vid, set_, source, method, label, repo, path):
    rows.append(dict(vid=vid, set=set_, source=source, method=method, label=label, repo=repo, path=path))


# ---- MAVOS-DD
arrow = hf_hub_download(MAV, "data-00000-of-00001.arrow", repo_type="dataset", local_dir=OUT / "meta" / "mavos")
with pa.memory_map(arrow) as src:
    try:
        meta = pa.ipc.open_stream(src).read_all().to_pylist()
    except pa.ArrowInvalid:
        meta = pa.ipc.open_file(src).read_all().to_pylist()


def pick(pool, k):
    pool = sorted(pool, key=lambda r: r["video_path"])
    return rng.sample(pool, min(k, len(pool)))


IN, OPEN = ("inswapper", "echomimic", "liveportrait", "memo"), ("hififace", "sonic", "roop")
plan = [("train", "train", False, IN, 60, 240), ("validation", "val", False, IN, 10, 40),
        ("test", "test", False, IN, 25, 100), ("test", "test_open", True, OPEN, 40, 60)]
for split, set_, open_model, methods, per_m, n_real in plan:
    sel = [r for r in meta if r["split"] == split and r["open_set_model"] == open_model and not r["open_set_language"]]
    for m in methods:
        for r in pick([x for x in sel if x["generative_method"] == m], per_m):
            add(f"MAV_{m}/{Path(r['video_path']).stem[:40]}", set_, "mavos", f"MAV_{m}", 1, MAV, r["video_path"])
    for r in pick([x for x in sel if x["label"] == "real"], n_real):
        add(f"MAV_real/{Path(r['video_path']).stem[:40]}", set_, "mavos", "MAV_R", 0, MAV, r["video_path"])

# ---- SynthForensics, CRF-23
for split, set_, per_pair in (("train", "train", 16), ("val", "val", 3), ("test", "test", 8)):
    paths = json.load(open(hf_hub_download(SF, f"{split}.json", repo_type="dataset", local_dir=OUT / "meta" / "sf")))
    by = collections.defaultdict(list)
    for p in paths:
        q = p.split("/")
        if q[2] == "crf23":
            by[(q[0], q[3])].append(p)
    for (mode, gen), ps in sorted(by.items()):
        for p in sorted(rng.sample(sorted(ps), min(per_pair, len(ps)))):
            add(f"SF_{gen}_{mode}/{Path(p).stem}", set_, "sf", f"SF_{mode}", 1, SF, p)

with open(OUT / "sets_gated.csv", "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0]))
    w.writeheader()
    w.writerows(rows)
print(collections.Counter((r["set"], r["source"], r["label"]) for r in rows), len(rows), "videos", flush=True)


def dl(r):
    try:
        hf_hub_download(r["repo"], r["path"], repo_type="dataset", local_dir=OUT / "videos" / r["source"])
    except Exception as e:  # noqa: BLE001
        return f"{r['vid']}: {type(e).__name__} {str(e)[:100]}"


with ThreadPoolExecutor(8) as ex:
    for k, err in enumerate(ex.map(dl, rows)):
        if err:
            print("FAILED", err, flush=True)
        if (k + 1) % 100 == 0:
            print(f"downloaded {k + 1}/{len(rows)}", flush=True)
print("download finished", flush=True)
