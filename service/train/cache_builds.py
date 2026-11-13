"""Build (CPU only) the aligned face crops of the research protocol once, so several models can be run on them later:
cached research geometry (data/prep), the authors' damage code (qa.conditions.build), 2 clips per video.
    python service/train/cache_builds.py fit  <out dir>    # 49 fit videos, clean + 7 damage types x levels 2/4 + RE_0
    python service/train/cache_builds.py test <out dir>    # 225 reported test videos, clean / GB_5 / VC_5
A video that fails (e.g. a video read failing when memory is low) is retried in up to 3 later passes; the script
prints "finished" only when every video with research geometry has its file, else "INCOMPLETE".
"""
import csv
import gc
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT)]
import numpy as np  # noqa: E402

from qa.conditions import build  # noqa: E402
from qa.config import DATA, conditions  # noqa: E402
from qa.manifest import local_video  # noqa: E402
from qa.prep import safe  # noqa: E402

WHICH, OUT = sys.argv[1], Path(sys.argv[2])
OUT.mkdir(parents=True, exist_ok=True)
splits = ("fit",) if WHICH == "fit" else ("test", "cross")
conds = conditions("fit") if WHICH == "fit" else ["clean", "GB_5", "VC_5"]
rows = [r for r in csv.DictReader(open(DATA / "manifest.csv")) if r["split"] in splits
        and (DATA / "prep" / (safe(r["vid"]) + ".npz")).exists()]
t0 = time.time()
for attempt in range(4):
    todo = [r for r in rows if not (OUT / (safe(r["vid"]) + ".npz")).exists()]
    if os.environ.get("SHARD"):
        i, n = map(int, os.environ["SHARD"].split("/"))
        todo = [r for j, r in enumerate(todo) if j % n == i]
    if not todo:
        break
    print(f"pass {attempt + 1}: {len(todo)} videos", flush=True)
    for k, r in enumerate(todo):
        out = OUT / (safe(r["vid"]) + ".npz")
        if out.exists():                                   # done meanwhile by another shard
            continue
        try:
            prep = dict(np.load(DATA / "prep" / (safe(r["vid"]) + ".npz")))
            built = build(str(local_video(r)), prep, r["vid"], conds)
            pos = {f: i for i, f in enumerate(prep["needed"])}
            clips = np.array([[pos[f] for f in c] for c in prep["layout"]])
            np.savez(str(out) + f".{os.getpid()}.tmp.npz", clip_idx=clips, label=int(r["label"]), method=r["method"], vid=r["vid"],
                     conds=np.array(conds), **{f"c:{c}": built[c][0] for c in conds})
            os.replace(str(out) + f".{os.getpid()}.tmp.npz", out)
            print(f"[{k + 1}/{len(todo)}] {r['vid']} {(time.time() - t0) / 60:.1f} min", flush=True)
        except Exception as e:  # noqa: BLE001
            print(f"skip {r['vid']} for now: {type(e).__name__}: {str(e)[:80]}", flush=True)
        built = None
        gc.collect()
    time.sleep(20)
missing = [r["vid"] for r in rows if not (OUT / (safe(r["vid"]) + ".npz")).exists()]
print("finished" if not missing else f"INCOMPLETE {missing}", flush=True)
