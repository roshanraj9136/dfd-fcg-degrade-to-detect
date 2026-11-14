"""How many more train-split videos the open datasets offer beyond sets.csv / sets_gated.csv."""
import collections
import csv
import json
from pathlib import Path

import pyarrow as pa

N = Path(r"D:\dfd-data\new")
used = {r["path"] for f in ("sets.csv", "sets_gated.csv") for r in csv.DictReader(open(N / f))}
arrow = next((N / "meta" / "mavos").rglob("*.arrow"))
with pa.memory_map(str(arrow)) as src:
    try:
        meta = pa.ipc.open_stream(src).read_all().to_pylist()
    except pa.ArrowInvalid:
        meta = pa.ipc.open_file(src).read_all().to_pylist()
c = collections.Counter((r["split"], r["generative_method"] if r["label"] != "real" else "real", r["open_set_model"],
                         r["open_set_language"], r["video_path"] in used) for r in meta if r["split"] == "train")
for k in sorted(c, key=str):
    print("mavos", k, c[k])
print(meta[0])
sf = json.load(open(next((N / "meta" / "sf").rglob("train.json"))))
c = collections.Counter((p.split("/")[0], p.split("/")[2], p.split("/")[3], p in used) for p in sf)
for k in sorted(c):
    if k[1] == "crf23":
        print("sf", k, c[k])
cf = list(csv.DictReader(open(next((N / "meta" / "synth").rglob("metadata.csv")), encoding="utf-8")))
cf = [r for r in cf if r.get("primary") == "1" and r.get("qc_pass") == "1" and r.get("qc_face") == "1"]
c = collections.Counter((r["generator"], int(r["counterfactual_group"]) <= 24, r["file_name"] in used) for r in cf)
for k in sorted(c):
    print("cf", k, c[k])
