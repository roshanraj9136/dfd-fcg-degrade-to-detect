"""
Shrink the manifest to the compute budget of a 4 GB laptop GPU.

Per (split, method) group a fixed number of videos is kept. Videos whose face alignment
is already finished are kept first; the rest are taken in a seeded random order, so the
choice never depends on the model's output.

    python -m qa.subset
"""
import csv
import random
import zlib

from qa.config import DATA
from qa.manifest import read_manifest
from qa.prep import safe

KEEP = {("test", "REAL"): 70, ("test", "DF"): 15, ("test", "F2F"): 15, ("test", "FS"): 15,
        ("test", "NT"): 15, ("test", "FSh"): 15, ("fit", "REAL"): 30, ("fit", "DF"): 5,
        ("fit", "F2F"): 5, ("fit", "FS"): 5, ("fit", "NT"): 5, ("demo", "DF"): 1}


def main():
    rows = read_manifest()
    groups = {}
    for r in rows:
        groups.setdefault((r["split"], r["method"]), []).append(r)
    keep = set()
    for g, rs in groups.items():
        rs = sorted(rs, key=lambda r: (not (DATA / "prep" / (safe(r["vid"]) + ".npz")).exists(),
                                       zlib.crc32(r["vid"].encode())))
        for r in rs[:KEEP[g]]:
            keep.add(r["vid"])
    out = [r for r in rows if r["vid"] in keep]
    full = DATA / "manifest_full.csv"
    if not full.exists():
        full.write_text((DATA / "manifest.csv").read_text())
    with open(DATA / "manifest.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(out[0]))
        w.writeheader()
        w.writerows(out)
    print(f"kept {len(out)} of {len(rows)} videos")


if __name__ == "__main__":
    random.seed(0)
    main()
