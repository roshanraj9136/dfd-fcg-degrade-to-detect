"""A smaller held-out test list from the newer datasets (time limit): a fixed number per dataset and label, taking
videos whose faces are already prepared first, the rest in a fixed random order. Writes <OUT>/sets_test_small.csv."""
import csv
import random
from pathlib import Path

NEW, CROPS = Path(r"D:\dfd-data\new"), Path(r"D:\dfd-data\crops")
QUOTA = {("test", "celebdf", "0"): 30, ("test", "celebdf", "1"): 30, ("test", "social", "0"): 50, ("test", "social", "1"): 50,
         ("test", "synth", "1"): 40, ("test", "mavos", "0"): 40, ("test", "mavos", "1"): 40,
         ("test_open", "mavos", "0"): 30, ("test_open", "mavos", "1"): 30, ("test", "sf", "1"): 60}
rows = [r for f in ("sets.csv", "sets_gated.csv") for r in csv.DictReader(open(NEW / f))]
rng = random.Random(7)
out = []
for key, k in QUOTA.items():
    pool = [r for r in rows if (r["set"], r["source"], r["label"]) == key]
    rng.shuffle(pool)
    done = lambda r: (CROPS / (r["vid"].replace("/", "__") + ".npz")).exists()  # noqa: E731
    pool.sort(key=lambda r: not done(r))
    out += pool[:k]
with open(NEW / "sets_test_small.csv", "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0]))
    w.writeheader()
    w.writerows(out)
todo = sum(not (CROPS / (r["vid"].replace("/", "__") + ".npz")).exists() for r in out)
print(len(out), "test videos,", todo, "still to prepare")
