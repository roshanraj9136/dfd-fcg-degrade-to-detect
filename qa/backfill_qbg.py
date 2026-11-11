"""
Compute the background-referenced quality descriptors for videos whose features were
extracted before they existed (CPU only; rebuilds the same crops deterministically).

    python -m qa.backfill_qbg [--workers 4]
"""
import argparse
import pickle
from multiprocessing import Pool

import numpy as np

from qa.conditions import build
from qa.config import DATA, p
from qa.extract import _low_priority, load_feats, summarise_built
from qa.manifest import local_video, read_manifest
from qa.prep import safe


def qbg_path(vid):
    return DATA / "qbg" / (safe(vid) + ".pkl")


def load_qbg(vid):
    f = qbg_path(vid)
    if f.exists():
        with open(f, "rb") as fh:
            return pickle.load(fh)
    return {}


def _one(args):
    row, conds = args
    prep = dict(np.load(DATA / "prep" / (safe(row["vid"]) + ".npz")))
    out = summarise_built(prep, build(local_video(row), prep, row["vid"], conds))
    have = load_qbg(row["vid"])
    for c, (_, _, q, qb) in out.items():
        have[c] = dict(q_bg=qb, q=q)
    with open(p("qbg", safe(row["vid"]) + ".pkl"), "wb") as fh:
        pickle.dump(have, fh)
    return row["vid"], len(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    tasks = []
    for r in read_manifest():
        feats = load_feats(r["vid"])
        if not feats:
            continue
        have = load_qbg(r["vid"])
        miss = [c for c, e in feats.items() if "q_bg" not in e and c not in have]
        if miss:
            tasks.append((r, miss))
    print(f"{len(tasks)} videos need background quality", flush=True)
    with Pool(a.workers, initializer=_low_priority) as pool:
        for i, (vid, n) in enumerate(pool.imap_unordered(_one, tasks)):
            print(f"[{i + 1}/{len(tasks)}] {vid}: {n} conditions", flush=True)


if __name__ == "__main__":
    main()
