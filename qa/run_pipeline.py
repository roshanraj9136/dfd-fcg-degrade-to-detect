"""
Drive the whole extraction on a single small GPU: alternate a face-alignment process and an
inference process, so each one has the card to itself (sharing a 4 GB WDDM GPU between the two
models made DFD-FCG spill into system memory and run 3x slower).

    python -m qa.run_pipeline [--chunk 30]
"""
import argparse
import os
import subprocess
import sys
import time

from qa.config import DATA
from qa.extract import load_feats, priority_order, wanted_conditions
from qa.manifest import read_manifest
from qa.prep import safe


def status():
    rows = priority_order(read_manifest())
    unprepped = [r for r in rows if not (DATA / "prep" / (safe(r["vid"]) + ".npz")).exists()
                 and not (DATA / "prep" / (safe(r["vid"]) + ".fail")).exists()]
    pending = [r for r in rows if (DATA / "prep" / (safe(r["vid"]) + ".npz")).exists()
               and any(c not in load_feats(r["vid"]) for c in wanted_conditions(r))]
    return unprepped, pending


def run(args, env):
    print(">>", " ".join(args), flush=True)
    subprocess.run([sys.executable, "-u", "-m"] + args, env=env, check=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunk", type=int, default=30)
    a = ap.parse_args()
    env = dict(os.environ, PREP_BATCH="1", PREP_LIMIT=str(a.chunk), TORCHDYNAMO_DISABLE="1")
    t0 = time.time()
    while True:
        unprepped, pending = status()
        print(f"== {time.time() - t0:.0f}s: {len(unprepped)} to align, {len(pending)} aligned and pending inference",
              flush=True)
        if not unprepped and not pending:
            break
        if pending and (len(pending) >= a.chunk or not unprepped):
            run(["qa.extract", "--no-prep", "--workers", "3"], env)
        elif unprepped:
            run(["qa.prep"], env)
    print(f"all done in {(time.time() - t0) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
