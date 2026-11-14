"""End-to-end run of the live checker pipeline (face finding -> every detector -> fusion -> verdict) on one video file.
    python tests/run_live_once.py <video> [cpu|cuda] [none|blur|compress]
Prints the verdict, the confidence, every member's score and the timing."""
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "service" / "app"), str(ROOT)]
os.environ.setdefault("MODEL_DIR", r"D:\dfd-data\ens\models")
from pipeline import Analyzer  # noqa: E402

video = sys.argv[1]
dev = sys.argv[2] if len(sys.argv) > 2 else "cpu"
deg = sys.argv[3] if len(sys.argv) > 3 else "none"
t = time.time()
an = Analyzer(device=dev)
print(f"loaded in {time.time() - t:.0f}s; fusion members: {an.fusion and an.fusion['members']}", flush=True)
t = time.time()
r = an.run(video, None if deg == "none" else deg)
small = {k: v for k, v in r.items() if not k.endswith("_png") and k not in ("timeline",)}
print(json.dumps(small, indent=1, default=float))
print(f"VERDICT {r['verdict']}  confidence {r['confidence']:.3f}  p_fake {r['p']:.3f}  ({time.time() - t:.0f}s)")
