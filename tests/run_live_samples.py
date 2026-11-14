"""Live pipeline on a few video files with one model load: verdict, final score, per-clip timeline, detectors.
    python tests/run_live_samples.py <cuda|cpu> <video> [<video> ...]"""
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "service" / "app"), str(ROOT)]
os.environ.setdefault("MODEL_DIR", r"D:\dfd-data\ens\models")
from pipeline import Analyzer  # noqa: E402

an = Analyzer(device=sys.argv[1])
for v in sys.argv[2:]:
    r = an.run(v)
    print(Path(v).name, r["verdict"], round(r["p"], 3), "clips", [round(c["p"], 2) for c in r["timeline"]],
          "detectors", r["detectors"], flush=True)
