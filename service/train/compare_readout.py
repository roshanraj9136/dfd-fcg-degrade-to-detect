"""Does fit_readout.py reproduce the deployed read-out? Compares the predicted quality shift g on the cached fit
features. python service/train/compare_readout.py <refit model.json> <features.npz> [deployed model.json]"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
a = json.load(open(sys.argv[1]))["selfcal"]
b = json.load(open(sys.argv[3] if len(sys.argv) > 3 else ROOT / "service" / "app" / "model.json"))["selfcal"]
T = np.load(sys.argv[2], allow_pickle=True)
Y = np.concatenate([T["y_s"], T["y_t"]], 1)
g = lambda m: ((Y - np.array(m["mean"])) / np.array(m["scale"])) @ np.array(m["coef"]).T + np.array(m["intercept"])  # noqa: E731
ga, gb = g(a), g(b)
print("rows", len(Y), "| mean |g| deployed", np.abs(gb).mean().round(4), "refit", np.abs(ga).mean().round(4),
      "| max |diff|", np.abs(ga - gb).max().round(4), "| corr", np.corrcoef(ga.ravel(), gb.ravel())[0, 1].round(5))
