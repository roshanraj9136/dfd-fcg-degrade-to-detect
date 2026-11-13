"""Refit the self-calibration read-out (qa.methods.SelfCal) for a fine-tuned checkpoint, with exactly the research
protocol: the 49 'fit' videos (FF++ train split), their cached research face geometry (data/prep), clean + the 7 damage
types at levels 2 and 4 + the re-encode control (qa.config.conditions('fit')), the authors' damage code
(qa.conditions.build), 2 clips per video, fp16 on the GPU.
    python service/train/fit_readout.py <checkpoint> <out model.json> [features.npz cache]
With the released checkpoint this must reproduce service/app/model.json (checked by compare_readout.py)."""
import csv
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CK, OUT = sys.argv[1], Path(sys.argv[2])
CACHE = Path(sys.argv[3]) if len(sys.argv) > 3 else OUT.with_suffix(".feats.npz")
os.environ["DFD_FCG_CKPT"] = CK
sys.path[:0] = [str(ROOT)]
import numpy as np  # noqa: E402
import torch  # noqa: E402

from qa.conditions import build  # noqa: E402
from qa.config import DATA, conditions  # noqa: E402
from qa.engine import Recorder, analyse, load_model  # noqa: E402
from qa.manifest import local_video  # noqa: E402
from qa.methods import SelfCal  # noqa: E402
from qa.prep import safe  # noqa: E402

if CACHE.exists():
    T = dict(np.load(CACHE, allow_pickle=True))
else:
    model = load_model(device="cuda")
    rec = Recorder(model)
    rows = [r for r in csv.DictReader(open(DATA / "manifest.csv")) if r["split"] == "fit"]
    conds = conditions("fit")
    T = {k: [] for k in ("vid", "clip", "cond", "label", "d", "y_s", "y_t")}
    part_dir = CACHE.with_suffix("")                     # one file per video, so an interrupted run resumes
    part_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    for k, r in enumerate(rows):
        part = part_dir / (safe(r["vid"]) + ".npz")
        if part.exists():
            z = np.load(part, allow_pickle=True)
            for key in T:
                T[key] += list(z[key])
            continue
        pp = DATA / "prep" / (safe(r["vid"]) + ".npz")
        if not pp.exists():
            print("no prep", r["vid"], flush=True)
            continue
        prep = dict(np.load(pp))
        built = build(str(local_video(r)), prep, r["vid"], conds)
        pos = {f: i for i, f in enumerate(prep["needed"])}
        one = {key: [] for key in T}
        for cond in conds:
            crops = built[cond][0]
            for ci, clip in enumerate(prep["layout"]):
                fr = torch.from_numpy(np.ascontiguousarray(crops[[pos[f] for f in clip]])).permute(0, 3, 1, 2).contiguous()
                with torch.inference_mode():
                    res = analyse(model, rec, model.transform(fr)[None], fp16=True)[0]
                for key, v in (("vid", r["vid"]), ("clip", ci), ("cond", cond), ("label", int(r["label"])),
                               ("d", res["d"]), ("y_s", res["y_s"]), ("y_t", res["y_t"])):
                    one[key].append(v)
        np.savez(part, **{key: np.array(v) for key, v in one.items()})
        for key in T:
            T[key] += one[key]
        print(f"[{k + 1}/{len(rows)}] {r['vid']} {(time.time() - t0) / 60:.1f} min", flush=True)
    T = {k: np.array(v) for k, v in T.items()}
    np.savez_compressed(CACHE, **T)

M = SelfCal().fit(T, np.ones(len(T["vid"]), bool))
old = json.load(open(ROOT / "service" / "app" / "model.json"))
new = dict(old, selfcal=dict(mean=M.sc.mean_.tolist(), scale=M.sc.scale_.tolist(), coef=M.reg.coef_.tolist(),
                             intercept=M.reg.intercept_.tolist()), checkpoint=Path(CK).name)
OUT.write_text(json.dumps(new, separators=(",", ":")))
print("targets", {c: [round(x, 3) for x in v] for c, v in M.targets.items()})
print("wrote", OUT, "rows", len(T["vid"]))
