"""Accuracy study for the live checker (service/app/pipeline.py).

For every video and website option (clean, GB_5 = strong blur, VC_5 = heavy compression) it stores, per face (up to 2)
and per clip, DFD-FCG's head logits d and the self-calibration's g for the normal and the mirrored view, and (optional)
the blur-test response r. Clips: the version-2 layout (8 clips from the first 10 s) and the version-1 / research layout
(2 clips from the first 144 frames), so every setting can then be scored offline by study_report.py and
upgrade_summary.py with the same formulas as Analyzer.finish.

    python service/study/study.py <manifest.csv> <splits> <out.jsonl> [vid1,vid2,...]

Resumable (videos already in out.jsonl are skipped); face finding of the next video runs in a background thread while
the GPU works. Environment: STUDY_FACE_DEVICE (cuda|cpu, default cuda; the video scores are within 0.002 of CPU face
finding), STUDY_PROBE (1 = also the blur test, needed for strict mode), MODEL_BATCH, DET_EVERY, LM_EVERY.
"""
import csv
import json
import os
import sys
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

os.environ.setdefault("DET_EVERY", "12")
os.environ.setdefault("LM_EVERY", "3")
os.environ.setdefault("MODEL_BATCH", "2")
FACE_DEVICE = os.environ.get("STUDY_FACE_DEVICE", "cuda")
PROBE = os.environ.get("STUDY_PROBE", "1") == "1"
ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "service" / "app"), str(ROOT)]
import numpy as np  # noqa: E402

from pipeline import Analyzer, NoFace, clip_layout, window_frames  # noqa: E402
from qa.config import DATA  # noqa: E402
from qa.manifest import local_video  # noqa: E402

CONDS = ["clean", "GB_5", "VC_5"]


def local(row):
    if row.get("repo"):
        return str(DATA / "hf_dfdc" / row["hf_path"])
    return str(local_video(row))


def rnd(a):
    return np.round(np.asarray(a, float), 5).tolist()


def main():
    man, splits, out = sys.argv[1], sys.argv[2].split(","), sys.argv[3]
    rows = [r for r in csv.DictReader(open(man)) if r["split"] in splits]
    if len(sys.argv) > 4:
        rows = [r for r in rows if r["vid"] in sys.argv[4].split(",")]
    done = {json.loads(line)["vid"] for line in open(out, encoding="utf-8")} if os.path.exists(out) else set()
    rows = [r for r in rows if r["vid"] not in done]
    print(f"{len(rows)} videos to do, {len(done)} done", flush=True)
    an = Analyzer(device="cuda", face_device=FACE_DEVICE)

    def prep(row):
        t0 = time.time()
        path = local(row)
        small, fps, s, hw = an.read(path)
        n = len(small)
        new = clip_layout(fps, n, 8, window_frames(fps))
        old = clip_layout(fps, n, 2, 144)
        allc = np.unique(np.vstack([new, old]), axis=0)
        needed = np.unique(allc)
        geo = an.geometry(small, s, hw, needed, max_faces=2)
        del small
        crops = an.crops(path, fps, n, needed, geo, CONDS)
        pos = {f: i for i, f in enumerate(needed)}
        clip_idx = np.array([[pos[f] for f in c] for c in allc])
        starts = [int(c[0]) for c in allc]
        info = dict(fps=fps, n=n, w=int(hw[1]), h=int(hw[0]), starts=starts,
                    new=[starts.index(int(c[0])) for c in new], old=[starts.index(int(c[0])) for c in old],
                    faces=len(geo), face_s=round(time.time() - t0, 1))
        return crops, clip_idx, info

    def job(row):
        try:
            return prep(row), None
        except NoFace as e:
            return None, f"NoFace: {e}"
        except Exception as e:  # noqa: BLE001
            traceback.print_exc()
            return None, f"{type(e).__name__}: {e}"

    ex = ThreadPoolExecutor(1)
    fut = ex.submit(job, rows[0]) if rows else None
    t_all = time.time()
    for k, row in enumerate(rows):
        t0 = time.time()
        res, err = fut.result()
        fut = ex.submit(job, rows[k + 1]) if k + 1 < len(rows) else None
        rec = dict(vid=row["vid"], split=row["split"], method=row["method"], label=int(row["label"]))
        if err:
            rec["error"] = err
        else:
            crops, clip_idx, info = res
            rec.update(info)
            try:
                for c in CONDS:
                    outs, _ = an.infer(crops[c], clip_idx, flip=True, probe=PROBE)
                    rec[c] = []
                    for fo in outs:
                        d0, g0 = an.readout(fo["base"])
                        d1, g1 = an.readout(fo["flip"])
                        r0 = np.stack([p["d"] for p in fo["base_probe"]]) - d0 if PROBE else None
                        rec[c].append(dict(d=rnd(d0), g=rnd(g0), df=rnd(d1), gf=rnd(g1), r=rnd(r0) if PROBE else None))
            except Exception as e:  # noqa: BLE001
                traceback.print_exc()
                rec = {k2: rec[k2] for k2 in ("vid", "split", "method", "label")} | {"error": f"{type(e).__name__}: {e}"}
        rec["gpu_s"] = round(time.time() - t0, 1)
        with open(out, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(rec) + "\n")
        el = time.time() - t_all
        print(f"[{k + 1}/{len(rows)}] {row['vid']:<24} faces={rec.get('faces', '-')} face_s={rec.get('face_s', '-')} "
              f"gpu_s={rec['gpu_s']} {rec.get('error', '')}  elapsed {el / 60:.1f} min, "
              f"eta {(len(rows) - k - 1) * el / (k + 1) / 60:.0f} min", flush=True)
    print("finished", flush=True)


if __name__ == "__main__":
    main()
