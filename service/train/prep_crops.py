"""Find, track and align the faces of every fine-tuning video once, with the live checker's own face code
(service/app/pipeline.py), and save the aligned 150x150 face clips the model reads.

  train / val videos: 4 clips of 10 frames from the first 10 s, main face only
  test / test_open:   the live checker's layout (8 clips from the first 10 s), up to 2 faces

Several workers can run at once (one on the GPU, one on the CPU): each video is claimed with a lock file. Videos
that are not downloaded yet are retried until the download logs say "download finished".
    python service/train/prep_crops.py <cuda|cpu> <out dir> <sets.csv> [<sets.csv> ...]
Output per video: <out>/<vid with / -> __>.npz  (face0, [face1], clip_idx, label, fps, w, h)  or .fail (reason)
"""
import csv
import os
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "service" / "app"), str(ROOT)]
os.environ.setdefault("DET_EVERY", "12")
os.environ.setdefault("LM_EVERY", "3")
import numpy as np  # noqa: E402
import torch  # noqa: E402

from pipeline import Analyzer, NoFace, _warm_numba, clip_layout, usable_cpus, window_frames  # noqa: E402

DEV, OUT, SETS = sys.argv[1], Path(sys.argv[2]), sys.argv[3:]
VIDEOS = Path(r"D:\dfd-data\new\videos")
LOGS = [Path(r"D:\dfd-data\make_sets.log"), Path(r"D:\dfd-data\make_sets_gated.log")]
OUT.mkdir(parents=True, exist_ok=True)


def face_only(device):
    """An Analyzer with only the face models (no DFD-FCG): find_faces / landmarks / geometry / crops."""
    import face_alignment
    an = Analyzer.__new__(Analyzer)
    n = usable_cpus() if device == "cuda" else max(1, (os.cpu_count() or 4) // 2)
    torch.set_num_threads(n)
    an.threads = n
    an.fa = face_alignment.FaceAlignment(face_alignment.LandmarksType.TWO_D, face_detector="sfd",
                                         flip_input=False, device="cpu", compile=False)
    if device != "cpu":
        an.fa.face_detector.face_detector.to(device)
        an.fa.face_detector.device = device
        an.fa.face_alignment_net.to(device)
        an.fa.device = device
    _warm_numba()
    return an


def safe(vid):
    return vid.replace("/", "__")


def one(an, row):
    path = VIDEOS / row["source"] / row["path"]
    small, fps, s, hw = an.read(path)
    if not small:
        raise NoFace("unreadable video")
    n = len(small)
    test = row["set"].startswith("test")
    layout = clip_layout(fps, n, 8 if test else 4, window_frames(fps))
    needed = np.unique(layout)
    geo = an.geometry(small, s, hw, needed, max_faces=2 if test else 1)
    del small
    crops = an.crops(path, fps, int(needed.max()) + 1, needed, geo, ["clean"])["clean"]
    pos = {f: i for i, f in enumerate(needed)}
    out = dict(clip_idx=np.array([[pos[f] for f in c] for c in layout]), label=int(row["label"]), fps=fps,
               w=int(hw[1]), h=int(hw[0]))
    for k, c in enumerate(crops):
        out[f"face{k}"] = c
    return out


def main():
    an = face_only(DEV)
    rows = []
    for f in SETS:
        rows += list(csv.DictReader(open(f)))
    order = {"train": 0, "val": 1, "test": 2, "test_open": 3}
    rows.sort(key=lambda r: order[r["set"]])
    print(f"{DEV}: {len(rows)} videos", flush=True)
    t0, done = time.time(), 0
    while True:
        waiting = 0
        for r in rows:
            base = OUT / safe(r["vid"])
            if base.with_suffix(".npz").exists() or base.with_suffix(".fail").exists():
                continue
            if not (VIDEOS / r["source"] / r["path"]).exists():
                waiting += 1
                continue
            lock = base.with_suffix(".lock")
            try:
                fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                os.close(fd)
            except FileExistsError:
                continue
            t = time.time()
            try:
                res = one(an, r)
                np.savez(str(base) + ".tmp.npz", **res)
                os.replace(str(base) + ".tmp.npz", base.with_suffix(".npz"))
                msg = f"faces={sum(k.startswith('face') for k in res)} clips={len(res['clip_idx'])}"
            except NoFace as e:
                base.with_suffix(".fail").write_text(f"NoFace: {e}")
                msg = f"NoFace: {e}"
            except Exception as e:  # noqa: BLE001
                if "out of memory" in str(e).lower():           # GPU shared with training: wait and retry later
                    torch.cuda.empty_cache()
                    waiting += 1
                    msg = "GPU out of memory, will retry"
                    time.sleep(20)
                else:
                    traceback.print_exc()
                    base.with_suffix(".fail").write_text(f"{type(e).__name__}: {e}")
                    msg = f"ERROR {type(e).__name__}: {e}"
            finally:
                lock.unlink(missing_ok=True)
            done += 1
            print(f"[{DEV} {done}] {r['set']:<9} {r['vid']:<48} {time.time() - t:5.1f}s {msg}  "
                  f"({(time.time() - t0) / 60:.0f} min)", flush=True)
        finished = all(p.exists() and "download finished" in p.read_text(errors="ignore") for p in LOGS)
        if waiting == 0:
            break
        if finished and not any((VIDEOS / r["source"] / r["path"]).exists() for r in rows
                                if not (OUT / safe(r["vid"])).with_suffix(".npz").exists()
                                and not (OUT / safe(r["vid"])).with_suffix(".fail").exists()):
            print(f"{DEV}: {waiting} videos never downloaded; stopping", flush=True)
            break
        time.sleep(10)
    print(f"{DEV}: finished", flush=True)


if __name__ == "__main__":
    main()
