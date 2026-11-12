"""
One uploaded video -> verdict + a short, honest explanation.

Steps (each reported to the user):
  1. find and align the face        (2D-FAN landmarks, the DFD-FCG / RealForensics alignment)
  2. run DFD-FCG                    (released checkpoint, exact per-patch attribution)
  3. quality check                  (self-calibration: the model's own features estimate how much
                                     FAKE evidence the video's blur / compression added; removed)
  4. blur test                      (one extra pass on the face blurred by 2 px)

Speed-up for CPU hosting: the face detector runs on every 12th frame and the landmarks on every
3rd frame (plus every frame the model uses); the landmark smoothing then averages the available
frames in the same +-6 frame window as the research pipeline.
"""
import base64
import copy
import json
import math
import os
import time
from pathlib import Path

import cv2
import numpy as np
import torch

from qa import probes
from qa.conditions import build
from qa.config import CROP_SIZE, MAX_FRAMES, TARGET_SIZE
from qa.engine import FACE_PARTS, Recorder, analyse, load_model
from qa.prep import MEAN_FACE, STABLE_POINTS, clip_layout, read_frames, roi_for

MODEL_JSON = Path(os.environ.get("DFD_MODEL_JSON", Path(__file__).resolve().parent / "model.json"))
DET_EVERY = int(os.environ.get("DET_EVERY", "12"))
LM_EVERY = int(os.environ.get("LM_EVERY", "3"))
DET_RES = 800
REGIONS = ["eyes", "nose", "lips", "skin"]
LM_GROUPS = {"eyes": list(range(36, 48)) + list(range(17, 27)), "nose": list(range(27, 36)),
             "lips": list(range(48, 68))}


def sig(x):
    return 1.0 / (1.0 + math.exp(-x))


class NoFace(Exception):
    pass


def usable_cpus():
    """CPU cores this process may really use: OMP_NUM_THREADS, else the affinity mask and the cgroup quota."""
    if os.environ.get("OMP_NUM_THREADS"):
        return max(1, int(os.environ["OMP_NUM_THREADS"]))
    n = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else (os.cpu_count() or 1)
    try:
        quota, period = open("/sys/fs/cgroup/cpu.max").read().split()[:2]
        if quota != "max":
            n = min(n, max(1, int(int(quota) / int(period))))
    except (OSError, ValueError):
        pass
    return max(1, n)


def _warm_numba():
    """Compile face-alignment's numba helpers once, in the main process (about 8 s). ZeroGPU runs every
    check in a forked worker, which then inherits the compiled code instead of compiling it again."""
    from face_alignment.utils import get_preds_fromhm
    hm = np.random.rand(1, 68, 64, 64).astype(np.float32)
    get_preds_fromhm(hm, np.array([100.0, 100.0]), np.float64(1.5))


class Analyzer:
    def __init__(self, device="cpu", face_device=None):
        import face_alignment
        face_device = face_device or device
        n = usable_cpus()                      # os.cpu_count() ignores container CPU limits
        torch.set_num_threads(n)
        cv2.setNumThreads(n)
        self.threads = n
        self.device = device
        self.fp16 = device == "cuda"          # the research features (and the fitted read-out) used fp16 on GPU
        self.fa = face_alignment.FaceAlignment(face_alignment.LandmarksType.TWO_D, face_detector="sfd",
                                               flip_input=False, device="cpu", compile=False)
        if face_device != "cpu":              # load on CPU, then move (the pattern ZeroGPU expects)
            self.fa.face_detector.face_detector.to(face_device)
            self.fa.face_detector.device = face_device
            self.fa.face_alignment_net.to(face_device)
            self.fa.device = face_device
        # cuDNN autotuning would be repeated in every ZeroGPU worker process; plain kernels are faster here
        torch.backends.cudnn.benchmark = False
        _warm_numba()
        self.model = load_model(device=device)
        self.rec = Recorder(self.model)
        m = json.load(open(MODEL_JSON))
        sc = m["selfcal"]
        self.mu, self.sd = np.array(sc["mean"]), np.array(sc["scale"])
        self.W, self.b = np.array(sc["coef"]), np.array(sc["intercept"])
        self.lam = float(m["strict_lam"])

    def cpu_twin(self):
        """Same analyser with its own CPU copy of DFD-FCG (shares the face models and the read-out).
        Used on ZeroGPU when a visitor has no free GPU time left; gives the same scores, a few seconds slower."""
        twin = copy.copy(self)
        twin.device, twin.fp16 = "cpu", False
        twin.model = load_model(device="cpu")
        twin.rec = Recorder(twin.model)
        return twin

    # ------------------------------------------------------------------ face
    def _landmarks(self, frames, needed, progress):
        h, w = frames[0].shape[:2]
        s = DET_RES / max(h, w) if max(h, w) > DET_RES else 1.0
        small = [cv2.cvtColor(cv2.resize(f, None, fx=s, fy=s), cv2.COLOR_BGR2RGB) if s != 1.0
                 else cv2.cvtColor(f, cv2.COLOR_BGR2RGB) for f in frames]
        n = len(frames)
        # 1. detector on a sparse grid; follow the largest face
        boxes, prev = {}, None
        det_frames = sorted(set(range(0, n, DET_EVERY)) | {int(needed[0])})
        for k, i in enumerate(det_frames):
            d = self.fa.face_detector.detect_from_image(small[i].copy())
            d = [b for b in d if b[4] > 0.5]
            if d:
                if prev is None:
                    b = max(d, key=lambda b: (b[2] - b[0]) * (b[3] - b[1]))
                else:
                    c = np.array([(prev[0] + prev[2]) / 2, (prev[1] + prev[3]) / 2])
                    b = min(d, key=lambda b: np.linalg.norm(np.array([(b[0] + b[2]) / 2, (b[1] + b[3]) / 2]) - c))
                boxes[i] = np.asarray(b[:4], float)
                prev = boxes[i]
            progress("face", 0.4 * (k + 1) / len(det_frames))
        if len(boxes) < 0.6 * len(det_frames):
            raise NoFace("no clear face was found in most of the video")
        keys = np.array(sorted(boxes))
        # 2. landmarks on every LM_EVERY-th frame and on every frame the model will read (batched FAN)
        lm_frames = sorted(set(range(0, n, LM_EVERY)) | {int(x) for x in needed})
        progress("landmarks", 0.4)
        near = [boxes[int(keys[np.argmin(np.abs(keys - i))])] for i in lm_frames]
        pts = self._fan_batch([small[i] for i in lm_frames], near,
                              on_batch=lambda f: progress("face", 0.4 + 0.6 * f))
        return {i: p / s for i, p in zip(lm_frames, pts)}

    @torch.no_grad()
    def _fan_batch(self, imgs, boxes, batch=16, on_batch=lambda frac: None):
        """2D-FAN landmarks for many (image, box) pairs in batches. Same crop, network and heat-map decoding as
        face_alignment.FaceAlignment.get_landmarks_from_image (flip_input=False), one network call per batch."""
        from face_alignment.utils import CENTER_Y_OFFSET, crop, get_preds_fromhm
        ref = self.fa.face_detector.reference_scale
        centers, scales, crops = [], [], []
        for img, d in zip(imgs, boxes):
            c = np.array([d[2] - (d[2] - d[0]) / 2.0, d[3] - (d[3] - d[1]) / 2.0])
            c[1] = c[1] - (d[3] - d[1]) * CENTER_Y_OFFSET
            sc = (d[2] - d[0] + d[3] - d[1]) / ref
            centers.append(c)
            scales.append(sc)
            crops.append(crop(img, c, sc).transpose((2, 0, 1)).astype(np.float32) / 255.0)
        x = torch.from_numpy(np.stack(crops))
        heat = []
        for k in range(0, len(x), batch):
            o = self.fa.face_alignment_net(x[k:k + batch].to(self.fa.device, dtype=self.fa.dtype))
            o = o[-1] if isinstance(o, list) else o
            heat.append(o.detach().to("cpu", torch.float32))
            on_batch(min(1.0, (k + batch) / len(x)))
        heat = torch.cat(heat).numpy()
        out = []
        for i in range(len(imgs)):
            _, p, _ = get_preds_fromhm(heat[i:i + 1], centers[i], scales[i])
            out.append(np.asarray(p, np.float64).reshape(68, 2))
        return out

    @staticmethod
    def _transforms(lms, needed, n):
        ref = np.vstack([MEAN_FACE[x] for x in STABLE_POINTS]) * (TARGET_SIZE / 256)
        Ms, offs, crop_lms = [], [], []
        for fi in needed:
            if fi not in lms:
                raise NoFace("the face was lost in part of the video")
            margin = min(6, fi, n - 1 - fi)
            win = [lms[i] for i in range(fi - margin, fi + margin + 1) if i in lms]
            sm = np.mean(win, axis=0)
            sm += lms[fi].mean(axis=0) - sm.mean(axis=0)
            M = cv2.estimateAffinePartial2D(np.vstack([sm[x] for x in STABLE_POINTS]), ref, method=cv2.LMEDS)[0]
            tl = sm @ M[:, :2].T + M[:, 2]
            size = CROP_SIZE // 2
            cx, cy = np.mean(tl[15:68], axis=0)
            cy = min(max(cy, size + 1), TARGET_SIZE - size - 1)
            cx = min(max(cx, size + 1), TARGET_SIZE - size - 1)
            uy, lx = int(cy - size), int(cx - size)
            Ms.append(M)
            offs.append((lx, uy))
            crop_lms.append(tl - [lx, uy])
        return np.array(Ms), np.array(offs), np.array(crop_lms)

    # ------------------------------------------------------------------ main
    def run(self, path, degrade=None, progress=lambda step, frac: None):
        """Whole check in one process (CPU service, local GPU)."""
        st = self.prepare(path, degrade, progress)
        return self.finish(st, *self.infer(st["crops"], st["clip_idx"], progress))

    def prepare(self, path, degrade=None, progress=lambda step, frac: None):
        """CPU part: read the video, find and align the face, build the (optionally degraded) face crops."""
        t0 = time.time()
        marks, report = {}, progress

        def progress(step, frac):                 # noqa: F811  (records when each step starts)
            marks.setdefault(step, round(time.time() - t0, 2))
            report(step, frac)

        frames, fps = read_frames(path, MAX_FRAMES)
        marks["read"] = round(time.time() - t0, 2)
        if not frames:
            raise NoFace("the file could not be read as a video")
        fps_int = int(round(fps)) if fps and fps > 1 else 30
        n = len(frames)
        try:
            layout = clip_layout(fps_int, n)
        except RuntimeError:
            raise NoFace("the video is too short: at least 4 seconds are needed")
        needed = np.unique(layout)
        progress("face", 0.0)
        lms = self._landmarks(frames, needed, progress)
        Ms, offs, clms = self._transforms(lms, needed, n)
        h, w = frames[0].shape[:2]
        prep = dict(fps=fps_int, n_frames=n, size=np.array([h, w]), layout=layout, needed=needed,
                    M=Ms, off=offs, crop_lm=clms, roi=roi_for(Ms, offs, h, w))
        del frames
        marks["landmarks_done"] = round(time.time() - t0, 2)
        cond = {"blur": "GB_5", "compress": "VC_5"}.get(degrade or "", "clean")
        crops = build(path, prep, "upload", [cond])[cond][0]
        marks["crops_done"] = round(time.time() - t0, 2)
        pos = {f: i for i, f in enumerate(needed)}
        clip_idx = np.array([[pos[f] for f in c] for c in layout])
        return dict(t0=t0, marks=marks, crops=crops, clip_idx=clip_idx, clms=clms, fps=fps_int, needed=needed,
                    w=w, h=h, degrade=degrade)

    def infer(self, crops, clip_idx, progress=lambda step, frac: None):
        """GPU part: DFD-FCG with exact attribution on the face crops, then on the same crops blurred by 2 px."""
        def forward(x, step):
            out = []
            for k, ci in enumerate(clip_idx):
                fr = torch.from_numpy(np.ascontiguousarray(x[ci])).permute(0, 3, 1, 2).contiguous()
                with torch.inference_mode():
                    out += analyse(self.model, self.rec, self.model.transform(fr)[None], fp16=self.fp16)
                progress(step, (k + 1) / len(clip_idx))
            return out

        t = time.time()
        progress("model", 0.0)
        base = forward(crops, "model")
        progress("blur", 0.0)
        probed = forward(probes.apply("pB2", crops), "blur")
        keep = ("d", "y_s", "y_t", "cs_patch", "cs_lq")              # only what the explanation needs
        slim = lambda rs: [{k: r[k] for k in keep} for r in rs]      # noqa: E731
        return slim(base), slim(probed), round(time.time() - t, 2)

    def finish(self, st, base, probed, model_seconds=None):
        """CPU part: self-calibration, strict mode, explanation."""
        marks, t0, clip_idx, clms = st["marks"], st["t0"], st["clip_idx"], st["clms"]
        d = np.stack([r["d"] for r in base])                                     # (K,3)
        y = np.stack([np.r_[r["y_s"], r["y_t"]] for r in base])
        g = ((y - self.mu) / self.sd) @ self.W.T + self.b                        # spurious evidence
        r = np.stack([p["d"] for p in probed]) - d
        marks["model_seconds"] = model_seconds
        P = lambda D: float(np.mean([np.mean([sig(v) for v in row]) for row in D]))  # noqa: E731
        p_orig, p_cal, p_strict = P(d), P(d - g), P(d - g - self.lam * r)

        # explanation: where the spatial head saw FAKE evidence (exact per-patch split)
        sp = np.mean([x["cs_patch"] for x in base], 0)
        lm = clms.mean(0)
        reg = region_labels(lm)
        region_ev = {k: float(sp[reg == k].sum()) for k in REGIONS}
        face = st["crops"][clip_idx[0][0]]
        return dict(
            verdict="FAKE" if p_cal >= 0.5 else "REAL",
            p=p_cal, p_original=p_orig, p_strict=p_strict,
            verdict_original="FAKE" if p_orig >= 0.5 else "REAL",
            quality_shift=p_orig - p_cal, spurious_logit=g.mean(0).tolist(),
            blur_response=r.mean(0).tolist(), heads=d.mean(0).tolist(),
            regions=region_ev, parts=dict(zip(FACE_PARTS, np.mean([x["cs_lq"].sum(0) for x in base], 0).tolist())),
            face_png=png64(cv2.resize(face, (300, 300), interpolation=cv2.INTER_CUBIC)),
            heat_png=png64(heat_overlay(face, sp)),
            meta=dict(fps=st["fps"], frames_used=int(len(st["needed"])), clips=int(len(clip_idx)), width=int(st["w"]),
                      height=int(st["h"]), degrade=st["degrade"] or "none", seconds=round(time.time() - t0, 1),
                      timing=marks),
        )


def region_labels(lm, grid=16, crop=150, near=12.0):
    """Same rule as the research analysis: nearest landmark group within 12 px, else skin inside the face."""
    step = crop / grid
    hull = cv2.convexHull(lm.astype(np.float32))
    out = []
    for r in range(grid):
        for c in range(grid):
            p = np.array([(c + .5) * step, (r + .5) * step])
            best, bd = None, 1e9
            for k, idx in LM_GROUPS.items():
                dd = np.linalg.norm(lm[idx] - p, axis=1).min()
                if dd < bd:
                    best, bd = k, dd
            if bd <= near:
                out.append(best)
            elif cv2.pointPolygonTest(hull, (float(p[0]), float(p[1])), False) >= 0:
                out.append("skin")
            else:
                out.append("background")
    return np.array(out)


def heat_overlay(face_rgb, patch, size=300):
    """Face crop with the 16x16 spatial evidence: red = towards FAKE, green = towards REAL."""
    img = cv2.resize(face_rgb, (size, size), interpolation=cv2.INTER_CUBIC).astype(np.float32)
    v = patch.reshape(16, 16)
    m = max(np.abs(v).max(), 1e-6)
    a = cv2.resize(np.abs(v) / m, (size, size), interpolation=cv2.INTER_NEAREST)[..., None]
    col = np.where(cv2.resize(v, (size, size), interpolation=cv2.INTER_NEAREST)[..., None] >= 0,
                   np.array([239, 68, 68], np.float32), np.array([34, 197, 94], np.float32))
    out = img * (1 - 0.6 * a) + col * (0.6 * a)
    return np.clip(out, 0, 255).astype(np.uint8)


def png64(rgb):
    ok, buf = cv2.imencode(".png", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    return "data:image/png;base64," + base64.b64encode(buf).decode()
