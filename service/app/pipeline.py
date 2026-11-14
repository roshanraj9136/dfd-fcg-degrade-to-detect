"""
One uploaded video -> verdict + a short, honest explanation.

Steps (each reported to the user):
  1. find and align the face(s)     (S3FD + 2D-FAN landmarks, the DFD-FCG / RealForensics alignment)
  2. run DFD-FCG                    (released checkpoint, exact per-patch attribution)
  3. quality check                  (self-calibration: the model's own features estimate how much
                                     FAKE evidence the video's blur / compression added; removed)
  4. blur test                      (one extra pass on the face blurred by 2 px)

Accuracy settings (environment variables, defaults = the live website, chosen on 117 separate FF++ / DFDC videos that
are in no reported result, see results/accuracy_study.md):
  CLIPS=8, WINDOW_SEC=10   8 evenly spaced 3-s clips from the first 10 s of the video (the research run used
                           2 clips from the first 144 frames; CLIPS=2 WINDOW_SEC=0 gives exactly that layout)
  MAX_FACES=2              every large face that is in the video most of the time is checked; the most
                           suspicious one decides the verdict (a swap may be on either person)
  FLIP_TTA=0               1 = also check each clip mirrored and average (tested: it did not help, worse on blur)
  CERT_MARGIN=0            the answer is always a firm REAL or FAKE (with its confidence); a value > 0 brings back the
                           version-2 "not sure" label for scores within that distance of the 0.5 line
  DFD_FUSION_JSON          second detector (GenD) + fusion weights (fusion.json next to this file; absent = DFD-FCG only)

Speed-up for CPU hosting: the face detector runs on every 12th frame and the landmarks on every
3rd frame (plus every frame the model uses); the landmark smoothing then averages the available
frames in the same +-6 frame window as the research pipeline.
"""
import base64
import copy
import json
import math
import os
import tempfile
import time
from pathlib import Path

import cv2
import numpy as np
import torch

from reliability import (InvalidScores, finite_array, fusion_logit, logit, mean_probability,
                         probability, readout_shift, sigmoid, track_candidate, validate_readout)

from qa import probes
from qa.conditions import FRAME_FN, FULL_FRAME, _seed, _x264, _xvid_then_x264, shift_M
from qa.config import (CROP_SIZE, DATA, MAX_FRAMES, NUM_FRAMES, PERTURB_PARAMS, SMOOTH_MARGIN, STRIDE_SEC,
                       TARGET_SIZE)
from qa.engine import FACE_PARTS, Recorder, analyse, load_model
from qa.prep import MEAN_FACE, STABLE_POINTS, crop, roi_for

_MJ = os.environ.get("DFD_MODEL_JSON", "model.json")           # relative names are looked up next to this file
MODEL_JSON = Path(_MJ) if os.path.isabs(_MJ) else Path(__file__).resolve().parent / _MJ
DET_EVERY = int(os.environ.get("DET_EVERY", "12"))
LM_EVERY = int(os.environ.get("LM_EVERY", "3"))
CLIPS = int(os.environ.get("CLIPS", "8"))
WINDOW_SEC = float(os.environ.get("WINDOW_SEC", "10"))
MAX_FACES = int(os.environ.get("MAX_FACES", "2"))
FLIP_TTA = os.environ.get("FLIP_TTA", "0") == "1"
CERT_MARGIN = float(os.environ.get("CERT_MARGIN", "0"))      # 0 = always a firm REAL / FAKE answer (see certainty())
CERT_AGREE = float(os.environ.get("CERT_AGREE", "0"))
MAX_WINDOW_FRAMES = 360          # memory cap (high-fps videos): 6 s at 60 fps
DET_RES = 800
SECOND_FACE_PRESENCE = 0.8       # an extra face must be found in 80% of the detector frames ...
SECOND_FACE_SIZE = 0.5           # ... and be at least half as wide as the main face
REGIONS = ["eyes", "nose", "lips", "skin"]
LM_GROUPS = {"eyes": list(range(36, 48)) + list(range(17, 27)), "nose": list(range(27, 36)),
             "lips": list(range(48, 68))}


def sig(x):
    return float(sigmoid(x))


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


def window_frames(fps_int, window_sec=None):
    """How many frames are read: WINDOW_SEC seconds (never fewer than the research run's 144 frames)."""
    window_sec = WINDOW_SEC if window_sec is None else window_sec
    return int(min(MAX_WINDOW_FRAMES, max(MAX_FRAMES, round(window_sec * fps_int))))


def clip_layout(fps_int, n, clips=None, max_frames=None):
    """Frame indices of the clips (K, 10). Same offsets as the upstream demo; with clips=2 and max_frames=144 this
    is exactly qa.prep.clip_layout (the research run)."""
    clips = CLIPS if clips is None else clips
    offsets = np.array([int(math.floor(i * STRIDE_SEC * fps_int)) for i in range(NUM_FRAMES)])
    usable = min(n, max_frames or n) - SMOOTH_MARGIN
    n_avail = usable - offsets[-1]
    if n_avail <= 0:
        raise NoFace("the video is too short: at least 4 seconds are needed")
    starts = np.unique(np.linspace(0, n_avail - 1, clips).round().astype(int))
    return np.stack([s + offsets for s in starts])


def _centre(b):
    return np.array([(b[0] + b[2]) / 2, (b[1] + b[3]) / 2])


def _width(b):
    return max(b[2] - b[0], b[3] - b[1])


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
        self.mu, self.sd, self.W, self.b = validate_readout(sc, outputs=3, features=1280)
        self.lam = float(finite_array(m["strict_lam"], "strict weight", ndim=0))
        # more detectors + fusion (round 2): extra DFD-FCG checkpoints and GenD on the same aligned face crops, combined
        # by a logistic regression fitted on validation videos only (fusion.json; absent = DFD-FCG alone, as before)
        from gend_live import FUSION_JSON, load_fusion, make_member
        fj = Path(FUSION_JSON) if os.path.isabs(FUSION_JSON) else Path(__file__).resolve().parent / FUSION_JSON
        self.fusion = load_fusion(fj)
        shared = {}
        self.extra = {n: make_member(self.fusion["detectors"][n], device, shared, here=fj.parent)
                      for n in (self.fusion or {}).get("members", []) if n != "dfd"}

    def cpu_twin(self):
        """Same analyser with its own CPU copy of DFD-FCG (shares the face models and the read-out).
        Used on ZeroGPU when a visitor has no free GPU time left; gives the same scores, a few seconds slower."""
        twin = copy.copy(self)
        twin.device, twin.fp16 = "cpu", False
        twin.model = load_model(device="cpu")
        twin.rec = Recorder(twin.model)
        twin.extra, done_models = {}, {}
        for n, m in self.extra.items():                  # members sharing a network keep sharing it on the CPU
            key = id(m.model)
            if key not in done_models:
                done_models[key] = m.twin("cpu")
                twin.extra[n] = done_models[key]
            else:
                t = copy.copy(done_models[key])
                t.cfg, t.sc = m.cfg, getattr(m, "sc", None)
                if hasattr(m, "frames"):
                    t.frames = m.frames
                twin.extra[n] = t
        return twin

    # ------------------------------------------------------------------ face
    @staticmethod
    def read(path, window_sec=None):
        """First WINDOW_SEC seconds, downscaled to DET_RES for the face models (full frames are re-read later)."""
        cap = cv2.VideoCapture(str(path))
        fps = cap.get(cv2.CAP_PROP_FPS)
        fps_int = int(round(fps)) if math.isfinite(fps) and fps > 1 else 30
        limit = window_frames(fps_int, window_sec)
        small, s, hw = [], 1.0, None
        while len(small) < limit:
            ok, fr = cap.read()
            if not ok:
                break
            if hw is None:
                hw = fr.shape[:2]
                s = DET_RES / max(hw) if max(hw) > DET_RES else 1.0
            small.append(cv2.cvtColor(cv2.resize(fr, None, fx=s, fy=s) if s != 1.0 else fr, cv2.COLOR_BGR2RGB))
        cap.release()
        return small, fps_int, s, hw

    def find_faces(self, small, needed, progress=lambda step, frac: None, max_faces=None):
        """Detector on a sparse grid. Face 1 is the one the research-matched website always used (largest face at
        the first detection, then the nearest face in each frame). Further faces are followed the same way among the
        remaining detections, without ever jumping across the frame, and are kept only if they are large and
        present most of the time. Returns [{frame: box}] (box in `small` coordinates), main face first."""
        max_faces = MAX_FACES if max_faces is None else max_faces
        n = len(small)
        det_frames = sorted(set(range(0, n, DET_EVERY)) | {int(needed[0])})
        free = {}
        for k, i in enumerate(det_frames):
            d = self.fa.face_detector.detect_from_image(small[i].copy())
            free[i] = [np.asarray(b[:4], float) for b in d if b[4] > 0.5]
            progress("face", 0.3 * (k + 1) / len(det_frames))
        faces = []
        for f in range(max_faces):
            boxes, prev = {}, None
            for i in det_frames:
                cand = free[i]
                if not cand:
                    continue
                j = track_candidate(cand, prev)
                if j is None:
                    continue
                boxes[i] = cand.pop(j)
                prev = boxes[i]
            if f == 0:
                if len(boxes) < 0.6 * len(det_frames):
                    raise NoFace("no clear face could be followed through most of the video")
                keys = np.array(sorted(boxes))
                if any(np.abs(keys - int(x)).min() > DET_EVERY for x in needed):
                    raise NoFace("the main face was lost in part of the clips")
            else:
                if len(boxes) < SECOND_FACE_PRESENCE * len(det_frames):
                    continue
                size = np.median([_width(b) for b in boxes.values()])
                if size < SECOND_FACE_SIZE * np.median([_width(b) for b in faces[0].values()]):
                    continue
                keys = np.array(sorted(boxes))
                if any(np.abs(keys - int(x)).min() > DET_EVERY for x in needed):
                    continue          # this person is missing in part of the clips the model reads
            faces.append(boxes)
        return faces

    def landmarks(self, small, s, boxes, needed, progress=lambda step, frac: None):
        """2D-FAN on every LM_EVERY-th frame near a model frame (the +-6 smoothing window) and on every model frame;
        each frame uses the box of the nearest detector frame. Returns {frame: (68,2)} in full-resolution pixels."""
        n = len(small)
        need = np.asarray(needed, int)
        lm_frames = sorted({i for i in range(0, n, LM_EVERY) if np.abs(need - i).min() <= SMOOTH_MARGIN}
                           | {int(x) for x in needed})
        keys = np.array(sorted(boxes))
        near = [boxes[int(keys[np.argmin(np.abs(keys - i))])] for i in lm_frames]
        pts = self._fan_batch([small[i] for i in lm_frames], near, on_batch=lambda f: progress("landmarks", f))
        return {i: p / s for i, p in zip(lm_frames, pts)}

    @torch.no_grad()
    def _fan_batch(self, imgs, boxes, batch=16, on_batch=lambda frac: None):
        """2D-FAN landmarks for many (image, box) pairs in batches. Same crop, network and heat-map decoding as
        face_alignment.FaceAlignment.get_landmarks_from_image (flip_input=False), one network call per batch."""
        from face_alignment.utils import CENTER_Y_OFFSET, crop as fa_crop, get_preds_fromhm
        ref = self.fa.face_detector.reference_scale
        centers, scales, crops = [], [], []
        for img, d in zip(imgs, boxes):
            c = np.array([d[2] - (d[2] - d[0]) / 2.0, d[3] - (d[3] - d[1]) / 2.0])
            c[1] = c[1] - (d[3] - d[1]) * CENTER_Y_OFFSET
            sc = (d[2] - d[0] + d[3] - d[1]) / ref
            centers.append(c)
            scales.append(sc)
            crops.append(fa_crop(img, c, sc).transpose((2, 0, 1)).astype(np.float32) / 255.0)
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
            if M is None or not np.isfinite(M).all():
                raise NoFace("the face could not be aligned reliably")
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

    def geometry(self, small, s, hw, needed, progress=lambda step, frac: None, max_faces=None):
        """Faces -> per-face alignment for every model frame: [dict(M, off, crop_lm, roi)]."""
        faces = self.find_faces(small, needed, progress, max_faces)
        h, w = hw
        out = []
        for k, boxes in enumerate(faces):
            lms = self.landmarks(small, s, boxes, needed,
                                 lambda st, f: progress("landmarks", 0.3 + 0.7 * (k + f) / len(faces)))
            Ms, offs, clms = self._transforms(lms, needed, len(small))
            out.append(dict(M=Ms, off=offs, crop_lm=clms, roi=roi_for(Ms, offs, h, w)))
        return out

    @staticmethod
    def crops(path, fps_int, n_read, needed, geo, conds, vid="upload"):
        """Aligned 150x150 RGB face crops of every face, for each condition: {cond: [crops (n_needed,150,150,3)]}.
        Same pixels as qa.conditions.build (authors' robustness protocol: the clean geometry is reused, the
        degradation is applied to the video first), but the video is read once and only the face regions are kept."""
        rois = [g["roi"] for g in geo]
        degraded = [c for c in conds if c != "clean"]
        need_set = {int(i) for i in needed}
        clean_src = [dict() for _ in geo]
        deg_src = {c: [[] for _ in geo] for c in degraded}
        if degraded:
            _seed(vid, degraded[0])       # exact for one condition (the website); GB and VC use no random numbers
        cap = cv2.VideoCapture(str(path))
        i = 0
        while i < n_read:
            ok, fr = cap.read()
            if not ok:
                break
            for k, (x0, y0, x1, y1) in enumerate(rois):
                if i in need_set:
                    clean_src[k][i] = np.ascontiguousarray(fr[y0:y1, x0:x1])
            for c in degraded:
                kind, lv = c.split("_")
                if kind in FULL_FRAME:                         # geometry-dependent: whole frame first, then cut
                    g = FRAME_FN[kind](fr.copy(), PERTURB_PARAMS[kind][int(lv) - 1])
                    for k, (x0, y0, x1, y1) in enumerate(rois):
                        deg_src[c][k].append(np.ascontiguousarray(g[y0:y1, x0:x1]))
                else:
                    for k, (x0, y0, x1, y1) in enumerate(rois):
                        deg_src[c][k].append(np.ascontiguousarray(fr[y0:y1, x0:x1]))
            i += 1
        cap.release()
        out = {}
        os.makedirs(DATA / "tmp", exist_ok=True)
        for c in conds:
            per_face = []
            for k, g in enumerate(geo):
                x0, y0 = g["roi"][:2]
                if c == "clean":
                    src = [clean_src[k][int(f)] for f in needed]
                else:
                    kind, lv = c.split("_")
                    _seed(vid, c)
                    frames = deg_src[c][k]
                    with tempfile.TemporaryDirectory(dir=DATA / "tmp") as tmpdir:
                        if kind == "VC":
                            rf = _x264(frames, fps_int, PERTURB_PARAMS["VC"][int(lv) - 1], os.path.join(tmpdir, "vc.mp4"))
                        else:
                            if kind not in FULL_FRAME:
                                frames = [np.ascontiguousarray(FRAME_FN[kind](f.copy(), PERTURB_PARAMS[kind][int(lv) - 1]))
                                          for f in frames]
                            rf = _xvid_then_x264(frames, fps_int, tmpdir)
                    src = [rf[int(f)] for f in needed]
                per_face.append(np.stack([cv2.cvtColor(crop(sf, shift_M(M, x0, y0), off), cv2.COLOR_BGR2RGB)
                                          for sf, M, off in zip(src, g["M"], g["off"])]))
            out[c] = per_face
        return out

    # ------------------------------------------------------------------ main
    def run(self, path, degrade=None, progress=lambda step, frac: None):
        """Whole check in one process (CPU service, local GPU)."""
        st = self.prepare(path, degrade, progress)
        return self.finish(st, *self.infer(st["crops"], st["clip_idx"], progress=progress))

    def prepare(self, path, degrade=None, progress=lambda step, frac: None):
        """CPU part: read the video, find and align the face(s), build the (optionally degraded) face crops."""
        t0 = time.time()
        marks, report = {}, progress

        def progress(step, frac):                 # noqa: F811  (records when each step starts)
            marks.setdefault(step, round(time.time() - t0, 2))
            report(step, frac)

        small, fps_int, s, hw = self.read(path)
        marks["read"] = round(time.time() - t0, 2)
        if not small:
            raise NoFace("the file could not be read as a video")
        n = len(small)
        layout = clip_layout(fps_int, n, CLIPS, window_frames(fps_int))
        needed = np.unique(layout)
        progress("face", 0.0)
        geo = self.geometry(small, s, hw, needed, progress)
        del small
        marks["landmarks_done"] = round(time.time() - t0, 2)
        cond = {"blur": "GB_5", "compress": "VC_5"}.get(degrade or "", "clean")
        crops = self.crops(path, fps_int, int(needed.max()) + 1 if cond == "clean" else n, needed, geo, [cond])[cond]
        marks["crops_done"] = round(time.time() - t0, 2)
        pos = {f: i for i, f in enumerate(needed)}
        clip_idx = np.array([[pos[f] for f in c] for c in layout])
        h, w = hw
        return dict(t0=t0, marks=marks, crops=crops, clip_idx=clip_idx, clms=[g["crop_lm"] for g in geo],
                    fps=fps_int, needed=needed, w=w, h=h, degrade=degrade, seconds_covered=round(n / fps_int, 1))

    def infer(self, crops, clip_idx, flip=None, probe=True, progress=lambda step, frac: None):
        """GPU part. For every face: DFD-FCG with exact attribution on the clips, the same clips mirrored (FLIP_TTA),
        and the blur test (the clips blurred by 2 px). Returns ([per-face dict], model seconds)."""
        flip = FLIP_TTA if flip is None else flip
        bs = int(os.environ.get("MODEL_BATCH", "4" if self.device == "cuda" else "2"))
        kinds = [("base", False)] + ([("flip", False)] if flip else []) + ([("base", True)] if probe else [])
        total = len(crops) * len(kinds) * len(clip_idx)
        done = [0]

        def forward(x, step, full):
            out = []
            for k in range(0, len(clip_idx), bs):
                batch = [self.model.transform(torch.from_numpy(np.ascontiguousarray(x[ci])).permute(0, 3, 1, 2)
                                              .contiguous()) for ci in clip_idx[k:k + bs]]
                with torch.inference_mode():
                    res = analyse(self.model, self.rec, torch.stack(batch), fp16=self.fp16)
                keep = ("d", "y_s", "y_t", "cs_patch", "cs_lq") if full else ("d", "y_s", "y_t")
                out += [{key: r[key] for key in keep} for r in res]
                done[0] += len(batch)
                progress(step, done[0] / total)
            return out

        t = time.time()
        progress("model", 0.0)
        result = []
        for fc in crops:
            fo = {}
            for view, is_probe in kinds:
                x = fc[:, :, ::-1] if view == "flip" else fc
                if is_probe:
                    x = probes.apply("pB2", np.ascontiguousarray(x))
                fo[view + ("_probe" if is_probe else "")] = forward(x, "blur" if is_probe else "model",
                                                                    full=view == "base" and not is_probe)
            if self.extra:                                  # the other detectors of the fusion, same face crops
                progress("model", done[0] / total)
                cache = {}                                  # one forward pass per network
                fo["extra"] = {}
                for n, m in self.extra.items():
                    key = (id(m.model), getattr(m, "frames", None))
                    if key not in cache:
                        cache[key] = m.frame_scores(fc, clip_idx)
                    fo["extra"][n] = cache[key]
            result.append(fo)
        return result, round(time.time() - t, 2)

    def readout(self, rows):
        """Per-clip head logits d (K,3) and the self-calibration's spurious evidence g (K,3)."""
        d = finite_array(np.stack([r["d"] for r in rows]), "DFD head logits", ndim=2)
        y = np.stack([np.r_[r["y_s"], r["y_t"]] for r in rows])
        if d.shape[1] != 3:
            raise InvalidScores("DFD must return three head logits per clip")
        return d, readout_shift(y, self.mu, self.sd, self.W, self.b)

    @staticmethod
    def prob(D):
        """DFD-FCG's video score: mean over clips (and views) of the mean of the three head sigmoids."""
        return mean_probability(D)

    def face_scores(self, fo):
        """Scores of one face: original DFD-FCG, self-calibrated (the verdict) and strict, averaged over the views.
        The blur test response r of a clip (measured on the normal view) is used for its mirrored view too."""
        d0, g0 = self.readout(fo["base"])
        r0 = np.stack([p["d"] for p in fo["base_probe"]]) - d0 if "base_probe" in fo else np.zeros_like(d0)
        D, G, R = [d0], [g0], [r0]
        if "flip" in fo:
            d1, g1 = self.readout(fo["flip"])
            D.append(d1)
            G.append(g1)
            R.append(r0)
        d, g, r = np.concatenate(D), np.concatenate(G), np.concatenate(R)
        per = np.array([np.mean([sig(v) for v in row]) for row in d - g]).reshape(len(D), -1)
        out = dict(p_original=self.prob(d), p=self.prob(d - g), p_strict=self.prob(d - g - self.lam * r),
                   d=d0, g=g0, r=r0, clip_p=per.mean(0), view_p=per)
        out["p_dfd"] = out["p"]
        if self.fusion:
            x = {"dfd": logit(out["p_dfd"])}
            for n, raw in fo.get("extra", {}).items():
                x[n] = self.extra[n].video_logit(*raw)
            z = fusion_logit(self.fusion, x)
            out["members"] = {k: sig(v) for k, v in x.items()}
            out["p"] = sig(z)
            # timeline: each clip's own DFD-FCG evidence, combined with the other detectors' video scores by the same
            # combiner, so the dots sit on the same side of the line as the final answer unless one clip disagrees
            coef = dict(zip(self.fusion["members"], self.fusion["coef"]))
            rest = z - coef.get("dfd", 0.0) * x["dfd"]
            cp = np.clip(np.asarray(out["clip_p"], float), 1e-6, 1 - 1e-6)
            out["clip_p"] = sigmoid(coef.get("dfd", 0.0) * np.log(cp / (1 - cp)) + rest)
        return out

    def finish(self, st, outs, model_seconds=None):
        """CPU part: self-calibration, strict mode, the most suspicious face, explanation."""
        marks, t0, clip_idx = st["marks"], st["t0"], st["clip_idx"]
        marks["model_seconds"] = model_seconds
        if not outs or len(outs) != len(st["crops"]):
            raise InvalidScores("missing inference output for one or more faces")
        scores = [self.face_scores(fo) for fo in outs]
        k = int(np.argmax([sc["p"] for sc in scores]))
        sc, base = scores[k], outs[k]["base"]

        # explanation: where the spatial head saw FAKE evidence (exact per-patch split), on the deciding face
        sp = np.mean([x["cs_patch"] for x in base], 0)
        lm = st["clms"][k].mean(0)
        reg = region_labels(lm)
        region_ev = {name: float(sp[reg == name].sum()) for name in REGIONS}
        face = st["crops"][k][clip_idx[0][0]]
        p_orig, p_cal, p_strict = (probability(sc[key]) for key in ("p_original", "p", "p_strict"))
        fps, needed = st["fps"], st["needed"]
        timeline = [dict(start=round(float(needed[ci[0]]) / fps, 2), end=round(float(needed[ci[-1]]) / fps, 2),
                         p=float(sc["clip_p"][j]), faces=[float(s["clip_p"][j]) for s in scores])
                    for j, ci in enumerate(clip_idx)]
        sure = certainty(p_cal, sc["view_p"])
        return dict(
            verdict="FAKE" if p_cal >= 0.5 else "REAL",
            p=p_cal, p_original=p_orig, p_strict=p_strict, **sure,
            p_dfd=sc["p_dfd"], members=sc.get("members"),
            confidence=max(p_cal, 1 - p_cal),
            detectors=(list(dict.fromkeys(self.fusion["names"].get(n, n).split(",")[0] for n in self.fusion["members"]))
                       if self.fusion else ["DFD-FCG"]),
            verdict_original="FAKE" if p_orig >= 0.5 else "REAL",
            quality_shift=p_orig - sc["p_dfd"], spurious_logit=sc["g"].mean(0).tolist(),
            blur_response=sc["r"].mean(0).tolist(), heads=sc["d"].mean(0).tolist(),
            regions=region_ev, parts=dict(zip(FACE_PARTS, np.mean([x["cs_lq"].sum(0) for x in base], 0).tolist())),
            faces=[dict(p=s["p"], p_original=s["p_original"], p_strict=s["p_strict"], p_dfd=s["p_dfd"],
                        members=s.get("members")) for s in scores],
            timeline=timeline,
            face_png=png64(cv2.resize(face, (300, 300), interpolation=cv2.INTER_CUBIC)),
            heat_png=png64(heat_overlay(face, sp)),
            meta=dict(fps=st["fps"], frames_used=int(len(st["needed"])), clips=int(len(clip_idx)),
                      faces_checked=len(outs), face_shown=k + 1, mirrored="flip" in outs[k],
                      seconds_covered=st.get("seconds_covered"), width=int(st["w"]), height=int(st["h"]),
                      degrade=st["degrade"] or "none", seconds=round(time.time() - t0, 1), timing=marks),
        )


def certainty(p, view_p, margin=None, agree=None):
    """Clip agreement and distance from the 0.5 line. The verdict is always REAL or FAKE (no "not sure" answer);
    `sure` is kept for old clients and is always True. CERT_MARGIN > 0 restores the old "not sure" rule."""
    margin = CERT_MARGIN if margin is None else margin
    agree = CERT_AGREE if agree is None else agree
    p = probability(p)
    v = finite_array(view_p, "clip probabilities").ravel()
    if ((v < 0) | (v > 1)).any():
        raise InvalidScores("clip probabilities must lie in [0, 1]")
    same = float(np.mean((v >= 0.5) == (p >= 0.5)))
    return dict(sure=bool(abs(p - 0.5) >= margin and same >= agree), agreement=same, margin=abs(p - 0.5))


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
