"""
Build the aligned 150x150 face crops of a video under one test condition.

Conditions follow the authors' robustness protocol
(src/preprocess/robustness/phase1_apply_all_to_videos.py):

  frame-level types (CS, CC, BW, GNC, GB, JPEG):
      perturb every frame -> write XVID .avi -> ffmpeg default re-encode (libx264, CRF 23)
  VC:  ffmpeg -crf <param> on the original video (libx264)
  RE_0 (our control): the XVID + CRF-23 round trip with no perturbation, which isolates
      the re-encoding that every frame-level condition silently contains.

Then the clean-video landmarks are reused to crop the face, as in the authors' phase 2.

Speed-ups that do not change what the model sees:
  * only the first MAX_FRAMES frames are processed;
  * encoding is done on a region of interest that contains every source pixel the face
    crops read (+40 px), with its origin on the 16-px macroblock grid of the full frame.
    BW, JPEG and GB are applied to the full frame before the ROI is cut, so their
    geometry (block density, resampling grid, blur at borders) is unchanged.
"""
import os
import random
import tempfile
import zlib

import av
import cv2
import numpy as np

from qa.config import DATA, MAX_FRAMES, PERTURB_PARAMS, UPSTREAM
from qa.prep import crop, read_frames
import sys

sys.path.insert(0, str(UPSTREAM))
from src.preprocess.robustness.distortions import (block_wise, color_contrast,  # noqa: E402
                                                   color_saturation, gaussian_blur,
                                                   gaussian_noise_color, jpeg_compression)

FRAME_FN = {"CS": color_saturation, "CC": color_contrast, "BW": block_wise,
            "GNC": gaussian_noise_color, "GB": gaussian_blur, "JPEG": jpeg_compression}
FULL_FRAME = {"BW", "JPEG", "GB"}       # geometry-dependent -> applied before the ROI cut


def _seed(vid, cond):
    s = zlib.crc32(f"{vid}|{cond}".encode())
    random.seed(s)
    np.random.seed(s % (2 ** 32))


def _x264(frames, fps, crf, path):
    out = av.open(path, "w")
    st = out.add_stream("libx264", rate=fps)
    st.width, st.height, st.pix_fmt = frames[0].shape[1], frames[0].shape[0], "yuv420p"
    st.options = {"crf": str(crf), "preset": "medium"}
    st.codec_context.thread_count = 2      # leave CPU cores for the GPU-feeding main process
    for f in frames:
        for pkt in st.encode(av.VideoFrame.from_ndarray(f, format="bgr24")):
            out.mux(pkt)
    for pkt in st.encode():
        out.mux(pkt)
    out.close()
    return _decode(path)


def _decode(path):
    inp = av.open(path)
    fr = [f.to_ndarray(format="bgr24") for f in inp.decode(video=0)]
    inp.close()
    return fr


def _xvid_then_x264(frames, fps, tmpdir):
    h, w = frames[0].shape[:2]
    avi = os.path.join(tmpdir, "tmp.avi")
    wr = cv2.VideoWriter(avi, cv2.VideoWriter_fourcc(*"XVID"), fps, (w, h))
    for f in frames:
        wr.write(f)
    wr.release()
    return _x264(_decode(avi), fps, 23, os.path.join(tmpdir, "tmp.mp4"))


def degraded_roi_frames(frames, fps, roi, vid, cond, tmpdir):
    """Return the (perturbed, re-encoded) ROI frames for one condition."""
    x0, y0, x1, y1 = roi
    kind, lv = cond.split("_")
    _seed(vid, cond)
    if kind == "VC":
        return _x264([f[y0:y1, x0:x1] for f in frames], fps,
                     PERTURB_PARAMS["VC"][int(lv) - 1], os.path.join(tmpdir, "vc.mp4"))
    if kind == "RE":
        return _xvid_then_x264([np.ascontiguousarray(f[y0:y1, x0:x1]) for f in frames], fps, tmpdir)
    fn, prm = FRAME_FN[kind], PERTURB_PARAMS[kind][int(lv) - 1]
    out = []
    for f in frames:
        if kind in FULL_FRAME:
            g = fn(f.copy(), prm)[y0:y1, x0:x1]
        else:
            g = fn(np.ascontiguousarray(f[y0:y1, x0:x1]).copy(), prm)
        out.append(np.ascontiguousarray(g))
    return _xvid_then_x264(out, fps, tmpdir)


def shift_M(M, x0, y0):
    """Affine for ROI coordinates: M(p + (x0, y0))."""
    M2 = M.copy()
    M2[:, 2] = M[:, 2] + M[:, :2] @ np.array([x0, y0], float)
    return M2


def build(video_path, prep, vid, conds, tmp_root=None):
    """
    Returns {cond: (crops_rgb (n_needed,150,150,3) uint8, roi_gray list for blockiness)}.
    """
    frames, _ = read_frames(video_path, MAX_FRAMES)
    fps = int(prep["fps"])
    needed, Ms, offs, roi = prep["needed"], prep["M"], prep["off"], prep["roi"]
    x0, y0, x1, y1 = roi
    res = {}
    tmp_root = tmp_root or str(DATA / "tmp")
    os.makedirs(tmp_root, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=tmp_root) as tmpdir:
        for cond in conds:
            if cond == "clean":
                src = [np.ascontiguousarray(frames[i][y0:y1, x0:x1]) for i in needed]
            else:
                rf = degraded_roi_frames(frames, fps, roi, vid, cond, tmpdir)
                src = [rf[i] for i in needed]
            crops = np.stack([cv2.cvtColor(crop(s, shift_M(M, x0, y0), off), cv2.COLOR_BGR2RGB)
                              for s, M, off in zip(src, Ms, offs)])
            grays = [cv2.cvtColor(s, cv2.COLOR_BGR2GRAY) for s in src]
            res[cond] = (crops, grays, src)
    return res
