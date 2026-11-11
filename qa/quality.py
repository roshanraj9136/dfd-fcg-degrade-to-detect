"""
Label-free image-quality descriptors, computed only from the input video.

These are what a deployed system can measure at test time; no ground truth and no
clean reference are needed.
"""
import cv2
import numpy as np

Q_NAMES = ["log_lapvar", "log_hf_ratio", "log_noise", "log_blockiness",
           "log_contrast", "brightness", "log_chroma", "log_face_scale"]

_IMMERKAER = np.array([[1, -2, 1], [-2, 4, -2], [1, -2, 1]], np.float32)


def _hf_ratio(g):
    g = g - g.mean()
    F = np.abs(np.fft.fftshift(np.fft.fft2(g))) ** 2
    h, w = g.shape
    fy = np.fft.fftshift(np.fft.fftfreq(h))[:, None]
    fx = np.fft.fftshift(np.fft.fftfreq(w))[None, :]
    r = np.sqrt(fx ** 2 + fy ** 2)
    return F[r > 0.25].sum() / (F.sum() + 1e-9)


def _noise(g):
    h, w = g.shape
    s = np.abs(cv2.filter2D(g, -1, _IMMERKAER))[1:-1, 1:-1].sum()
    return np.sqrt(np.pi / 2) * s / (6 * (w - 2) * (h - 2))


def _blockiness(g):
    """Mean |step| across 8-px block boundaries / mean |step| elsewhere (ROI is grid-aligned)."""
    g = g.astype(np.float32)
    dx = np.abs(np.diff(g, axis=1))           # dx[:, j] = |g[:, j+1]-g[:, j]|
    dy = np.abs(np.diff(g, axis=0))
    bx = (np.arange(dx.shape[1]) % 8) == 7
    by = (np.arange(dy.shape[0]) % 8) == 7
    rx = dx[:, bx].mean() / (dx[:, ~bx].mean() + 1e-6)
    ry = dy[by].mean() / (dy[~by].mean() + 1e-6)
    return 0.5 * (rx + ry)


def frame_features(crop_rgb, roi_gray, face_scale):
    g = cv2.cvtColor(crop_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
    ycc = cv2.cvtColor(crop_rgb, cv2.COLOR_RGB2YCrCb).astype(np.float32)
    chroma = np.sqrt((ycc[..., 1] - 128) ** 2 + (ycc[..., 2] - 128) ** 2).mean()
    return np.array([
        np.log(cv2.Laplacian(g, cv2.CV_32F).var() + 1.0),
        np.log(_hf_ratio(g) + 1e-6),
        np.log(_noise(g) + 1e-3),
        np.log(_blockiness(roi_gray) + 1e-3),
        np.log(g.std() + 1.0),
        g.mean() / 255.0,
        np.log(chroma + 1.0),
        np.log(face_scale),
    ], np.float32)


def face_scale_of(M):
    """Linear scale of the alignment transform (source px -> crop px); small = big face."""
    return float(np.sqrt(abs(np.linalg.det(M[:, :2]))))


# --------------------------------------------------------------------------------------------
# Background-referenced quality: the same descriptors measured OUTSIDE the (dilated) face.
# FaceForensics++ manipulations only change the face, so these pixels carry the video's
# quality but no forgery evidence by construction.
# --------------------------------------------------------------------------------------------
QB_NAMES = ["bg_log_lapvar", "bg_log_noise", "bg_log_blockiness", "bg_log_hf", "bg_log_contrast",
            "bg_brightness", "bg_log_chroma", "log_face_scale"]


def face_mask(shape, pts, grow=1.35, up=0.35):
    """Dilated convex hull of the 68 landmarks, extended upwards to cover the forehead."""
    c = pts.mean(0)
    p = c + (pts - c) * grow
    h = pts[:, 1].max() - pts[:, 1].min()
    top = p[p[:, 1] < c[1]].copy()
    top[:, 1] -= up * h
    hull = cv2.convexHull(np.concatenate([p, top]).astype(np.float32)).astype(np.int32)
    m = np.zeros(shape[:2], np.uint8)
    cv2.fillConvexPoly(m, hull, 1)
    return m.astype(bool)


def background_features(roi_bgr, bg, face_scale):
    """Masked versions of the crop descriptors; bg is a boolean mask of background pixels."""
    g = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    ycc = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2YCrCb).astype(np.float32)
    inner = cv2.erode(bg.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)   # avoid hull edge
    if inner.sum() < 400:
        inner = bg
    lap = cv2.Laplacian(g, cv2.CV_32F)
    imm = np.abs(cv2.filter2D(g, -1, _IMMERKAER))
    noise = np.sqrt(np.pi / 2) * imm[inner].mean() / 6.0
    gm = g[inner]
    dx = np.abs(np.diff(g, axis=1))
    dy = np.abs(np.diff(g, axis=0))
    mx = inner[:, 1:] & inner[:, :-1]
    my = inner[1:, :] & inner[:-1, :]
    bx = np.zeros_like(mx)
    bx[:, (np.arange(mx.shape[1]) % 8) == 7] = True
    by = np.zeros_like(my)
    by[(np.arange(my.shape[0]) % 8) == 7, :] = True
    rx = dx[mx & bx].mean() / (dx[mx & ~bx].mean() + 1e-6) if (mx & bx).any() else 1.0
    ry = dy[my & by].mean() / (dy[my & ~by].mean() + 1e-6) if (my & by).any() else 1.0
    chroma = np.sqrt((ycc[..., 1] - 128) ** 2 + (ycc[..., 2] - 128) ** 2)[inner].mean()
    return np.array([
        np.log(lap[inner].var() + 1.0),
        np.log(noise + 1e-3),
        np.log(0.5 * (rx + ry) + 1e-3),
        np.log((lap[inner] ** 2).mean() / (gm.var() + 1.0) + 1e-6),
        np.log(gm.std() + 1.0),
        gm.mean() / 255.0,
        np.log(chroma + 1.0),
        np.log(face_scale),
    ], np.float32)


def landmarks_in_roi(crop_lm, M, off, roi):
    """Crop-space landmarks -> ROI pixel coordinates."""
    Minv = cv2.invertAffineTransform(M)
    p = (crop_lm + np.asarray(off, float)) @ Minv[:, :2].T + Minv[:, 2]
    return p - np.array([roi[0], roi[1]], float)
