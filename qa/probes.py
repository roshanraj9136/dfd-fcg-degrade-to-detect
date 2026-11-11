"""
Test-time probes: known, mild re-degradations applied by the detector itself to the aligned
150x150 face crops (crop space, so the probe strength is the same for every face size).
"""
import cv2
import numpy as np


def blur(crops, sigma):
    return np.stack([cv2.GaussianBlur(c, (0, 0), sigma) for c in crops])


def jpeg(crops, quality):
    out = []
    for c in crops:
        ok, buf = cv2.imencode(".jpg", cv2.cvtColor(c, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, quality])
        out.append(cv2.cvtColor(cv2.imdecode(buf, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB))
    return np.stack(out)


def resample(crops, factor):
    h, w = crops.shape[1:3]
    return np.stack([cv2.resize(cv2.resize(c, (w // factor, h // factor), interpolation=cv2.INTER_AREA),
                                (w, h), interpolation=cv2.INTER_LINEAR) for c in crops])


def noise(crops, sigma, seed=0):
    rng = np.random.default_rng(seed)
    return np.clip(crops.astype(np.float32) + rng.normal(0, sigma, crops.shape), 0, 255).astype(np.uint8)


PROBES = {
    "pB1": lambda x: blur(x, 1.0),
    "pB2": lambda x: blur(x, 2.0),
    "pJ50": lambda x: jpeg(x, 50),
    "pR2": lambda x: resample(x, 2),
    "pN3": lambda x: noise(x, 3.0),
}


def apply(name, crops):
    return PROBES[name](crops)
