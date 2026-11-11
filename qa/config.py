"""Paths and protocol constants. Every path can be overridden with an environment variable."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# upstream DFD-FCG checkout (https://github.com/aiiu-lab/DFD-FCG): provides src/ and misc/
UPSTREAM = Path(os.environ.get("DFD_FCG_UPSTREAM", ROOT.parent / "DFD-FCG")).resolve()
# authors' released checkpoint (weights.ckpt, 1.2 GB)
CKPT = Path(os.environ.get("DFD_FCG_CKPT", UPSTREAM / "checkpoint" / "weights.ckpt")).resolve()
# where videos, landmarks and extracted features are written
DATA = Path(os.environ.get("DFD_DATA", ROOT.parent / "data")).resolve()

HF_REPO = "gonnerthetooner/FaceForensics"          # public mirror of FaceForensics++ (c23)
HF_PREFIX = "faceforensics_c23/FaceForensics++_C23/"
METHODS = {                                        # folder name in the mirror -> short code
    "original": "REAL",
    "Deepfakes": "DF",
    "Face2Face": "F2F",
    "FaceSwap": "FS",
    "NeuralTextures": "NT",
    "FaceShifter": "FSh",                          # not in DFD-FCG's training set
}

# ---- clip protocol (same as the upstream demo / xai engine) ----
NUM_FRAMES = 10            # frames per clip
STRIDE_SEC = 0.333         # frame i of a clip is at i * 0.333 * fps  -> 3 s clip
CLIPS_PER_VIDEO = 2        # evenly spaced clips per video
MAX_FRAMES = 144           # only the first 144 frames (~4.8 s at 30 fps) are processed
SMOOTH_MARGIN = 6          # upstream landmark smoothing uses a 12-frame window (+-6)
CROP_SIZE = 150            # upstream crop_main_face defaults
TARGET_SIZE = 256

# ---- the authors' robustness protocol (src/preprocess/robustness) ----
PERTURB_PARAMS = {
    "CS": [0.4, 0.3, 0.2, 0.1, 0.0],
    "CC": [0.85, 0.725, 0.6, 0.475, 0.35],
    "BW": [16, 32, 48, 64, 80],
    "GNC": [0.001, 0.002, 0.005, 0.01, 0.05],
    "GB": [7, 9, 13, 17, 21],
    "JPEG": [2, 3, 4, 5, 6],
    "VC": [30, 32, 35, 38, 40],
}
PERTURB_NAMES = {
    "CS": "colour saturation", "CC": "colour contrast", "BW": "block-wise occlusion",
    "GNC": "Gaussian noise", "GB": "Gaussian blur", "JPEG": "down/up-sampling",
    "VC": "H.264 compression", "RE": "re-encode only (control)", "clean": "clean c23",
}
LEVELS = [1, 3, 5]          # severities evaluated on the test split ...
# ... for the three degradations that cause false alarms; one (medium) level for the other four
TEST_LEVELS = {"GB": [1, 3, 5], "JPEG": [1, 3, 5], "VC": [1, 3, 5],
               "CS": [3], "CC": [3], "BW": [3], "GNC": [3]}
FIT_LEVELS = [2, 4]         # severities used to FIT corrections (never evaluated -> severity shift too)


def conditions(split="test"):
    """Conditions per split: clean, re-encode control, 7 types x levels."""
    out = ["clean", "RE_0"]
    for k in PERTURB_PARAMS:
        for lv in (FIT_LEVELS if split == "fit" else TEST_LEVELS[k]):
            out.append(f"{k}_{lv}")
    return out


def p(*parts):
    path = DATA.joinpath(*parts)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path
