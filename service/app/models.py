"""Fetch the model files at startup (DFD-FCG checkpoint from a private Hugging Face repo, CLIP, 2D-FAN).

Each file is tried from the private weights repo first (3 attempts). CLIP and the face models also have
public fallback URLs (the same files the upstream libraries download themselves).
"""
import os
import shutil
import time
import urllib.request
from pathlib import Path

REPO = os.environ.get("WEIGHTS_REPO", "roshan9136/dfd-fcg-weights")
ROOT = Path(os.environ.get("MODEL_DIR", "/tmp/models"))

PUBLIC = {
    "clip/ViT-L-14.pt": "https://openaipublic.azureedge.net/clip/models/"
                        "b8cca3fd41ae0c99ba7e8951adf17d267cdb84cd88be6f7c2e0eca1737a03836/ViT-L-14.pt",
    "face_alignment/s3fd-619a316812.pth": "https://www.adrianbulat.com/downloads/python-fan/s3fd-619a316812.pth",
    "face_alignment/2DFAN4-11f355bf06.pth.tar":
        "https://www.adrianbulat.com/downloads/python-fan/2DFAN4-11f355bf06.pth.tar",
}


def _from_hub(name, token):
    from huggingface_hub import hf_hub_download
    return hf_hub_download(REPO, name, token=token, local_dir=ROOT / "dl")


def _from_url(url, dest):
    tmp = dest.with_suffix(dest.suffix + ".part")
    with urllib.request.urlopen(url, timeout=60) as r, open(tmp, "wb") as fh:
        shutil.copyfileobj(r, fh, 1 << 20)
    tmp.replace(dest)


def _fetch(name, dest, token):
    if dest.exists():
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    last = None
    for attempt in range(3):
        try:
            shutil.move(_from_hub(name, token), dest)
            print(f"[models] {name} <- {REPO}", flush=True)
            return
        except Exception as e:  # noqa: BLE001
            last = e
            print(f"[models] hub download of {name} failed ({type(e).__name__}: {e}); attempt {attempt + 1}",
                  flush=True)
            time.sleep(5 * (attempt + 1))
    if name in PUBLIC:
        _from_url(PUBLIC[name], dest)
        print(f"[models] {name} <- public URL", flush=True)
        return
    raise RuntimeError(f"could not download {name}: {last}")


def ensure_models():
    token = os.environ.get("HF_TOKEN")
    ROOT.mkdir(parents=True, exist_ok=True)
    ckpt = Path(os.environ.get("DFD_FCG_CKPT") or ROOT / "weights.ckpt")
    _fetch("weights.ckpt", ckpt, token)
    clip_dir = Path(os.path.expanduser("~/.cache/clip"))
    _fetch("clip/ViT-L-14.pt", clip_dir / "ViT-L-14.pt", token)
    hub = Path(os.environ.get("TORCH_HOME", os.path.expanduser("~/.cache/torch"))) / "hub" / "checkpoints"
    for name in ["s3fd-619a316812.pth", "2DFAN4-11f355bf06.pth.tar"]:
        _fetch(f"face_alignment/{name}", hub / name, token)
    os.environ["DFD_FCG_CKPT"] = str(ckpt)
    shutil.rmtree(ROOT / "dl", ignore_errors=True)
