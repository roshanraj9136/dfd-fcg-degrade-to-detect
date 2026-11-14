"""GenD (Yermakov et al., WACV 2026, github.com/yermandy/GenD): a frame-level deepfake detector made of a frozen
foundation-model image encoder (CLIP ViT-L/14, Perception Encoder L or DINOv3 L) whose LayerNorm parameters were tuned
on FF++, with an L2-normalised class token and a linear real/fake head. Label 1 = fake.

This wrapper loads the released weights from Hugging Face with the authors' own modelling code (vendored from their
repository, MIT licence) and scores uint8 RGB face crops.
"""
import os
import sys
from pathlib import Path

os.environ.setdefault("HF_HUB_CACHE", r"D:\hf-cache")
GEND_SRC = Path(os.environ.get("GEND_SRC", Path(__file__).resolve().parents[3] / "GenD"))
sys.path.insert(0, str(GEND_SRC))

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

MODELS = {"clip": "yermandy/GenD_CLIP_L_14", "pe": "yermandy/GenD_PE_L", "dino": "yermandy/GenD_DINOv3_L"}


def _dinov3_offline():
    """facebook/dinov3-* is gated on Hugging Face. GenD's checkpoint holds the whole backbone, so only the
    architecture is needed: the public DINOv3 ViT-L/16 (LVD-1689M) config, ImageNet normalisation."""
    import transformers
    from types import SimpleNamespace
    from transformers import DINOv3ViTConfig, DINOv3ViTModel
    cfg = DINOv3ViTConfig(patch_size=16, hidden_size=1024, intermediate_size=4096, num_hidden_layers=24,
                          num_attention_heads=16, hidden_act="gelu", layer_norm_eps=1e-5, rope_theta=100.0,
                          image_size=224, query_bias=True, key_bias=False, value_bias=True, proj_bias=True,
                          mlp_bias=True, layerscale_value=1e-5, use_gated_mlp=False, num_register_tokens=4,
                          pos_embed_rescale=2.0)
    a_img, a_mod = transformers.AutoImageProcessor.from_pretrained, transformers.AutoModel.from_pretrained

    def img(name, *a, **k):
        if "dinov3" in name:
            return SimpleNamespace(image_mean=[0.485, 0.456, 0.406], image_std=[0.229, 0.224, 0.225])
        return a_img(name, *a, **k)

    def mod(name, *a, **k):
        if "dinov3" in name:
            return DINOv3ViTModel(cfg)
        return a_mod(name, *a, **k)

    transformers.AutoImageProcessor.from_pretrained, transformers.AutoModel.from_pretrained = img, mod


def load(name, device="cuda", dtype=torch.float16):
    """Builds GenD from its config (the backbone is fetched by the authors' own code), then loads the released
    weights. (transformers >= 5 forbids GenD.from_pretrained, because GenD calls from_pretrained inside __init__.)"""
    import json
    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file
    from src.hf.modeling_gend import GenD, GenDConfig
    repo = MODELS.get(name, name)
    cfg = json.load(open(hf_hub_download(repo, "config.json")))
    if "dinov3" in cfg["backbone"]:
        _dinov3_offline()
    m = GenD(GenDConfig(backbone=cfg["backbone"], head=cfg.get("head", "linear")))
    sd = load_file(hf_hub_download(repo, "model.safetensors"))
    own = set(m.state_dict())
    for k in [k for k in sd if k not in own]:               # newer transformers nest some backbones one level deeper
        for alt in (k.replace("backbone.", "backbone.model.", 1), k.replace("backbone.model.", "backbone.", 1)):
            if alt in own:
                sd[alt] = sd.pop(k)
                break
    missing, unexpected = m.load_state_dict(sd, strict=False)
    missing = [k for k in missing if "position_ids" not in k and "inv_freq" not in k]
    if missing or unexpected:
        raise RuntimeError(f"{repo}: missing {missing[:5]} unexpected {unexpected[:5]}")
    m.eval().to(device=device, dtype=dtype)
    m.requires_grad_(False)
    return m


def norm_params(model):
    """Mean / std of the encoder's own preprocessing (the resize is done in to_tensor)."""
    fe = model.feature_extractor
    if hasattr(fe, "_preprocess") and hasattr(fe._preprocess, "image_processor"):     # CLIP / DINO (transformers)
        ip = fe._preprocess.image_processor
        return np.array(ip.image_mean), np.array(ip.image_std)
    if hasattr(fe, "_preprocess") and hasattr(fe._preprocess, "image_mean"):
        return np.array(fe._preprocess.image_mean), np.array(fe._preprocess.image_std)
    for t in getattr(fe._preprocess, "transforms", []):                                # timm (PE)
        if t.__class__.__name__ == "Normalize":
            return np.array(t.mean), np.array(t.std)
    raise RuntimeError("normalisation not found")


class Scorer:
    def __init__(self, name, device="cuda", size=224):
        self.name, self.device, self.size = name, device, size
        self.model = load(name, device)
        mean, std = norm_params(self.model)
        self.mean = torch.tensor(mean, dtype=torch.float32, device=device).view(1, 3, 1, 1)
        self.std = torch.tensor(std, dtype=torch.float32, device=device).view(1, 3, 1, 1)

    def to_tensor(self, frames):
        """uint8 RGB (N,H,W,3) -> normalised (N,3,size,size). Bicubic resize with antialiasing, as the encoders'
        own processors do for a square crop."""
        x = torch.from_numpy(np.ascontiguousarray(frames)).to(self.device).permute(0, 3, 1, 2).float() / 255.0
        if x.shape[-1] != self.size or x.shape[-2] != self.size:
            x = F.interpolate(x, size=(self.size, self.size), mode="bicubic", antialias=True, align_corners=False)
            x = x.clamp(0, 1)
        return (x - self.mean) / self.std

    @torch.inference_mode()
    def features(self, frames, bs=32):
        """L2-normalised class tokens (N, 1024) and fake-minus-real logits (N,)."""
        feats, logits = [], []
        for k in range(0, len(frames), bs):
            x = self.to_tensor(frames[k:k + bs]).to(next(self.model.parameters()).dtype)
            f = self.model.feature_extractor(x)
            out = self.model.model(f)
            feats.append(F.normalize(f.float(), dim=1).cpu())
            logits.append((out[:, 1] - out[:, 0]).float().cpu())
        return torch.cat(feats).numpy(), torch.cat(logits).numpy()
