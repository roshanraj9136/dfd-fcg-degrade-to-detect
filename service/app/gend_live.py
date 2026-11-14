"""GenD (Yermakov et al., WACV 2026; github.com/yermandy/GenD, MIT) for the live checker: a foundation image encoder
(CLIP ViT-L/14 or Perception Encoder L), L2-normalised class token, linear real/fake head. The encoder is built from
its architecture only (no second backbone download); weights: the authors' released checkpoint plus, optionally, our
fine-tuned LayerNorm + head tensors (service/ens/train_gend.py) and the self-calibration read-out on the class token.

Same preprocessing as the offline evaluation (service/ens/gend.py): the 150x150 aligned DFD-FCG face crop, bicubic
resize to 224, the encoder's mean / std; video score = logit of the mean frame probability over FRAMES evenly spaced
frames of the clips.
"""
import json
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from reliability import (InvalidScores, finite_array, logit, mean_probability,
                         readout_shift, validate_fusion, validate_readout)

REPOS = {"clip": "yermandy/GenD_CLIP_L_14", "pe": "yermandy/GenD_PE_L"}
NORM = {"clip": ((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))}


class GenDNet(nn.Module):
    def __init__(self, backbone):
        super().__init__()
        self.kind = backbone
        if backbone == "clip":
            from transformers import CLIPVisionConfig, CLIPVisionModel
            cfg = CLIPVisionConfig(hidden_size=1024, intermediate_size=4096, num_hidden_layers=24,
                                   num_attention_heads=16, image_size=224, patch_size=14, projection_dim=768,
                                   hidden_act="quick_gelu", layer_norm_eps=1e-5)
            self.enc = CLIPVisionModel(cfg)
        elif backbone == "pe":
            import timm
            self.enc = timm.create_model("vit_pe_core_large_patch14_336", pretrained=False, dynamic_img_size=True)
            self.enc.head = nn.Identity()
            dc = timm.data.resolve_model_data_config(self.enc)
            NORM["pe"] = (tuple(dc["mean"]), tuple(dc["std"]))
        else:
            raise ValueError(backbone)
        self.linear = nn.Linear(1024, 2)

    def forward(self, x):
        f = self.enc(pixel_values=x).pooler_output if self.kind == "clip" else self.enc(x)
        return f, self.linear(F.normalize(f, dim=1))

    def load_gend(self, sd):
        """GenD checkpoint names -> this module (works with transformers 4 and 5, whose CLIPVisionModel nests the
        weights differently)."""
        pre = {"clip": "feature_extractor.vision_model.", "pe": "feature_extractor.backbone."}[self.kind]
        keys = set(self.state_dict())
        cands = {"clip": ["enc.vision_model.", "enc."], "pe": ["enc."]}[self.kind]
        probe = next(k for k in sd if k.startswith(pre))[len(pre):]
        dst = next((c for c in cands if c + probe in keys), cands[0])
        own = {}
        for k, v in sd.items():
            if k.startswith(pre):
                own[dst + k[len(pre):]] = v
            elif k.startswith("model.linear."):
                own["linear." + k[len("model.linear."):]] = v
        missing, unexpected = self.load_state_dict(own, strict=False)
        missing = [k for k in missing if "position_ids" not in k and not k.endswith("head.weight")
                   and not k.endswith("head.bias")]
        if missing or unexpected:
            raise RuntimeError(f"GenD {self.kind} weights: missing {missing[:5]} unexpected {unexpected[:5]}")


class GenDMember:
    """One GenD detector. cfg: dict(backbone=clip|pe, frames=24, ln=<file of fine-tuned tensors or null>,
    selfcal=dict(mean, scale, coef, intercept) or null)."""

    def __init__(self, cfg, device="cpu", here=Path(__file__).resolve().parent):
        from huggingface_hub import hf_hub_download
        from safetensors.torch import load_file
        self.cfg, self.device, self.frames = cfg, device, int(cfg.get("frames", 24))
        if self.frames < 1:
            raise InvalidScores("GenD requires at least one sampled frame")
        kind = cfg.get("backbone", "clip")
        m = GenDNet(kind)
        sd = load_file(hf_hub_download(REPOS[kind], "model.safetensors"))
        if cfg.get("ln"):
            sd.update(torch.load(here / cfg["ln"], map_location="cpu"))
        m.load_gend(sd)
        self.fp16 = device == "cuda"
        self.model = m.eval().to(device=device, dtype=torch.float16 if self.fp16 else torch.float32)
        mean, std = NORM[kind]
        self.mean = torch.tensor(mean, device=device).view(1, 3, 1, 1)
        self.std = torch.tensor(std, device=device).view(1, 3, 1, 1)
        sc = cfg.get("selfcal")
        self.sc = None if sc is None else dict(zip(("mean", "scale", "coef", "intercept"),
                                                   validate_readout(sc, outputs=1, features=1024)))

    def twin(self, device):
        """A copy of this member on another device (CPU fallback)."""
        import copy
        t = copy.copy(self)
        t.device, t.fp16 = device, device == "cuda"
        t.model = copy.deepcopy(self.model).to(device=device, dtype=torch.float16 if t.fp16 else torch.float32)
        t.mean, t.std = self.mean.to(device), self.std.to(device)
        return t

    def pick(self, clip_idx):
        u = np.unique(np.asarray(clip_idx).ravel())
        if len(u) <= self.frames:
            return u
        return u[np.linspace(0, len(u) - 1, self.frames).round().astype(int)]

    @torch.inference_mode()
    def frame_scores(self, crops, clip_idx, bs=24):
        """crops: (N,150,150,3) uint8 RGB of one face. Returns frame logits (fake - real) and L2 class tokens."""
        fr = crops[self.pick(clip_idx)]
        if len(fr) == 0:
            raise InvalidScores("no frames available for GenD")
        lg, ft = [], []
        for k in range(0, len(fr), bs):
            x = torch.from_numpy(np.ascontiguousarray(fr[k:k + bs])).to(self.device).permute(0, 3, 1, 2).float() / 255
            x = F.interpolate(x, size=(224, 224), mode="bicubic", antialias=True, align_corners=False).clamp(0, 1)
            x = ((x - self.mean) / self.std).to(next(self.model.parameters()).dtype)
            f, out = self.model(x)
            lg.append((out[:, 1] - out[:, 0]).float().cpu().numpy())
            ft.append(F.normalize(f.float(), dim=1).cpu().numpy())
        return np.concatenate(lg), np.concatenate(ft)

    def video_logit(self, frame_logits, feats):
        """Logit of the mean frame probability, after removing the self-calibration's spurious evidence."""
        z = finite_array(frame_logits, "GenD frame logits", ndim=1)
        f = finite_array(feats, "GenD features", ndim=2)
        if f.shape != (len(z), 1024):
            raise InvalidScores("GenD frame logits and features do not match")
        if self.sc is not None:
            z = z - readout_shift(f, self.sc["mean"], self.sc["scale"],
                                  self.sc["coef"], self.sc["intercept"]).ravel()
        return logit(mean_probability(z))


def load_fusion(path):
    """fusion.json: {"members": [...], "coef": [...], "intercept": b, "detectors": {name: cfg}} or None.
    Member kinds: "dfd" (the main DFD-FCG with the live read-out), dfd-type extras (cfg type=dfd: weights, model_json)
    and GenD members (cfg type=gend)."""
    p = Path(path)
    if not p.exists():
        if os.environ.get("DFD_FUSION_JSON"):
            raise FileNotFoundError(f"configured fusion file does not exist: {p}")
        return None
    with p.open(encoding="utf-8") as fh:
        return validate_fusion(json.load(fh))


class DfdMember:
    """An extra DFD-FCG checkpoint (e.g. the round-1 fine-tuned one) with its own self-calibration read-out.
    cfg: dict(type=dfd, weights=<file in MODEL_DIR>, model_json=<file next to this module>). Video score: logit of the
    mean over clips of the mean of the three corrected head sigmoids (as service/train/eval_new.py)."""

    def __init__(self, cfg, device="cpu", here=Path(__file__).resolve().parent):
        import qa.engine as E
        self.cfg, self.device = cfg, device
        ck = Path(os.environ.get("MODEL_DIR", "/tmp/models")) / cfg["weights"]
        if cfg.get("weights_path"):
            ck = Path(cfg["weights_path"])
        old = E.CKPT
        E.CKPT = ck
        try:
            self.model = E.load_model(device=device)
        finally:
            E.CKPT = old
        self.rec = E.Recorder(self.model)
        m = json.load(open(here / cfg["model_json"]))["selfcal"]
        self.mu, self.sd, self.W, self.b = validate_readout(m, outputs=3, features=1280)
        self.fp16 = device == "cuda"

    def twin(self, device):
        import copy
        t = copy.copy(self)
        t.device, t.fp16 = device, device == "cuda"
        import qa.engine as E
        old = E.CKPT
        E.CKPT = Path(os.environ.get("MODEL_DIR", "/tmp/models")) / self.cfg["weights"]
        if self.cfg.get("weights_path"):
            E.CKPT = Path(self.cfg["weights_path"])
        try:
            t.model = E.load_model(device=device)
        finally:
            E.CKPT = old
        t.rec = E.Recorder(t.model)
        return t

    def frame_scores(self, crops, clip_idx, bs=None):
        """Head logits D (K,3) and features Y (K,1280) of every clip."""
        from qa.engine import analyse
        bs = bs or (4 if self.device == "cuda" else 2)
        D, Y = [], []
        for k in range(0, len(clip_idx), bs):
            x = torch.stack([self.model.transform(torch.from_numpy(np.ascontiguousarray(crops[ci])).permute(0, 3, 1, 2)
                                                  .contiguous()) for ci in clip_idx[k:k + bs]])
            with torch.inference_mode():
                for r in analyse(self.model, self.rec, x, fp16=self.fp16):
                    D.append(r["d"])
                    Y.append(np.r_[r["y_s"], r["y_t"]])
        return np.array(D), np.array(Y)

    def video_logit(self, D, Y):
        D = finite_array(D, "DFD head logits", ndim=2)
        G = readout_shift(Y, self.mu, self.sd, self.W, self.b)
        if D.shape != G.shape:
            raise InvalidScores("DFD head logits and features do not match")
        return logit(mean_probability(D - G))


def make_member(cfg, device, shared=None):
    """shared: dict reused across members, so GenD members with the same network (backbone + fine-tuned tensors) share
    one model and one forward pass (e.g. a member with and one without self-calibration)."""
    if cfg.get("type") == "dfd":
        return DfdMember(cfg, device)
    shared = {} if shared is None else shared
    key = (cfg.get("backbone", "clip"), cfg.get("ln"))
    if key in shared:
        import copy
        m = copy.copy(shared[key])
        m.cfg = cfg
        m.frames = int(cfg.get("frames", 24))
        if m.frames < 1:
            raise InvalidScores("GenD requires at least one sampled frame")
        sc = cfg.get("selfcal")
        m.sc = None if sc is None else dict(zip(("mean", "scale", "coef", "intercept"),
                                               validate_readout(sc, outputs=1, features=1024)))
        return m
    m = GenDMember(cfg, device)
    shared[key] = m
    return m


FUSION_JSON = os.environ.get("DFD_FUSION_JSON", "fusion.json")
