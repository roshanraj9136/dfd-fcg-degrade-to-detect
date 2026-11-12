r"""
Exact additive attribution for DFD-FCG, extended from
github.com/roshanraj9136/dfd-fcg-explainability (xai/engine.py, MIT).

DFD-FCG:  P_final = [ sigma(d_s) + sigma(d_t) + sigma(d_a) ] / 3, with d = logit_FAKE - logit_REAL and
    d_s = w_s . LN_s(y_s) + b_s
    d_t = w_t . LN_t(y_t) + b_t
    d_a = w_as . LN_s(y_s) + w_at . LN_t(y_t) + b_a
    y_s = sum_{layer l, query q, patch k} y_s[l,q,k]     y_t = sum_l y_t[l]

For any additive split y = sum_c y^(c):
    w . LN(y) + b  =  sum_c  (1/sigma) sum_j w_j g_j (y^(c)_j - mean(y^(c)))   +   (w . beta + b)
so every head logit is an exact sum of per-term contributions plus a constant.
New in this version: the same split is applied to the fusion head, and the spatial
branch is also split per *patch* (exact, via the attention weights), so all three heads
are decomposed over the same (layer x component) and (patch) terms.
"""
import math

import numpy as np
import torch

from qa.config import CKPT, UPSTREAM

REAL, FAKE = 0, 1
FACE_PARTS = ["lips", "skin", "eyes", "nose"]


def load_model(device="cuda"):
    import os
    import sys
    sys.path.insert(0, str(UPSTREAM))
    cwd = os.getcwd()
    os.chdir(UPSTREAM)                     # the checkpoint refers to misc/ by relative path
    try:
        from src.model.clip.svl import FFGSynoVideoLearner
        try:
            model = FFGSynoVideoLearner.load_from_checkpoint(str(CKPT), map_location="cpu",
                                                             attn_record=False)
        except Exception:
            model = FFGSynoVideoLearner.load_from_checkpoint(str(CKPT), map_location="cpu",
                                                             attn_record=False, strict=False)
    finally:
        os.chdir(cwd)
    model.eval().to(device)
    return model


class Recorder:
    """Patches SynoBlock.spatial_detection / temporal_detection to keep the additive parts."""

    def __init__(self, model):
        self.blocks = list(model.model.encoder.decoder.decoder_layers)
        self.reset()
        for blk in self.blocks:
            blk.spatial_detection = self._spatial(blk)
            blk.temporal_detection = self._temporal(blk)

    def reset(self):
        self.s_parts, self.t_layer, self.s_patch = [], [], None

    def _spatial(self, blk):
        rec = self

        def fn(attrs):
            _k = attrs[blk.s_k_attr][:, :, 1:]
            _v = attrs[blk.s_v_attr][:, :, 1:]
            b = _k.shape[0]
            s_q = blk.syno_embedding.unsqueeze(0).repeat(b, 1, 1)
            if _k.dim() == 5:
                _k = _k.flatten(-2).contiguous()
            if _v.dim() == 5:
                _v = _v.flatten(-2).contiguous()
            s_aff = torch.einsum("nqw,ntkw->ntqk",
                                 s_q / (s_q.norm(dim=-1, keepdim=True) + 1e-4),
                                 _k / (_k.norm(dim=-1, keepdim=True) + 1e-4))
            s_aff = (s_aff * 100).softmax(dim=-1)
            s_mix = torch.einsum("ntql,ntlw->ntqw", s_aff, _v)
            tq = s_mix.shape[1] * s_mix.shape[2]
            y_parts = s_mix.sum(dim=1) / tq                      # (b,q,w)
            y = y_parts.sum(dim=1)                               # == original mean over (t,q)
            # exact per-patch split of the same y: sum_t sum_q aff[t,q,k] v[t,k] / tq
            patch = torch.einsum("ntql,ntlw->nlw", s_aff.float(), _v.float()) / tq
            rec.s_patch = patch if rec.s_patch is None else rec.s_patch + patch
            rec.s_parts.append(y_parts.float())
            return dict(s_q=s_q, y=y)

        return fn

    def _temporal(self, blk):
        rec = self
        orig = type(blk).temporal_detection

        def fn(attrs):
            ret = orig(blk, attrs)
            rec.t_layer.append(ret["y"].float())
            return ret

        return fn


class _LN:
    """Read-only float64 copy of a LayerNorm's parameters (the model itself is not touched)."""

    def __init__(self, ln):
        self.weight = ln.weight.detach().double().cpu()
        self.bias = ln.bias.detach().double().cpu()
        self.eps = ln.eps


def _split(Y, ln, w, bias):
    """Exact contributions of the rows of Y (K,D) to  w . LN(sum Y) + bias."""
    Y = Y.double()
    y = Y.sum(0)
    g, beta = ln.weight, ln.bias
    sig = torch.sqrt(y.var(unbiased=False) + ln.eps)
    c = ((Y - Y.mean(1, keepdim=True)) * (w * g)).sum(1) / sig
    const = float((w * beta).sum() + bias)
    return c, const


def _sig(x):
    return 1.0 / (1.0 + math.exp(-x))


@torch.inference_mode()
def analyse(model, rec, clips, fp16=True, timing=None):
    """
    clips: (B, T, 3, 224, 224) normalised tensor.
    Returns a list of per-clip dicts of numpy arrays (all exact, float64 algebra).
    """
    import time
    t0 = time.time()
    m = model.model
    rec.reset()
    dev = next(m.parameters()).device
    with torch.autocast("cuda", dtype=torch.float16, enabled=fp16 and dev.type == "cuda"):
        out = m(clips.to(dev))
    model_p = out["logits"].float().softmax(-1)[:, FAKE].cpu().numpy()   # logits = log(mean softmax)
    if timing is not None:
        timing["forward"] = timing.get("forward", 0) + time.time() - t0

    S = torch.stack(rec.s_parts, 1).double().cpu()    # (B, L, Q, 1024)
    T = torch.stack(rec.t_layer, 1).double().cpu()    # (B, L, 256)
    SP = rec.s_patch.double().cpu()                   # (B, 256, 1024)
    B, L, Q, Ds = S.shape

    def wdiff(lin):
        W = lin.weight.detach().double().cpu()
        b = lin.bias.detach().double().cpu()
        return W[FAKE] - W[REAL], float(b[FAKE] - b[REAL])

    s_ln, t_ln = _LN(m.s_ln), _LN(m.t_ln)
    ws, bs = wdiff(m.s_head)
    wt, bt = wdiff(m.t_head)
    wa, ba = wdiff(m.a_head)
    was, wat = wa[:Ds], wa[Ds:]

    res = []
    for i in range(B):
        lq = S[i].reshape(L * Q, Ds)
        cs_lq, k_s = _split(lq, s_ln, ws, bs)                         # spatial head
        ct_l, k_t = _split(T[i], t_ln, wt, bt)                        # temporal head
        ca_lq, k_as = _split(lq, s_ln, was, 0.0)                      # fusion head, spatial input
        ca_tl, k_at = _split(T[i], t_ln, wat, 0.0)                    # fusion head, temporal input
        cs_patch, _ = _split(SP[i], s_ln, ws, bs)                     # spatial head per patch
        ct_patch, _ = _split(torch.diag(T[i].sum(0)), t_ln, wt, bt)   # temporal head per patch
        ca_spatch, _ = _split(SP[i], s_ln, was, 0.0)
        ca_tpatch, _ = _split(torch.diag(T[i].sum(0)), t_ln, wat, 0.0)
        d_s = float(cs_lq.sum()) + k_s
        d_t = float(ct_l.sum()) + k_t
        d_a = float(ca_lq.sum() + ca_tl.sum()) + k_as + k_at + ba
        p_f = (_sig(d_s) + _sig(d_t) + _sig(d_a)) / 3
        res.append(dict(
            d=np.array([d_s, d_t, d_a]),
            const=np.array([k_s, k_t, k_as + k_at + ba]),
            p_final=p_f, model_p=float(model_p[i]),
            cs_lq=cs_lq.reshape(L, Q).numpy().astype(np.float32),
            ct_l=ct_l.numpy().astype(np.float32),
            ca_lq=ca_lq.reshape(L, Q).numpy().astype(np.float32),
            ca_tl=ca_tl.numpy().astype(np.float32),
            cs_patch=cs_patch.numpy().astype(np.float32),
            ct_patch=ct_patch.numpy().astype(np.float32),
            ca_patch=(ca_spatch + ca_tpatch).numpy().astype(np.float32),
            y_s=S[i].sum((0, 1)).float().numpy(),
            y_t=T[i].sum(0).float().numpy(),
        ))
    rec.reset()
    return res
