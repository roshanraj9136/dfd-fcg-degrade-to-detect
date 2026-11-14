"""Loads every stored score so detectors and ensembles can be compared offline (no GPU).

Per video and face (and, for the 225 reported test videos, per condition clean / GB_5 / VC_5):
  dfd   released DFD-FCG with the live self-calibration read-out: P(fake)        (eval_new_old.jsonl, dfd_val_old.jsonl,
                                                                                   feats_test_old.npz + model.json)
  gend_<name>  a GenD model: per-frame fake-minus-real logits                   (gend_<name>_<new|rep>.npz)
"""
import csv
import json
import os
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
D = Path(os.environ.get("DFD_DATA_ROOT", r"D:\dfd-data"))
E = Path(os.environ.get("DFD_ENS_DIR", str(D / "ens")))
SETS = [D / "new" / "sets.csv", D / "new" / "sets_gated.csv", D / "new" / "sets_more.csv"]
NAMES = {"celebdf": "Celeb-DF v2", "social": "Social-media deepfakes (2026)",
         "mavos": "MAVOS-DD, methods seen in training (2025)", "mavos_open": "MAVOS-DD, methods never seen (2025)",
         "sf": "SynthForensics, AI-generated people (2026)", "synth": "AI-generated talking heads (2026)",
         "ffpp": "FF++ test (reported)", "fsh": "FaceShifter (reported, never trained on)",
         "dfdc": "DFDC part 0 (reported, never trained on)"}
sig = lambda x: 1 / (1 + np.exp(-np.asarray(x, float)))  # noqa: E731
logit = lambda p: np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6)))  # noqa: E731


def cache_file(name):
    """New experiment scores can live separately; existing scores stay read-only."""
    candidate = E / name
    return candidate if candidate.exists() else D / "ens" / name


def rows():
    """vid -> dict(set, source, label, method) for the fine-tuning sets (sets.csv, sets_gated.csv)."""
    out = {}
    for f in SETS:
        for r in csv.DictReader(open(f)):
            out.setdefault(r["vid"], dict(set=r["set"], source=r["source"], label=int(r["label"]), method=r["method"]))
    return out


def group(r):
    """Evaluation group of a fine-tuning-set video."""
    return "mavos_open" if r["set"] == "test_open" else r["source"]


def gend(name, which):
    """{(vid, cond, face): per-frame logits} and {(vid, cond, face): per-frame L2 features (fp16)}."""
    p = cache_file(f"gend_{name}_{which}.npz")
    if not p.exists():
        return None, None
    z = np.load(p)
    keys, st, lg, ft = z["keys"], z["start"], z["logit"], z["feats"]
    L, Fm = {}, {}
    for k, key in enumerate(keys):
        vid, cond, face = str(key).split("|")
        L[(vid, cond, int(face))] = lg[st[k]:st[k + 1]]
        Fm[(vid, cond, int(face))] = ft[st[k]:st[k + 1]]
    return L, Fm


def dfd_new(which="old"):
    """DFD-FCG self-calibrated P(fake) per (vid, 'clean', face) for the test / val videos of the fine-tuning sets.
    which: 'old' = released checkpoint, 'new' = round-1 fine-tuned checkpoint (each with its own read-out)."""
    out = {}
    for f in (D / f"eval_new_{which}.jsonl", cache_file(f"dfd_val_{which}.jsonl")):
        if not f.exists():
            continue
        for l in open(f, encoding="utf-8"):
            r = json.loads(l)
            faces = r.get("faces") or [r["p"]]
            for k, p in enumerate(faces):
                out[(r["vid"], "clean", k)] = float(p)
    return out


def dfd_rep(which="old"):
    """DFD-FCG self-calibrated P(fake) per (vid, cond, 0) for the 225 reported test videos (research protocol, 2 clips)
    and vid -> (label, method)."""
    z = np.load(D / f"feats_test_{which}.npz")
    mj = ROOT / "service" / "app" / "model.json" if which == "old" else D / "readout_new.json"
    m = json.load(open(mj))["selfcal"]
    mu, sd, W, b = (np.array(m[k]) for k in ("mean", "scale", "coef", "intercept"))
    Y = np.c_[z["y_s"], z["y_t"]]
    G = ((Y - mu) / sd) @ W.T + b
    P = sig(z["d"] - G).mean(1)
    acc, meta = {}, {}
    for vid, cond, lab, meth, p in zip(z["vid"], z["cond"], z["label"], z["method"], P):
        acc.setdefault((str(vid), str(cond), 0), []).append(p)
        meta[str(vid)] = (int(lab), str(meth))
    return {k: float(np.mean(v)) for k, v in acc.items()}, meta


def rep_group(vid, method):
    if method in ("DFDC_R", "DFDC_F") or vid.startswith("DFDC"):
        return "dfdc"
    if method == "FSh":
        return "fsh"
    return "ffpp"


def video_score(frame_logits, how="prob"):
    """GenD video score as a logit: mean frame probability (the paper's rule) or mean logit."""
    if how == "prob":
        return float(logit(sig(frame_logits).mean()))
    return float(np.mean(frame_logits))


class GenDSelfCal:
    """Self-calibration (our DFD-FCG add-on) carried over to GenD. GenD's L2-normalised class token also encodes how
    degraded the face is; a ridge read-out on it predicts the spurious FAKE evidence a degradation adds to the frame
    logit, and it is subtracted. Same recipe as qa.methods.SelfCal: the target is class-agnostic (for a frame of a fit
    video under condition c: the mean paired shift (degraded - clean) of the video logit that c causes on the REAL fit
    videos; 0 for clean), fitted on the 49 FF++ train-split fit videos at damage levels 2 and 4 (+ re-encode), alpha
    1000, standardised features."""

    def __init__(self, name, alpha=1000.0):
        from sklearn.linear_model import Ridge
        from sklearn.preprocessing import StandardScaler
        L, Fm = gend(name, "fit")
        if L is None:
            raise FileNotFoundError(f"gend_{name}_fit.npz")
        z = np.load(D / "feats_fit_old.npz")
        lab = {str(v): int(l) for v, l in zip(z["vid"], z["label"])}
        vlog = {k: video_score(v) for k, v in L.items()}
        shift = {}
        for (vid, cond, f), s in vlog.items():
            if cond != "clean" and lab.get(vid) == 0 and (vid, "clean", f) in vlog:
                shift.setdefault(cond, []).append(s - vlog[(vid, "clean", f)])
        self.targets = {"clean": 0.0, **{c: float(np.mean(v)) for c, v in shift.items()}}
        X, y = [], []
        for k, fr in Fm.items():
            if k[1] in self.targets and k[0] in lab:
                X.append(fr.astype(np.float32))
                y.append(np.full(len(fr), self.targets[k[1]]))
        X, y = np.concatenate(X), np.concatenate(y)
        self.sc = StandardScaler().fit(X)
        self.reg = Ridge(alpha=alpha).fit(self.sc.transform(X), y)

    def correct(self, frame_logits, frame_feats):
        return frame_logits - self.reg.predict(self.sc.transform(frame_feats.astype(np.float32)))


def metrics(y, s, thr=0.0):
    """y labels, s scores as logits (verdict FAKE when s >= thr)."""
    from sklearn.metrics import roc_auc_score
    y, s = np.asarray(y), np.asarray(s)
    out = dict(n=int(len(y)))
    pred = s >= thr
    out["acc"] = float(np.mean(pred == (y == 1)))
    if (y == 0).any():
        out["false_alarm"] = float(np.mean(pred[y == 0]))
    if (y == 1).any():
        out["caught"] = float(np.mean(pred[y == 1]))
    if (y == 0).any() and (y == 1).any():
        out["bacc"] = (1 - out["false_alarm"] + out["caught"]) / 2
        out["auroc"] = float(roc_auc_score(y, s))
    return out
