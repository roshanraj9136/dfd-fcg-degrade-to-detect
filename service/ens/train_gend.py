"""Fine-tune GenD (Yermakov et al., WACV 2026) on newer face-forgery data, with the authors' recipe: only the LayerNorm
parameters of the image encoder and the linear head are trained (about 0.03% of the weights), the class token is
L2-normalised, and the loss is cross-entropy + 0.1 x alignment + 0.5 x uniformity on the unit sphere.

Start: the released FF++ checkpoint (yermandy/GenD_CLIP_L_14 by default). Data: the train split of the fine-tuning
sets (FF++ train, Celeb-DF v2 non-test, MAVOS-DD train, SynthForensics train, Counterfactual groups 1-24, social-media
2026 train part), main face, 40 frames per video, read from the memory-mapped store of build_store.py. DFDC, every
test split and every reported test video are never used. Checkpoint selection: the val splits only.

Sampling: label 50:50, then a source uniformly among the sources with that label, then a video, then a frame.
Augmentation (per frame): mirror; small affine; one quality change with p 0.6 (JPEG 15-95, Gaussian blur 0.4-3.0,
down/up-scaling x1.5-4, noise 2-8), a second one with p 0.2; colour jitter with p 0.3.

    python service/ens/train_gend.py <out dir> [steps=4000] [lr=1e-4] [model=clip]
Output: <out>/best.pt (trainable tensors only), train_log.jsonl
"""
import csv
import json
import math
import os
import random
import sys
import time
import zlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "service" / "ens")]
import cv2  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402

from gend import Scorer  # noqa: E402
from selection import validate_split_rows, validation_score  # noqa: E402
from sampling import PairedSampler  # noqa: E402

OUT = Path(sys.argv[1])
STEPS = int(sys.argv[2]) if len(sys.argv) > 2 else 4000
LR = float(sys.argv[3]) if len(sys.argv) > 3 else 1e-4
NAME = sys.argv[4] if len(sys.argv) > 4 else "clip"
BS = int(os.environ.get("BS", "32"))
VAL_EVERY = int(os.environ.get("VAL_EVERY", "500"))
SELECTION_METRIC = os.environ.get("SELECTION_METRIC", "auc")
if SELECTION_METRIC not in ("bacc", "auc"):
    raise ValueError("SELECTION_METRIC must be bacc or auc")
WARMUP = int(os.environ.get("WARMUP", "200"))
W_ALIGN, W_UNIF = float(os.environ.get("W_ALIGN", "0.1")), float(os.environ.get("W_UNIF", "0.5"))
STORE = Path(os.environ.get("STORE", r"D:\dfd-data\ens\store"))
LN_NAMES = {"clip": ["pre_layrnorm", "layer_norm1", "layer_norm2", "post_layernorm"],
            "pe": ["norm"], "dino": ["norm1", "norm2", "layernorm", "norm."]}
OUT.mkdir(parents=True, exist_ok=True)
LOG = open(OUT / "train_log.jsonl", "a", encoding="utf-8")


def log(**kw):
    LOG.write(json.dumps(kw) + "\n")
    LOG.flush()
    try:                                  # the console may go away (parent script stopped): never let that stop training
        print(json.dumps(kw), flush=True)
    except (OSError, ValueError):
        pass


def rows(split):
    return [dict(r, row=int(r["row"]), label=int(r["label"]), n=int(r["n_frames"]))
            for r in csv.DictReader(open(STORE / "index.csv")) if r["set"] == split and int(r["n_frames"]) > 0]


# ------------------------------------------------------------------ augmentation (uint8 RGB 150x150x3)
def degrade(f, rng, kind=None):
    kind = kind or rng.choice(["jpeg", "blur", "scale", "noise"])
    if kind == "jpeg":
        ok, buf = cv2.imencode(".jpg", f, [cv2.IMWRITE_JPEG_QUALITY, int(rng.uniform(15, 95))])
        return cv2.imdecode(buf, cv2.IMREAD_UNCHANGED)
    if kind == "blur":
        return cv2.GaussianBlur(f, (0, 0), rng.uniform(0.4, 3.0))
    if kind == "scale":
        k = rng.uniform(1.5, 4.0)
        h, w = f.shape[:2]
        return cv2.resize(cv2.resize(f, (int(w / k), int(h / k)), interpolation=cv2.INTER_AREA), (w, h),
                          interpolation=cv2.INTER_LINEAR)
    noise = np.random.default_rng(rng.randrange(1 << 30)).normal(0, rng.uniform(2, 8), f.shape)
    return np.clip(f.astype(np.float32) + noise, 0, 255).astype(np.uint8)


def augment(f, rng):
    f = np.ascontiguousarray(f)
    if rng.random() < 0.5:
        f = f[:, ::-1]
    if rng.random() < 0.5:
        h, w = f.shape[:2]
        M = cv2.getRotationMatrix2D((w / 2, h / 2), rng.uniform(-6, 6), rng.uniform(0.94, 1.06))
        M[:, 2] += [rng.uniform(-0.03, 0.03) * w, rng.uniform(-0.03, 0.03) * h]
        f = cv2.warpAffine(np.ascontiguousarray(f), M, (w, h), borderMode=cv2.BORDER_REFLECT)
    if rng.random() < 0.6:
        f = degrade(np.ascontiguousarray(f), rng)
        if rng.random() < 0.2:
            f = degrade(f, rng)
    if rng.random() < 0.3:
        hsv = cv2.cvtColor(np.ascontiguousarray(f), cv2.COLOR_RGB2HSV).astype(np.float32)
        hsv[..., 1] *= rng.uniform(0.7, 1.3)
        f = cv2.cvtColor(np.clip(hsv, 0, 255).astype(np.uint8), cv2.COLOR_HSV2RGB)
        f = np.clip(f.astype(np.float32) * rng.uniform(0.8, 1.2) + rng.uniform(-20, 20), 0, 255).astype(np.uint8)
    return np.ascontiguousarray(f)


class Frames(torch.utils.data.Dataset):
    """Infinite balanced sampler: index -> (uint8 frame 150x150x3, label)."""

    def __init__(self, rs, seed):
        self.by = {}
        for r in rs:
            self.by.setdefault(r["label"], {}).setdefault(r["source"], []).append(r)
        self.seed, self.mm = seed, None

    def __len__(self):
        return 10 ** 9

    def __getitem__(self, i):
        if self.mm is None:
            self.mm = np.load(STORE / "frames.npy", mmap_mode="r")
        rng = random.Random(self.seed * 1_000_003 + i)
        lab = rng.randrange(2)
        src = rng.choice(sorted(self.by[lab]))
        r = rng.choice(self.by[lab][src])
        f = np.array(self.mm[r["row"], rng.randrange(r["n"])])
        return torch.from_numpy(augment(f, rng)), lab


class PairedFrames(Frames):
    def __init__(self, rs, seed, pairs):
        self.sampler = PairedSampler(rs, pairs, seed)
        self.mm = None

    def __getitem__(self, i):
        if self.mm is None:
            self.mm = np.load(STORE / "frames.npy", mmap_mode="r")
        row, position, augmentation_seed, _ = self.sampler.plan(i)
        frame_index = round(position * (row["n"] - 1))
        frame = np.array(self.mm[row["row"], frame_index])
        return torch.from_numpy(augment(frame, random.Random(augmentation_seed))), row["label"]


# ------------------------------------------------------------------ validation
def val_items(rs, per_video=8):
    """8 evenly spaced frames per val video, clean and with one fixed degradation (JPEG q40 / blur 1.5 / x2.5)."""
    mm = np.load(STORE / "frames.npy", mmap_mode="r")
    items = []
    for r in rs:
        idx = np.linspace(0, r["n"] - 1, per_video).round().astype(int)
        fr = np.array(mm[r["row"], idx])
        rng = random.Random(zlib.crc32(r["vid"].encode()))
        kind = ["jpeg", "blur", "scale"][zlib.crc32(r["vid"].encode()) % 3]
        deg = {"jpeg": lambda f: cv2.imdecode(cv2.imencode(".jpg", f, [cv2.IMWRITE_JPEG_QUALITY, 40])[1], -1),
               "blur": lambda f: cv2.GaussianBlur(f, (0, 0), 1.5),
               "scale": lambda f: cv2.resize(cv2.resize(f, (60, 60), interpolation=cv2.INTER_AREA), (150, 150))}[kind]
        items.append((r, "clean", fr))
        items.append((r, "degraded", np.stack([deg(np.ascontiguousarray(f)) for f in fr])))
        del rng
    return items


@torch.inference_mode()
def evaluate(sc, items):
    sc.model.eval()
    allf = np.concatenate([it[2] for it in items])
    probs = []
    for k in range(0, len(allf), 64):
        x = sc.to_tensor(allf[k:k + 64])
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = sc.model.model(sc.model.feature_extractor(x))
        probs.append(out.float().softmax(-1)[:, 1].cpu().numpy())
    probs = np.concatenate(probs)
    pos, res = 0, {}
    vp = []
    for r, cond, fr in items:
        vp.append((r, cond, float(probs[pos:pos + len(fr)].mean())))
        pos += len(fr)
    for cond in ("clean", "degraded"):
        vs = [(r, p) for r, c, p in vp if c == cond]
        reals = [p for r, p in vs if r["label"] == 0]
        per = {}
        for src in sorted({r["source"] for r, _ in vs}):
            fk = [p for r, p in vs if r["source"] == src and r["label"] == 1]
            rl = [p for r, p in vs if r["source"] == src and r["label"] == 0] or reals
            if not fk:
                continue
            y = np.r_[np.zeros(len(rl)), np.ones(len(fk))]
            s = np.r_[rl, fk]
            per[src] = dict(auc=round(float(roc_auc_score(y, s)), 4),
                            bacc=round(float((np.mean(np.array(rl) < 0.5) + np.mean(np.array(fk) >= 0.5)) / 2), 4))
        res[cond] = dict(per_source=per, auc=round(float(np.mean([v["auc"] for v in per.values()])), 4),
                         bacc=round(float(np.mean([v["bacc"] for v in per.values()])), 4))
    res["selection_metric"] = SELECTION_METRIC
    res["score"] = round(validation_score(res, SELECTION_METRIC), 4)
    sc.model.train()
    return res


def trainable(model):
    keys = LN_NAMES[NAME]
    out = {}
    for n, p in model.named_parameters():
        on = n.startswith("model.") or (n.startswith("feature_extractor.") and any(k in n for k in keys))
        p.requires_grad_(on)
        if on:
            out[n] = p
    return out


def loader_options():
    workers = int(os.environ.get("WORKERS", "4"))
    if workers < 0:
        raise ValueError("WORKERS must be nonnegative")
    options = dict(num_workers=workers)
    if workers:
        options.update(persistent_workers=True, prefetch_factor=4)
    return options


def main():
    torch.manual_seed(0)
    sc = Scorer(NAME)
    sc.model.float()
    if os.environ.get("INIT"):
        sd = torch.load(os.environ["INIT"], map_location="cpu")
        sc.model.load_state_dict({**sc.model.state_dict(), **sd})
    params = trainable(sc.model)
    log(event="params", trainable=sum(p.numel() for p in params.values()), tensors=len(params),
        total=sum(p.numel() for p in sc.model.parameters()))
    enc = sc.model.feature_extractor
    layers = getattr(getattr(getattr(enc, "vision_model", None), "encoder", None), "layers", None)
    if layers is not None:                                     # CLIP (transformers): gradient checkpointing per layer
        from functools import partial
        for l in layers:
            l.gradient_checkpointing = True
            l._gradient_checkpointing_func = partial(torch.utils.checkpoint.checkpoint, use_reentrant=False)
    elif hasattr(getattr(enc, "backbone", None), "set_grad_checkpointing"):
        enc.backbone.set_grad_checkpointing(True)
    tr, va = rows("train"), rows("val")
    excluded = set(filter(None, os.environ.get("VAL_EXCLUDE_SOURCES", "").split(",")))
    va = [r for r in va if r["source"] not in excluded]
    validate_split_rows(tr, va)
    if any(r["source"] == "dfdc" for r in tr + va):
        raise ValueError("DFDC is reserved and must not enter training or checkpoint selection")
    if os.environ.get("VAL_LIMIT"):
        va = va[::max(1, len(va) // int(os.environ["VAL_LIMIT"]))]
        validate_split_rows(tr, va)
    log(event="data", train=len(tr), val=len(va), validation_excluded_sources=sorted(excluded),
        by={f"{s}/{l}": sum(1 for r in tr if r["source"] == s and r["label"] == l)
            for s in sorted({r["source"] for r in tr}) for l in (0, 1)})
    items = val_items(va)
    base = evaluate(sc, items)
    log(event="val", step=0, **base)
    best = base["score"]
    torch.save({k: v.detach().cpu().clone() for k, v in params.items()}, OUT / "best.pt")
    opt = torch.optim.Adam(list(params.values()), lr=LR, betas=(0.9, 0.999))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / WARMUP) * (0.01 + 0.99 * 0.5 * (1 + math.cos(math.pi * min(s, STEPS) / STEPS))))
    seed = int(os.environ.get("SEED", "1"))
    if os.environ.get("PAIRS"):
        if BS % 2:
            raise ValueError("paired batches need an even BS")
        with open(os.environ["PAIRS"], encoding="utf-8") as fh:
            dataset = PairedFrames(tr, seed, list(csv.DictReader(fh)))
        log(event="pairing", available=len(dataset.sampler.real_of), unavailable=dataset.sampler.unavailable)
        if len(dataset.sampler.real_of) < int(os.environ.get("MIN_PAIRS", "1")):
            raise ValueError("insufficient prepared real/fake pairs")
    else:
        dataset = Frames(tr, seed)
    dl = torch.utils.data.DataLoader(dataset, batch_size=BS, **loader_options())
    it = iter(dl)
    sc.model.train()
    t0, run = time.time(), []
    for step in range(1, STEPS + 1):
        x, y = next(it)
        x = sc.to_tensor(x.numpy())
        y = y.cuda()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            feat = sc.model.feature_extractor(x)
            logits = sc.model.model(feat)
        z = F.normalize(feat.float(), dim=1)
        ce = F.cross_entropy(logits.float(), y)
        eq = (y[:, None] == y[None, :]).triu(1).nonzero()
        al = (z[eq[:, 0]] - z[eq[:, 1]]).norm(dim=1).pow(2).mean() if len(eq) else z.sum() * 0
        un = torch.pdist(z).pow(2).mul(-2).exp().mean().clamp(min=1e-6).log()
        loss = ce + W_ALIGN * al + W_UNIF * un
        if not torch.isfinite(loss):
            raise ValueError(f"non-finite training loss at step {step}")
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(list(params.values()), 1.0, error_if_nonfinite=True)
        opt.step()
        sched.step()
        run.append([ce.item(), al.item(), un.item(), (logits.argmax(1) == y).float().mean().item()])
        if step % 50 == 0:
            m = np.mean(run[-50:], 0)
            log(event="train", step=step, ce=round(m[0], 4), align=round(m[1], 4), unif=round(m[2], 4),
                acc=round(m[3], 3), lr=sched.get_last_lr()[0], min=round((time.time() - t0) / 60, 1),
                mem=round(torch.cuda.max_memory_allocated() / 1e9, 2))
        if step % VAL_EVERY == 0 or step == STEPS:
            res = evaluate(sc, items)
            log(event="val", step=step, **res)
            torch.save({k: v.detach().cpu().clone() for k, v in params.items()}, OUT / f"step{step}.pt")
            if res["score"] > best:
                best = res["score"]
                torch.save({k: v.detach().cpu().clone() for k, v in params.items()}, OUT / "best.pt")
                log(event="saved", step=step, score=best)
    log(event="done", best=best, base=base["score"])


if __name__ == "__main__":
    main()
