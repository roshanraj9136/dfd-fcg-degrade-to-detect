"""Fine-tune DFD-FCG on newer data (CLIP ViT-L/14 stays frozen, exactly as in the paper; only the authors' side network
and heads, 0.25 M parameters, are trained, with the authors' loss: focal loss on the three heads + face-feature-guided
attention loss).

Data: the aligned face clips written by prep_crops.py for the "train" and "val" sets of make_sets.py /
make_sets_gated.py (DFDC excluded, see load_rows). Every test video, our FF++ test / fit / selection videos and DFDC
are never seen.

Each step draws a clip with label real/fake 50:50, then a source uniformly among the sources that have that label,
then a video and one of its 4 clips. Augmentations (same for all 10 frames of a clip): mirror (p 0.5), one quality
change (p 0.6: JPEG 30-95, Gaussian blur sigma 0.4-2.0, down/up-scaling x1.5-3, noise sigma 2-8), brightness/contrast
(p 0.3).

The released checkpoint is evaluated on the validation clips first (step 0); the checkpoint with the best validation
score is kept. Score = mean over sources of balanced accuracy at 0.5 (fake-only sources are paired with all validation
real videos), on clean and on degraded validation clips.
    python service/train/train.py <out dir> [steps=3000] [lr=3e-5]
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
sys.path[:0] = [str(ROOT)]
import cv2  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from qa.config import CKPT  # noqa: E402
from qa.engine import load_model  # noqa: E402

OUT = Path(sys.argv[1])
STEPS = int(sys.argv[2]) if len(sys.argv) > 2 else 3000
LR = float(sys.argv[3]) if len(sys.argv) > 3 else 3e-5
ACCUM, BS, VAL_EVERY = int(os.environ.get("ACCUM", "4")), int(os.environ.get("BS", "2")), int(os.environ.get("VAL_EVERY", "500"))
VAL_LIMIT = int(os.environ.get("VAL_LIMIT", "0"))       # smoke tests only
CROPS = Path(r"D:\dfd-data\crops")
SETS = [Path(r"D:\dfd-data\new\sets.csv"), Path(r"D:\dfd-data\new\sets_gated.csv")]
OUT.mkdir(parents=True, exist_ok=True)
LOG = open(OUT / "train_log.jsonl", "a", encoding="utf-8")


def log(**kw):
    print(json.dumps(kw), flush=True)
    LOG.write(json.dumps(kw) + "\n")
    LOG.flush()


def load_rows(set_name):
    """Rows of one set with prepared clips. DFDC is not used: its parts can share actors with the DFDC test part, so
    training on it could leak into the reported DFDC test videos."""
    rows = []
    for f in SETS:
        for r in csv.DictReader(open(f)):
            p = CROPS / (r["vid"].replace("/", "__") + ".npz")
            if r["set"] == set_name and r["source"] != "dfdc" and p.exists():
                rows.append(dict(r, npz=p, label=int(r["label"])))
    return rows


# ------------------------------------------------------------------ augmentation (uint8 RGB clips, T x 150 x 150 x 3)
def degrade(clip, rng, kind=None):
    kind = kind or rng.choice(["jpeg", "blur", "scale", "noise"])
    if kind == "jpeg":
        q = int(rng.uniform(30, 95))
        out = []
        for f in clip:
            ok, buf = cv2.imencode(".jpg", cv2.cvtColor(f, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, q])
            out.append(cv2.cvtColor(cv2.imdecode(buf, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB))
        return np.stack(out)
    if kind == "blur":
        s = rng.uniform(0.4, 2.0)
        return np.stack([cv2.GaussianBlur(f, (0, 0), s) for f in clip])
    if kind == "scale":
        k = rng.uniform(1.5, 3.0)
        h, w = clip.shape[1:3]
        return np.stack([cv2.resize(cv2.resize(f, (int(w / k), int(h / k)), interpolation=cv2.INTER_AREA), (w, h),
                                    interpolation=cv2.INTER_LINEAR) for f in clip])
    s = rng.uniform(2, 8)
    return np.clip(clip.astype(np.float32) + np.random.default_rng(rng.randrange(1 << 30)).normal(0, s, clip.shape),
                   0, 255).astype(np.uint8)


def augment(clip, rng):
    if rng.random() < 0.5:
        clip = clip[:, :, ::-1]
    if rng.random() < 0.6:
        clip = degrade(np.ascontiguousarray(clip), rng)
    if rng.random() < 0.3:
        a, b = rng.uniform(0.85, 1.15), rng.uniform(-20, 20)
        clip = np.clip(clip.astype(np.float32) * a + b, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(clip)


class Clips(torch.utils.data.Dataset):
    """Infinite balanced sampler: index -> (frames (10,3,224,224), label)."""

    def __init__(self, rows, transform, seed):
        self.by = {}
        for r in rows:
            self.by.setdefault(r["label"], {}).setdefault(r["source"], []).append(r)
        self.transform, self.seed = transform, seed

    def __len__(self):
        return 10 ** 8

    def __getitem__(self, i):
        rng = random.Random(self.seed * 1_000_003 + i)
        lab = rng.randrange(2)
        src = rng.choice(sorted(self.by[lab]))
        r = rng.choice(self.by[lab][src])
        z = np.load(r["npz"])
        ci = z["clip_idx"][rng.randrange(len(z["clip_idx"]))]
        clip = augment(z["face0"][ci], rng)
        x = self.transform(torch.from_numpy(clip).permute(0, 3, 1, 2).contiguous())
        return x, lab


# ------------------------------------------------------------------ validation
def val_clips(rows, transform):
    """Fixed validation set: clips 0 and 2 of every video, clean and with one fixed degradation each (kept as uint8;
    the CLIP transform is applied batch by batch in evaluate, to keep memory small)."""
    items = []
    for r in rows:
        z = np.load(r["npz"])
        rng = random.Random(zlib.crc32(r["vid"].encode()))
        for k in (0,):
            if k >= len(z["clip_idx"]):
                continue
            clip = z["face0"][z["clip_idx"][k]]
            for cond in ("clean", "degraded"):
                c = clip if cond == "clean" else degrade(clip, rng, ["jpeg", "blur", "scale"][k % 3])
                items.append((r, cond, np.ascontiguousarray(c)))
    return items


@torch.inference_mode()
def evaluate(model, items):
    model.eval()
    probs = []
    for k in range(0, len(items), 4):
        x = torch.stack([model.transform(torch.from_numpy(it[2]).permute(0, 3, 1, 2).contiguous())
                         for it in items[k:k + 4]]).cuda()
        with torch.autocast("cuda", dtype=torch.float16):
            out = model.model(x)
        probs += out["logits"].float().softmax(-1)[:, 1].cpu().tolist()
    vids = {}
    for (r, cond, _), p in zip(items, probs):
        vids.setdefault((r["vid"], cond), dict(r=r, cond=cond, p=[]))["p"].append(p)
    res = {}
    for cond in ("clean", "degraded"):
        vs = [v for v in vids.values() if v["cond"] == cond]
        real_all = [np.mean(v["p"]) < 0.5 for v in vs if v["r"]["label"] == 0]
        per = {}
        for src in sorted({v["r"]["source"] for v in vs}):
            reals = [np.mean(v["p"]) < 0.5 for v in vs if v["r"]["source"] == src and v["r"]["label"] == 0]
            fakes = [np.mean(v["p"]) >= 0.5 for v in vs if v["r"]["source"] == src and v["r"]["label"] == 1]
            tnr = np.mean(reals) if reals else np.mean(real_all)
            per[src] = round(float((tnr + np.mean(fakes)) / 2), 4) if fakes else round(float(np.mean(reals)), 4)
        res[cond] = dict(per_source=per, mean=round(float(np.mean(list(per.values()))), 4))
    res["score"] = round((res["clean"]["mean"] + res["degraded"]["mean"]) / 2, 4)
    model.train()
    return res


def save(model, path):
    ck = torch.load(CKPT, map_location="cpu", weights_only=False)
    sd = model.state_dict()
    for k in ck["state_dict"]:
        if not k.startswith("model.encoder.model."):
            ck["state_dict"][k] = sd[k].detach().cpu().to(ck["state_dict"][k].dtype)
    for k in ("optimizer_states", "lr_schedulers", "loops", "callbacks"):
        ck.pop(k, None)
    torch.save(ck, path)


def main():
    torch.manual_seed(0)
    model = load_model(device="cuda")
    model.log = lambda *a, **k: None                      # no Lightning trainer here
    train_rows, val_rows = load_rows("train"), load_rows("val")
    if VAL_LIMIT:
        val_rows = val_rows[:VAL_LIMIT]
    log(event="data", train=len(train_rows), val=len(val_rows),
        train_by={f"{s}/{l}": sum(1 for r in train_rows if r["source"] == s and r["label"] == l)
                  for s in sorted({r["source"] for r in train_rows}) for l in (0, 1)})
    items = val_clips(val_rows, model.transform)
    base = evaluate(model, items)
    log(event="val", step=0, **base)
    best = base["score"]
    save(model, OUT / "best.ckpt")
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=LR, weight_decay=1e-3)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / 100) * 0.5 * (1 + math.cos(math.pi * min(s, STEPS) / STEPS)))
    scaler = torch.amp.GradScaler("cuda")
    ds = Clips(train_rows, model.transform, seed=1)
    nw = 2
    try:
        dl = torch.utils.data.DataLoader(ds, batch_size=BS, num_workers=nw, persistent_workers=True, prefetch_factor=4)
        it = iter(dl)
        next(it)
    except Exception as e:  # noqa: BLE001  (transform not picklable etc.)
        log(event="loader", note=f"workers off: {type(e).__name__}")
        dl = torch.utils.data.DataLoader(ds, batch_size=BS, num_workers=0)
        it = iter(dl)
    model.train()
    t0, run = time.time(), []
    for step in range(1, STEPS + 1):
        opt.zero_grad(set_to_none=True)
        for _ in range(ACCUM):
            x, y = next(it)
            batch = dict(xyz=(x.cuda(non_blocking=True), y.cuda(), {}), indices=torch.zeros(len(y)), dts_name="train",
                         names=["x"] * len(y))
            with torch.autocast("cuda", dtype=torch.float16):
                loss = model.shared_step(batch, "train")["loss"] / ACCUM
            scaler.scale(loss).backward()
            run.append(loss.item() * ACCUM)
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(params, 0.1)
        scaler.step(opt)
        scaler.update()
        sched.step()
        if step % 50 == 0:
            log(event="train", step=step, loss=round(float(np.mean(run[-200:])), 4), lr=sched.get_last_lr()[0],
                min=round((time.time() - t0) / 60, 1))
        if step % VAL_EVERY == 0 or step == STEPS:
            res = evaluate(model, items)
            log(event="val", step=step, **res)
            if res["score"] > best:
                best = res["score"]
                save(model, OUT / "best.ckpt")
                log(event="saved", step=step, score=best)
    log(event="done", best=best, base=base["score"])


if __name__ == "__main__":
    main()
