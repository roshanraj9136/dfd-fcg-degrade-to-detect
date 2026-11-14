"""Round 2b: score the validation videos under two damages applied to the stored aligned face crops, so the combiner
can learn how each detector behaves on low-quality video (results/ensemble/PLAN.md, round 2b).
  blur:  Gaussian blur sigma 2.5 px on the 150-px crop, then JPEG quality 75 (every frame)
  comp:  every 10-frame clip encoded with H.264 CRF 40 (qa.conditions._x264, 3 fps)
Members: dfd (released DFD-FCG + live read-out), dfdft (round-1 checkpoint + its read-out), pe / peft (GenD-PE released /
fine-tuned; 24 evenly spaced frames of the clips, as score_gend.py).
    python service/ens/score_valdmg.py <dfd|dfdft|pe|peft|clip|clipft> <out.npz>
Output for dfd*: keys "vid|cond|0", p (self-calibrated P(fake)); for GenD: keys, start, feats, logit (as score_gend.py).
"""
import csv
import json
import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "service" / "ens")]
import cv2  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

NAME, OUT = sys.argv[1], Path(sys.argv[2])
CROPS = Path(r"D:\dfd-data\crops")
SETS = [Path(r"D:\dfd-data\new\sets.csv"), Path(r"D:\dfd-data\new\sets_gated.csv")]
CONDS = ("blur", "comp")


def _x264(frames, fps, crf, path):
    """Same encoder settings as qa.conditions._x264 (libx264, preset medium), without importing the DFD-FCG code
    (GenD's `src` package would shadow the upstream one)."""
    import av
    out = av.open(path, "w")
    st = out.add_stream("libx264", rate=fps)
    st.width, st.height, st.pix_fmt = frames[0].shape[1], frames[0].shape[0], "yuv420p"
    st.options = {"crf": str(crf), "preset": "medium"}
    for f in frames:
        for pkt in st.encode(av.VideoFrame.from_ndarray(f, format="bgr24")):
            out.mux(pkt)
    for pkt in st.encode():
        out.mux(pkt)
    out.close()
    inp = av.open(path)
    fr = [f.to_ndarray(format="bgr24") for f in inp.decode(video=0)]
    inp.close()
    return fr


def damage(clip, cond, tmp):
    """clip: (10,150,150,3) uint8 RGB."""
    if cond == "blur":
        out = []
        for f in clip:
            g = cv2.GaussianBlur(f, (0, 0), 2.5)
            ok, buf = cv2.imencode(".jpg", cv2.cvtColor(g, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 75])
            out.append(cv2.cvtColor(cv2.imdecode(buf, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB))
        return np.stack(out)
    fr = _x264([np.ascontiguousarray(cv2.cvtColor(f, cv2.COLOR_RGB2BGR)) for f in clip], 3, 40, os.path.join(tmp, "c.mp4"))
    return np.stack([cv2.cvtColor(f, cv2.COLOR_BGR2RGB) for f in fr[:len(clip)]])


def rows():
    out, seen = [], set()
    for f in SETS:
        for r in csv.DictReader(open(f)):
            p = CROPS / (r["vid"].replace("/", "__") + ".npz")
            if r["set"] == "val" and r["source"] != "dfdc" and r["vid"] not in seen and p.exists():
                seen.add(r["vid"])
                out.append((r, p))
    return out


def gend_settings(name):
    if name not in ("pe", "peft", "clip", "clipft"):
        raise ValueError("member must be dfd, dfdft, pe, peft, clip or clipft")
    kind = "clip" if name.startswith("clip") else "pe"
    checkpoint = os.environ.get("GEND_LN")
    if checkpoint is None and name.endswith("ft"):
        checkpoint = str(Path(r"D:\dfd-data\ens") / f"ft_{kind}" / "best.pt")
    return kind, checkpoint


def score(rs, tmp):
    if NAME in ("dfd", "dfdft"):
        if NAME == "dfdft":
            os.environ["DFD_FCG_CKPT"] = r"D:\dfd-data\train_v1\best.ckpt"
        import qa.engine as E
        E.CKPT = Path(os.environ["DFD_FCG_CKPT"])
        mj = ROOT / "service" / "app" / "model.json" if NAME == "dfd" else Path(r"D:\dfd-data\readout_new.json")
        m = json.load(open(mj))["selfcal"]
        mu, sd, W, b = (np.array(m[k]) for k in ("mean", "scale", "coef", "intercept"))
        model = E.load_model(device="cuda")
        rec = E.Recorder(model)
        keys, ps = [], []
        for k, (r, p) in enumerate(rs):
            z = np.load(p)
            for cond in CONDS:
                D, Y = [], []
                clips = [damage(z["face0"][ci], cond, tmp) for ci in z["clip_idx"]]
                for c in range(0, len(clips), 2):
                    x = torch.stack([model.transform(torch.from_numpy(np.ascontiguousarray(cl)).permute(0, 3, 1, 2).contiguous())
                                     for cl in clips[c:c + 2]])
                    with torch.inference_mode():
                        for q in E.analyse(model, rec, x, fp16=True):
                            D.append(q["d"])
                            Y.append(np.r_[q["y_s"], q["y_t"]])
                D, Y = np.array(D), np.array(Y)
                G = ((Y - mu) / sd) @ W.T + b
                keys.append(f"{r['vid']}|{cond}|0")
                ps.append(float((1 / (1 + np.exp(-(D - G)))).mean()))
            if (k + 1) % 50 == 0:
                print(f"{NAME} {k + 1}/{len(rs)}", flush=True)
        np.savez(OUT, keys=np.array(keys), p=np.array(ps))
    else:
        os.environ.setdefault("HF_HUB_CACHE", r"D:\hf-cache")
        from gend import Scorer

        def pick(clip_idx, n):                       # same frames as score_gend.py
            u = np.unique(np.asarray(clip_idx).ravel())
            return u if len(u) <= n else u[np.linspace(0, len(u) - 1, n).round().astype(int)]
        kind, checkpoint = gend_settings(NAME)
        sc = Scorer(kind)
        if checkpoint:
            sd_ = torch.load(checkpoint, map_location="cpu", weights_only=True)
            own = sc.model.state_dict()
            sc.model.load_state_dict({**own, **{k: v.to(own[k].dtype) for k, v in sd_.items()}})
        keys, start, feats, logits, n = [], [], [], [], 0
        for k, (r, p) in enumerate(rs):
            z = np.load(p)
            fr = pick(z["clip_idx"], 24)
            pos = {int(f): i for i, f in enumerate(fr)}
            for cond in CONDS:
                dmg = {}
                for ci in z["clip_idx"]:
                    d = damage(z["face0"][ci], cond, tmp)
                    for j, f in enumerate(ci):
                        if int(f) in pos and int(f) not in dmg:
                            dmg[int(f)] = d[j]
                frames = np.stack([dmg[int(f)] for f in fr])
                f_, lg = sc.features(frames)
                keys.append(f"{r['vid']}|{cond}|0")
                start.append(n)
                n += len(lg)
                feats.append(f_.astype(np.float16))
                logits.append(lg.astype(np.float32))
            if (k + 1) % 50 == 0:
                print(f"{NAME} {k + 1}/{len(rs)}", flush=True)
        np.savez(OUT, keys=np.array(keys), start=np.array(start + [n]), feats=np.concatenate(feats),
                 logit=np.concatenate(logits))
    print("finished", len(rs), "videos", flush=True)


def main():
    if NAME not in ("dfd", "dfdft"):
        gend_settings(NAME)
    rs = rows()
    if os.environ.get("LIMIT"):
        rs = rs[:int(os.environ["LIMIT"])]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="valdmg-", dir=OUT.parent) as tmp:
        score(rs, tmp)


if __name__ == "__main__":
    main()
