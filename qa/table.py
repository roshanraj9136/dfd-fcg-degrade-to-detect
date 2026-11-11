"""
Flatten the cached per-(video, condition) results into one clip-level table.

    python -m qa.table          # writes data/table.npz
"""
import numpy as np

from qa.config import DATA
from qa.extract import load_feats
from qa.manifest import read_manifest

ARRAYS = ["d", "const", "cs_lq", "ct_l", "ca_lq", "ca_tl", "cs_patch", "ct_patch", "ca_patch",
          "y_s", "y_t"]


def build():
    from qa.backfill_qbg import load_qbg
    cols = {k: [] for k in ARRAYS}
    meta = {k: [] for k in ["vid", "split", "method", "label", "cond", "kind", "level", "clip"]}
    p_final, model_p, q, qb = [], [], [], []
    for r in read_manifest():
        store = load_feats(r["vid"])
        extra = load_qbg(r["vid"]) if store and any("q_bg" not in e for e in store.values()) else {}
        for cond, entry in store.items():
            kind, level = ("clean", 0) if cond == "clean" else (cond.split("_")[0], int(cond.split("_")[1]))
            q_bg = entry.get("q_bg")
            if q_bg is None:
                q_bg = extra.get(cond, {}).get("q_bg")
            for ci, rec in enumerate(entry["clips"]):
                for k in ARRAYS:
                    cols[k].append(np.asarray(rec[k], np.float32))
                p_final.append(rec["p_final"])
                model_p.append(rec["model_p"])
                q.append(entry["q"][ci])
                qb.append(q_bg[ci] if q_bg is not None else np.full(8, np.nan, np.float32))
                for k, v in [("vid", r["vid"]), ("split", r["split"]), ("method", r["method"]),
                             ("label", int(r["label"])), ("cond", cond), ("kind", kind),
                             ("level", level), ("clip", ci)]:
                    meta[k].append(v)
    out = {k: np.stack(v) for k, v in cols.items()}
    out.update({k: np.array(v) for k, v in meta.items()})
    out["p_final"] = np.array(p_final, np.float64)
    out["model_p"] = np.array(model_p, np.float64)
    out["q"] = np.stack(q).astype(np.float32)
    out["q_bg"] = np.stack(qb).astype(np.float32)
    return out


def load():
    z = np.load(DATA / "table.npz", allow_pickle=False)
    return {k: z[k] for k in z.files}


def main():
    t = build()
    np.savez_compressed(DATA / "table.npz", **t)
    n_v = len(set(t["vid"].tolist()))
    print(f"{len(t['vid'])} clips, {n_v} videos, conditions: {sorted(set(t['cond'].tolist()))}")


if __name__ == "__main__":
    main()
