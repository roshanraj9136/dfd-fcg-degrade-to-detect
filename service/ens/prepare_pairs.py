"""Complete FF++ real counterparts for already-prepared training fakes.

Only official train originals are eligible; existing fit/selection/test videos
and any non-train manifest entry are excluded. Existing manifests are read-only.
Outputs are separate manifests consumed by prep_crops.py and build_store.py.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import time

from weighted_fusion import metadata


def counterpart(vid):
    method, name = vid.split("/")
    parts = name.split("_")
    if method not in ("FF_DF", "FF_F2F", "FF_FS", "FF_NT") or len(parts) != 2 \
            or any(len(p) != 3 or not p.isdigit() for p in parts):
        raise ValueError(f"unsupported FF++ fake ID: {vid}")
    return "FF_REAL/" + parts[0]


def eligible_pairs(fakes, meta, train_ids, reserved):
    pairs, reals, rejected = [], {}, []
    for fake in fakes:
        if fake["set"] != "train" or fake["source"] != "ffpp" or fake["label"] != 1:
            raise ValueError("pair construction only accepts FF++ training fakes")
        real_id = counterpart(fake["vid"])
        original = real_id.split("/")[1]
        known = meta.get(real_id)
        if original not in train_ids or original in reserved or (known and known["set"] != "train"):
            rejected.append(fake["vid"])
            continue
        if known and (known["label"] != 0 or known["source"] != "ffpp"):
            raise ValueError(f"conflicting real counterpart metadata: {real_id}")
        reals[real_id] = known or dict(
            vid=real_id, set="train", source="ffpp", method="REAL", label=0,
            repo=fake["repo"], path="faceforensics_c23/FaceForensics++_C23/original/" + original + ".mp4")
        pairs.append(dict(real_vid=real_id, fake_vid=fake["vid"], source="ffpp"))
    return pairs, list(reals.values()), rejected


def write_csv(path, rows, columns):
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def download(reals, data):
    from huggingface_hub import hf_hub_download
    for i, row in enumerate(reals):
        target = data / "new" / "videos" / "ffpp" / row["path"]
        if not target.exists():
            for attempt in range(5):
                try:
                    hf_hub_download(row["repo"], row["path"], repo_type="dataset",
                                    local_dir=data / "new" / "videos" / "ffpp")
                    break
                except Exception:
                    if attempt == 4:
                        raise
                    time.sleep(5 * (attempt + 1))
        print(f"real {i + 1}/{len(reals)} available", flush=True)
    print("download finished", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path(r"D:\dfd-data"))
    parser.add_argument("--protocol", type=Path, default=Path(r"C:\Users\hp\dfd-work\data"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--resume-download", action="store_true")
    args = parser.parse_args()
    if args.resume_download:
        audit = json.loads((args.out / "audit.json").read_text(encoding="utf-8"))
        if any(hashlib.sha256(Path(p).read_bytes()).hexdigest() != digest for p, digest in audit["snapshot"].items()):
            raise ValueError("source manifests changed after pair preparation")
        download(list(csv.DictReader((args.out / "reals.csv").open(encoding="utf-8"))), args.data)
        return
    if (args.out / "audit.json").exists():
        raise FileExistsError("pair manifest already frozen; use a new output directory")
    meta = metadata(args.data)
    fakes = [r for r in meta.values() if r["set"] == "train" and r["source"] == "ffpp" and r["label"] == 1
             and (args.data / "crops" / (r["vid"].replace("/", "__") + ".npz")).exists()]
    split = args.protocol / "splits" / "train.json"
    train_ids = {v for p in json.loads(split.read_text()) for v in p}
    manifests = [args.protocol / n for n in ("manifest_full.csv", "manifest.csv", "manifest_select.csv")]
    reserved = {r["vid"].split("/")[1] for f in manifests for r in csv.DictReader(f.open())
                if r["vid"].startswith("REAL/")}
    pairs, reals, rejected = eligible_pairs(fakes, meta, train_ids, reserved)
    args.out.mkdir(parents=True, exist_ok=True)
    write_csv(args.out / "reals.csv", reals, ("vid", "set", "source", "method", "label", "repo", "path"))
    write_csv(args.out / "pairs.csv", pairs, ("real_vid", "fake_vid", "source"))
    files = [split, *manifests, *(args.data / "new" / n for n in ("sets.csv", "sets_gated.csv", "sets_more.csv"))]
    audit = dict(prepared_fakes=len(fakes), eligible_pairs=len(pairs), real_videos=len(reals),
                 excluded_reserved_fakes=rejected,
                 real_crops_already_present=sum((args.data / "crops" / (r["vid"].replace("/", "__") + ".npz")).exists()
                                                for r in reals),
                 snapshot={str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in files})
    (args.out / "audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    print(json.dumps(audit, indent=2), flush=True)
    if args.download:
        download(reals, args.data)


if __name__ == "__main__":
    main()
