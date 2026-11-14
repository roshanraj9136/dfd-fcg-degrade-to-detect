"""Checkpoint selection at the decision threshold, using validation only."""
import math


def validation_score(result, metric="bacc"):
    """Equal clean/degraded weight. AUROC remains an explicit reproduction option."""
    if metric not in ("bacc", "auc"):
        raise ValueError("selection metric must be bacc or auc")
    values = [float(result[c][metric]) for c in ("clean", "degraded")]
    if any(not math.isfinite(v) or not 0 <= v <= 1 for v in values):
        raise ValueError("validation metrics must be finite values in [0, 1]")
    return sum(values) / len(values)


def validate_split_rows(train, val):
    """Catch accidental train/validation overlap before loading the frame store.

    A shared identity/source-video group is checked when callers supply 'group'.
    Video IDs alone cannot establish identity-disjoint datasets.
    """
    for name, rows in (("train", train), ("val", val)):
        if not rows:
            raise ValueError(f"{name} split is empty")
        if any(not r.get("vid") or r.get("label") not in (0, 1) for r in rows):
            raise ValueError(f"{name} rows need video IDs and binary labels")
        if {r["label"] for r in rows} != {0, 1}:
            raise ValueError(f"{name} split must contain both real and fake videos")
    for key in ("vid", "group"):
        a = {r[key] for r in train if r.get(key)}
        b = {r[key] for r in val if r.get(key)}
        if a & b:
            raise ValueError(f"train/validation overlap in {key}: {sorted(a & b)[:5]}")
