"""Numerical and tracking checks shared by the live inference pipeline.

Invalid or missing evidence is an analysis error, never a REAL prediction.
These helpers need only NumPy and can be tested without downloading models.
"""
import numpy as np


class InvalidScores(ValueError):
    """The detector did not produce a complete, finite set of scores."""


def finite_array(value, name="scores", ndim=None):
    try:
        a = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise InvalidScores(f"{name} must be numeric") from exc
    if not a.size or not np.isfinite(a).all():
        raise InvalidScores(f"{name} must be nonempty and finite")
    if ndim is not None and a.ndim != ndim:
        raise InvalidScores(f"{name} must have {ndim} dimensions")
    return a


def sigmoid(logits):
    """Stable at large finite logits; reject NaN/Inf instead of propagating them."""
    z = finite_array(logits, "logits")
    e = np.exp(-np.abs(z))
    return np.where(z >= 0, 1.0 / (1.0 + e), e / (1.0 + e))


def probability(value):
    p = finite_array(value, "probability", ndim=0)
    if not 0.0 <= float(p) <= 1.0:
        raise InvalidScores("probability must lie in [0, 1]")
    return float(p)


def logit(value):
    p = probability(value)
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return float(np.log(p) - np.log1p(-p))


def mean_probability(logits):
    return probability(sigmoid(logits).mean())


def validate_readout(cfg, outputs, features):
    """Validate the saved ridge read-out before any video is accepted."""
    try:
        mu, scale, coef, intercept = (
            finite_array(cfg[k], f"readout {k}") for k in ("mean", "scale", "coef", "intercept")
        )
    except (KeyError, TypeError) as exc:
        raise InvalidScores("incomplete calibration readout") from exc
    expected = ((features,), (features,), (outputs, features), (outputs,))
    if tuple(a.shape for a in (mu, scale, coef, intercept)) != expected:
        raise InvalidScores("calibration readout dimensions do not match the detector")
    if (scale <= 0).any():
        raise InvalidScores("calibration scales must be positive")
    return mu, scale, coef, intercept


def readout_shift(features, mu, scale, coef, intercept):
    y = finite_array(features, "features", ndim=2)
    if y.shape[1] != mu.size:
        raise InvalidScores("feature dimensions do not match the calibration readout")
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        shift = ((y - mu) / scale) @ coef.T + intercept
    return finite_array(shift, "calibration correction", ndim=2)


def validate_fusion(cfg):
    if not isinstance(cfg, dict):
        raise InvalidScores("fusion must be an object")
    members = cfg.get("members")
    if (not isinstance(members, list) or not members
            or any(not isinstance(m, str) or not m for m in members)
            or len(set(members)) != len(members)):
        raise InvalidScores("fusion members must be unique nonempty names")
    coef = finite_array(cfg.get("coef"), "fusion coefficients", ndim=1)
    if coef.shape != (len(members),):
        raise InvalidScores("one fusion coefficient is required for each member")
    finite_array(cfg.get("intercept"), "fusion intercept", ndim=0)
    detectors = cfg.get("detectors", {})
    if not isinstance(detectors, dict):
        raise InvalidScores("fusion detectors must be an object")
    for member in members:
        if member != "dfd" and not isinstance(detectors.get(member), dict):
            raise InvalidScores(f"missing detector configuration: {member}")
    return cfg


def fusion_logit(cfg, member_logits):
    """Require every configured member; never silently downgrade the ensemble."""
    validate_fusion(cfg)
    missing = set(cfg["members"]) - member_logits.keys()
    if missing:
        raise InvalidScores(f"missing detector scores: {', '.join(sorted(missing))}")
    x = finite_array([member_logits[m] for m in cfg["members"]], "member logits", ndim=1)
    with np.errstate(over="ignore", invalid="ignore"):
        z = np.dot(cfg["coef"], x) + cfg["intercept"]
    return float(finite_array(z, "fusion logit", ndim=0))


def valid_box(box):
    b = np.asarray(box, dtype=float)
    return bool(b.shape == (4,) and np.isfinite(b).all() and b[2] > b[0] and b[3] > b[1])


def track_candidate(candidates, previous):
    """Largest initial face, then nearest face within one previous face width.

    Apply the same displacement gate to the main and secondary tracks. This
    prevents a disappearing main face from jumping to a distant bystander.
    It is a geometric gate, not an identity recognizer.
    """
    indices = [i for i, b in enumerate(candidates) if valid_box(b)]
    if not indices:
        return None
    if previous is None:
        return max(indices, key=lambda i: (candidates[i][2] - candidates[i][0])
                   * (candidates[i][3] - candidates[i][1]))
    if not valid_box(previous):
        return None
    prev = np.asarray(previous, dtype=float)
    centre = (prev[:2] + prev[2:]) / 2
    width = max(prev[2] - prev[0], prev[3] - prev[1])
    distances = {i: np.linalg.norm((np.asarray(candidates[i])[:2] + np.asarray(candidates[i])[2:]) / 2 - centre)
                 for i in indices}
    nearest = min(indices, key=distances.get)
    return nearest if distances[nearest] <= width else None
