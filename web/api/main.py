"""
Results API + the self-calibrated decision rule, served from Neon Postgres.

    GET  /api/health              liveness + database check
    GET  /api/summary             headline numbers and the method comparison table
    GET  /api/analysis            explainability findings (replication, drift, fragility, ...)
    GET  /api/conditions          per-condition metrics for every method and regime
    GET  /api/videos              test videos with their per-condition scores
    GET  /api/videos/{m}/{name}   one video: scores and 16x16 evidence maps
    GET  /api/example             a real clip to try /api/decide with
    POST /api/decide              released, self-calibrated and strict verdicts for one clip

Environment: DATABASE_URL (a read-only role is enough), ALLOWED_ORIGINS (comma separated).
"""
import json
import math
import os
from functools import lru_cache
from typing import Dict, List

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator

try:
    from .db import connect
except ImportError:          # run as a plain module (uvicorn main:app)
    from db import connect

DATABASE_URL = os.environ.get("DATABASE_URL", "")
ORIGINS = [o.strip() for o in os.environ.get(
    "ALLOWED_ORIGINS", "http://localhost:5173,http://127.0.0.1:5173").split(",") if o.strip()]
ORIGIN_REGEX = os.environ.get("ALLOWED_ORIGIN_REGEX", r"https://dfd-fcg-degrade-to-detect[a-z0-9-]*\.vercel\.app")

app = FastAPI(title="Self-calibrated DFD-FCG API", version="1.0.0")
app.add_middleware(CORSMiddleware, allow_origins=ORIGINS, allow_origin_regex=ORIGIN_REGEX,
                   allow_methods=["GET", "POST"], allow_headers=["Content-Type"])


def _query(sql, **params):
    if not DATABASE_URL:
        raise HTTPException(503, "database not configured")
    try:
        conn = connect(DATABASE_URL)
        try:
            return conn.run(sql, **params)
        finally:
            conn.close()
    except HTTPException:
        raise
    except Exception as e:  # noqa: BLE001  (driver / network errors -> 503, no details leaked)
        raise HTTPException(503, f"database error: {type(e).__name__}") from e


def _json(v):
    return json.loads(v) if isinstance(v, str) else v


@lru_cache(maxsize=16)
def _kv(key):
    rows = _query("select value from kv where key = :k", k=key)
    if not rows:
        raise HTTPException(404, f"{key} not found")
    return _json(rows[0][0])


@app.get("/api/health")
def health():
    try:
        n = _query("select count(*) from videos")[0][0]
        return {"status": "ok", "database": "ok", "videos": n}
    except HTTPException as e:
        return {"status": "degraded", "database": e.detail}


@app.get("/api/summary")
def summary():
    return _kv("summary")


@app.get("/api/analysis")
def analysis():
    return _kv("analysis")


@app.get("/api/conditions")
def conditions():
    return _kv("conditions")


@app.get("/api/example")
def example():
    return _kv("example")


@app.get("/api/videos")
def videos():
    rows = _query("select vid, split, method, label, public_frames, scores from videos order by vid")
    return [dict(vid=r[0], split=r[1], method=r[2], label=r[3], public=r[4], scores=_json(r[5])) for r in rows]


@app.get("/api/videos/{method}/{name}")
def video(method: str, name: str):
    vid = f"{method}/{name}"
    if len(vid) > 64:
        raise HTTPException(400, "bad id")
    rows = _query("select vid, split, method, label, public_frames, scores, detail from videos where vid = :v",
                  v=vid)
    if not rows:
        raise HTTPException(404, "unknown video")
    r = rows[0]
    return dict(vid=r[0], split=r[1], method=r[2], label=r[3], public=r[4], scores=_json(r[5]),
                detail=_json(r[6]))


# ---------------------------------------------------------------- the decision rule
def _finite(v, n, what):
    if len(v) != n or any((not math.isfinite(x)) or abs(x) > 1e4 for x in v):
        raise ValueError(f"{what}: expected {n} finite values with |x| <= 1e4")
    return v


class Clip(BaseModel):
    """One clip as produced by qa/engine.py: DFD-FCG's head logits d = (d_s, d_t, d_a), the inputs of its
    two LayerNorm heads (y_s: 1024, y_t: 256) and, optionally, the head logits of the blurred copy."""
    d: List[float] = Field(..., description="head logits (3), logit FAKE - logit REAL")
    y_s: List[float] = Field(..., description="spatial head input (1024)")
    y_t: List[float] = Field(..., description="temporal head input (256)")
    d_probe: Dict[str, List[float]] | None = Field(None, description='optional, e.g. {"pB2": [3 logits]}')

    @field_validator("d")
    @classmethod
    def _d(cls, v):
        return _finite(v, 3, "d")

    @field_validator("y_s")
    @classmethod
    def _ys(cls, v):
        return _finite(v, 1024, "y_s")

    @field_validator("y_t")
    @classmethod
    def _yt(cls, v):
        return _finite(v, 256, "y_t")

    @field_validator("d_probe")
    @classmethod
    def _dp(cls, v):
        if v is None:
            return v
        if len(v) > 4:
            raise ValueError("at most 4 probes")
        return {k[:8]: _finite(x, 3, "d_probe") for k, x in v.items()}


def _sig(x):
    return 1.0 / (1.0 + math.exp(-x))


@lru_cache(maxsize=1)
def _model():
    return _kv("model")


@app.post("/api/decide")
def decide(x: Clip):
    m = _model()
    sc = m["selfcal"]
    feats = list(x.y_s) + list(x.y_t)
    z = [(f - mu) / sd for f, mu, sd in zip(feats, sc["mean"], sc["scale"])]
    g = [sum(c * zi for c, zi in zip(row, z)) + b for row, b in zip(sc["coef"], sc["intercept"])]
    p0 = sum(_sig(v) for v in x.d) / 3
    p1 = sum(_sig(v - gi) for v, gi in zip(x.d, g)) / 3
    out = dict(p_released=p0, p_selfcal=p1, spurious_evidence=g,
               verdict_released="FAKE" if p0 >= 0.5 else "REAL",
               verdict_selfcal="FAKE" if p1 >= 0.5 else "REAL")
    if x.d_probe and m["probe"] in x.d_probe:
        r = [b - a for a, b in zip(x.d, x.d_probe[m["probe"]])]
        lam = float(m["strict_lam"])
        p2 = sum(_sig(v - gi - lam * ri) for v, gi, ri in zip(x.d, g, r)) / 3
        out.update(response=r, p_strict=p2, verdict_strict="FAKE" if p2 >= 0.5 else "REAL",
                   reading=("the FAKE evidence fell under the blur probe, which is rare for authentic videos" if r[2] < -0.25
                            else "the FAKE evidence grew under the blur probe, as it does for most authentic videos"
                            if r[2] > 0.25 else "the evidence barely moved under the blur probe"))
    return out
