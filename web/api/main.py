"""
Results API + the Degrade-to-Detect decision rule, served from Neon Postgres.

    GET  /api/health              liveness + database check
    GET  /api/summary             headline numbers and the method comparison table
    GET  /api/analysis            explainability findings (replication, drift, fragility, ...)
    GET  /api/conditions          per-condition metrics for every method and regime
    GET  /api/videos              test videos with their per-condition scores
    GET  /api/videos/{m}/{name}   one video: scores and 16x16 evidence maps
    GET  /api/example             a real clip to try /api/decide with
    POST /api/decide              Degrade-to-Detect verdict from the model's head logits on the
                                  input and on its probed copy

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

app = FastAPI(title="DFD-FCG Degrade-to-Detect API", version="1.0.0")
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
def _finite3(v):
    if len(v) != 3 or any((not math.isfinite(x)) or abs(x) > 1e4 for x in v):
        raise ValueError("expected 3 finite values with |x| <= 1e4")
    return v


class Clip(BaseModel):
    """Head logits (d_s, d_t, d_a = logit FAKE - logit REAL) of DFD-FCG on one clip and on the
    same clip after the probe(s) of qa/probes.py were applied to the aligned face crops."""
    d: List[float] = Field(..., description="head logits on the input clip (3)")
    d_probe: Dict[str, List[float]] = Field(..., description='e.g. {"pB2": [3 logits], "pJ50": [3 logits]}')

    @field_validator("d")
    @classmethod
    def _d(cls, v):
        return _finite3(v)

    @field_validator("d_probe")
    @classmethod
    def _dp(cls, v):
        if len(v) > 4:
            raise ValueError("at most 4 probes")
        return {k[:8]: _finite3(x) for k, x in v.items()}


def _sig(x):
    return 1.0 / (1.0 + math.exp(-x))


@app.post("/api/decide")
def decide(x: Clip):
    m = _kv("model")
    probe, lam = m["probe"], float(m["lam"])
    if probe not in x.d_probe:
        raise HTTPException(422, f"d_probe must contain '{probe}'")
    r = [b - a for a, b in zip(x.d, x.d_probe[probe])]
    p0 = sum(_sig(v) for v in x.d) / 3
    p1 = sum(_sig(v - lam * ri) for v, ri in zip(x.d, r)) / 3
    out = dict(p_released=p0, p_degrade_to_detect=p1, response=r, lam=lam,
               verdict_released="FAKE" if p0 >= 0.5 else "REAL",
               verdict_degrade_to_detect="FAKE" if p1 >= 0.5 else "REAL",
               reading=("the fake evidence collapsed under the probe" if r[2] < -0.25 else
                        "the evidence grew under the probe, typical of authentic detail being removed" if r[2] > 0.25
                        else "the evidence barely moved under the probe"))
    lr = m["lr"]
    if all(p in x.d_probe for p in lr["probes"]):
        feats = list(x.d) + [b - a for p in lr["probes"] for a, b in zip(x.d, x.d_probe[p])]
        z = sum(c * (f - mu) / sd for c, f, mu, sd in zip(lr["coef"], feats, lr["mean"], lr["scale"])) + lr["intercept"]
        out["p_learned"] = _sig(z)
    return out
