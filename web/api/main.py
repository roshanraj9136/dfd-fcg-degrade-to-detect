"""
Read-only results API + the evidence-nulling correction, served from Neon Postgres.

    GET  /api/health              liveness + database check
    GET  /api/summary             headline numbers and the method comparison table
    GET  /api/analysis            explainability findings (replication, drift, geometry, ...)
    GET  /api/conditions          per-condition metrics for every method and regime
    GET  /api/videos              test videos with their per-condition scores
    GET  /api/videos/{vid}        one video: scores, nuisance split, 16x16 evidence maps
    POST /api/nullify             apply evidence nulling to an exact-attribution vector

Environment: DATABASE_URL (a read-only role is enough), ALLOWED_ORIGINS (comma separated).
"""
import json
import math
import os
from functools import lru_cache
from typing import List

import numpy as np
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
ORIGIN_REGEX = os.environ.get("ALLOWED_ORIGIN_REGEX", r"https://dfd-fcg-evidence-nulling[a-z0-9-]*\.vercel\.app")

app = FastAPI(title="DFD-FCG Evidence Nulling API", version="1.0.0")
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


# ---------------------------------------------------------------- the correction itself
class Attribution(BaseModel):
    """One clip: the three head logits, the background quality descriptors and (optionally) the
    exact per-term contributions (see qa/engine.py and qa/quality.py in the repository)."""
    d: List[float] = Field(..., description="head logits d_s, d_t, d_a (3)")
    q_bg: List[float] = Field(..., description="background-referenced quality descriptors (8)")
    s: List[float] | None = Field(None, description="spatial terms: 24 layers x 4 queries, row-major (96)")
    t: List[float] | None = Field(None, description="temporal terms: 24 layers (24)")
    a: List[float] | None = Field(None, description="fusion terms: 96 spatial + 24 temporal (120)")

    @field_validator("s", "t", "a", "d", "q_bg")
    @classmethod
    def finite(cls, v):
        if v is not None and any((not math.isfinite(x)) or abs(x) > 1e4 for x in v):
            raise ValueError("values must be finite and |x| <= 1e4")
        return v


SIZES = {"s": 96, "t": 24, "a": 120}
PARTS = ["lips", "skin", "eyes", "nose"]


def _term_name(h, k):
    if h == "s":
        return f"spatial L{k // 4 + 1} {PARTS[k % 4]}"
    if h == "t":
        return f"temporal L{k + 1}"
    return f"fusion spatial L{k // 4 + 1} {PARTS[k % 4]}" if k < 96 else f"fusion temporal L{k - 95}"


@lru_cache(maxsize=1)
def _model():
    m = _kv("projection")
    return dict(phi0=np.array(m["phi0"], float), scale=np.array(m["scale"], float),
                W={h: np.array(m["W"][h], float) for h in "sta"})


def _sig(x):
    return 1.0 / (1.0 + math.exp(-x))


@app.post("/api/nullify")
def nullify(x: Attribution):
    if len(x.d) != 3 or len(x.q_bg) != 8:
        raise HTTPException(422, "'d' needs 3 values and 'q_bg' 8 values")
    for h, n in SIZES.items():
        v = getattr(x, h)
        if v is not None and len(v) != n:
            raise HTTPException(422, f"'{h}' must have {n} values")
    m = _model()
    q = np.array(x.q_bg[:7], float)
    z = (np.concatenate([q, q ** 2]) - m["phi0"]) / m["scale"]
    per_term = {h: m["W"][h] @ z for h in "sta"}                 # predicted quality evidence per term
    nuis = [float(per_term[h].sum()) for h in "sta"]
    p0 = sum(_sig(v) for v in x.d) / 3
    p1 = sum(_sig(v - n) for v, n in zip(x.d, nuis)) / 3
    top = sorted(((float(v), h, k) for h in "sta" for k, v in enumerate(per_term[h])), key=lambda t: -abs(t[0]))[:6]
    return dict(p_original=p0, p_corrected=p1, nuisance_logit=nuis,
                verdict_original="FAKE" if p0 >= 0.5 else "REAL",
                verdict_corrected="FAKE" if p1 >= 0.5 else "REAL",
                top_quality_terms=[dict(term=_term_name(h, k), evidence=v) for v, h, k in top])


@app.get("/api/example")
def example():
    """A real attribution vector from the test set (degraded real video), for trying /api/nullify."""
    return _kv("example")
