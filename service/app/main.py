"""
Video deepfake checker API (Hugging Face Space, CPU).

    GET  /api/health          ready / loading, queue length, videos analysed
    POST /api/analyze         multipart upload (field "file"), optional form field "degrade" = none|blur|compress
                              -> {"job": id}
    GET  /api/jobs/{id}       status, current step, progress, result

Uploaded videos are written to a temporary file, analysed, and deleted. Only anonymous numbers
(verdict, scores, duration) are logged, to Neon Postgres if DATABASE_URL is set.
"""
import os
import queue
import tempfile
import threading
import time
import traceback
import uuid

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

MAX_MB = int(os.environ.get("MAX_UPLOAD_MB", "60"))
MAX_QUEUE = int(os.environ.get("MAX_QUEUE", "6"))
ORIGINS = [o.strip() for o in os.environ.get("ALLOWED_ORIGINS", "*").split(",") if o.strip()]
ALLOWED_EXT = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v", ".3gp"}

app = FastAPI(title="Deepfake video checker (self-calibrated DFD-FCG)")
app.add_middleware(CORSMiddleware, allow_origins=ORIGINS, allow_methods=["GET", "POST"], allow_headers=["*"])

state = dict(ready=False, error=None, analyzer=None, done=0)
jobs = {}
work = queue.Queue()
STEPS = ["queued", "face", "model", "quality", "blur", "done"]


def log_result(res):
    url = os.environ.get("DATABASE_URL")
    if not url:
        return
    try:
        from db import connect
        c = connect(url)
        c.run("insert into analyses (verdict, p_calibrated, p_original, p_strict, blur_response, degrade, seconds, "
              "width, height) values (:v, :p, :po, :ps, :br, :dg, :s, :w, :h)",
              v=res["verdict"], p=res["p"], po=res["p_original"], ps=res["p_strict"],
              br=res["blur_response"][2], dg=res["meta"]["degrade"], s=res["meta"]["seconds"],
              w=res["meta"]["width"], h=res["meta"]["height"])
        c.close()
    except Exception:  # noqa: BLE001  (logging must never break an analysis)
        traceback.print_exc()


def count_logged():
    url = os.environ.get("DATABASE_URL")
    if not url:
        return None
    try:
        from db import connect
        c = connect(url)
        n = c.run("select count(*) from analyses")[0][0]
        c.close()
        return int(n)
    except Exception:  # noqa: BLE001
        return None


def worker():
    try:
        from models import ensure_models
        ensure_models()
        from pipeline import Analyzer
        state["analyzer"] = Analyzer()
        state["ready"] = True
    except Exception as e:  # noqa: BLE001
        state["error"] = f"model failed to load: {e}"
        traceback.print_exc()
        return
    from pipeline import NoFace
    while True:
        jid, path, degrade = work.get()
        job = jobs[jid]
        job.update(status="running", started=time.time())

        def progress(step, frac):
            job.update(step=step, progress=round(float(frac), 3))

        try:
            res = state["analyzer"].run(path, degrade=degrade, progress=progress)
            job.update(status="done", step="done", progress=1.0, result=res)
            state["done"] += 1
            log_result(res)
        except NoFace as e:
            job.update(status="error", error=str(e))
        except Exception as e:  # noqa: BLE001
            traceback.print_exc()
            job.update(status="error", error=f"analysis failed ({type(e).__name__})")
        finally:
            try:
                os.remove(path)
            except OSError:
                pass
            job["finished"] = time.time()
            # forget old jobs (results are only kept for one hour)
            for k in [k for k, j in jobs.items() if j.get("finished", time.time()) < time.time() - 3600]:
                jobs.pop(k, None)


threading.Thread(target=worker, daemon=True).start()


@app.get("/")
def root():
    return {"service": "deepfake video checker", "docs": "/docs", "health": "/api/health"}


@app.get("/api/health")
def health():
    waiting = sum(1 for j in jobs.values() if j["status"] in ("queued", "running"))
    return dict(status="ready" if state["ready"] else ("error" if state["error"] else "loading"),
                error=state["error"], queue=waiting, analysed_since_start=state["done"],
                analysed_total=count_logged())


@app.post("/api/analyze")
async def analyze(file: UploadFile = File(...), degrade: str = Form("none")):
    if degrade not in ("none", "blur", "compress"):
        raise HTTPException(422, "degrade must be none, blur or compress")
    ext = os.path.splitext(file.filename or "")[1].lower()
    if ext not in ALLOWED_EXT:
        raise HTTPException(415, "please upload a video file (mp4, mov, avi, mkv, webm)")
    if sum(1 for j in jobs.values() if j["status"] in ("queued", "running")) >= MAX_QUEUE:
        raise HTTPException(429, "the checker is busy, please try again in a minute")
    fd, path = tempfile.mkstemp(suffix=ext, dir=os.environ.get("UPLOAD_DIR"))
    size = 0
    with os.fdopen(fd, "wb") as fh:
        while chunk := await file.read(1 << 20):
            size += len(chunk)
            if size > MAX_MB << 20:
                fh.close()
                os.remove(path)
                raise HTTPException(413, f"the video is larger than {MAX_MB} MB")
            fh.write(chunk)
    jid = uuid.uuid4().hex[:16]
    jobs[jid] = dict(status="queued", step="queued", progress=0.0, created=time.time(),
                     position=work.qsize() + 1)
    work.put((jid, path, None if degrade == "none" else degrade))
    return {"job": jid}


@app.get("/api/jobs/{jid}")
def job(jid: str):
    j = jobs.get(jid)
    if j is None:
        raise HTTPException(404, "unknown or expired job")
    out = {k: v for k, v in j.items() if k in ("status", "step", "progress", "error", "result")}
    if j["status"] == "queued":
        out["ahead"] = sum(1 for k, x in jobs.items() if x["status"] in ("queued", "running")
                           and x["created"] < j["created"])
    out["ready"] = state["ready"]
    return out
