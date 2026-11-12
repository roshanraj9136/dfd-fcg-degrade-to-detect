"""
Deepfake video checker: Hugging Face Space on ZeroGPU (free GPU per request).

    DFD-FCG (Han et al., CVPR 2025, released checkpoint)
  + self-calibration (our add-on): a linear read-out of the model's own features estimates how much FAKE
    evidence the video's blur / compression added, and removes it
  + blur test and per-region evidence map as a short explanation.

Face finding and alignment run on the Space's CPU; only the DFD-FCG passes book the GPU (8 clips from the first 10 s
plus the blur test, for every person in the video). When a visitor's free ZeroGPU time is used up, or no GPU is free,
those passes run on the CPU instead (same scores, slower).

API (used by the website through @gradio/client):
    /analyze  (file, degrade: none|blur|compress) -> result dict
    /health   () -> status dict

Uploaded videos are deleted right after the check; only anonymous numbers are logged (Neon Postgres).
"""
import os
import subprocess
import sys
import threading
import time
import traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent
UPSTREAM = Path(os.environ.get("DFD_FCG_UPSTREAM", "/tmp/DFD-FCG"))
UPSTREAM_COMMIT = "3c0b8408ea5027b2b642d8d17ea13b8aa43e02b4"     # aiiu-lab/DFD-FCG, the version we evaluated
os.environ["DFD_FCG_UPSTREAM"] = str(UPSTREAM)
os.environ.setdefault("DFD_DATA", "/tmp/dfd-data")
os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")
os.environ.setdefault("DET_EVERY", "12")      # face finding runs on the CPU (no GPU quota): detector every
os.environ.setdefault("LM_EVERY", "3")        # 12th frame, landmarks every 3rd frame and on every frame the model reads
sys.path[:0] = [str(HERE), str(HERE / "checker")]

import spaces  # noqa: E402  (must come before torch on ZeroGPU)

if not (UPSTREAM / "src").exists():
    UPSTREAM.mkdir(parents=True, exist_ok=True)
    for cmd in (["git", "init", "-q"], ["git", "remote", "add", "origin", "https://github.com/aiiu-lab/DFD-FCG"],
                ["git", "fetch", "-q", "--depth", "1", "origin", UPSTREAM_COMMIT], ["git", "checkout", "-q", "FETCH_HEAD"]):
        subprocess.run(cmd, cwd=UPSTREAM, check=True)

import gradio as gr  # noqa: E402
import torch  # noqa: E402
from models import ensure_models  # noqa: E402

t0 = time.time()
ensure_models()
from pipeline import FLIP_TTA, Analyzer, NoFace  # noqa: E402

ZERO = os.environ.get("SPACES_ZERO_GPU", "").lower() in ("1", "true")
DEVICE = "cuda" if ZERO or torch.cuda.is_available() else "cpu"
AN = Analyzer(device=DEVICE, face_device="cpu")
# CPU copy of DFD-FCG for visitors whose free ZeroGPU time is used up (or when no GPU is free): same scores,
# a few seconds slower (on the Space's 16 CPU cores both passes take about 7 s, on the GPU about 2 s).
# Loaded here, before the first GPU worker is forked, so a fork never copies a half-done load.
AN_CPU = AN.cpu_twin() if DEVICE == "cuda" else AN
print(f"[checker] models ready on {DEVICE} (+cpu fallback) in {time.time() - t0:.0f}s; cpu_count={os.cpu_count()} "
      f"affinity={len(os.sched_getaffinity(0)) if hasattr(os, 'sched_getaffinity') else '?'} threads={AN.threads}",
      flush=True)
ALLOWED_EXT = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v", ".3gp"}
MAX_MB = 60
state = dict(done=0, cpu_fallbacks=0)
# progress messages: "<step id>: <text>"; the website reads the step id, the Gradio page shows the text
STEP_TEXT = {"face": "face: finding and aligning the face", "landmarks": "face: finding and aligning the face",
             "gpu": "gpu: waiting for a free GPU", "cpu": "cpu: no free GPU time left, using the server CPU",
             "quality": "quality: quality check"}


def _gpu_seconds(crops, clip_idx):
    """GPU time to book for one check: grows with the number of DFD-FCG passes (faces x clips x views + blur test).
    ZeroGPU checks the booked time against the visitor's remaining quota, so it is kept close to the real need."""
    passes = len(crops) * len(clip_idx) * (3 if FLIP_TTA else 2)
    return int(min(90, 10 + 0.5 * passes))


@spaces.GPU(duration=_gpu_seconds)   # only the DFD-FCG passes use the GPU; face finding stays on the CPU
def _run_gpu(crops, clip_idx):
    return AN.infer(crops, clip_idx)


def _model_passes(st, progress):
    """DFD-FCG + blur test on a ZeroGPU GPU; on the CPU when the GPU cannot be used for this visitor."""
    if AN_CPU is not AN:
        progress(0.0, desc=STEP_TEXT["gpu"])
        try:
            return _run_gpu(st["crops"], st["clip_idx"]), "cuda"
        except Exception as e:  # noqa: BLE001  ZeroGPU raises gr.Error for used-up quota, no free GPU, aborted task
            if "closed by visitor" in str(e).lower():
                raise
            print(f"[checker] GPU not usable ({type(e).__name__}: {str(e)[:200]}); running on the CPU", flush=True)
            state["cpu_fallbacks"] += 1
    progress(0.0, desc=STEP_TEXT["cpu"])
    return AN_CPU.infer(st["crops"], st["clip_idx"]), "cpu"


def _log(res):
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
    except Exception:  # noqa: BLE001  (logging must never break a check)
        traceback.print_exc()


def analyze(file, degrade="none", progress=gr.Progress()):
    """Check one video. Returns the verdict, scores and a short explanation (two small PNG images)."""
    path = file if isinstance(file, str) else getattr(file, "name", None) or (file or {}).get("path")
    if not path:
        raise gr.Error("Please upload a video.")
    try:
        if degrade not in ("none", "blur", "compress"):
            raise gr.Error("degrade must be none, blur or compress")
        if os.path.splitext(path)[1].lower() not in ALLOWED_EXT:
            raise gr.Error("Please upload a video file (mp4, mov, avi, mkv or webm).")
        if os.path.getsize(path) > MAX_MB << 20:
            raise gr.Error(f"The video is larger than {MAX_MB} MB.")
        t = time.time()
        try:
            st = AN.prepare(path, None if degrade == "none" else degrade,          # CPU: face, alignment, crops
                            lambda step, frac: progress(frac, desc=STEP_TEXT.get(step, STEP_TEXT["face"])))
        except NoFace as e:
            raise gr.Error(str(e)) from None
        except Exception as e:  # noqa: BLE001
            traceback.print_exc()
            raise gr.Error(f"The video could not be processed ({type(e).__name__}). Please try another video.") from None
        t_model = time.time()
        try:
            (outs, model_s), used = _model_passes(st, progress)        # DFD-FCG (+ mirrored view) + blur test
        except gr.Error:
            raise
        except Exception as e:  # noqa: BLE001
            traceback.print_exc()
            raise gr.Error(f"The analysis failed ({type(e).__name__}). Please try again.") from None
        model_wall = round(time.time() - t_model, 1)
        progress(1.0, desc=STEP_TEXT["quality"])
        res = AN.finish(st, outs, model_s)
        res["meta"]["gpu_seconds" if used == "cuda" else "cpu_model_seconds"] = model_wall
        res["meta"]["seconds"] = round(time.time() - t, 1)
        res["meta"]["device"] = used
        print(f"[checker] {res['verdict']} p={res['p']:.3f} total={res['meta']['seconds']}s "
              f"{used}={model_wall}s timing={res['meta'].get('timing')}", flush=True)
        state["done"] += 1
        threading.Thread(target=_log, args=(res,), daemon=True).start()
        return res
    finally:
        try:
            os.remove(path)
        except OSError:
            pass


def health():
    """Service status and the number of videos checked so far."""
    total = None
    url = os.environ.get("DATABASE_URL")
    if url:
        try:
            from db import connect
            c = connect(url, timeout=5)
            total = int(c.run("select count(*) from analyses")[0][0])
            c.close()
        except Exception:  # noqa: BLE001
            pass
    return dict(status="ready", device=DEVICE, analysed_since_start=state["done"], analysed_total=total,
                cpu_fallbacks_since_start=state["cpu_fallbacks"])


def _img(data_url):
    import base64
    import io

    import numpy as np
    from PIL import Image
    return np.array(Image.open(io.BytesIO(base64.b64decode(data_url.split(",", 1)[1]))).convert("RGB"))


def ui_check(video, degrade, progress=gr.Progress()):
    res = analyze(video, {"As it is": "none", "Add strong blur": "blur", "Add heavy compression": "compress"}[degrade],
                  progress)
    fake = res["verdict"] == "FAKE"
    md = (f"## {'🔴 Looks like a deepfake' if fake else '🟢 Looks like a real video'}"
          f"{'' if res['sure'] else ' (not sure)'}\n"
          f"**{res['p'] * 100:.0f}% fake** after the quality correction "
          f"(original DFD-FCG: {res['p_original'] * 100:.0f}%). Blur test: {res['blur_response'][2]:+.2f} "
          f"(real faces usually react with a positive number).")
    small = {k: v for k, v in res.items() if not k.endswith("_png")}
    return md, _img(res["face_png"]), _img(res["heat_png"]), small


with gr.Blocks(title="Deepfake Video Checker", delete_cache=(1800, 3600)) as demo:
    gr.Markdown("# Deepfake Video Checker\nSelf-calibrated DFD-FCG. Website: "
                "[dfd-fcg-degrade-to-detect.vercel.app](https://dfd-fcg-degrade-to-detect.vercel.app) · "
                "[code](https://github.com/roshanraj9136/dfd-fcg-degrade-to-detect)")
    with gr.Row():
        with gr.Column():
            v = gr.Video(label="Face video (at least 4 s)", sources=["upload"])
            d = gr.Radio(["As it is", "Add strong blur", "Add heavy compression"], value="As it is",
                         label="Optional: test a low-quality copy")
            b = gr.Button("Check the video", variant="primary")
        with gr.Column():
            out = gr.Markdown()
            with gr.Row():
                face = gr.Image(label="Aligned face", type="numpy")
                heat = gr.Image(label="Evidence (red = fake, green = real)", type="numpy")
            js = gr.JSON(label="Details")
    b.click(ui_check, [v, d], [out, face, heat, js], api_visibility="private")

    # programmatic endpoints for the website
    f_in = gr.File(visible=False)
    d_in = gr.Textbox(visible=False)
    r_out = gr.JSON(visible=False)
    f_in.change(analyze, [f_in, d_in], r_out, api_name="analyze")
    gr.api(health, api_name="health", queue=False)

demo.queue(default_concurrency_limit=1, max_size=20).launch(max_file_size=f"{MAX_MB}mb", ssr_mode=False)
