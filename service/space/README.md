---
title: Deepfake Video Checker
emoji: 🎬
colorFrom: indigo
colorTo: green
sdk: gradio
sdk_version: 6.31.0
python_version: "3.12"
app_file: app.py
pinned: false
license: mit
short_description: Upload a face video, get REAL or FAKE with a short reason
---

# Deepfake video checker (self-calibrated DFD-FCG)

Backend of the website https://dfd-fcg-degrade-to-detect.vercel.app
(code: https://github.com/roshanraj9136/dfd-fcg-degrade-to-detect).

* DFD-FCG (Han et al., CVPR 2025), the authors' released checkpoint, on ZeroGPU.
* Self-calibration (our add-on): a linear read-out of the model's own features estimates how much FAKE
  evidence blur and compression added to the video, and removes it. No retraining.
* Blur test and a per-region evidence map as a short explanation.
* Face finding runs on the Space's CPU; only the two DFD-FCG passes use ZeroGPU (about 2 s). When a visitor's
  free GPU time is used up, those passes run on a CPU copy of the model instead (same scores, a few seconds slower).

API: `/analyze` (file, degrade = none | blur | compress) and `/health`, e.g. with `@gradio/client`.
Uploaded videos are deleted right after the check; only anonymous numbers are logged.

Space secrets: `HF_TOKEN` (read access to the private weights repo; the checkpoint is for research use
only, as required by its authors) and optionally `DATABASE_URL` (anonymous result log in Neon Postgres).
