---
title: Deepfake Video Checker
emoji: 🎬
colorFrom: indigo
colorTo: green
sdk: docker
app_port: 7860
pinned: false
license: mit
short_description: Upload a face video, get REAL or FAKE with a short reason
---

# Deepfake video checker (self-calibrated DFD-FCG)

Backend of https://github.com/roshanraj9136/dfd-fcg-degrade-to-detect.

* DFD-FCG (Han et al., CVPR 2025), released checkpoint, CPU inference.
* Self-calibration: a linear read-out of the model's own features removes the FAKE evidence that
  blur and compression add to authentic videos.
* Blur test and per-region evidence map as a short explanation.

Endpoints: `GET /api/health`, `POST /api/analyze` (multipart `file`), `GET /api/jobs/{id}`.
Uploaded videos are deleted right after analysis.

Requires Space secrets `HF_TOKEN` (read access to the private weights repo) and, optionally,
`DATABASE_URL` (anonymous result log in Neon Postgres).
