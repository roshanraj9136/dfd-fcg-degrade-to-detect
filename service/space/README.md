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

# Deepfake video checker (DFD-FCG + GenD, self-calibrated, combined)

Backend of the website https://dfd-fcg-degrade-to-detect.vercel.app. Made by Roshan Raj.

* Detectors on the same aligned faces: DFD-FCG (Han et al., CVPR 2025, released checkpoint; 8 clips of 10 frames)
  and GenD-PE (Yermakov et al., WACV 2026; 24 single frames), released and retrained on 2025-2026 fakes.
* Self-calibration (our add-on): a linear read-out of each detector's own features estimates how much FAKE
  evidence blur and compression added, and removes it. No retraining.
* A combiner fitted on validation videos only turns the detectors' scores into one firm answer, REAL or FAKE,
  with its confidence.
* Every clear face in the video is checked (up to 2 people; a swap can be on either), from the first 10 seconds.
  The most suspicious face decides.
* Face finding runs on the Space's CPU; only the detector passes use ZeroGPU (the booked time grows with the number
  of faces). When a visitor's free GPU time is used up, those passes run on a CPU copy of the models instead
  (same scores, slower).

API: `/analyze` (file, degrade = none | blur | compress) and `/health`, e.g. with `@gradio/client`.
Uploaded videos are deleted right after the check; only anonymous numbers are logged.

Space secrets: `HF_TOKEN` (read access to the private weights repo; the checkpoint is for research use
only, as required by its authors) and optionally `DATABASE_URL` (anonymous result log in Neon Postgres).
