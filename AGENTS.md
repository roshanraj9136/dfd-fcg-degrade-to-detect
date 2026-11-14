# AGENTS.md: briefing for any coding agent working on this project

Written 14 Nov 2026, 1:35 PM, by the Kiro agent that did rounds 1 and 2 (overnight 13-14 Nov), for Astra (Codex),
at the owner's request. Everything below was checked against the files at the time of writing. If a number here
disagrees with `results/ensemble/results.json`, the file wins.

Owner: Roshan Raj (B.Tech project). Goal: a free, non-commercial deepfake video checker for the public good, as
accurate as possible on modern (2025-2026) videos, that always answers **REAL or FAKE** (never "not sure"; the owner
asked for this explicitly). He wants results he can publish. Commit as `Roshan Raj <roshanraj9136@users.noreply.github.com>`.

---------------------------------------------------------------------------------------------------------------------

## 1. What is live right now

| part | where | built from |
|---|---|---|
| checker website | https://dfd-fcg-degrade-to-detect.vercel.app | `web/checker` (Vercel project `dfd-fcg-degrade-to-detect`) |
| backend (Gradio API `/analyze`, `/health`) | HF Space `roshan9136/dfd-fcg-checker`, ZeroGPU, CPU fallback | `service/space/build.ps1` → `C:\Users\hp\dfd-work\space-zerogpu` → upload |
| viva guide | https://dfd-fcg-viva.vercel.app | `web/viva` (Vercel project `dfd-fcg-viva`) |
| results API | Render `dfd-fcg-degrade-to-detect-api.onrender.com`, Neon Postgres | `web/api` (unchanged since round 1) |

Live detector = **fusion of 5 scores from 3 networks** on the same aligned face crops (`service/app/fusion.json`):

| member | network | weights | video score |
|---|---|---|---|
| `dfd` | DFD-FCG (Han et al., CVPR 2025), released checkpoint | private HF repo `roshan9136/dfd-fcg-weights` → `weights.ckpt` | 8 clips × 10 frames, our self-calibration read-out (`service/app/model.json`) |
| `pe_ft` / `pe_ft_sc` | GenD-PE (Yermakov et al., WACV 2026), **our fine-tune** (LayerNorm + head only) | public `yermandy/GenD_PE_L` + `service/app/gend_pe_ft.pt` (our 104 tensors) | 24 frames, mean prob; `_sc` = with self-calibration on the class token |
| `pe` / `pe_sc` | GenD-PE, released | `yermandy/GenD_PE_L` | same |

Combiner: logistic regression on the 5 logits, coef `dfd 1.020, pe_ft 0.086, pe_ft_sc 1.403, pe 0.116, pe_sc -0.362`,
intercept 1.607, fitted on 353 validation videos × {clean, blur, compression}. Most suspicious face decides. Verdict
= combined p ≥ 0.5. Members sharing a network share one forward pass (`gend_live.make_member`).

## 2. Measured results (held-out test videos, never used for training, checkpoint choice, combiner fit or selection)

Balanced accuracy at 0.5 (fake-only sets: share of fakes caught). Live = `dfd+pe_ft+pe_ft_sc+pe+pe_sc`.

| test set | released DFD-FCG | live system |
|---|---|---|
| Celeb-DF v2 official test (120) | 78.3% | 92.5% |
| Social-media deepfakes 2026, test part (87) | 55.0% | 89.2% |
| MAVOS-DD 2025, methods seen (78) | 54.7% | 83.4% |
| MAVOS-DD 2025, methods never seen: hififace, sonic, roop (56) | 60.7% | 75.0% |
| SynthForensics 2026, AI-made people (59 fakes) | 35.6% caught | 76.3% caught |
| Counterfactual DeepFake Bench 2026, AI talking heads (80 fakes) | 15.0% caught | 100% caught |
| FF++ test (130), mean of clean / strong blur GB_5 / heavy compression VC_5 | 84.6% | **79.0%** |
| FaceShifter (15 fakes + FF++ reals), never trained on | 70.5% | **63.5%** |
| DFDC part 0 (80), never trained on | 58.8% | 60.4% |
| **all 705 test videos as uploaded (clean)** | 63.1% correct | **85.1% correct** |

Confidence bands (all 705, clean): 50-65% → 58.1% right (12.2% of answers); 65-80% → 76.4% (17.4%); 80%+ → 91.9% (70.4%).
Bootstrap 95% range of the gain on the 4 newer sets with reals: +17.4 to +28.3 points.

**Honest caveats you must keep in anything you publish:**
- The pre-registered switch rule (`results/ensemble/PLAN.md`) was **not met**: FF++ dropped 5.6 points (rule ≤ 2),
  mainly **false alarms on clean FF++ real videos: 18.6% vs 1.4%** for released DFD-FCG. The owner chose to deploy
  anyway for modern videos; the viva page and README say so. Do not hide this.
- Single run, single seed, ~50-120 videos per test set. Differences of a few points are within noise.
- DFD-FCG weights are research-only; that is why the service is free and non-commercial. GenD is MIT.

Every option tried (73 fusions × 2 rounds, with validation CV scores and all test numbers):
`results/ensemble/all_options_round2.json` (combiner fit on clean val) and `all_options_round2b.json` (clean+damaged
val, the live one). Round 1 (fine-tuning DFD-FCG itself, rule not met, DFDC −5.0): `results/retrain/`.

## 3. Answers to the questions you raised in your session

- **Where are the Kiro chats?** `C:\Users\hp\.kiro\sessions\<workspace-hash>\<session-id>\messages.jsonl` (JSON lines;
  `payload.type` = user / assistant / tool_call / tool_result). Relevant ones:
  `6156ed8f99646ea5\sess_5e70bff9-...` (round 1 retraining), `bebe3adce36d0150\sess_2b2b8b34-...` (round 2, this one),
  `bebe3adce36d0150\sess_30639443-...` and `sess_6e56c69d-...` (website / Space). You should not need them: this file,
  `HANDOFF.md`, `results/*/PLAN.md` and the code docstrings hold everything.
- **"Invalid scores can fall through to REAL"**: correct, `pipeline.finish` does `verdict = FAKE if p >= 0.5 else REAL`
  with no finite check. A NaN guard is welcome. Do not turn it into a "not sure" answer; raise a clear error instead
  (the website shows errors), or fall back to the DFD-FCG score.
- **"Face tracking can jump between people"**: the main face follows the nearest box from the largest first
  detection (`find_faces`); only extra faces have the `dist > width` guard. A guard for face 1 too is reasonable;
  re-run `tests/check_fusion_live.py` and the two samples afterwards.
- **"Checkpoints are selected by AUROC while the app decides at a fixed threshold"**: true for `train_gend.py`
  (score = mean val AUROC, clean + degraded), but the live threshold is set by the **combiner**, which is refitted on
  validation videos after training. So AUROC is the right criterion for a member whose threshold is recalibrated;
  switching to balanced accuracy at 0.5 is likely to pick worse members. If you add it, keep AUROC as the default
  and compare both on validation only. (`service/train/train.py`, round 1, used balanced accuracy at 0.5.)

## 4. Data, environments, hardware

- **Data root `D:\dfd-data`** (not in git):
  - `new\sets.csv`, `new\sets_gated.csv`: train / val / test / test_open splits (columns vid,set,source,method,label,repo,path);
    `new\sets_more.csv`: 1,480 extra **train-split** videos (MAVOS-DD, SynthForensics, Counterfactual groups 1-24,
    FF++ train, Celeb-DF non-test); videos in `new\videos\<source>\`.
  - `crops\<vid>.npz`: aligned 150×150 RGB face clips (`face0`, `face1`, `clip_idx`, label, fps, w, h).
    **Only 378 of the 1,480 extra videos are prepared** (CPU, ~1 min/video/worker; `D:\dfd-data\ens\prep_more.ps1`).
  - `ens\store\frames.npy` (+ `index.csv`): memory-mapped training frames (1,669 train + 353 val videos);
    `ens\store2\`: same plus the 342 extra videos prepared at that time (2,011 train).
  - `builds\test\*.npz`: the 225 reported test videos under clean / GB_5 / VC_5 (authors' damage code);
    `builds\fit\*.npz`: the 49 self-calibration fit videos × 16 conditions.
  - Stored scores used by all evaluations: `eval_new_old.jsonl`, `feats_test_old.npz`, `ens\gend_<model>_<new|rep|fit>.npz`,
    `ens\dfd_val_*.jsonl`, `ens\valdmg_*.npz`. Re-scoring a fusion needs no GPU (`service/ens/evaluate.py`).
- **Never used for training, tuning or selection**: test / test_open splits, the 225 reported test videos (FF++ test,
  FaceShifter, DFDC part 0), all of DFDC (its parts share actors), Celeb-DF official test list, Counterfactual groups 25-36.
- **Checkpoints**: released DFD-FCG `C:\Users\hp\B.Tech-Projects\checkpoint\weights.ckpt`; round-1 DFD-FCG
  `D:\dfd-data\train_v1\best.ckpt` (+ `readout_new.json`); GenD-PE fine-tune `D:\dfd-data\ens\ft_pe\best.pt`
  (val 0.719 → 0.9185); GenD-CLIP fine-tune on store2 `D:\dfd-data\ens\ft_clip\best.pt` (val 0.635 → 0.851, **not yet
  scored or used**; job `D:\dfd-data\ens\jobs\061_score_clipft.ps1.hold`).
- **Two Pythons**: GenD code runs in the system Python 3.12 (torch 2.6 cu124, transformers 5.20.2, timm 1.0.29);
  DFD-FCG needs `C:\Users\hp\dfd-explain\DFD-FCG\.venv\Scripts\python.exe` with `DFD_FCG_UPSTREAM=C:\Users\hp\dfd-work\DFD-FCG`,
  `DFD_FCG_CKPT=...weights.ckpt`, `DFD_DATA=D:\dfd-data\work`, `TORCHDYNAMO_DISABLE=1`. GenD's repo is
  `C:\Users\hp\dfd-work\GenD`; its `src` package shadows DFD-FCG's `src`, so never import both in one process
  (except through `service/app/gend_live.py`, which needs no GenD code). `HF_HUB_CACHE=D:\hf-cache`.
  DINOv3 is gated on HF; `service/ens/gend.py` rebuilds it locally from GenD's own checkpoint.
- **GPU: RTX 3050 Laptop, 4 GB, 16 GB RAM.** Run **one GPU job at a time** (two jobs crashed with out-of-memory in
  round 1). GenD-PE fine-tuning: batch 32, gradient checkpointing, 2.5 GB, ~1.8 s/step, 4,000 steps ≈ 2 h.
  Each CPU face-prep worker commits ~2.5 GB; with training running, use at most one. Start long jobs with output
  redirected to files (`Start-Process -RedirectStandardOutput`): a training run died once when its parent console
  was killed. Job queue: `D:\dfd-data\ens\runner.ps1` runs `D:\dfd-data\ens\jobs\*.ps1` in name order (not running now).
- No cloud GPU is set up (no Kaggle token on this machine). Kaggle (2×T4 16 GB, ~30 h/week) would need the owner to
  create `C:\Users\hp\.kaggle\kaggle.json`.

## 5. How to change the live system safely

1. Branch from `main`; do not touch `main` until tests pass.
2. Fusion change: `VAL_CONDS=clean,blur,comp python service/ens/evaluate.py <members...>` (prints validation CV score and
   every test set), then `report.py <option> <candidates...>` and `export_live.py <option>` (writes `fusion.json` and
   tensors into `service/app/`). Choose by **validation CV only**; write the rule into `results/ensemble/PLAN.md` before
   looking at test numbers, and report every option.
3. Parity: `tests/check_gend_live.py clip|pe` (live GenD == evaluation code, exact), `tests/check_fusion_live.py cuda`
   (live pipeline == offline scores, all members; DFD-FCG venv), `tests/run_live_samples.py cuda <videos>`.
4. Deploy backend: `powershell service/space/build.ps1`, then `python C:\Users\hp\dfd-work\scratch\make_space.py --upload --msg "..."`,
   wait (`scratch/space_watch.py`), test (`scratch/space_test.py <video>[@blur|@compress]`). Rollback: re-upload the
   previous build, or delete `service/app/fusion.json` (= DFD-FCG only). Space secrets are already set; never print them.
5. Websites: `npm run build` in `web/checker` / `web/viva`, then `vercel deploy --prod --yes --project <name>`.
   `web/checker/src/accuracy.json` and `web/viva/src/ensemble.json` are generated by `service/ens/report.py`; never
   hand-edit numbers. Screenshot check: `scratch/shot_full.py <url> <prefix>`, upload test `scratch/upload_test.py`.
6. The checker must never show "not sure" / "unsure" / "can't tell" (`scratch/live_text_check.py` verifies).
7. Never commit `data/`, `.env*`, `.vercel`, tokens, database URLs, DFD-FCG weights or dataset videos/frames.
   On Windows PowerShell 5, edit non-ASCII files with Python or an editor, not `Get-Content`/`Out-File` (it corrupts `–`, `“`).

## 6. What would improve accuracy most (ranked by expected gain per GPU hour)

1. **Cut the FF++ false alarms (biggest known weakness).** 18.6% of clean FF++ reals are called FAKE, almost all
   from the fine-tuned frame detector. Options: (a) finish preparing the 1,102 remaining extra videos (they include
   120 more FF++ reals and 400 MAVOS reals), rebuild `store2`, retrain GenD-PE on it; (b) **paired real/fake
   sampling** in `train_gend.py` (GenD's main finding: sample a fake together with its own source real; FF++ and
   Celeb-DF have pairs, MAVOS-DD has same-video reals) to stop it learning "2025 video style = fake"; (c) give FF++
   reals more weight in the combiner fit.
2. **Score the finished GenD-CLIP fine-tune** (`061_score_clipft`, ~30 min), add `clip_ft` damaged-validation
   scores (extend `service/ens/score_valdmg.py` with a clip branch), and re-run the fusion search. Different backbone
   = more independent errors.
3. **Several seeds** (3 per fine-tune, average the tensors or the scores): cheap variance reduction and needed for
   publishable numbers with confidence intervals.
4. **Calibrate confidence** (temperature or isotonic on validation) so "70% confidence" means 70% right; the 50-65%
   band is only 58% right today.
5. **More frames at test time** for GenD (24 → 48, plus horizontal flip): costs ~1 s on the GPU, often +1-2 points.
6. **FaceShifter / identity-swap generalisation** (63.5%): more swap-type training data that is not a test set,
   e.g. DeepSpeak v2, FFIW train split (check licences, keep test splits out).
7. **Audio-visual detector** for lip-sync-only fakes (MAVOS-DD has fake audio); none of the current members can see it.
8. Larger GPU (Kaggle) would allow unfreezing the last CLIP/PE blocks; risky for generalisation, try last.

Ideas already tested that did **not** help (don't repeat without a new reason): flip test-time augmentation for
DFD-FCG, clip agreement as a doubt signal, moving the 0.5 line, quality from the background, the blur test as the
fix, fine-tuning DFD-FCG's side network alone (round 1), fusion fitted on clean validation only (round 2).

## 7. Machine state (14 Nov, 1:35 PM)

**Division of work (14 Nov, 1:45 PM):** Astra owns the GPU and all code / model changes (branch
`codex/reliability-and-validation`, GenD-CLIP evaluation, fusion updates). Kiro only writes the paper draft in
`paper/` and a CPU-only script `service/ens/selfcal_transfer.py` that reads stored scores. No overlap in files.

All Kiro background jobs are finished; GPU and CPU are free; keep-awake script stopped. Power plan (AC) was set to
never sleep and lid-close = do nothing (restore commands in `HANDOFF.md`). Working copy used for every deployment:
`C:\Users\hp\dfd-work\dfd-fcg-reliability`; a fresh clone also exists at `C:\Users\hp\dfd-fcg-degrade-to-detect`.
