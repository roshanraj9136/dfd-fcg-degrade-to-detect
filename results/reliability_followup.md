# Reliability follow-up (14 November 2026)

This change fixes inference failure modes and prepares a better checkpoint-selection
criterion. It does **not** contain newly trained weights or a measured increase in
deepfake-detection accuracy.

## What the existing results establish

The checked-in `results/ensemble/results.json` reports the current ensemble at
85.1% accuracy across 705 clean held-out videos. That pooled number does not describe
every domain: on DFDC its fake recall is 37.5% on clean input, 15.0% under strong
blur and 12.5% under heavy compression. Its average DFDC false-alarm rate across
those conditions is only 0.83%. Missing fakes is therefore a demonstrated limitation,
not something that can be resolved by claiming higher confidence.

The result file records the owner's deliberate choice to deploy the ensemble for
its gains on newer videos despite a 5.6-point FF++ balanced-accuracy regression.
This change preserves that ensemble, its weights, calibration and 0.5 threshold.

## Changes

- Reject empty, NaN, infinite and incorrectly shaped scores/calibration parameters.
  A failed detector or a missing ensemble member can no longer silently return a
  DFD-only decision, or let a NaN comparison fall through to REAL.
- Use a stable sigmoid for large logits. Ordinary finite predictions retain the
  existing mean-probability aggregation and fitted fusion coefficients.
- Check that inference returns every prepared face. Reject missing explicit
  fusion-file paths instead of silently using a different detector.
- Apply the existing secondary-face displacement gate to the main face as well.
  Reject a main track that is absent around sampled frames. This reduces distant
  person switches; it is not identity recognition and needs video-level validation.
- Preserve each shared GenD member's frame count, including in CPU fallback, and
  key cached inference by both network and sampling count.
- For **new GenD training runs**, select the checkpoint by validation balanced
  accuracy at the app's actual 0.5 threshold, equally weighting clean/degraded
  conditions. Keep AUROC in the log and support `SELECTION_METRIC=auc` to reproduce
  the previous selection criterion. Check train/validation video overlap and
  explicit group overlap when group IDs are present.
- Support `WORKERS=0` for debugging training on Windows.
- Include the new inference helper in the Hugging Face Space build.

The existing REAL/FAKE result contract is retained. Invalid inference returns an
analysis error through the existing API/Space exception handling.

## Checks

```powershell
python -m unittest discover -s tests -p "test_reliability.py" -v
python -m compileall -q service/app service/ens tests/test_reliability.py
```

The new GitHub Actions workflow runs these CPU checks. They cover numerical
failures, missing ensemble evidence, score parity on finite synthetic inputs,
tracking gates, checkpoint selection, split overlap and zero-worker loading.
The lightweight integration tests compile the real scoring methods from their
source AST without importing the GPU stack. They do not test model loading,
full-video preprocessing, GPU/CPU precision parity or classification accuracy.

Before deployment, run the existing `tests/check_fusion_live.py` against the saved
crops/checkpoints and compare the complete old/new pipeline on the same videos.
Report analysis failures/coverage as well as accuracy; rejected videos must not
silently disappear from the denominator.

## Continuing the accuracy work on the training laptop

Repository code points to `D:\dfd-data`, including `ens\store\index.csv`,
`ens\store\frames.npy`, `ens\ft_pe\best.pt` and saved per-video scores.
These are code defaults, not paths verified on the laptop. The user has identified
Kiro CLI chats on the Windows laptop as the source of training history. The local
tool connection was unavailable while this follow-up was prepared; those chats,
the datasets and checkpoints have not yet been inspected.

1. Read the project-specific Kiro sessions and confirm the current training
   process, dataset manifests, model artifacts and failed example videos.
2. Reproduce the current ensemble's per-domain fake recall, false alarms,
   balanced accuracy and failures on the same saved inputs.
3. Audit source-video/identity separation. Video-ID checks alone cannot detect
   leakage between different videos of the same person.
4. Train into a fresh output directory, selecting only on validation. Example
   after confirming the existing store and GPU environment:

   ```powershell
   $env:STORE = 'D:\dfd-data\ens\store'
   $env:SELECTION_METRIC = 'bacc'
   python service/ens/train_gend.py D:\dfd-data\ens\ft_pe_bacc_candidate 4000 0.0001 pe
   ```

   Do not overwrite `ft_pe/best.pt` or export the candidate directly to the live app.
   If a custom existing run used other settings, preserve and record those settings.
5. Score the candidate, refit any dependent self-calibration and fusion on
   validation data only, and freeze the candidate before testing. Since prior test
   results have already informed development, use a fresh final holdout for any
   new generalization claim.
6. Compare fake recall at matched false-alarm rates by domain and degradation,
   along with balanced accuracy and confidence calibration. Lowering a threshold
   alone can catch more fakes while falsely accusing more real videos.
7. Revisit temporal coverage and face coverage with measured examples: the current
   app samples the beginning of a video (10 s, capped at 360 frames), follows at
   most two sufficiently persistent faces and can miss later/brief manipulations.
   Changing sampling requires matching evaluation and refitting affected read-outs.

No training job, website deployment or live-model replacement is performed by
this branch.
