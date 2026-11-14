# Paired PE continuation, frozen before training

The preceding FF++ reweighting and CLIP comparison both failed their declared
release rules. Retain the deployed five-score ensemble.

Hypothesis: matching real source videos to training fakes reduces reliance on
video style and improves generalization. This is a joint data/sampling change,
not an isolated causal ablation. It follows the paired-data motivation in
[GenD](https://arxiv.org/html/2508.06248v2#S4.SS4).

- Complete real counterparts for already-prepared FF++ training fakes. Only
  official train originals are allowed; exclude fit/selection/test originals.
  Frozen manifests and source hashes: `D:/dfd-data/ens/paired_v1/`.
  143 prepared fakes yield 131 eligible pairs / 119 real originals; 12 fakes have
  reserved originals. Prepare the 96 missing real crops; all completed successfully.
- Build a separate store from existing manifests and the new real manifest.
  DFDC and all test/test_open videos remain excluded. For checkpoint selection,
  also exclude synth validation because of the documented split ambiguity.
- Start from the existing fine-tuned PE checkpoint. Train only LayerNorms and
  head for 1,000 steps at learning rate 1e-5, batch 32, warmup 100, seed 1 for
  sampling and seed 0 for Torch. Preserve the existing augmentations and losses.
- Each adjacent sample pair contains one real and one fake. Select a fake source
  uniformly, then a fake video. When its verified counterpart exists, use it with
  matching relative time and augmentation seed. Otherwise select a real source
  uniformly and use independent frames/augmentations. This intentionally changes
  real-source sampling frequencies; do not attribute any gain solely to pairing.
- Require at least 100 available pairs. Validate every 250 steps. Select by mean
  clean/degraded validation AUROC, including step zero; ties retain the earlier
  checkpoint. Stop on non-finite loss/gradients. Never select on test results.
- Score the chosen checkpoint at 24 frames, including the same damaged validation
  protocol. Compare the current five members plus this member and its self-calibrated
  score using the Round 3 CLIP protocol: common complete validation cohort, all 127
  subsets, C=1, five video-grouped folds, same weights and advancement thresholds.
  Freeze the choice before evaluating the previously reported test benchmarks.
- Release only if the existing Round 3 rule versus the exact deployed baseline
  passes, followed by parity and live samples. No automatic deployment from the
  training job. Report failed options and coverage. A single continuation run is
  exploratory and is not a multi-seed or fresh-holdout result.
