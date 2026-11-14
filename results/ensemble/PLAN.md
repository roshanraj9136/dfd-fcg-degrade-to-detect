# Round 2: a second, complementary detector and a fitted fusion. Plan and decision rule

Written on 14 Nov 2026, 1:40 AM, before any test result of the new detectors or of any fusion existed. (At this
time only a 20-video smoke test on validation videos had been run.)

## Why
Round 1 (`results/retrain/`) fine-tuned DFD-FCG's own side network. It gained on the newer sets (+5.1 points) but
lost on DFDC (-5.0) and FF++ (-1.9), so the live checker kept the released model. DFD-FCG reads 10-frame clips
with frozen CLIP features; its errors on 2025-2026 videos (MAVOS-DD 53%, social media 55% balanced accuracy) suggest
it needs a detector that sees each frame at full detail.

## Members
* `dfd`: released DFD-FCG + the live self-calibration read-out (exactly the live checker, unchanged).
* `clip`, `pe`, `dino`: GenD (Yermakov et al., WACV 2026), released FF++ checkpoints with CLIP ViT-L/14, Perception
  Encoder L and DINOv3 L backbones. Frame-level; video score = mean frame probability (the authors' rule), on 24
  evenly spaced frames of the same aligned face crops DFD-FCG reads.
* `clipft`: GenD CLIP fine-tuned with the authors' own recipe (LayerNorm + head only, cross-entropy + 0.1 alignment
  + 0.5 uniformity) on the train splits of round 1 (FF++ train, Celeb-DF v2 non-test, MAVOS-DD train, SynthForensics
  train, Counterfactual groups 1-24, social-media train part), with our quality augmentations (JPEG, blur,
  down-scaling, noise). 4000 steps x 32 frames. Checkpoint chosen on the validation splits only.

## Fusion
Logistic regression on the members' video logits, fitted on the validation splits only (sources weighted equally,
classes balanced). A video with two faces is FAKE if either face is. Verdict at 0.5: always REAL or FAKE.

## Choice of the final detector (validation only)
Among all members and fusions, the one with the best mean balanced accuracy in 5-fold cross-validation on the
validation videos (folds by video; ties: fewer members). Test numbers of every option are reported, but the choice
uses validation only.

## Decision rule for the live checker (same as round 1)
The chosen detector replaces the live one only if, on the held-out tests:
- on the newer test sets (Celeb-DF v2, social media, MAVOS-DD seen and unseen methods), mean balanced accuracy
  improves by at least 5 points;
- on the FF++ reported test videos, balanced accuracy over the three website options (as it is, strong blur, heavy
  compression) drops by at most 2 points;
- on DFDC (reported test, part 0, never trained on), balanced accuracy over the three options drops by at most 2
  points.

## Never used for training, fusion fitting or selection
The 225 reported test videos (FF++ test, FaceShifter, DFDC part 0), all of DFDC, the test and test_open splits of the
newer sets (Celeb-DF v2 official test list, social-media test part, MAVOS-DD test and open-set test, SynthForensics
test, Counterfactual groups 31-36).

## Amendments (written before any test result of a fine-tuned GenD model existed)

14 Nov 2026, 3:20 AM. The first GenD-CLIP fine-tuning run crashed at step 350 (its console pipe was closed), so the
order of the runs changed. At this point test results existed only for released models (DFD-FCG, round-1 DFD-FCG,
released GenD-CLIP and GenD-PE, and their fusions); they are reported as they are.
1. **Members added.** `dfdft`: the round-1 fine-tuned DFD-FCG with its own refitted read-out (`results/retrain/`).
   `pe`: released GenD with the Perception Encoder backbone. `<member>_sc`: a GenD member with self-calibration carried
   over unchanged (ridge read-out on GenD's L2-normalised class token, class-agnostic target from the real fit
   videos, the same 49 fit videos and damage levels 2 and 4, alpha 1000).
2. **Which GenD is fine-tuned first: PE**, because its validation score before fine-tuning was higher (0.719 vs
   0.635 for CLIP, `train_log.jsonl` step 0). Same recipe, data and step count as planned for CLIP; LayerNorms of PE
   (all `norm` tensors, as in the authors' PE configs) + head.
3. **GenD-CLIP is fine-tuned second**, on the same train splits plus the extra train-split videos of
   `service/train/make_sets_more.py` that are prepared by then (MAVOS-DD, SynthForensics, Counterfactual groups 1-24,
   FF++ train, Celeb-DF v2 non-test). Validation and test sets are unchanged.
4. The selection rule is unchanged: among every member and fusion available at the end, the best 5-fold CV mean
   balanced accuracy on the validation videos. The decision rule for the live checker is unchanged.
5. The live checker always answers REAL or FAKE (the user asked for a firm answer); the confidence and the measured
   accuracy at that confidence are shown instead of "not sure".

## Round 2 result, and round 2b (written 14 Nov 2026, 5:45 AM, before any round-2b score existed)

**Round 2 outcome.** The option chosen on validation, `dfd+pe_ft_sc+pe_sc`, gains +24.9 points on the newer test sets
and 0.0 on DFDC but loses 6.2 on FF++ (rule: at most 2), so **the rule is not met**. Per condition the loss is on
clean FF++ reals (14% false alarms vs 1%) and on damaged FF++ fakes (fewer caught under strong blur and compression).
The combiner was fitted on clean validation videos only, while the FF++ and DFDC tests average clean, strong blur
and heavy compression.

**Round 2b: the combiner also sees damaged validation videos.** Nothing else changes (members, checkpoints, test sets,
the switch rule).
1. Every validation video (main face, the 4 stored clips) is scored by `dfd`, `dfdft`, `pe_ft` and `pe` also under two
   damages applied to the aligned face crops: `blur` (Gaussian, sigma 2.5 px on the 150-px crop, about the test's
   strong blur after alignment, then JPEG quality 75) and `comp` (each 10-frame clip encoded with H.264 CRF 40). These approximate the website's strong blur and heavy compression
   options; they are not the authors' code and are not the test protocol.
2. Combiner fitted on clean + blur + comp validation videos (each condition weighted equally, within it sources and
   classes balanced as before).
3. Selection among all members and fusions: the best 5-fold CV mean balanced accuracy on clean + blur + comp
   validation videos (folds by video, so a video's three versions stay in one fold).
4. Same switch rule. Round-2 and round-2b test numbers are both reported.

## Round 3 follow-up (14 Nov 2026, before evaluating GenD-CLIP test results)

The validation-only FF++ reweighting protocol was recorded in `ROUND3_PLAN.md`.
Its 15 options produced no qualifying candidate, so the live model is retained.

Next evaluate the completed GenD-CLIP fine-tune (the existing `ft_clip/best.pt`,
chosen by AUROC before this round). Score the same 24 frames, fit videos and
clean/blur/compression validation protocol. No new network training is performed
for this comparison. Add `clip_ft` and `clip_ft_sc` to the five live members.

**Split ambiguity:** AGENTS reserves Counterfactual groups 25-36 from tuning,
while the older manifests use groups 25-30 as validation. Until the owner resolves
this, exclude all `synth` validation rows from this new selection/fitting. All
DFDC remains excluded. Refit the baseline on this same restricted validation
cohort for CV comparisons; compare against the exact deployed baseline at test.
Prior use of groups 25-30 is disclosed, not erased by this exclusion.

**Selection, validation only:**
- Require all seven members to have every clean/blur/comp row of a video, and
  compare every subset on that same complete cohort. Report coverage explicitly.
- Evaluate every nonempty subset with the same source/class/condition weighting,
  C=1 logistic regression, and five video-grouped folds (seed 0) as round 2b.
- Advance only if CV macro balanced accuracy beats the five-member baseline by
  at least 0.5 points, macro fake recall loses at most 1 point, and clean FF++
  false alarms do not increase. Rank by macro balanced accuracy; ties rounded to
  0.1 points favor fewer networks, then fewer scores.
- Freeze the chosen coefficients and all CV results before computing test metrics.
  Report every option, including failures; test metrics never change the choice.

**Release rule versus the current live model:** mean balanced accuracy on the four
newer two-class sets must increase, mean FF++ balanced accuracy must not decrease,
and neither DFDC nor FaceShifter may lose more than 2 points. Neither fake-only set
may lose more than 2 points of recall. Full model parity, both sample videos and
the existing binary-answer UI checks must pass. If the rule fails, retain the
current model and publish the tradeoff. This is a follow-up benchmark on previously
reported tests, not a new blind holdout.
