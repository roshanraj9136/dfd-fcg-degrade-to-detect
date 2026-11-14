# Round 3: validation-only reweighting of the existing ensemble

Written 14 November 2026 before scoring any new candidate on test data.

The Kiro briefing identifies excessive FF++ real-video false alarms and recommends
increasing their weight when fitting the combiner. The existing five members,
checkpoints, 24-frame GenD sampling and 0.5 decision threshold stay fixed. This
experiment uses their saved outputs and needs no GPU. GenD checkpoint selection
continues to default to AUROC because the combiner recalibrates its members.

## Selection

- Use only the same validation sources as round 2b, excluding all DFDC, with
  clean/blur/compression scores. Compare every option on exactly the same complete
  rows; report excluded validation videos and require all three conditions.
- Keep all versions of a video in one of five folds, with the round-2b seed and
  assignment. This is video-grouped CV; no claim of identity-disjoint CV is made.
- Fit the same regularized logistic combiner with C in {0.3, 1, 3} and an extra
  FF++ real-row weight in {1, 1.5, 2, 3, 4}. All other source/class/condition weights
  follow round 2b. Baseline CV is C=1, real weight=1. Also preserve the exact live
  coefficients for the final comparison.
- A candidate must improve source/condition macro balanced accuracy by at least
  0.5 percentage points over baseline CV, reduce clean FF++ false alarms, and lose
  no more than 1 point of macro fake recall. Among eligible candidates choose the
  highest macro balanced accuracy; ties prefer weight nearest 1, then C nearest 1.
- Freeze the selected option, coefficients and validation report before a separate
  test command runs. If no candidate qualifies, retain the current live model.

## Verification and release

The NumPy logistic solver must match scikit-learn on synthetic weighted data, and
its baseline fit on saved validation scores must be compared with the exported
live coefficients. Report any differences; do not attribute optimizer drift to a
new model. Record all candidate CV results, including unsuccessful candidates.

Evaluate the single frozen choice once on the existing held-out test cohorts,
alongside the exact live fusion. This is a follow-up benchmark, not a fresh blind
holdout: previous test results are already public within the project.

A candidate is eligible for deployment only if clean FF++ false alarms decrease,
average FF++ balanced accuracy increases, mean balanced accuracy on the four newer
two-class sets does not fall, and neither DFDC nor FaceShifter loses more than 2
points. Fake-only recall must not fall by more than 2 points on either fake-only
set. Report every domain even if the rule fails. Full checkpoint parity and the
two real/fake video samples must also pass before deployment.

The public result remains REAL or FAKE. Invalid inference is an analysis error.
