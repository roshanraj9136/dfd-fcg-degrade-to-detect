# Reliability follow-up (14 November 2026)

**Status: model experiments completed; reliability deployment pending. No newly
measured accuracy gain.** The result
remains REAL or FAKE. Invalid inference produces an analysis error.

## Reliability changes

- Reject empty/non-finite logits, malformed calibration parameters, missing
  ensemble members and missing face outputs. NaN cannot fall through to REAL.
- Stabilize sigmoid calculations while preserving ordinary finite scores.
- Apply the existing face displacement gate to the main track as well as secondary
  tracks. This prevents distant person switches, but needs video-level validation.
- Preserve per-member sampling counts when GenD networks or CPU copies are shared.
- Report a clear API/Space error when scoring fails.
- Validate train/validation video overlap (and groups when provided), support
  zero-worker debugging, and retain **AUROC as the default** GenD selection metric.
  Balanced accuracy is an explicit experimental option; member thresholds are
  recalibrated by the combiner, as the Kiro briefing explains.
- Add CLIP/CLIP-fine-tune support to damaged-validation scoring, a runner for the
  already-completed CLIP checkpoint, and a separate experiment-cache directory
  so existing scores remain untouched.

## Actual accuracy experiment

The preregistered rule is in `ensemble/ROUND3_PLAN.md`. Reweighting real FF++ videos
was evaluated on the same **353 validation videos / 1,059 condition rows** as round
2b. Nine manifest validation videos lack complete cached scores and were excluded
from every option equally. All 15 options are in `ensemble/round3/selection.json`.

| Validation result | Original weighting, C=1 | FF++ real weight x3, C=1 |
|---|---:|---:|
| Macro balanced accuracy | 80.58% | 79.77% |
| Macro fake recall | 77.02% | 68.80% |
| Clean FF++ false alarms | 7.50% | 2.50% |

**No candidate passed the selection rule.** Lower false alarms came with more
missed fakes. No new candidate was tested or exported. The NumPy baseline reproduces
the previous CV balanced accuracy exactly; its full-fit probabilities differ from
the stored live combiner by at most 0.000666 (optimizer tolerance).

The existing published result remains 85.1% correct on 705 clean test videos.
It is a single-run estimate, not a promise for arbitrary uploads. The original
switch rule was not met: FF++ balanced accuracy fell 5.6 points. The owner chose
the modern-video gains deliberately. DFD-FCG remains research-only; the service
is free and non-commercial.

## Verification and remaining work

```powershell
python -m unittest discover -s tests -p "test_*.py" -v
python -m compileall -q service/app service/ens service/space tests
```

Local result: 34 passed in the existing Python 3.12 environment, including the
sklearn comparison. CI installs sklearn to run that comparison too.
Tests cover the actual scoring methods isolated from GPU imports, numerical
failures, tracking gates, split checks, CLIP routing and weighted logistic fitting.
They do not replace full-video accuracy or GPU/CPU parity checks.

Read `AGENTS.md` and `HANDOFF.md` before continuing. The Kiro session location,
data root and completed CLIP checkpoint were located. The CLIP log confirms 4,000
steps completed and a best validation AUROC of 0.8508. Local Python/GPU access was
restored after the initial session restriction. CLIP scoring and evaluation are complete.

Runner using the existing GPU environment:

```powershell
powershell -NoProfile -File C:\Users\hp\dfd-fcg-degrade-to-detect\scripts\score_clip_candidate.ps1
```

It writes only candidate scores to `C:\Users\hp\dfd-work\round3-cache`; it does
not select or deploy a model. Run one GPU job at a time. Use
`DFD_ENS_DIR` to read these candidate scores alongside the original cached scores.
Freeze a validation-selected candidate before viewing its new test metrics.

Completed model checks (`scripts/validate_models.ps1`): released CLIP and PE
implementations matched exactly; seven held-out fusion examples matched the
offline scores (largest probability difference rounded to four decimals: 0.0000).
The full real sample returned REAL, p(fake)=0.212, and the fake sample returned
FAKE, p(fake)=1.000. These checks establish implementation parity, not a new
accuracy estimate. A selected replacement needs its own parity checks. No model
or website has been replaced.

A split-rule ambiguity was sent to the owner: AGENTS reserves Counterfactual
groups 25–36 from tuning, while previous code uses 25–30 for validation. The above
experiment reconstructs that previous validation protocol. The new CLIP selection
excludes all synth validation rows conservatively. The earlier checkpoint's use
of those validation rows is not undone by excluding them now. The complete rule,
frozen before new test metrics, is in `ensemble/PLAN.md`; selection and test
evaluation run as separate commands in `service/ens/select_clip_fusion.py`.

## Completed CLIP comparison

All 127 subsets were compared on 313 validation videos / 939 condition rows;
40 synth validation videos were excluded under the conservative split policy and
nine other videos lacked complete scores. The selected option was
`dfd+pe_ft_sc+pe_sc` (no CLIP): validation balanced accuracy 78.90% versus 78.20%
for the refitted five-member baseline, fake recall 75.87% versus 75.29%, and the
same 7.50% clean FF++ false alarms. Its weights were committed before testing.

The selected option failed the release rule on the exact same 480 newer and 225
reported test videos as the deployed model:

| Test balanced accuracy | Current live model | Selected candidate |
|---|---:|---:|
| Mean of four newer two-class sets | 85.02% | 82.72% |
| FF++, mean of three conditions | 79.05% | 79.44% |
| FaceShifter, mean of three conditions | 63.49% | 64.44% |
| DFDC, mean of three conditions | 60.42% | 60.00% |

The newer-set regression rules out deployment. The live model weights remain
unchanged. Every CV option and test result, the frozen candidate and environment
versions are retained in `ensemble/round3_clip/`. These previously reported tests
are follow-up benchmarks, not fresh blind holdouts. Neither the reweighting nor
CLIP experiment demonstrated a replacement meeting the declared rule.

The training-store audit found 26 of 142 prepared FF++ fakes with the matching
first-ID real video in the store. There are 1,094 additional training videos still
awaiting preprocessing and eight recorded failures (1,480 total, 378 prepared).
The combined manifests have 4,798 unique video IDs, no ID pointing to conflicting
files, and no identical repository/path occurring across declared splits. This
does not establish identity-disjoint evaluation. GenD's paired-data motivation
is documented in [the authors' paper](https://arxiv.org/html/2508.06248v2#S4.SS4).
