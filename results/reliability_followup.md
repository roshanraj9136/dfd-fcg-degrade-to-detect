# Reliability follow-up (14 November 2026)

**Status: draft, no deployment and no newly measured accuracy gain.** The result
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

Local result: 28 passed, one sklearn comparison skipped because sklearn is absent
from the accessible runtime. CI installs sklearn to run that comparison too.
Tests cover the actual scoring methods isolated from GPU imports, numerical
failures, tracking gates, split checks, CLIP routing and weighted logistic fitting.
They do not replace full-video accuracy or GPU/CPU parity checks.

Read `AGENTS.md` and `HANDOFF.md` before continuing. The Kiro session location,
data root and completed CLIP checkpoint were located. The CLIP log confirms 4,000
steps completed and a best validation AUROC of 0.8508. Scoring that complementary
model is the next GPU task.

This execution session receives **Access denied** for
`C:\Users\hp\AppData\Local\Programs\Python\Python312\python.exe`.
The DFD venv depends on that installation. The accessible bundled Python can run
NumPy evaluation, but has no Torch/CUDA stack. No GPU inference or training was run.

Prepared runner, for a Windows session with access to the existing GPU environment:

```powershell
powershell -NoProfile -File C:\Users\hp\dfd-fcg-degrade-to-detect\scripts\score_clip_candidate.ps1
```

It writes only candidate scores to `C:\Users\hp\dfd-work\round3-cache`; it does
not select or deploy a model. Run one GPU job at a time. Use
`DFD_ENS_DIR` to read these candidate scores alongside the original cached scores.
Freeze a validation-selected candidate before viewing its new test metrics.

Before merging/deploying, run `tests/check_gend_live.py`,
`tests/check_fusion_live.py cuda` and both sample videos using the documented
environments. Count rejected videos in coverage reporting. No model or website
has been replaced.

A split-rule ambiguity was sent to the owner: AGENTS reserves Counterfactual
groups 25–36 from tuning, while previous code uses 25–30 for validation. The above
experiment reconstructs that previous validation protocol; no further tuning
should proceed until the intended rule is confirmed.
