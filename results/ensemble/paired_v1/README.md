# Paired PE continuation: no qualifying improvement

Completed all 1,000 steps and the frozen evaluation protocol in `../PAIRED_PLAN.md`.
The run used 2,143 training videos, 131 verified pairs and 313 validation videos.
All 96 missing real counterparts were prepared before training.

| Training step | Mean clean/degraded validation AUROC |
|---|---:|
| 0 | 0.9038 |
| 250 | 0.9033 |
| 500 | 0.9024 |
| 750 | 0.9024 |
| 1,000 | 0.9024 |

The selected `best.pt` retains the starting checkpoint: all 104 tensors were
checked for exact equality. The newly trained checkpoints did not improve the
declared selection metric. `training.jsonl` contains the full training record.

All 127 fusion options are in `selection.json` and `test.json`. The validation
choice was `dfd+pe_ft_sc+pe_sc` (78.90% CV balanced accuracy versus 78.20% for the
five-member refitted baseline on the same restricted validation cohort).
It was frozen before test metrics were computed.

On the exact same 480 newer and 225 reported test videos, newer-set mean balanced
accuracy fell from the deployed 85.02% to 82.72%. The release rule failed. The
live model and published accuracy figures are unchanged. The added candidate
checkpoint is identical to the existing fine-tune, so it supplies no independent
new model evidence. This outcome is not a fresh blind-holdout result and does not
rule out other paired-data training recipes.
