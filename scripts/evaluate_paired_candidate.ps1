# Sequential GPU queue after the training process; writes candidate results only.
param([int]$TrainingProcessId, [string]$Output = 'D:\dfd-data\ens\paired_scores_v1')
$ErrorActionPreference = 'Stop'
$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$modelPython = 'C:\Users\hp\AppData\Local\Programs\Python\Python312\python.exe'
if ($TrainingProcessId -and (Get-Process -Id $TrainingProcessId -ErrorAction SilentlyContinue)) {
    Wait-Process -Id $TrainingProcessId
}
$log = 'D:\dfd-data\ens\ft_pe_paired_v1\train_log.jsonl'
if (-not (Test-Path -LiteralPath $log)) { throw 'Training log missing' }
$last = Get-Content -LiteralPath $log -Tail 1 | ConvertFrom-Json
if ($last.event -ne 'done') { throw 'Training did not finish; candidate evaluation stopped' }
# The first run retained step zero and redundantly rescored the same weights.
# Future runs can stop here: train_gend only replaces best.pt on strict gain.
if ($last.best -le $last.base) {
    Write-Output 'No checkpoint improved the starting model; duplicate scoring/evaluation skipped. No deployment performed.'
    exit 0
}
New-Item -ItemType Directory -Force -Path $Output | Out-Null
$env:HF_HUB_CACHE = 'D:\hf-cache'
$env:HF_HUB_OFFLINE = '1'
$env:TRANSFORMERS_OFFLINE = '1'
$env:GEND_SRC = 'C:\Users\hp\dfd-work\GenD'
$env:GEND_LN = 'D:\dfd-data\ens\ft_pe_paired_v1\best.pt'
$env:DFD_ENS_DIR = $Output
$env:SCORE_SPLITS = 'val,test,test_open'
$env:FUSION_CANDIDATE = 'pe_pair'
$env:PYTHONIOENCODING = 'utf-8'
Set-Location $repoRoot
foreach ($split in @('fit', 'new', 'rep')) {
    $target = Join-Path $Output "gend_pe_pair_$split.npz"
    if (Test-Path -LiteralPath $target) { throw "Candidate scores already exist: $target" }
    & $modelPython service/ens/score_gend.py pe $split $target 24
    if ($LASTEXITCODE -ne 0) { throw "Scoring failed: $split" }
}
$target = Join-Path $Output 'valdmg_pe_pair.npz'
if (Test-Path -LiteralPath $target) { throw "Candidate scores already exist: $target" }
& $modelPython service/ens/score_valdmg.py pe_pair $target
if ($LASTEXITCODE -ne 0) { throw 'Damaged-validation scoring failed' }
& $modelPython service/ens/select_clip_fusion.py select --output results/ensemble/paired_v1
if ($LASTEXITCODE -ne 0) { throw 'Validation selection failed' }
# select persists coefficients and hashes before test is allowed to run.
& $modelPython service/ens/select_clip_fusion.py test --output results/ensemble/paired_v1
if ($LASTEXITCODE -ne 0) { throw 'Frozen test evaluation failed' }
Write-Output 'PAIRED CANDIDATE EVALUATION COMPLETE; no deployment performed'
