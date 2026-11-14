# Run from a Windows session that can execute the existing GPU Python installation.
# Uses existing cached weights and writes candidate scores only; no deployment.
param(
    [string]$Python = 'C:\Users\hp\AppData\Local\Programs\Python\Python312\python.exe',
    [string]$Output = 'C:\Users\hp\dfd-work\round3-cache'
)
$ErrorActionPreference = 'Stop'
$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
if (-not (Test-Path -LiteralPath $Python)) { throw "GPU Python is unavailable: $Python" }
New-Item -ItemType Directory -Force -Path $Output | Out-Null
$Output = (Resolve-Path -LiteralPath $Output).Path
$env:HF_HUB_CACHE = 'D:\hf-cache'
$env:HF_HUB_OFFLINE = '1'
$env:TRANSFORMERS_OFFLINE = '1'
$env:GEND_SRC = 'C:\Users\hp\dfd-work\GenD'
$env:GEND_LN = 'D:\dfd-data\ens\ft_clip\best.pt'
$env:SCORE_SPLITS = 'val,test,test_open'
$env:PYTHONIOENCODING = 'utf-8'
Push-Location $repoRoot
try {
    & $Python -c 'import torch; assert torch.cuda.is_available(), "CUDA is required"'
    if ($LASTEXITCODE -ne 0) { throw 'GPU environment check failed' }
    foreach ($split in @('fit', 'new', 'rep')) {
        $target = Join-Path $Output "gend_clipft_$split.npz"
        if (Test-Path -LiteralPath $target) { throw "Output already exists; inspect it before rerunning: $target" }
        & $Python service/ens/score_gend.py clip $split $target 24
        if ($LASTEXITCODE -ne 0) { throw "Scoring failed: $split" }
    }
    $target = Join-Path $Output 'valdmg_clipft.npz'
    if (Test-Path -LiteralPath $target) { throw "Output already exists: $target" }
    & $Python service/ens/score_valdmg.py clipft $target
    if ($LASTEXITCODE -ne 0) { throw 'Damaged-validation scoring failed' }
    Write-Output "Candidate scores saved in $Output. Select the fusion on validation before viewing test metrics."
} finally {
    Pop-Location
}
