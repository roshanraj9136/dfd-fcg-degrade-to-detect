# One GPU preparation job; no model weights or test videos are changed.
param([string]$Output = 'D:\dfd-data\ens\paired_v1')
$ErrorActionPreference = 'Stop'
$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$modelPython = 'C:\Users\hp\AppData\Local\Programs\Python\Python312\python.exe'
$dfdPython = 'C:\Users\hp\dfd-explain\DFD-FCG\.venv\Scripts\python.exe'
$env:PYTHONIOENCODING = 'utf-8'
$env:DFD_FCG_UPSTREAM = 'C:\Users\hp\dfd-work\DFD-FCG'
$env:DFD_DATA = 'D:\dfd-data\work'
$env:TORCHDYNAMO_DISABLE = '1'
$env:PREP_THREADS = '4'
$env:DET_EVERY = '24'
$env:LM_EVERY = '3'
$env:PREP_LOGS = Join-Path $Output 'run.out'
Set-Location $repoRoot
if (Test-Path -LiteralPath (Join-Path $Output 'audit.json')) {
    & $modelPython service/ens/prepare_pairs.py --out $Output --resume-download
} else {
    & $modelPython service/ens/prepare_pairs.py --out $Output --download
}
if ($LASTEXITCODE -ne 0) { throw 'Pair manifest/download failed' }
& $dfdPython service/train/prep_crops.py cuda 'D:\dfd-data\crops' (Join-Path $Output 'reals.csv')
if ($LASTEXITCODE -ne 0) { throw 'Pair face preparation failed' }
Write-Output 'PAIRED REAL PREPARATION COMPLETE'
