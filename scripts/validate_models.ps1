# Sequential parity and sample checks on the existing local GPU environment.
$ErrorActionPreference = 'Stop'
$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$modelPython = 'C:\Users\hp\AppData\Local\Programs\Python\Python312\python.exe'
$dfdPython = 'C:\Users\hp\dfd-explain\DFD-FCG\.venv\Scripts\python.exe'
$env:HF_HUB_CACHE = 'D:\hf-cache'
$env:HF_HUB_OFFLINE = '1'
$env:TRANSFORMERS_OFFLINE = '1'
$env:GEND_SRC = 'C:\Users\hp\dfd-work\GenD'
$env:DFD_FCG_UPSTREAM = 'C:\Users\hp\dfd-work\DFD-FCG'
$env:DFD_FCG_CKPT = 'C:\Users\hp\B.Tech-Projects\checkpoint\weights.ckpt'
$env:DFD_DATA = 'D:\dfd-data\work'
$env:TORCHDYNAMO_DISABLE = '1'
$env:PYTHONIOENCODING = 'utf-8'
$env:MODEL_DIR = 'D:\dfd-data\ens\models'
Set-Location $repoRoot
foreach ($kind in @('clip', 'pe')) {
    Write-Output "BEGIN GenD $kind parity"
    & $modelPython tests/check_gend_live.py $kind
    if ($LASTEXITCODE -ne 0) { throw "GenD $kind parity failed" }
}
Write-Output 'BEGIN fusion parity'
& $dfdPython tests/check_fusion_live.py cuda
if ($LASTEXITCODE -ne 0) { throw 'Fusion parity failed' }
Write-Output 'BEGIN real and fake video samples'
& $dfdPython tests/run_live_samples.py cuda 'C:\Users\hp\dfd-work\DFD-FCG\resources\videos\000.mp4' 'C:\Users\hp\dfd-work\DFD-FCG\resources\videos\000_003.mp4'
if ($LASTEXITCODE -ne 0) { throw 'Sample-video check failed' }
Write-Output 'ALL MODEL CHECKS PASSED'
