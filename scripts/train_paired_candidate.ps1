param([string]$Output = 'D:\dfd-data\ens\ft_pe_paired_v1')
$ErrorActionPreference = 'Stop'
$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$modelPython = 'C:\Users\hp\AppData\Local\Programs\Python\Python312\python.exe'
if (Test-Path -LiteralPath $Output) { throw 'Training output already exists; inspect before resuming' }
$env:HF_HUB_CACHE = 'D:\hf-cache'
$env:HF_HUB_OFFLINE = '1'
$env:TRANSFORMERS_OFFLINE = '1'
$env:GEND_SRC = 'C:\Users\hp\dfd-work\GenD'
$env:PYTHONIOENCODING = 'utf-8'
$env:STORE = 'D:\dfd-data\ens\store_paired_v1'
$env:STORE_SETS = 'D:\dfd-data\new\sets.csv;D:\dfd-data\new\sets_gated.csv;D:\dfd-data\new\sets_more.csv;D:\dfd-data\ens\paired_v1\reals.csv'
$env:PAIRS = 'D:\dfd-data\ens\paired_v1\pairs.csv'
$env:MIN_PAIRS = '100'
$env:VAL_EXCLUDE_SOURCES = 'synth'
$env:SELECTION_METRIC = 'auc'
$env:INIT = 'D:\dfd-data\ens\ft_pe\best.pt'
$env:BS = '32'
$env:SEED = '1'
$env:WORKERS = '2'
$env:VAL_EVERY = '250'
$env:WARMUP = '100'
Set-Location $repoRoot
if (Test-Path -LiteralPath $env:STORE) { throw 'Candidate frame store already exists; inspect before rebuilding' }
& $modelPython service/ens/build_store.py $env:STORE
if ($LASTEXITCODE -ne 0) { throw 'Frame-store build failed' }
& $modelPython service/ens/train_gend.py $Output 1000 1e-5 pe
if ($LASTEXITCODE -ne 0) { throw 'Paired PE continuation failed' }
Write-Output 'PAIRED PE TRAINING COMPLETE; evaluate on validation before using any test metrics'
