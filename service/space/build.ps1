# Assemble the Hugging Face Space folder (Gradio + ZeroGPU): app.py, checker/ (service code), qa/ (pipeline modules).
param([string]$Out = "$PSScriptRoot\..\..\..\space-zerogpu")
$svc = Resolve-Path "$PSScriptRoot\.."
$root = Resolve-Path "$PSScriptRoot\..\.."
if (Test-Path $Out) { Get-ChildItem $Out -Force | Where-Object { $_.Name -ne ".git" } | Remove-Item -Recurse -Force }
New-Item -ItemType Directory -Force "$Out\qa", "$Out\checker" | Out-Null
Copy-Item "$PSScriptRoot\app.py", "$PSScriptRoot\requirements.txt", "$PSScriptRoot\packages.txt", "$PSScriptRoot\README.md" $Out
Copy-Item "$svc\app\pipeline.py", "$svc\app\models.py", "$svc\app\db.py", "$svc\app\model.json", "$svc\app\gend_live.py", "$svc\app\reliability.py" "$Out\checker"
# fusion of several detectors (round 2): weights, read-outs and fine-tuned GenD tensors, if present
foreach ($f in Get-ChildItem "$svc\app" -File | Where-Object { $_.Name -match '^(fusion\.json|model_.*\.json|gend_.*\.pt)$' }) {
    Copy-Item $f.FullName "$Out\checker"
}
foreach ($m in "__init__", "config", "conditions", "engine", "prep", "probes", "manifest", "quality") {
    Copy-Item "$root\qa\$m.py" "$Out\qa"
}
# Linux xargs preserves CR characters in packages.txt as part of package names.
# Make artifacts independent of a Windows checkout's core.autocrlf setting.
$spaceEncoding = New-Object System.Text.UTF8Encoding($false)
foreach ($entry in Get-ChildItem -LiteralPath $Out -Recurse -File | Where-Object { $_.Extension -in '.py', '.txt', '.md', '.json' }) {
    $textContent = [System.IO.File]::ReadAllText($entry.FullName).Replace("`r`n", "`n")
    [System.IO.File]::WriteAllText($entry.FullName, $textContent, $spaceEncoding)
}
"space folder in $((Resolve-Path $Out).Path)"
