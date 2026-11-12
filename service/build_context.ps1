# Assemble the Docker build context (service files + the qa modules the pipeline needs).
param([string]$Out = "$PSScriptRoot\..\..\space-build")
$root = Resolve-Path "$PSScriptRoot\.."
if (Test-Path $Out) { Get-ChildItem $Out -Force | Where-Object { $_.Name -ne ".git" } | Remove-Item -Recurse -Force }
New-Item -ItemType Directory -Force "$Out\qa", "$Out\app" | Out-Null
Copy-Item "$PSScriptRoot\Dockerfile", "$PSScriptRoot\requirements.txt", "$PSScriptRoot\README.md" $Out
Copy-Item "$PSScriptRoot\app\*.py", "$PSScriptRoot\app\model.json" "$Out\app"
foreach ($m in "__init__", "config", "conditions", "engine", "prep", "probes", "manifest", "quality") {
    Copy-Item "$root\qa\$m.py" "$Out\qa"
}
"build context in $Out"
