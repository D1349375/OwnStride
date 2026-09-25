# OwnStride: start the local Ollama server used by the weekly planner.
#
# On this machine the default model folder (%USERPROFILE%\.ollama\models) inherits
# permissions without "delete", so Ollama cannot prune or finish downloads there.
# We keep models in %LOCALAPPDATA%\OwnStride\ollama-models instead and disable pruning.
#
# Usage (PowerShell):  .\scripts\start_ollama.ps1

$ollama = Join-Path $env:LOCALAPPDATA "Programs\Ollama\ollama.exe"
if (-not (Test-Path $ollama)) {
    Write-Error "Ollama not found at $ollama. Install it with: winget install --id Ollama.Ollama -e"
    exit 1
}

$env:OLLAMA_MODELS = Join-Path $env:LOCALAPPDATA "OwnStride\ollama-models"
$env:OLLAMA_NOPRUNE = "1"
New-Item -ItemType Directory -Force $env:OLLAMA_MODELS | Out-Null

# Stop any instance started without these settings (e.g. the tray app)
Get-Process | Where-Object { $_.ProcessName -like "ollama*" } | Stop-Process -Force -Confirm:$false

Start-Process -FilePath $ollama -ArgumentList "serve" -WindowStyle Hidden
Start-Sleep -Seconds 5

& $ollama list
Write-Host "Ollama running at http://localhost:11434 (models in $env:OLLAMA_MODELS)."
Write-Host "If qwen2.5:1.5b is missing, run: & '$ollama' pull qwen2.5:1.5b"
