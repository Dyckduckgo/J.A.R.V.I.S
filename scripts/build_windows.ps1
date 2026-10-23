# Build a standalone Windows distribution in dist\JARVIS\ (requires the dev venv).
# Usage (from the repository root):  powershell -ExecutionPolicy Bypass -File scripts\build_windows.ps1
$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root

$python = Join-Path $root ".venv\Scripts\python.exe"
if (-not (Test-Path $python)) {
    throw "Virtual environment not found. Create it first: python -m venv .venv; .venv\Scripts\python -m pip install -e .[dev,build]"
}

& $python -m pip install --quiet -e ".[build]"
if ($LASTEXITCODE -ne 0) { throw "pip install failed" }

& $python -m PyInstaller --noconfirm --clean packaging\jarvis.spec
if ($LASTEXITCODE -ne 0) { throw "PyInstaller failed" }

$exe = Join-Path $root "dist\JARVIS\JARVIS.exe"
Write-Host "Built $exe"

# Verify the frozen build starts and exits cleanly without a display.
$env:QT_QPA_PLATFORM = "offscreen"
$env:JARVIS_HOME = Join-Path $env:TEMP "jarvis-build-smoke"
$proc = Start-Process -FilePath $exe -ArgumentList "--smoke-test" -PassThru -Wait
if ($proc.ExitCode -ne 0) { throw "Smoke test of the frozen build failed with exit code $($proc.ExitCode)" }
Write-Host "Smoke test passed."
