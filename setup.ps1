# setup.ps1 - Idempotent environment bootstrap
$Root = "C:\BybitBacktest"
Set-Location $Root

Write-Host "=== BybitBacktest Setup ===" -ForegroundColor Cyan

# Create venv if missing
if (-not (Test-Path "$Root\.venv")) {
    Write-Host "Creating virtual environment..." -ForegroundColor Yellow
    python -m venv .venv
}

# Activate
& "$Root\.venv\Scripts\Activate.ps1"

# Upgrade pip & install
python -m pip install --upgrade pip
pip install -r requirements.txt

Write-Host "`nSetup complete. Virtual environment activated." -ForegroundColor Green
Write-Host "You can now run the pipeline modules." -ForegroundColor Green
