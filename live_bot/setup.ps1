# ============================================================
#  BYBIT LIVE BOT - ONE-CLICK SETUP
# ============================================================
$OutputEncoding = [System.Text.Encoding]::UTF8
$env:PYTHONIOENCODING="utf-8"

Write-Host "===================================================" -ForegroundColor Cyan
Write-Host "  BYBIT LIVE BOT - SETUP" -ForegroundColor Cyan
Write-Host "===================================================" -ForegroundColor Cyan

Set-Location "C:\BybitLiveBot"

Write-Host "`n[1/5] Checking Python installation..." -ForegroundColor Yellow
if (-not (Get-Command python -ErrorAction SilentlyContinue)) {
    Write-Host "ERROR: Python not found. Install from python.org" -ForegroundColor Red
    exit 1
}
$pyVer = python --version 2>&1
Write-Host "      [OK] $pyVer" -ForegroundColor Green

Write-Host "[2/5] Installing Python dependencies..." -ForegroundColor Yellow
python -m pip install --quiet --disable-pip-version-check -r requirements.txt
if ($LASTEXITCODE -ne 0) {
    Write-Host "WARNING: pip install had errors. Check requirements.txt" -ForegroundColor Yellow
}
Write-Host "      [OK] Dependencies installed" -ForegroundColor Green

Write-Host "[3/5] Setting up .env file..." -ForegroundColor Yellow
if (-not (Test-Path ".env")) {
    Copy-Item ".env.example" ".env"
    Write-Host "      [OK] .env created from template" -ForegroundColor Green
    Write-Host "      [ACTION REQUIRED] Edit C:\BybitLiveBot\.env and fill in your Bybit API keys" -ForegroundColor Yellow
} else {
    Write-Host "      [OK] .env already exists" -ForegroundColor Green
}

Write-Host "[4/5] Creating logs folder..." -ForegroundColor Yellow
New-Item -ItemType Directory -Force -Path "logs" | Out-Null
Write-Host "      [OK] logs\ folder ready" -ForegroundColor Green

Write-Host "[5/5] Setup complete!" -ForegroundColor Green

Write-Host "`n===================================================" -ForegroundColor Cyan
Write-Host "  NEXT STEPS" -ForegroundColor Cyan
Write-Host "===================================================" -ForegroundColor Cyan
Write-Host "  1. Edit .env with your Bybit API keys:"
Write-Host "     notepad C:\BybitLiveBot\.env" -ForegroundColor Yellow
Write-Host ""
Write-Host "  2. Run the symbol scanner to check tradeability:"
Write-Host "     python C:\BybitLiveBot\symbol_scanner.py" -ForegroundColor Yellow
Write-Host ""
Write-Host "  3. If satisfied with the scan, start the bot:"
Write-Host "     python C:\BybitLiveBot\live_bot.py" -ForegroundColor Yellow
Write-Host ""
Write-Host "  4. Monitor logs in real-time:"
Write-Host "     Get-Content C:\BybitLiveBot\logs\bot_*.log -Wait -Tail 20" -ForegroundColor Yellow
Write-Host "==================================================="  -ForegroundColor Cyan
