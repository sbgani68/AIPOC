# AVD AI Monitoring PoC - Demo Script
# This script demonstrates the capabilities without requiring Azure credentials

Write-Host "`n=================================" -ForegroundColor Cyan
Write-Host "AVD AI Monitoring PoC - Quick Demo" -ForegroundColor Cyan
Write-Host "=================================`n" -ForegroundColor Cyan

# Check if venv is activated
if (-not $env:VIRTUAL_ENV) {
    Write-Host "Activating virtual environment..." -ForegroundColor Yellow
    .\venv\Scripts\Activate.ps1
}

Write-Host "Running tests to demonstrate functionality...`n" -ForegroundColor Green

# Run tests
pytest -v --tb=short

Write-Host "`n=================================" -ForegroundColor Cyan
Write-Host "Demo Complete!" -ForegroundColor Cyan
Write-Host "=================================`n" -ForegroundColor Cyan

Write-Host "Next steps:" -ForegroundColor Yellow
Write-Host "1. Configure .env with your Azure credentials" -ForegroundColor White
Write-Host "2. Run: python main.py (select demo option)" -ForegroundColor White
Write-Host "3. Or start API: uvicorn api:app --reload" -ForegroundColor White
Write-Host "4. Access docs at: http://localhost:8000/docs`n" -ForegroundColor White
