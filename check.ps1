# One command for all main checks: .\check.ps1
$ErrorActionPreference = "Stop"

function Invoke-Check($label, [scriptblock]$command) {
    Write-Host "`n=== $label ===" -ForegroundColor Cyan
    & $command
    if ($LASTEXITCODE -ne 0) { throw "$label failed" }
}

Invoke-Check "ruff lint" { python -m ruff check . }
Invoke-Check "ruff format" { python -m ruff format --check . }
Invoke-Check "mypy" { python -m mypy }
Invoke-Check "pytest" { python -m pytest -q }
Write-Host "`nAll checks passed." -ForegroundColor Green
