param(
    [int]$Port = 8000
)

$ErrorActionPreference = "Stop"

$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
$Python = Join-Path $Root ".venv\Scripts\python.exe"

if (-not (Test-Path -LiteralPath $Python)) {
    Write-Error "Missing .venv. Run: py -3.13 -m venv .venv; .\.venv\Scripts\python.exe -m pip install -r backend\requirements.txt"
    exit 1
}

& $Python -c "import fastapi, uvicorn, mediapipe" 2>$null
if ($LASTEXITCODE -ne 0) {
    Write-Error "Backend dependencies are incomplete. Run: .\.venv\Scripts\python.exe -m pip install -r backend\requirements.txt"
    exit 1
}

Write-Host "Starting backend with project virtual environment: $Python"
Write-Host "API: http://127.0.0.1:$Port"
& $Python -m uvicorn backend.main:app --host 127.0.0.1 --port $Port --reload
