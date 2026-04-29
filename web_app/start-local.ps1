Write-Host "Starting LiveKit..." -ForegroundColor Cyan
Set-Location "$PSScriptRoot"

$dockerCmd = Get-Command docker -ErrorAction SilentlyContinue
if (-not $dockerCmd) {
  Write-Host "Docker CLI is not installed. Install Docker Desktop, or run LiveKit on another host and update backend .env." -ForegroundColor Red
  exit 1
}

docker info *> $null
if ($LASTEXITCODE -ne 0) {
  Write-Host "Docker Desktop engine is not running (cannot access //./pipe/dockerDesktopLinuxEngine)." -ForegroundColor Red
  Write-Host "Fix:" -ForegroundColor Yellow
  Write-Host "1) Open Docker Desktop and wait until it shows 'Engine running'."
  Write-Host "2) Retry: docker compose -f .\docker-compose.livekit.yml up -d"
  Write-Host ""
  Write-Host "Fallback (no Docker): use a hosted LiveKit and set in backend\.env:" -ForegroundColor Yellow
  Write-Host "LIVEKIT_URL=wss://<your-livekit-host>"
  Write-Host "LIVEKIT_API_KEY=<key>"
  Write-Host "LIVEKIT_API_SECRET=<secret>"
  exit 1
}

docker compose -f .\docker-compose.livekit.yml up -d
if ($LASTEXITCODE -ne 0) {
  Write-Host "Failed to start LiveKit container." -ForegroundColor Red
  exit 1
}

Write-Host "Set backend environment variables..." -ForegroundColor Cyan
$env:LIVEKIT_URL = "ws://localhost:7880"
$env:LIVEKIT_API_KEY = "devkey"
$env:LIVEKIT_API_SECRET = "secret"
$env:CORS_ORIGINS = "http://localhost:5173"

Write-Host "Open new terminals and run:" -ForegroundColor Green
Write-Host "1) cd $PSScriptRoot\backend; pip install -r requirements.txt; uvicorn main:app --reload --port 8000"
Write-Host "2) cd $PSScriptRoot\frontend; npm install; npm run dev"
Write-Host ""
Write-Host "Stop LiveKit with: docker compose -f .\docker-compose.livekit.yml down" -ForegroundColor Yellow

