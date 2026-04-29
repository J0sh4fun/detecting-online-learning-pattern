param(
  [switch]$SkipDocker
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $MyInvocation.MyCommand.Path
$backendDir = Join-Path $root "backend"
$frontendDir = Join-Path $root "frontend"
$venvActivate = Join-Path (Split-Path -Parent $root) ".venv\Scripts\Activate.ps1"

if (-not (Test-Path (Join-Path $backendDir ".env")) -and (Test-Path (Join-Path $backendDir ".env.example"))) {
  Copy-Item (Join-Path $backendDir ".env.example") (Join-Path $backendDir ".env")
  Write-Host "Created backend\.env from .env.example" -ForegroundColor Cyan
}

if (-not (Test-Path (Join-Path $frontendDir ".env")) -and (Test-Path (Join-Path $frontendDir ".env.example"))) {
  Copy-Item (Join-Path $frontendDir ".env.example") (Join-Path $frontendDir ".env")
  Write-Host "Created frontend\.env from .env.example" -ForegroundColor Cyan
}

if (-not $SkipDocker) {
  $dockerCmd = Get-Command docker -ErrorAction SilentlyContinue
  if ($dockerCmd) {
    docker info *> $null
    if ($LASTEXITCODE -eq 0) {
      Write-Host "Starting LiveKit container..." -ForegroundColor Cyan
      Set-Location $root
      docker compose -f .\docker-compose.livekit.yml up -d
    } else {
      Write-Host "Docker is installed but engine is not running. Skipping LiveKit startup." -ForegroundColor Yellow
      Write-Host "Start Docker Desktop or run this script with -SkipDocker and use hosted LiveKit." -ForegroundColor Yellow
    }
  } else {
    Write-Host "Docker is not installed. Skipping LiveKit startup." -ForegroundColor Yellow
  }
}

$backendCmd = @"
Set-Location '$backendDir'
if (Test-Path '$venvActivate') { . '$venvActivate' }
`$env:LIVEKIT_URL='ws://localhost:7880'
`$env:LIVEKIT_API_KEY='devkey'
`$env:LIVEKIT_API_SECRET='secret'
`$env:CORS_ORIGINS='http://localhost:5173'
uvicorn main:app --reload --port 8000
"@

$frontendCmd = @"
Set-Location '$frontendDir'
npm run dev
"@

Write-Host "Starting backend and frontend terminals..." -ForegroundColor Green
Start-Process powershell -ArgumentList "-NoExit", "-Command", $backendCmd | Out-Null
Start-Process powershell -ArgumentList "-NoExit", "-Command", $frontendCmd | Out-Null

Write-Host "Done. Backend: http://127.0.0.1:8000 | Frontend: http://localhost:5173" -ForegroundColor Green

