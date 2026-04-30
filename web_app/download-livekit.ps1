$ErrorActionPreference = 'Stop'

Write-Host "Fetching latest LiveKit release info..."
$release = Invoke-RestMethod -Uri "https://api.github.com/repos/livekit/livekit/releases/latest"
$asset = $release.assets | Where-Object { $_.name -match "windows_amd64.zip" }

if (-not $asset) {
    Write-Error "Could not find Windows binary in latest release."
    exit 1
}

Write-Host "Downloading $($asset.name)..."
Invoke-WebRequest -Uri $asset.browser_download_url -OutFile "livekit.zip"

Write-Host "Extracting..."
if (Test-Path "livekit-bin") { Remove-Item -Recurse -Force "livekit-bin" }
Expand-Archive -Path "livekit.zip" -DestinationPath "livekit-bin" -Force
Remove-Item "livekit.zip"

Write-Host ""
Write-Host "=============================================" -ForegroundColor Cyan
Write-Host "LiveKit downloaded successfully." -ForegroundColor Green
Write-Host "To start LiveKit Server natively on Windows, run this command in a new terminal:" -ForegroundColor Yellow
Write-Host ".\livekit-bin\livekit-server.exe --config .\backend\livekit.yaml" -ForegroundColor Green
Write-Host "=============================================" -ForegroundColor Cyan
