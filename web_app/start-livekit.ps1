param(
    [string]$ConfigPath = ".\backend\livekit.yaml",
    [string]$InstallDir = ".\livekit-bin",
    [switch]$ForceInstall,
    [switch]$Foreground
)

$ErrorActionPreference = 'Stop'

function Resolve-ScriptPath([string]$Path) {
    if ([System.IO.Path]::IsPathRooted($Path)) {
        return $Path
    }
    return Join-Path $PSScriptRoot $Path
}

function Install-LiveKit([string]$TargetDir) {
    $zipPath = Join-Path $PSScriptRoot "livekit.zip"

    Write-Host "Fetching latest LiveKit release info..."
    $release = Invoke-RestMethod -Uri "https://api.github.com/repos/livekit/livekit/releases/latest"
    $asset = $release.assets | Where-Object { $_.name -match "windows_amd64.zip" } | Select-Object -First 1

    if (-not $asset) {
        throw "Could not find a Windows AMD64 LiveKit binary in the latest release."
    }

    Write-Host "Downloading $($asset.name)..."
    Invoke-WebRequest -Uri $asset.browser_download_url -OutFile $zipPath

    Write-Host "Extracting LiveKit to $TargetDir..."
    if (Test-Path $TargetDir) {
        Remove-Item -Recurse -Force $TargetDir
    }
    Expand-Archive -Path $zipPath -DestinationPath $TargetDir -Force
    Remove-Item $zipPath
}

$resolvedInstallDir = Resolve-ScriptPath $InstallDir
$resolvedConfigPath = Resolve-ScriptPath $ConfigPath
$livekitExe = Join-Path $resolvedInstallDir "livekit-server.exe"

if ($ForceInstall -or -not (Test-Path $livekitExe)) {
    Install-LiveKit $resolvedInstallDir
} else {
    Write-Host "LiveKit is already installed at $livekitExe"
}

if (-not (Test-Path $resolvedConfigPath)) {
    throw "LiveKit config file not found: $resolvedConfigPath"
}

Write-Host "Starting LiveKit server..."
Write-Host "$livekitExe --config $resolvedConfigPath" -ForegroundColor Green
if ($Foreground) {
    & $livekitExe --config $resolvedConfigPath
    exit $LASTEXITCODE
}

$logsDir = Join-Path $PSScriptRoot "logs"
if (-not (Test-Path $logsDir)) {
    New-Item -ItemType Directory -Path $logsDir | Out-Null
}

$stdoutLog = Join-Path $logsDir "livekit.stdout.log"
$stderrLog = Join-Path $logsDir "livekit.stderr.log"

$process = Start-Process `
    -FilePath $livekitExe `
    -ArgumentList @("--config", $resolvedConfigPath) `
    -WorkingDirectory $PSScriptRoot `
    -RedirectStandardOutput $stdoutLog `
    -RedirectStandardError $stderrLog `
    -PassThru

Start-Sleep -Seconds 2
if ($process.HasExited) {
    throw "LiveKit exited immediately (exit code: $($process.ExitCode)). Check logs: $stdoutLog and $stderrLog"
}

Write-Host "LiveKit is running in background (PID: $($process.Id))." -ForegroundColor Green
Write-Host "Logs:" -ForegroundColor Cyan
Write-Host "  stdout: $stdoutLog"
Write-Host "  stderr: $stderrLog"
Write-Host "Stop command:" -ForegroundColor Cyan
Write-Host "  Stop-Process -Id $($process.Id)"
