param(
    [string]$ConfigPath = ".\backend\livekit.yaml",
    [string]$InstallDir = ".\livekit-bin",
    [switch]$ForceInstall
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
& $livekitExe --config $resolvedConfigPath
