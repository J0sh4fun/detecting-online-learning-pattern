# ============================================================
# reset_db.ps1
# MUC DICH : Toan bo reset Docker + MySQL volume + schema moi.
#             Dung khi can dev fresh start hoac doi sang schema moi.
#
# CANH BAO: XOA TOAN BO DU LIEU MYSQL!
#
# CACH CHAY:
#   cd web_app
#   .\scripts\reset_db.ps1
# ============================================================

param(
    [switch]$Force  # Bo qua xac nhan neu truyen -Force
)

$ErrorActionPreference = "Stop"
$ScriptDir  = Split-Path -Parent $MyInvocation.MyCommand.Path
$ComposeDir = Split-Path -Parent $ScriptDir   # web_app/
$ComposeFile = Join-Path $ComposeDir "docker-compose.yml"

Write-Host ""
Write-Host "=============================================" -ForegroundColor Cyan
Write-Host "  Focus Classroom -- Full DB Reset Script   " -ForegroundColor Cyan
Write-Host "=============================================" -ForegroundColor Cyan
Write-Host ""

if (-not $Force) {
    Write-Host "[CANH BAO] Script nay se:" -ForegroundColor Yellow
    Write-Host "  1. Dung TAT CA Docker containers (backend, frontend, mysql)"
    Write-Host "  2. Xoa volume mysql_data (TOAN BO DU LIEU BI XOA)"
    Write-Host "  3. Khoi dong lai containers"
    Write-Host "  4. Ap dung schema moi tu reset_db.sql"
    Write-Host ""
    $confirm = Read-Host "Go 'yes' de tiep tuc, bat ky phim nao khac de huy"
    if ($confirm -ne "yes") {
        Write-Host "Da huy." -ForegroundColor Yellow
        exit 0
    }
}

# ── Buoc 1: Dung containers ──────────────────────────────────
Write-Host ""
Write-Host "[1/5] Dung Docker containers..." -ForegroundColor Green
docker compose -f $ComposeFile down
if ($LASTEXITCODE -ne 0) {
    Write-Host "Khong the dung containers (co the chua chay). Tiep tuc..." -ForegroundColor Yellow
}

# ── Buoc 2: Xoa volume ───────────────────────────────────────
Write-Host ""
Write-Host "[2/5] Xoa MySQL data volume..." -ForegroundColor Green
# Thu nhieu ten volume (docker compose co the them prefix)
$volumeNames = @("web_app_mysql_data", "mysql_data", "focus_classroom_mysql_data")
foreach ($vol in $volumeNames) {
    docker volume rm $vol 2>$null
}
docker volume prune -f | Out-Null
Write-Host "  Volume da duoc xoa."

# ── Buoc 3: Khoi dong lai ────────────────────────────────────
Write-Host ""
Write-Host "[3/5] Khoi dong lai Docker containers..." -ForegroundColor Green
docker compose -f $ComposeFile up -d mysql
if ($LASTEXITCODE -ne 0) {
    Write-Host "Loi khi khoi dong MySQL container!" -ForegroundColor Red
    exit 1
}

# ── Buoc 4: Doi MySQL ready ──────────────────────────────────
Write-Host ""
Write-Host "[4/5] Doi MySQL san sang..." -ForegroundColor Green
$maxRetries = 30
$retries    = 0
$ready      = $false
do {
    Start-Sleep -Seconds 3
    $ping = docker exec focus-mysql mysqladmin ping -uroot -pfocusdev --silent 2>$null
    if ($LASTEXITCODE -eq 0) { $ready = $true; break }
    $retries++
    Write-Host "  Thu lan $retries/$maxRetries..."
} while ($retries -lt $maxRetries)

if (-not $ready) {
    Write-Host "MySQL khong san sang sau $maxRetries lan thu!" -ForegroundColor Red
    Write-Host "Kiem tra log: docker logs focus-mysql"
    exit 1
}
Write-Host "  MySQL san sang!"

# ── Buoc 5: Ap dung schema moi ───────────────────────────────
Write-Host ""
Write-Host "[5/5] Ap dung reset_db.sql (schema moi)..." -ForegroundColor Green
$sqlFile = Join-Path $ScriptDir "reset_db.sql"
Get-Content $sqlFile | docker exec -i focus-mysql mysql -uroot -pfocusdev
if ($LASTEXITCODE -ne 0) {
    Write-Host "Loi khi ap dung SQL!" -ForegroundColor Red
    exit 1
}

# ── Khoi dong phan con lai ───────────────────────────────────
Write-Host ""
Write-Host "Khoi dong backend va frontend..." -ForegroundColor Green
docker compose -f $ComposeFile up -d
if ($LASTEXITCODE -ne 0) {
    Write-Host "Loi khi khoi dong cac services!" -ForegroundColor Red
    exit 1
}

# ── Tong ket ─────────────────────────────────────────────────
Write-Host ""
Write-Host "=============================================" -ForegroundColor Green
Write-Host "  RESET THANH CONG!                        " -ForegroundColor Green
Write-Host "=============================================" -ForegroundColor Green
Write-Host ""
Write-Host "Trang thai containers:" -ForegroundColor Cyan
docker compose -f $ComposeFile ps
Write-Host ""
Write-Host "Backend API : http://localhost:8000"
Write-Host "Frontend    : http://localhost:5173"
Write-Host "MySQL port  : localhost:3307"
Write-Host ""
Write-Host "Ghi chu: Backend se tu dong tao lai tables khi khoi dong." -ForegroundColor Yellow
Write-Host "         Tuy nhien reset_db.sql da tao san tables voi schema moi." -ForegroundColor Yellow
