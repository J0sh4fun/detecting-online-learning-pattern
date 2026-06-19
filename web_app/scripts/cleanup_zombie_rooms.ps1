# ============================================================
# cleanup_zombie_rooms.ps1
# MUC DICH : Chi don dep cac phong zombie (status=active)
#             MA KHONG mat bat ky du lieu nao.
#             Container MySQL phai dang chay.
#
# CACH CHAY:
#   cd web_app
#   .\scripts\cleanup_zombie_rooms.ps1
# ============================================================

$ErrorActionPreference = "Stop"
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$SqlFile   = Join-Path $ScriptDir "cleanup_zombie_rooms.sql"

Write-Host ""
Write-Host "=============================================" -ForegroundColor Cyan
Write-Host "  Focus Classroom -- Cleanup Zombie Rooms   " -ForegroundColor Cyan
Write-Host "=============================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Script nay chi UPDATE trang thai, KHONG xoa du lieu." -ForegroundColor Green
Write-Host ""

# Kiem tra container dang chay
$running = docker inspect --format "{{.State.Running}}" focus-mysql 2>$null
if ($running -ne "true") {
    Write-Host "MySQL container (focus-mysql) chua chay!" -ForegroundColor Red
    Write-Host "Chay truoc: docker compose up -d mysql"
    exit 1
}

Write-Host "Dang chay cleanup_zombie_rooms.sql..." -ForegroundColor Yellow
Get-Content $SqlFile | docker exec -i focus-mysql mysql -uroot -pfocusdev

if ($LASTEXITCODE -eq 0) {
    Write-Host ""
    Write-Host "Don dep zombie rooms thanh cong!" -ForegroundColor Green
} else {
    Write-Host ""
    Write-Host "Co loi khi chay SQL. Kiem tra container logs." -ForegroundColor Red
    exit 1
}
