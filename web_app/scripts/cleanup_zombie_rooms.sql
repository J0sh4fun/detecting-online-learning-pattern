-- ============================================================
-- cleanup_zombie_rooms.sql
-- MỤC ĐÍCH: Dọn sạch các phòng "zombie" (status=active nhưng
--           server đã tắt) mà KHÔNG xóa bất kỳ dữ liệu nào.
-- AN TOÀN:  Script này chỉ UPDATE, không DROP hay DELETE.
--
-- CÁCH CHẠY (container đang chạy):
--   docker exec -i focus-mysql mysql -uroot -pfocusdev focus_classroom < web_app/scripts/cleanup_zombie_rooms.sql
--
-- HOẶC dùng PowerShell wrapper:
--   .\web_app\scripts\cleanup_zombie_rooms.ps1
-- ============================================================

USE focus_classroom;

-- Bước 1: Xem có bao nhiêu phòng zombie
SELECT
    COUNT(*) AS zombie_rooms_found,
    GROUP_CONCAT(room_code ORDER BY created_at DESC SEPARATOR ', ') AS room_codes
FROM rooms
WHERE status = 'active';

-- Bước 2: Đánh dấu tất cả active rooms là ended
UPDATE rooms
SET
    status    = 'ended',
    ended_at  = NOW()
WHERE status = 'active';

-- Bước 3: Xác nhận kết quả
SELECT
    room_code,
    room_name,
    status,
    DATE_FORMAT(created_at, '%Y-%m-%d %H:%i') AS created_at,
    DATE_FORMAT(ended_at,   '%Y-%m-%d %H:%i') AS ended_at
FROM rooms
ORDER BY created_at DESC
LIMIT 10;
