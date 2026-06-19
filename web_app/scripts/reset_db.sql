-- ============================================================
-- reset_db.sql
-- MỤC ĐÍCH: Xóa và tạo lại TOÀN BỘ database với schema mới.
--           Các thay đổi so với schema cũ:
--           - room_participants: ĐÃ XÓA cột role, camera_on,
--             current_score, current_status (dữ liệu live chỉ
--             lưu trong in-memory cache của server)
--           - left_at: giờ được ghi khi học sinh disconnect
--
-- !! CẢNH BÁO: XÓA TOÀN BỘ DỮ LIỆU !!
--
-- CÁCH CHẠY:
--   docker exec -i focus-mysql mysql -uroot -pfocusdev < web_app/scripts/reset_db.sql
--
-- HOẶC dùng PowerShell wrapper (khuyến nghị):
--   .\web_app\scripts\reset_db.ps1
-- ============================================================

-- Xóa và tạo lại database
DROP DATABASE IF EXISTS focus_classroom;
CREATE DATABASE focus_classroom
    CHARACTER SET utf8mb4
    COLLATE utf8mb4_unicode_ci;
USE focus_classroom;

-- ────────────────────────────────────────────────────────────
-- users
-- ────────────────────────────────────────────────────────────
CREATE TABLE users (
    id            INT          NOT NULL AUTO_INCREMENT,
    username      VARCHAR(64)  NOT NULL,
    password_hash VARCHAR(255) NOT NULL,
    role          ENUM('teacher','student') NOT NULL,
    created_at    DATETIME     NOT NULL DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (id),
    UNIQUE  KEY uq_username (username),
    INDEX        idx_username (username)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- ────────────────────────────────────────────────────────────
-- rooms
-- ────────────────────────────────────────────────────────────
CREATE TABLE rooms (
    id           INT          NOT NULL AUTO_INCREMENT,
    room_code    VARCHAR(12)  NOT NULL,
    room_name    VARCHAR(120) NOT NULL,
    teacher_id   INT          NOT NULL,
    max_students INT          NOT NULL DEFAULT 20,
    status       ENUM('active','ended') NOT NULL DEFAULT 'active',
    created_at   DATETIME     NOT NULL DEFAULT CURRENT_TIMESTAMP,
    ended_at     DATETIME     NULL,
    PRIMARY KEY (id),
    UNIQUE  KEY uq_room_code (room_code),
    INDEX        idx_room_code  (room_code),
    INDEX        idx_teacher_id (teacher_id),
    CONSTRAINT fk_rooms_teacher
        FOREIGN KEY (teacher_id) REFERENCES users(id) ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- ────────────────────────────────────────────────────────────
-- room_participants  (SCHEMA MỚI — đã xóa các cột cache)
-- ────────────────────────────────────────────────────────────
CREATE TABLE room_participants (
    id                 INT         NOT NULL AUTO_INCREMENT,
    room_id            INT         NOT NULL,
    user_id            INT         NOT NULL,
    display_id         VARCHAR(64) NOT NULL,
    -- role        đã xóa — tất cả participant đều là student
    -- camera_on   đã xóa — live state nằm trong server cache
    -- current_score   đã xóa — live state nằm trong server cache
    -- current_status  đã xóa — live state nằm trong server cache
    joined_at          DATETIME    NOT NULL DEFAULT CURRENT_TIMESTAMP,
    left_at            DATETIME    NULL,                -- ghi khi student disconnect
    last_score_update  DATETIME    NULL,
    last_ingest_epoch  FLOAT       NOT NULL DEFAULT 0.0,
    PRIMARY KEY (id),
    INDEX idx_room_id (room_id),
    INDEX idx_user_id (user_id),
    CONSTRAINT fk_rp_room FOREIGN KEY (room_id) REFERENCES rooms(id)  ON DELETE CASCADE,
    CONSTRAINT fk_rp_user FOREIGN KEY (user_id) REFERENCES users(id)  ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- ────────────────────────────────────────────────────────────
-- focus_scores  (không thay đổi)
-- ────────────────────────────────────────────────────────────
CREATE TABLE focus_scores (
    id             INT         NOT NULL AUTO_INCREMENT,
    participant_id INT         NOT NULL,
    room_id        INT         NOT NULL,
    score          FLOAT       NOT NULL,
    status_label   VARCHAR(80) NOT NULL,
    camera_on      TINYINT(1)  NOT NULL,
    recorded_at    DATETIME    NOT NULL DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (id),
    INDEX idx_participant_id (participant_id),
    INDEX idx_room_id        (room_id),
    INDEX idx_recorded_at    (recorded_at),
    CONSTRAINT fk_fs_participant FOREIGN KEY (participant_id) REFERENCES room_participants(id) ON DELETE CASCADE,
    CONSTRAINT fk_fs_room        FOREIGN KEY (room_id)        REFERENCES rooms(id)             ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- ────────────────────────────────────────────────────────────
-- verification_flags  (không thay đổi)
-- ────────────────────────────────────────────────────────────
CREATE TABLE verification_flags (
    id             INT         NOT NULL AUTO_INCREMENT,
    participant_id INT         NOT NULL,
    room_id        INT         NOT NULL,
    client_score   FLOAT       NOT NULL,
    server_score   FLOAT       NOT NULL,
    server_status  VARCHAR(80) NOT NULL,
    discrepancy    FLOAT       NOT NULL,
    resolved       TINYINT(1)  NOT NULL DEFAULT 0,
    created_at     DATETIME    NOT NULL DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (id),
    INDEX idx_participant_id (participant_id),
    INDEX idx_room_id        (room_id),
    CONSTRAINT fk_vf_participant FOREIGN KEY (participant_id) REFERENCES room_participants(id) ON DELETE CASCADE,
    CONSTRAINT fk_vf_room        FOREIGN KEY (room_id)        REFERENCES rooms(id)             ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- ────────────────────────────────────────────────────────────
-- room_reports  (không thay đổi)
-- ────────────────────────────────────────────────────────────
CREATE TABLE room_reports (
    id                  INT   NOT NULL AUTO_INCREMENT,
    room_id             INT   NOT NULL,
    class_average_score FLOAT NOT NULL,
    total_students      INT   NOT NULL,
    student_summaries   JSON  NOT NULL,
    generated_at        DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (id),
    UNIQUE  KEY uq_room_id (room_id),
    INDEX        idx_room_id (room_id),
    CONSTRAINT fk_rr_room FOREIGN KEY (room_id) REFERENCES rooms(id) ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- Xác nhận
SHOW TABLES;
SELECT 'Schema reset complete ✅' AS result;
