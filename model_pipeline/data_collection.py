# Tệp: data_collector.py
import cv2
import csv
import os
import time
import mediapipe as mp
from uuid import uuid4
from src.posture_classifier import PostureClassifier
from src.feature_schema import FEATURE_ALIASES, FEATURE_ORDER, METADATA_COLUMNS, NO_HAND_VISIBLE_RATIO, OUTPUT_COLUMNS

# Đường dẫn tệp CSV
DATA_DIR = "data"
CSV_FILE = os.path.join(DATA_DIR, "posture_dataset_train.csv")
COLLECTION_INTERVAL_SEC = 0.25
CANONICAL_FIELDNAMES = OUTPUT_COLUMNS
LEGACY_COLUMN_ALIASES = {**FEATURE_ALIASES, "label": ["label"]}

os.makedirs(DATA_DIR, exist_ok=True)

def get_csv_fieldnames():
    if os.path.exists(CSV_FILE):
        with open(CSV_FILE, mode="r", newline="") as f:
            reader = csv.DictReader(f)
            header = reader.fieldnames
            rows = list(reader)
        if header and all(name in header for name in CANONICAL_FIELDNAMES):
            return CANONICAL_FIELDNAMES
        if header:
            migrate_csv_schema(rows)
            return CANONICAL_FIELDNAMES

    fieldnames = CANONICAL_FIELDNAMES
    with open(CSV_FILE, mode="w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
    return fieldnames


def migrate_csv_schema(rows):
    migrated_rows = []
    for row in rows:
        migrated = {name: "" for name in CANONICAL_FIELDNAMES}
        for column, aliases in LEGACY_COLUMN_ALIASES.items():
            for alias in aliases:
                value = row.get(alias)
                if value not in (None, ""):
                    migrated[column] = value
                    break
        migrated_rows.append(migrated)

    with open(CSV_FILE, mode="w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CANONICAL_FIELDNAMES)
        writer.writeheader()
        writer.writerows(migrated_rows)


def clamp_hand_to_face_ratio(value):
    if value >= 999.0:
        return NO_HAND_VISIBLE_RATIO
    return min(value, NO_HAND_VISIBLE_RATIO)


def build_csv_row(fieldnames, metadata, features, label):
    canonical_values = {name: round(float(features.get(name, 0.0)), 4) for name in FEATURE_ORDER}
    canonical_values["hand_to_face_ratio"] = round(clamp_hand_to_face_ratio(features["hand_to_face_ratio"]), 4)
    canonical_values["wrist_elevated"] = int(features["wrist_elevated"])
    canonical_values["visible_wrist_count"] = int(features.get("visible_wrist_count", 0))
    canonical_values["hand_visible"] = int(features.get("hand_visible", 0))
    canonical_values["face_detected"] = int(features.get("face_detected", 0))
    canonical_values["label"] = label
    canonical_values.update(metadata)
    return {name: canonical_values.get(name, "") for name in fieldnames}

def main():
    csv_fieldnames = get_csv_fieldnames()

    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5)
    
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, min_detection_confidence=0.5)
    
    classifier = PostureClassifier()
    cap = cv2.VideoCapture(0)

    # Biến trạng thái
    is_collecting = False
    is_counting_down = False
    countdown_start_time = 0
    current_label = None
    session_id = uuid4().hex[:12]
    capture_group = None
    frame_index = 0
    last_saved_at = 0.0
    
    # Đã chỉnh sửa: Chỉ giữ lại 4 nhãn 0, 1, 2, 3
    label_map = {
        ord('0'): "Focused",
        ord('1'): "Slouching",
        ord('2'): "Leaning on Desk",
        ord('3'): "Looking Away"
    }

    # Đã chỉnh sửa: Cập nhật menu terminal
    print("=== DATA COLLECTOR (TOGGLE MODE) ===")
    print("Phím 0: Focused | 1: Slouching | 2: Leaning | 3: Looking Away")
    print("Nhấn lại phím đó để dừng. Nhấn 'q' để thoát.")

    while True:
        ret, frame = cap.read()
        if not ret: break
        
        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Đã chỉnh sửa: Cập nhật hướng dẫn trên cửa sổ (Window Menu)
        cv2.putText(frame, "0:Focus 1:Slouch 2:Lean 3:Away", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

        results = pose.process(image_rgb)
        mesh_results = face_mesh.process(image_rgb)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key in label_map:
            input_label = label_map[key]
            
            if is_collecting and current_label == input_label:
                is_collecting = False
                current_label = None
                capture_group = None
            elif not is_collecting and not is_counting_down:
                current_label = input_label
                is_counting_down = True
                countdown_start_time = time.time()
                capture_group = f"{session_id}_{input_label.replace(' ', '_')}_{int(countdown_start_time)}"
                last_saved_at = 0.0

        # Logic đếm ngược
        if is_counting_down:
            elapsed = time.time() - countdown_start_time
            remaining = 3 - int(elapsed)
            if remaining > 0:
                cv2.putText(frame, f"GET READY: {remaining}", (w//2 - 100, h//2), 
                            cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 5)
            else:
                is_counting_down = False
                is_collecting = True

        # Logic thu thập dữ liệu
        if is_collecting:
            data_to_save = None
            now = time.time()
            
            if results.pose_landmarks and now - last_saved_at >= COLLECTION_INTERVAL_SEC:
                landmarks = results.pose_landmarks.landmark
                face_landmarks = mesh_results.multi_face_landmarks[0] if mesh_results.multi_face_landmarks else None
                features = classifier.extract_features(landmarks, face_landmarks, w, h)

                metadata = {
                    "session_id": session_id,
                    "capture_group": capture_group,
                    "captured_at": round(now, 3),
                    "frame_index": frame_index,
                }
                data_to_save = build_csv_row(csv_fieldnames, metadata, features, current_label)
                last_saved_at = now
                frame_index += 1
            
            # Đã xóa phần kiểm tra "Absence" vì nhãn này không còn được sử dụng

            if data_to_save:
                with open(CSV_FILE, mode='a', newline='') as f:
                    writer = csv.DictWriter(f, fieldnames=csv_fieldnames)
                    writer.writerow(data_to_save)
                # Chỉ báo đang quay (Recording indicator)
                cv2.rectangle(frame, (0, 0), (w, h), (0, 0, 255), 10)
                cv2.putText(frame, f"REC: {current_label}", (10, h - 20), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        cv2.imshow('Data Collector', frame)

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
