# Tệp: data_collector.py
import cv2
import csv
import os
import time
import mediapipe as mp
from src.posture_classifier import PostureClassifier

# Đường dẫn tệp CSV
DATA_DIR = "data"
CSV_FILE = os.path.join(DATA_DIR, "posture_dataset.csv")

os.makedirs(DATA_DIR, exist_ok=True)

if not os.path.exists(CSV_FILE):
    with open(CSV_FILE, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'neck_ratio', 'forward_lean_z', 'shoulder_tilt', 'head_tilt', 
            'hand_to_face', 'pose_x', 'pose_y', 
            'wrist_elevated', 'label'
        ])

def main():
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
            elif not is_collecting and not is_counting_down:
                current_label = input_label
                is_counting_down = True
                countdown_start_time = time.time()

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
            
            if results.pose_landmarks:
                landmarks = results.pose_landmarks.landmark
                face_landmarks = mesh_results.multi_face_landmarks[0] if mesh_results.multi_face_landmarks else None
                features = classifier.extract_features(landmarks, face_landmarks, w, h)
                
                data_to_save = [
                    round(features['neck_ratio'], 4),
                    round(features['forward_lean_z'], 4),
                    round(features['shoulder_tilt_ratio'], 4),
                    round(features['head_tilt_ratio'], 4),
                    round(features['hand_to_face_ratio'], 4),
                    round(features['pose_x'], 4),
                    round(features['pose_y'], 4),
                    int(features['wrist_elevated']),
                    current_label
                ]
            
            # Đã xóa phần kiểm tra "Absence" vì nhãn này không còn được sử dụng

            if data_to_save:
                with open(CSV_FILE, mode='a', newline='') as f:
                    writer = csv.writer(f)
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
