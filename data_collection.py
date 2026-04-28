# Tệp: data_collector.py
import cv2
import csv
import os
import mediapipe as mp
from src.posture_classifier import PostureClassifier

# Đường dẫn tệp CSV
DATA_DIR = "data"
CSV_FILE = os.path.join(DATA_DIR, "posture_dataset.csv")

# Tạo thư mục nếu chưa có
os.makedirs(DATA_DIR, exist_ok=True)

# Khởi tạo CSV và viết Header
if not os.path.exists(CSV_FILE):
    with open(CSV_FILE, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'neck_ratio', 'forward_lean_z', 'shoulder_tilt', 'head_tilt', 
            'hand_to_face', 'pose_x', 'pose_y', 
            'wrist_elevated', 'has_phone', 'label'
        ])

def main():
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5)
    
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, min_detection_confidence=0.5)
    
    classifier = PostureClassifier()
    cap = cv2.VideoCapture(0)

    print("=== DATA COLLECTOR ===")
    print("Phím 0: Focused | 1: Slouching | 2: Leaning | 3: Looking Away | 4: Using Phone | 5: Absence")
    print("Nhấn 'q' để thoát.")

    while True:
        ret, frame = cap.read()
        if not ret: break
        
        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # UI Hướng dẫn
        cv2.putText(frame, "0:Focus 1:Slouch 2:Lean 3:Away 4:Phone 5:Absence", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

        results = pose.process(image_rgb)
        mesh_results = face_mesh.process(image_rgb)
        has_phone = classifier.detect_phone(frame)

        # --- FIX: Move waitKey outside the IF block so the window never freezes ---
        key = cv2.waitKey(1) & 0xFF
        label = None
        
        if key == ord('0'): label = "Focused"
        elif key == ord('1'): label = "Slouching"
        elif key == ord('2'): label = "Leaning on Desk"
        elif key == ord('3'): label = "Looking Away"
        elif key == ord('4'): label = "Using Phone"
        elif key == ord('5'): label = "Absence"
        elif key == ord('q'): break

        # Prepare feature values
        data_to_save = None

        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            face_landmarks = mesh_results.multi_face_landmarks[0] if mesh_results.multi_face_landmarks else None
            features = classifier.extract_features(landmarks, face_landmarks, w, h)
            
            if label:
                data_to_save = [
                    round(features['neck_ratio'], 4),
                    round(features['forward_lean_z'], 4),
                    round(features['shoulder_tilt_ratio'], 4),
                    round(features['head_tilt_ratio'], 4),
                    round(features['hand_to_face_ratio'], 4),
                    round(features['pose_x'], 4),
                    round(features['pose_y'], 4),
                    int(features['wrist_elevated']),
                    int(has_phone),
                    label
                ]
        
        # --- Handle Absence label specifically when landmarks are NOT found ---
        elif label == "Absence":
            data_to_save = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0, 0, "Absence"]

        # Save to CSV if we have a valid label and data
        if data_to_save:
            with open(CSV_FILE, mode='a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(data_to_save)
            print(f" Đã lưu mẫu: {data_to_save[-1]}")
            cv2.rectangle(frame, (0, 0), (w, h), (0, 255, 0), 10) 

        cv2.imshow('Data Collector', frame)

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
