# Tệp: src/posture_classifier.py
import mediapipe as mp
import statistics
import numpy as np
import joblib
from collections import deque
from src.feature_utils import calculate_distance, get_midpoint, estimate_head_pose
from ultralytics import YOLO

class PostureClassifier:
    def __init__(self):
        # 1. Khởi tạo MediaPipe
        self.mp_pose = mp.solutions.pose
        self.mp_face_mesh = mp.solutions.face_mesh
        
        # 2. Khởi tạo YOLO (Tăng Thresh lên 0.55 để không nhận nhầm bàn phím)
        self.yolo_model = YOLO('yolo26s.pt')
        self.CELL_PHONE_CLASS_ID = 67 
        self.THRESH_PHONE_CONF = 0.3
        
        # 3. TẢI BỘ NÃO AI (Mô hình và Scaler)
        try:
            self.scaler = joblib.load('models/scaler.pkl')
            self.model = joblib.load('models/best_posture_model.pkl')
        except Exception as e:
            print(f"LỖI: Không tìm thấy tệp mô hình. Chi tiết: {e}")

        # 4. Bộ đệm thời gian để làm mịn nhãn
        self.history_length = 15 
        self.label_history = deque(maxlen=self.history_length)

    def detect_phone(self, frame):
        """Sử dụng YOLO để phát hiện điện thoại"""
        results = self.yolo_model(frame, classes=[self.CELL_PHONE_CLASS_ID], device='0', verbose=False)
        for r in results:
            for box in r.boxes:
                if box.conf[0].item() > self.THRESH_PHONE_CONF:
                    return True 
        return False

    def get_landmark_px(self, landmark, w, h):
        return (int(landmark.x * w), int(landmark.y * h))

    def extract_features(self, landmarks, face_landmarks, w, h):
        """Trích xuất dữ liệu đặc trưng từ các điểm mốc (landmarks)"""
        nose = self.get_landmark_px(landmarks[self.mp_pose.PoseLandmark.NOSE.value], w, h)
        l_shoulder = self.get_landmark_px(landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value], w, h)
        r_shoulder = self.get_landmark_px(landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value], w, h)
        l_ear = self.get_landmark_px(landmarks[self.mp_pose.PoseLandmark.LEFT_EAR.value], w, h)
        r_ear = self.get_landmark_px(landmarks[self.mp_pose.PoseLandmark.RIGHT_EAR.value], w, h)
        l_wrist = self.get_landmark_px(landmarks[self.mp_pose.PoseLandmark.LEFT_WRIST.value], w, h)
        r_wrist = self.get_landmark_px(landmarks[self.mp_pose.PoseLandmark.RIGHT_WRIST.value], w, h)

        nose_z = landmarks[self.mp_pose.PoseLandmark.NOSE.value].z
        l_shoulder_z = landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value].z
        r_shoulder_z = landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value].z
        mid_shoulder_z = (l_shoulder_z + r_shoulder_z) / 2

        mid_shoulder = get_midpoint(l_shoulder, r_shoulder)
        mid_ear = get_midpoint(l_ear, r_ear)
        shoulder_width = calculate_distance(l_shoulder, r_shoulder) or 1 

        neck_ratio = abs(mid_shoulder[1] - mid_ear[1]) / shoulder_width
        forward_lean_z = mid_shoulder_z - nose_z 
        
        shoulder_tilt_ratio = abs(l_shoulder[1] - r_shoulder[1]) / shoulder_width
        head_tilt_ratio = abs(l_ear[1] - r_ear[1]) / shoulder_width

        chest_level = mid_shoulder[1] + (shoulder_width * 0.5)
        wrist_elevated = False
        min_hand_to_face = 999.0
        
        for wrist_lm, wrist_px in [(landmarks[15], l_wrist), (landmarks[16], r_wrist)]:
            if wrist_lm.visibility > 0.2:
                dist_face = min(calculate_distance(wrist_px, l_ear), calculate_distance(wrist_px, nose))
                min_hand_to_face = min(min_hand_to_face, dist_face / shoulder_width)
                if wrist_px[1] < chest_level:
                    wrist_elevated = True

        pose_x, pose_y, pose_z = 0, 0, 0
        if face_landmarks:
            pose_x, pose_y, pose_z = estimate_head_pose(face_landmarks, w, h)

        return {
            "neck_ratio": neck_ratio,
            "forward_lean_z": forward_lean_z,
            "shoulder_tilt_ratio": shoulder_tilt_ratio,
            "head_tilt_ratio": head_tilt_ratio,
            "hand_to_face_ratio": min_hand_to_face,
            "pose_x": pose_x,
            "pose_y": pose_y, 
            "wrist_elevated": wrist_elevated,
            "coords": {"nose": nose, "mid_shoulder": mid_shoulder, "mid_ear": mid_ear}
        }

    def _predict_ml(self, features, has_phone):
        """Dự đoán các tư thế khác bằng mô hình AI"""
        feature_vector = np.array([[
            features['neck_ratio'], 
            features['forward_lean_z'], 
            features['shoulder_tilt_ratio'], 
            features['head_tilt_ratio'], 
            features['hand_to_face_ratio'], 
            features['pose_x'], 
            features['pose_y'], 
            int(features['wrist_elevated']), 
            int(has_phone)
        ]])

        scaled_vector = self.scaler.transform(feature_vector)
        prediction = self.model.predict(scaled_vector)
        return prediction[0]

    def classify(self, features, landmarks, has_phone=False):
        """
        Luồng xử lý ưu tiên: 
        1. Kiểm tra vắng mặt (Absence)
        2. Ưu tiên kết quả YOLO (Using Phone)
        3. Sử dụng mô hình AI cho các tư thế còn lại
        """
        # 1. Kiểm tra vắng mặt bằng độ hiển thị của MediaPipe
        visibility_nose = landmarks[self.mp_pose.PoseLandmark.NOSE.value].visibility
        visibility_l_shoulder = landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value].visibility
        visibility_r_shoulder = landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value].visibility
        
        if visibility_nose < 0.3 and visibility_l_shoulder < 0.3 and visibility_r_shoulder < 0.3:
            raw_label = "Absence"
        
        # 2. ƯU TIÊN YOLO: Nếu YOLO thấy điện thoại, gán nhãn ngay lập tức
        elif has_phone:
            raw_label = "Using Phone"
            
        else:
            # 3. Dùng AI dự đoán các tư thế như Focused, Slouching, v.v.
            raw_label = self._predict_ml(features, has_phone)
            
        # Làm mịn kết quả bằng mode (nhãn xuất hiện nhiều nhất trong history)
        self.label_history.append(raw_label)
        return statistics.mode(self.label_history)