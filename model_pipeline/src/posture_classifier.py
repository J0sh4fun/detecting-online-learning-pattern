# Tệp: src/posture_classifier.py
import mediapipe as mp
import statistics
import numpy as np
import joblib
from collections import deque
from src.feature_utils import calculate_distance, get_midpoint, estimate_head_pose
from src.feature_schema import BASE_FEATURE_ORDER, FEATURE_ORDER
from ultralytics import YOLO

class PostureClassifier:
    def __init__(self):
        # 1. Khởi tạo MediaPipe
        self.mp_pose = mp.solutions.pose
        self.mp_face_mesh = mp.solutions.face_mesh
        
        # 2. Khởi tạo YOLO
        self.yolo_model = YOLO('yolo26s.pt')
        self.CELL_PHONE_CLASS_ID = 67 
        self.THRESH_PHONE_CONF = 0.55
        
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
        # Match training convention (cv2.flip(frame, 1)): mirror x-coordinate.
        return (int((1.0 - landmark.x) * w), int(landmark.y * h))

    def extract_features(self, landmarks, face_landmarks, w, h):
        """Trích xuất dữ liệu đặc trưng từ các điểm mốc (landmarks)"""
        nose_lm = landmarks[self.mp_pose.PoseLandmark.NOSE.value]
        l_shoulder_lm = landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value]
        r_shoulder_lm = landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
        l_ear_lm = landmarks[self.mp_pose.PoseLandmark.LEFT_EAR.value]
        r_ear_lm = landmarks[self.mp_pose.PoseLandmark.RIGHT_EAR.value]
        l_wrist_lm = landmarks[self.mp_pose.PoseLandmark.LEFT_WRIST.value]
        r_wrist_lm = landmarks[self.mp_pose.PoseLandmark.RIGHT_WRIST.value]

        nose = self.get_landmark_px(nose_lm, w, h)
        l_shoulder = self.get_landmark_px(l_shoulder_lm, w, h)
        r_shoulder = self.get_landmark_px(r_shoulder_lm, w, h)
        l_ear = self.get_landmark_px(l_ear_lm, w, h)
        r_ear = self.get_landmark_px(r_ear_lm, w, h)
        l_wrist = self.get_landmark_px(l_wrist_lm, w, h)
        r_wrist = self.get_landmark_px(r_wrist_lm, w, h)

        nose_z = nose_lm.z
        l_shoulder_z = l_shoulder_lm.z
        r_shoulder_z = r_shoulder_lm.z
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
        visible_wrist_count = 0
        
        for wrist_lm, wrist_px in [(l_wrist_lm, l_wrist), (r_wrist_lm, r_wrist)]:
            if wrist_lm.visibility > 0.2:
                visible_wrist_count += 1
                dist_face = min(calculate_distance(wrist_px, l_ear), calculate_distance(wrist_px, nose))
                min_hand_to_face = min(min_hand_to_face, dist_face / shoulder_width)
                if wrist_px[1] < chest_level:
                    wrist_elevated = True

        pose_x, pose_y, pose_z = 0, 0, 0
        if face_landmarks:
            pose_x, pose_y, pose_z = estimate_head_pose(face_landmarks, w, h)

        head_offset_x_ratio = (mid_ear[0] - mid_shoulder[0]) / shoulder_width
        head_offset_y_ratio = (mid_ear[1] - mid_shoulder[1]) / shoulder_width
        nose_shoulder_x_ratio = (nose[0] - mid_shoulder[0]) / shoulder_width
        nose_shoulder_y_ratio = (nose[1] - mid_shoulder[1]) / shoulder_width

        return {
            "neck_ratio": neck_ratio,
            "forward_lean_z": forward_lean_z,
            "shoulder_tilt_ratio": shoulder_tilt_ratio,
            "head_tilt_ratio": head_tilt_ratio,
            "hand_to_face_ratio": min_hand_to_face,
            "pose_x": pose_x,
            "pose_y": pose_y,
            "wrist_elevated": wrist_elevated,
            "shoulder_width_ratio": shoulder_width / max(float(w), 1.0),
            "head_offset_x_ratio": head_offset_x_ratio,
            "head_offset_y_ratio": head_offset_y_ratio,
            "nose_shoulder_x_ratio": nose_shoulder_x_ratio,
            "nose_shoulder_y_ratio": nose_shoulder_y_ratio,
            "abs_pose_x": abs(pose_x),
            "abs_pose_y": abs(pose_y),
            "face_detected": 1.0 if face_landmarks else 0.0,
            "nose_visibility": nose_lm.visibility,
            "shoulder_visibility_min": min(l_shoulder_lm.visibility, r_shoulder_lm.visibility),
            "ear_visibility_min": min(l_ear_lm.visibility, r_ear_lm.visibility),
            "left_wrist_visibility": l_wrist_lm.visibility,
            "right_wrist_visibility": r_wrist_lm.visibility,
            "visible_wrist_count": visible_wrist_count,
            "hand_visible": 1.0 if visible_wrist_count > 0 else 0.0,
            "coords": {"nose": nose, "mid_shoulder": mid_shoulder, "mid_ear": mid_ear}
        }

    def _predict_ml(self, features):
        """Dự đoán các tư thế khác bằng mô hình AI"""
        expected_feature_count = getattr(self.scaler, "n_features_in_", len(FEATURE_ORDER))
        feature_order = FEATURE_ORDER if expected_feature_count > len(BASE_FEATURE_ORDER) else BASE_FEATURE_ORDER
        feature_vector = np.array([[features[name] for name in feature_order[:expected_feature_count]]])

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
        # Only YOLO decides "Using Phone"
        if has_phone:
            raw_label = "Using Phone"
            
        else:
            # 3. Dùng AI dự đoán các tư thế như Focused, Slouching, v.v.
            raw_label = self._predict_ml(features)
            
        # Làm mịn kết quả bằng mode (nhãn xuất hiện nhiều nhất trong history)
        self.label_history.append(raw_label)
        return statistics.mode(self.label_history)
