from __future__ import annotations

from dataclasses import dataclass

import cv2
import joblib
import mediapipe as mp
import numpy as np

from app.core.config import settings


@dataclass
class VerificationResult:
    score: float
    status: str


class VerificationScorer:
    """
    Runs server-side posture verification from random frames.
    """

    def __init__(self) -> None:
        self.scaler = joblib.load(settings.scaler_path)
        self.classifier = joblib.load(settings.classifier_path)
        self.mp_pose = mp.solutions.pose
        self.mp_face_mesh = mp.solutions.face_mesh
        self.pose = self.mp_pose.Pose(
            static_image_mode=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            min_detection_confidence=0.5,
        )

    @staticmethod
    def _distance(p1: tuple[float, float], p2: tuple[float, float]) -> float:
        return float(np.hypot(p2[0] - p1[0], p2[1] - p1[1]))

    @staticmethod
    def _midpoint(p1: tuple[float, float], p2: tuple[float, float]) -> tuple[float, float]:
        return ((p1[0] + p2[0]) / 2.0, (p1[1] + p2[1]) / 2.0)

    @staticmethod
    def _head_pose(face_landmarks, w: int, h: int) -> tuple[float, float]:
        nose = face_landmarks.landmark[1]
        left_eye = face_landmarks.landmark[33]
        right_eye = face_landmarks.landmark[263]

        lx, ly, lz = (1.0 - left_eye.x) * w, left_eye.y * h, left_eye.z * w
        rx, ry, rz = (1.0 - right_eye.x) * w, right_eye.y * h, right_eye.z * w
        nx, ny, nz = (1.0 - nose.x) * w, nose.y * h, nose.z * w

        yaw = float(np.degrees(np.arctan2(rz - lz, rx - lx)))
        mid_eye_y, mid_eye_z = (ly + ry) / 2.0, (lz + rz) / 2.0
        pitch = float(np.degrees(np.arctan2(nz - mid_eye_z, ny - mid_eye_y)))
        return pitch, yaw

    def _check_absence(self, pose_landmarks):
        """Xác thực vắng mặt dựa trên độ hiển thị (visibility) thay vì chỉ check None"""
        if not pose_landmarks:
            return True
            
        nose = pose_landmarks.landmark[self.mp_pose.PoseLandmark.NOSE]
        left_shoulder = pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_SHOULDER]
        right_shoulder = pose_landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_SHOULDER]
        
        # Ngưỡng 0.3 đồng bộ với script training
        if nose.visibility < 0.3 or (left_shoulder.visibility < 0.3 and right_shoulder.visibility < 0.3):
            return True
            
        return False

    def _extract_features(self, frame: np.ndarray) -> list[float] | None:
        """
        Trích xuất đầy đủ 23 đặc trưng (8 base + 15 extra) đồng bộ với client và pipeline training.
        Trả về None nếu không phát hiện được dáng người hoặc rơi vào trạng thái vắng mặt.
        """
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pose_results = self.pose.process(rgb)
        
        # 1. Kiểm tra nếu MediaPipe không phát hiện ra khung xương dáng người
        if not pose_results.pose_landmarks:
            return None

        landmarks = pose_results.pose_landmarks.landmark
        
        # Lấy thông tin các điểm mốc chính
        nose_lm = landmarks[self.mp_pose.PoseLandmark.NOSE.value]
        l_shoulder_lm = landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value]
        r_shoulder_lm = landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
        l_ear_lm = landmarks[self.mp_pose.PoseLandmark.LEFT_EAR.value]
        r_ear_lm = landmarks[self.mp_pose.PoseLandmark.RIGHT_EAR.value]
        l_wrist_lm = landmarks[self.mp_pose.PoseLandmark.LEFT_WRIST.value]
        r_wrist_lm = landmarks[self.mp_pose.PoseLandmark.RIGHT_WRIST.value]

        # 2. Kiểm tra ngưỡng hiển thị (Visibility) để xác định trạng thái Vắng mặt (Absence)
        # Đồng bộ 100% với logic ưu tiên của file posture_classifier.py
        if nose_lm.visibility < 0.3 and l_shoulder_lm.visibility < 0.3 and r_shoulder_lm.visibility < 0.3:
            return None

        face_results = self.face_mesh.process(rgb)
        w = frame.shape[1]
        h = frame.shape[0]

        def to_px(index: int) -> tuple[float, float, float]:
            lm = landmarks[index]
            return (float((1.0 - lm.x) * w), float(lm.y * h), float(lm.z))

        # Chuyển đổi tọa độ pixel (đảo ngược trục X để đồng bộ góc nhìn gương giống lúc thu thập dữ liệu)
        nose = to_px(self.mp_pose.PoseLandmark.NOSE.value)
        l_shoulder = to_px(self.mp_pose.PoseLandmark.LEFT_SHOULDER.value)
        r_shoulder = to_px(self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value)
        l_ear = to_px(self.mp_pose.PoseLandmark.LEFT_EAR.value)
        r_ear = to_px(self.mp_pose.PoseLandmark.RIGHT_EAR.value)
        l_wrist = to_px(self.mp_pose.PoseLandmark.LEFT_WRIST.value)
        r_wrist = to_px(self.mp_pose.PoseLandmark.RIGHT_WRIST.value)

        # Tính toán BASE FEATURES (8 đặc trưng ban đầu)
        shoulder_width = self._distance(l_shoulder[:2], r_shoulder[:2]) or 1.0
        mid_shoulder = self._midpoint(l_shoulder[:2], r_shoulder[:2])
        mid_ear = self._midpoint(l_ear[:2], r_ear[:2])
        mid_shoulder_z = (l_shoulder[2] + r_shoulder[2]) / 2.0

        neck_ratio = abs(mid_shoulder[1] - mid_ear[1]) / shoulder_width
        forward_lean_z = mid_shoulder_z - nose[2]
        shoulder_tilt_ratio = abs(l_shoulder[1] - r_shoulder[1]) / shoulder_width
        head_tilt_ratio = abs(l_ear[1] - r_ear[1]) / shoulder_width

        chest_level = mid_shoulder[1] + (shoulder_width * 0.5)
        
        min_hand_to_face = 999.0
        wrist_elevated = False
        visible_wrist_count = 0

        for wrist_lm, wrist in [(l_wrist_lm, l_wrist), (r_wrist_lm, r_wrist)]:
            if wrist_lm.visibility > 0.2:
                visible_wrist_count += 1
                dist_face = min(
                    self._distance(wrist[:2], l_ear[:2]),
                    self._distance(wrist[:2], nose[:2]),
                )
                min_hand_to_face = min(min_hand_to_face, dist_face / shoulder_width)
                if wrist[1] < chest_level:
                    wrist_elevated = True

        pose_x = 0.0
        pose_y = 0.0
        if face_results.multi_face_landmarks:
            pose_x, pose_y = self._head_pose(face_results.multi_face_landmarks[0], w, h)

        # Tính toán EXTRA FEATURES (15 đặc trưng mở rộng từ bản cập nhật mới)
        head_offset_x_ratio = (mid_ear[0] - mid_shoulder[0]) / shoulder_width
        head_offset_y_ratio = (mid_ear[1] - mid_shoulder[1]) / shoulder_width
        nose_shoulder_x_ratio = (nose[0] - mid_shoulder[0]) / shoulder_width
        nose_shoulder_y_ratio = (nose[1] - mid_shoulder[1]) / shoulder_width

        # Trả về mảng một chiều phẳng sắp xếp chuẩn theo thứ tự FEATURE_ORDER của feature_schema.py
        return [
            # 8 thuộc tính thuộc BASE_FEATURE_ORDER
            float(neck_ratio),
            float(forward_lean_z),
            float(shoulder_tilt_ratio),
            float(head_tilt_ratio),
            float(min_hand_to_face),
            float(pose_x),
            float(pose_y),
            1.0 if wrist_elevated else 0.0,
            
            # 15 thuộc tính thuộc EXTRA_FEATURE_ORDER
            float(shoulder_width / max(float(w), 1.0)),
            float(head_offset_x_ratio),
            float(head_offset_y_ratio),
            float(nose_shoulder_x_ratio),
            float(nose_shoulder_y_ratio),
            float(abs(pose_x)),
            float(abs(pose_y)),
            1.0 if face_results.multi_face_landmarks else 0.0,
            float(nose_lm.visibility),
            float(min(l_shoulder_lm.visibility, r_shoulder_lm.visibility)),
            float(min(l_ear_lm.visibility, r_ear_lm.visibility)),
            float(l_wrist_lm.visibility),
            float(r_wrist_lm.visibility),
            float(visible_wrist_count),
            1.0 if visible_wrist_count > 0 else 0.0,
        ]

    def score_frame(self, frame: np.ndarray) -> VerificationResult:
        """
        Thực hiện chấm điểm frame ảnh ngẫu nhiên được gửi lên từ client.
        Phương thức này chạy đồng bộ và an toàn để tích hợp vào ThreadPool của FastAPI.
        """
        # 1. Trích xuất đặc trưng hình học cơ thể
        features = self._extract_features(frame)
        
        # 2. Nếu không có đặc trưng hoặc dính checkpoint vắng mặt, trả kết quả Absence ngay lập tức
        if features is None:
            return VerificationResult(score=5.0, status="Absence")

        try:
            # 3. Định hình lại mảng dữ liệu thành dạng 2D [1, N_features] để đưa vào bộ nén Scaler
            features_matrix = np.array([features], dtype=np.float64)
            scaled = self.scaler.transform(features_matrix)
            
            # 4. Dự đoán nhãn tư thế bằng mô hình học máy (classifier)
            label = str(self.classifier.predict(scaled)[0])
            
            # 5. Khớp nhãn với thang điểm tương ứng định nghĩa trong hệ thống
            return VerificationResult(score=self._label_to_score(label), status=self._display_label(label))
            
        except ValueError as e:
            # Cơ chế tự động bắt lỗi và báo cáo chính xác nếu có lệch số lượng đặc trưng đầu vào
            expected_features = getattr(self.scaler, "n_features_in_", "N/A")
            print(f"[CRITICAL ERROR] Mismatch dimension in server-side AI pipeline! "
                  f"Scaler expects {expected_features} features, but extracted {len(features)}. Detail: {e}")
            
            # Trả về fallback an toàn để tránh crash luồng chính của hệ thống lớp học trực tuyến
            return VerificationResult(score=55.0, status="Error")

    @staticmethod
    def _label_to_score(label: str) -> float:
        map_value = {
            "Focused": 95.0,
            "Slouched": 70.0,
            "Slouching": 70.0,
            "Looking Away": 62.0,
            "Leaning on Desk": 45.0,
            "Using Phone": 20.0,
            "Absence": 5.0,
        }
        return map_value.get(label, 55.0)

    @staticmethod
    def _display_label(label: str) -> str:
        if label == "Slouching":
            return "Slouched"
        return label

_verification_scorer_instance = None


def get_verification_scorer() -> VerificationScorer:
    global _verification_scorer_instance
    if _verification_scorer_instance is None:
        _verification_scorer_instance = VerificationScorer()
    return _verification_scorer_instance

