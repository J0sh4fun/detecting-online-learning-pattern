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

    def _extract_features(self, frame: np.ndarray) -> list[float] | None:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pose_results = self.pose.process(rgb)
        if not pose_results.pose_landmarks:
            return None

        face_results = self.face_mesh.process(rgb)
        landmarks = pose_results.pose_landmarks.landmark
        w = frame.shape[1]
        h = frame.shape[0]

        def to_px(index: int) -> tuple[float, float, float]:
            lm = landmarks[index]
            return (float((1.0 - lm.x) * w), float(lm.y * h), float(lm.z))

        nose = to_px(self.mp_pose.PoseLandmark.NOSE.value)
        l_shoulder = to_px(self.mp_pose.PoseLandmark.LEFT_SHOULDER.value)
        r_shoulder = to_px(self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value)
        l_ear = to_px(self.mp_pose.PoseLandmark.LEFT_EAR.value)
        r_ear = to_px(self.mp_pose.PoseLandmark.RIGHT_EAR.value)
        l_wrist = to_px(self.mp_pose.PoseLandmark.LEFT_WRIST.value)
        r_wrist = to_px(self.mp_pose.PoseLandmark.RIGHT_WRIST.value)

        shoulder_width = self._distance(l_shoulder[:2], r_shoulder[:2]) or 1.0
        mid_shoulder = self._midpoint(l_shoulder[:2], r_shoulder[:2])
        mid_ear = self._midpoint(l_ear[:2], r_ear[:2])
        mid_shoulder_z = (l_shoulder[2] + r_shoulder[2]) / 2.0

        neck_ratio = abs(mid_shoulder[1] - mid_ear[1]) / shoulder_width
        forward_lean_z = mid_shoulder_z - nose[2]
        shoulder_tilt_ratio = abs(l_shoulder[1] - r_shoulder[1]) / shoulder_width
        head_tilt_ratio = abs(l_ear[1] - r_ear[1]) / shoulder_width

        chest_level = mid_shoulder[1] + (shoulder_width * 0.5)
        wrists = [l_wrist, r_wrist]
        min_hand_to_face = 999.0
        wrist_elevated = False

        for wrist in wrists:
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

        has_phone_like_gesture = min_hand_to_face < 0.18 and wrist_elevated
        return [
            float(neck_ratio),
            float(forward_lean_z),
            float(shoulder_tilt_ratio),
            float(head_tilt_ratio),
            float(min_hand_to_face),
            float(pose_x),
            float(pose_y),
            1.0 if wrist_elevated else 0.0,
            1.0 if has_phone_like_gesture else 0.0,
        ]

    @staticmethod
    def _label_to_score(label: str) -> float:
        map_value = {
            "Focused": 95.0,
            "Slouching": 70.0,
            "Looking Away": 62.0,
            "Leaning on Desk": 45.0,
            "Using Phone": 20.0,
            "Absence": 5.0,
        }
        return map_value.get(label, 55.0)

    def score_frame(self, frame: np.ndarray) -> VerificationResult:
        features = self._extract_features(frame)
        if not features:
            return VerificationResult(score=5.0, status="Absence")

        scaled = self.scaler.transform(np.array([features], dtype=np.float64))
        label = str(self.classifier.predict(scaled)[0])
        return VerificationResult(score=self._label_to_score(label), status=label)


_verification_scorer_instance = None


def get_verification_scorer() -> VerificationScorer:
    global _verification_scorer_instance
    if _verification_scorer_instance is None:
        _verification_scorer_instance = VerificationScorer()
    return _verification_scorer_instance

