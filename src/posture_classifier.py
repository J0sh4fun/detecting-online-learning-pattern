import mediapipe as mp
import statistics
from collections import deque
from src.feature_utils import calculate_distance, get_midpoint, estimate_head_pose
from ultralytics import YOLO

class PostureClassifier:
    """
    Classification engine for student posture and attention detection.
    
    Uses MediaPipe Pose/FaceMesh for geometry and YOLO for phone detection,
    applying temporal smoothing to ensure stable status updates.
    """
    
    def __init__(self):
        """Initializes AI models, temporal buffers, and geometric thresholds."""
        self.mp_pose = mp.solutions.pose
        self.mp_face_mesh = mp.solutions.face_mesh
        self.baseline_features = None 
        
        # --- TEMPORAL SMOOTHING SETUP ---
        # Stores the labels of the last 20 frames (~0.7 - 1 second)
        self.history_length = 20
        self.label_history = deque(maxlen=self.history_length)
        
        # --- GEOMETRIC THRESHOLDS ---
        self.THRESH_HAND_TO_FACE = 0.55  # Distance ratio for hand close to face
        self.THRESH_SLOUCH_Y = 0.55      # Neck drop ratio (allows slight drop for writing)
        self.THRESH_SLOUCH_Z = 1.40      # Forward lean ratio (z-axis depth)

        # --- PHONE THRESHOLDS ---
        self.yolo_model = YOLO('yolov8s.pt')
        self.CELL_PHONE_CLASS_ID = 67 
        self.THRESH_PHONE_CONF = 0.35    # Confidence Score
        self.THRESH_CALLING = 0.25
        
    def detect_phone(self, frame):
        """
        Detects if a mobile phone is present in the current frame using YOLO.

        Args:
            frame: The current BGR image from the webcam.

        Returns:
            bool: True if a phone is detected with confidence > threshold.
        """
        results = self.yolo_model(frame, classes=[self.CELL_PHONE_CLASS_ID], device='0', verbose=False)
        for r in results:
            boxes = r.boxes
            for box in boxes:
                if box.conf[0].item() > self.THRESH_PHONE_CONF:
                    return True 
        return False

    def get_landmark_px(self, landmark, w, h):
        """
        Converts normalized landmark coordinates to absolute pixel coordinates.
        
        Args:
            landmark (object): MediaPipe landmark object containing x, y.
            w (int): Width of the image frame.
            h (int): Height of the image frame.
            
        Returns:
            tuple: (x, y) absolute pixel coordinates.
        """
        return (int(landmark.x * w), int(landmark.y * h))

    def extract_features(self, landmarks, face_landmarks, w, h):
        """
        Extracts all necessary posture features from raw MediaPipe data.

        Args:
            landmarks: MediaPipe Pose landmarks.
            face_landmarks: MediaPipe Face Mesh landmarks.
            w (int): Frame width.
            h (int): Frame height.

        Returns:
            dict: Dictionary of calculated ratios, angles, and coordinates.
        """
        # Extract absolute coordinates
        nose = self.get_landmark_px(landmarks[self.mp_pose.PoseLandmark.NOSE.value], w, h)
        l_shoulder = self.get_landmark_px(landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value], w, h)
        r_shoulder = self.get_landmark_px(landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value], w, h)
        l_ear = self.get_landmark_px(landmarks[self.mp_pose.PoseLandmark.LEFT_EAR.value], w, h)
        r_ear = self.get_landmark_px(landmarks[self.mp_pose.PoseLandmark.RIGHT_EAR.value], w, h)
        l_wrist = self.get_landmark_px(landmarks[self.mp_pose.PoseLandmark.LEFT_WRIST.value], w, h)
        r_wrist = self.get_landmark_px(landmarks[self.mp_pose.PoseLandmark.RIGHT_WRIST.value], w, h)

        # Extract depth (Z-axis) coordinates
        nose_z = landmarks[self.mp_pose.PoseLandmark.NOSE.value].z
        l_shoulder_z = landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value].z
        r_shoulder_z = landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value].z
        mid_shoulder_z = (l_shoulder_z + r_shoulder_z) / 2

        # Calculate midpoints and reference widths
        mid_shoulder = get_midpoint(l_shoulder, r_shoulder)
        mid_ear = get_midpoint(l_ear, r_ear)
        shoulder_width = calculate_distance(l_shoulder, r_shoulder) or 1 

        # Calculate core features
        neck_length_y = abs(mid_shoulder[1] - mid_ear[1])
        neck_ratio = neck_length_y / shoulder_width
        forward_lean_z = mid_shoulder_z - nose_z 
        
        shoulder_tilt_ratio = abs(l_shoulder[1] - r_shoulder[1]) / shoulder_width
        head_tilt_ratio = abs(l_ear[1] - r_ear[1]) / shoulder_width

        # Caculate head yaw
        left_dist = abs(nose[0] - l_ear[0])
        right_dist = abs(nose[0] - r_ear[0])
        # If looking straight ahead, the ratio is approximately 1. If turning your head, 
        # the ratio will be very large (>2.5) or very small (<0.4).
        head_yaw_ratio = left_dist / (right_dist + 0.001)
        
        # Check elevated wrist
        wrist_elevated = False
        # Wrists higher than chest (mid-shoulder length + half shoulder width)
        chest_level = mid_shoulder[1] + (shoulder_width * 0.5)
        
        # Calculate minimum distance from either hand to the face
        min_hand_to_face = 999.0
        min_hand_to_ear = 999.0
        
        # High sensitivity (visibility > 0.2) to detect partially occluded hands
        if landmarks[self.mp_pose.PoseLandmark.LEFT_WRIST.value].visibility > 0.2:
            dist = min(calculate_distance(l_wrist, l_ear), calculate_distance(l_wrist, nose))
            min_hand_to_face = min(min_hand_to_face, dist / shoulder_width)
            min_hand_to_ear = min(min_hand_to_ear, calculate_distance(l_wrist, l_ear) / shoulder_width)
            
        if landmarks[self.mp_pose.PoseLandmark.RIGHT_WRIST.value].visibility > 0.2:
            dist = min(calculate_distance(r_wrist, r_ear), calculate_distance(r_wrist, nose))
            min_hand_to_face = min(min_hand_to_face, dist / shoulder_width)
            min_hand_to_ear = min(min_hand_to_ear, calculate_distance(r_wrist, r_ear) / shoulder_width)

        if landmarks[self.mp_pose.PoseLandmark.LEFT_WRIST.value].visibility > 0.2:
            if l_wrist[1] < chest_level:
                wrist_elevated = True
                
        if landmarks[self.mp_pose.PoseLandmark.RIGHT_WRIST.value].visibility > 0.2:
            if r_wrist[1] < chest_level:
                wrist_elevated = True

        pose_x, pose_y, pose_z = 0, 0, 0
        head_direction = "forward"

        if face_landmarks:
            pose_x, pose_y, pose_z = estimate_head_pose(face_landmarks, w, h)

            if pose_y > 10:
                head_direction = "right"
            elif pose_y < -10:
                head_direction = "left"
            elif pose_x < -20: 
                head_direction = "down"
            else:
                head_direction = "forward"

        return {
            "neck_ratio": neck_ratio,
            "forward_lean_z": forward_lean_z,
            "shoulder_tilt_ratio": shoulder_tilt_ratio,
            "head_tilt_ratio": head_tilt_ratio,
            "hand_to_face_ratio": min_hand_to_face,
            "hand_to_ear_ratio": min_hand_to_ear,
            "head_yaw_ratio": head_yaw_ratio, 
            "wrist_elevated": wrist_elevated,
            "head_direction": head_direction,
            "pose_x": pose_x,
            "pose_y": pose_y, 
            "coords": {"nose": nose, "mid_shoulder": mid_shoulder, "mid_ear": mid_ear}
        }

    def set_baseline(self, features):
        """
        Saves the current features as the golden standard (baseline) for the user.
        
        Args:
            features (dict): The features extracted from the calibration frame.
        """
        self.baseline_features = {
            "neck_ratio": features["neck_ratio"],
            "forward_lean_z": features["forward_lean_z"]
        }
        self.label_history.clear()
        print("Calibration successful! Starting evaluation...")

    def _get_raw_label(self, features, landmarks, has_phone):
        """
        Internal method to classify posture based on a single frame.
        
        Args:
            features (dict): Extracted features.
            landmarks (list): MediaPipe landmarks.
            has_phone (boolean): Checked features.
            
        Returns:
            str: Raw posture label.
        """

        # Check if the user is out of frame (Absence)
        visibility_nose = landmarks[self.mp_pose.PoseLandmark.NOSE.value].visibility
        visibility_l_shoulder = landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value].visibility
        visibility_r_shoulder = landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value].visibility
        
        if visibility_nose < 0.3 and visibility_l_shoulder < 0.3 and visibility_r_shoulder < 0.3:
            return "Absence"

        # Require calibration before classification
        if not self.baseline_features:
            return "Not Calibrated"
        
        # Edge-cases
        is_head_turned = features["head_yaw_ratio"] > 2.5 or features["head_yaw_ratio"] < 0.4
        is_reading_distance = 0.5 < features["hand_to_face_ratio"] < 1.5 # Không quá sát mặt, không quá xa

        if has_phone or features["hand_to_ear_ratio"] < self.THRESH_CALLING:
            return "Using Phone"
        
        if is_head_turned and features["wrist_elevated"] and is_reading_distance:
            return "Using Phone"
        
        direction = features["head_direction"]
        if direction in ["left", "right"]:
            return "Looking Away"

        # 1. LEANING ON DESK: Hand is close to the face
        if features["hand_to_face_ratio"] < self.THRESH_HAND_TO_FACE:
            return "Leaning on Desk"

        # 2. SLOUCHING: Neck drops significantly or head leans too far forward
        neck_compare = features["neck_ratio"] / self.baseline_features["neck_ratio"]
        lean_compare = features["forward_lean_z"] / (self.baseline_features["forward_lean_z"] + 0.001)
        
        if neck_compare < self.THRESH_SLOUCH_Y or lean_compare > self.THRESH_SLOUCH_Z:
             return "Slouching"

        # 3. FOCUSED: Default state if no violations are detected
        return "Focused"

    def classify(self, features, landmarks, has_phone=False):
        """
        Returns a temporally smoothed posture label using majority voting.
        
        Args:
            features (dict): Extracted features.
            landmarks (list): MediaPipe landmarks.
            
        Returns:
            str: Smoothed posture label.
        """
        raw_label = self._get_raw_label(features, landmarks, has_phone)
        self.label_history.append(raw_label)
        
        if len(self.label_history) == 0:
            return raw_label
            
        # Returns the most common label in the recent history buffer
        return statistics.mode(self.label_history)