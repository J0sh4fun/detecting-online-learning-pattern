BASE_FEATURE_ORDER = [
    "neck_ratio",
    "forward_lean_z",
    "shoulder_tilt_ratio",
    "head_tilt_ratio",
    "hand_to_face_ratio",
    "pose_x",
    "pose_y",
    "wrist_elevated",
]

EXTRA_FEATURE_ORDER = [
    "shoulder_width_ratio",
    "head_offset_x_ratio",
    "head_offset_y_ratio",
    "nose_shoulder_x_ratio",
    "nose_shoulder_y_ratio",
    "abs_pose_x",
    "abs_pose_y",
    "face_detected",
    "nose_visibility",
    "shoulder_visibility_min",
    "ear_visibility_min",
    "left_wrist_visibility",
    "right_wrist_visibility",
    "visible_wrist_count",
    "hand_visible",
]

FEATURE_ORDER = BASE_FEATURE_ORDER + EXTRA_FEATURE_ORDER

METADATA_COLUMNS = ["session_id", "capture_group", "captured_at", "frame_index"]
OUTPUT_COLUMNS = METADATA_COLUMNS + FEATURE_ORDER + ["label"]

FEATURE_ALIASES = {
    "neck_ratio": ["neck_ratio", "neckneck_ratio"],
    "forward_lean_z": ["forward_lean_z"],
    "shoulder_tilt_ratio": ["shoulder_tilt_ratio", "shoulder_tilt"],
    "head_tilt_ratio": ["head_tilt_ratio", "head_tilt"],
    "hand_to_face_ratio": ["hand_to_face_ratio", "hand_to_face"],
    "pose_x": ["pose_x"],
    "pose_y": ["pose_y"],
    "wrist_elevated": ["wrist_elevated"],
    "shoulder_width_ratio": ["shoulder_width_ratio"],
    "head_offset_x_ratio": ["head_offset_x_ratio"],
    "head_offset_y_ratio": ["head_offset_y_ratio"],
    "nose_shoulder_x_ratio": ["nose_shoulder_x_ratio"],
    "nose_shoulder_y_ratio": ["nose_shoulder_y_ratio"],
    "abs_pose_x": ["abs_pose_x"],
    "abs_pose_y": ["abs_pose_y"],
    "face_detected": ["face_detected"],
    "nose_visibility": ["nose_visibility"],
    "shoulder_visibility_min": ["shoulder_visibility_min"],
    "ear_visibility_min": ["ear_visibility_min"],
    "left_wrist_visibility": ["left_wrist_visibility"],
    "right_wrist_visibility": ["right_wrist_visibility"],
    "visible_wrist_count": ["visible_wrist_count"],
    "hand_visible": ["hand_visible"],
}

FEATURE_DEFAULTS = {
    "shoulder_width_ratio": 0.0,
    "head_offset_x_ratio": 0.0,
    "head_offset_y_ratio": -1.0,
    "nose_shoulder_x_ratio": 0.0,
    "nose_shoulder_y_ratio": -1.0,
    "abs_pose_x": 0.0,
    "abs_pose_y": 0.0,
    "face_detected": 0.0,
    "nose_visibility": 1.0,
    "shoulder_visibility_min": 1.0,
    "ear_visibility_min": 1.0,
    "left_wrist_visibility": 0.0,
    "right_wrist_visibility": 0.0,
    "visible_wrist_count": 0.0,
    "hand_visible": 0.0,
}

NO_HAND_VISIBLE_RATIO = 5.0
