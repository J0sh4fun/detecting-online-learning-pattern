import math
import cv2
import numpy as np

def calculate_distance(p1, p2):
    """
    Calculates the Euclidean distance between two 2D points.

    Args:
        p1 (tuple): The (x, y) coordinates of the first point.
        p2 (tuple): The (x, y) coordinates of the second point.

    Returns:
        float: The distance between p1 and p2.
    """
    return math.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)

def get_midpoint(p1, p2):
    """
    Calculates the midpoint between two 2D points.

    Args:
        p1 (tuple): The (x, y) coordinates of the first point.
        p2 (tuple): The (x, y) coordinates of the second point.

    Returns:
        tuple: The (x, y) coordinates of the midpoint.
    """
    return (int((p1[0] + p2[0]) / 2), int((p1[1] + p2[1]) / 2))

def calculate_vertical_angle(p_top, p_bottom):
    """
    Calculates the angle between a line segment and the vertical axis.

    Args:
        p_top (tuple): (x, y) coordinates of the upper point.
        p_bottom (tuple): (x, y) coordinates of the lower point.

    Returns:
        float: The angle in degrees (0 is perfectly vertical).
    """
    dx = p_top[0] - p_bottom[0]
    dy = p_bottom[1] - p_top[1] 
    
    angle_rad = math.atan2(abs(dx), abs(dy))
    return math.degrees(angle_rad)

def estimate_head_pose(face_landmarks, frame_width, frame_height):
    """
    Estimates head rotation (Pitch and Yaw) using 3D vector logic from Face Mesh.
    
    This method uses the relative depth (Z-axis) of facial landmarks to 
    calculate angles without needing the complex solvePnP matrix.

    Args:
        face_landmarks: MediaPipe Face Mesh landmarks object.
        frame_width (int): Width of the video frame.
        frame_height (int): Height of the video frame.

    Returns:
        tuple: (pitch, yaw, roll) angles in degrees.
    """
    # Take 3 reference points from Face Mesh: Nose (1), Left Eye (33), Right Eye (263)
    nose = face_landmarks.landmark[1]
    left_eye = face_landmarks.landmark[33]
    right_eye = face_landmarks.landmark[263]

    # Convert to pixel coordinates with horizontal flip to match training
    # frames that were mirrored via cv2.flip(frame, 1).
    lx, ly, lz = (1.0 - left_eye.x) * frame_width, left_eye.y * frame_height, left_eye.z * frame_width
    rx, ry, rz = (1.0 - right_eye.x) * frame_width, right_eye.y * frame_height, right_eye.z * frame_width
    nx, ny, nz = (1.0 - nose.x) * frame_width, nose.y * frame_height, nose.z * frame_width

    # 1. Calculate the YAW (Left/Right Rotation) along the X and Z axes of both eyes.
    dx = rx - lx
    dz = rz - lz
    pose_y = math.degrees(math.atan2(dz, dx))

    # 2. Measure the pitch (tilting/tilting) of the nose relative to the eyes along the Y and Z axes.
    mid_eye_y = (ly + ry) / 2
    mid_eye_z = (lz + rz) / 2
    dy = ny - mid_eye_y
    dz_pitch = nz - mid_eye_z
    pose_x = math.degrees(math.atan2(dz_pitch, dy))

    return pose_x, pose_y, 0.0