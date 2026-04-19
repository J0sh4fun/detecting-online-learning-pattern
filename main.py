import cv2
import mediapipe as mp
from src.posture_classifier import PostureClassifier

def main():
    """Main pipeline for reading webcam feed, processing poses, and rendering UI."""
    # Initialize MediaPipe Pose
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(
        static_image_mode=False, 
        min_detection_confidence=0.5, 
        min_tracking_confidence=0.5
    )
    mp_drawing = mp.solutions.drawing_utils
    classifier = PostureClassifier()

    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, min_detection_confidence=0.5)

    cap = cv2.VideoCapture(0)
    print("Starting Webcam. Press 'q' to exit.")

    while True:
        ret, frame = cap.read()
        if not ret: 
            break
            
        # Mirror the frame for a natural user experience
        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process the frame to find poses
        results = pose.process(image_rgb)
        mesh_results = face_mesh.process(image_rgb)
        
        if not results.pose_landmarks:
            # Handle user absence
            cv2.putText(frame, "Status: Absence", (20, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (150, 150, 150), 2, cv2.LINE_AA)
        else:
            # Handle active user
            landmarks = results.pose_landmarks.landmark

            face_landmarks = mesh_results.multi_face_landmarks[0] if mesh_results.multi_face_landmarks else None
            
            # Draw faded pose landmarks in the background
            mp_drawing.draw_landmarks(
                frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(200, 200, 200), thickness=1, circle_radius=1),
                mp_drawing.DrawingSpec(color=(200, 200, 200), thickness=1)
            )
            
            # Extract features and specific coordinates
            features = classifier.extract_features(landmarks, face_landmarks, w, h)
            coords = features["coords"]

            # Detech phone
            has_phone = classifier.detect_phone(frame)
            
            # Draw visual indicator lines
            cv2.line(frame, coords["mid_shoulder"], coords["mid_ear"], (0, 255, 255), 3)
            cv2.line(frame, coords["mid_shoulder"], coords["nose"], (255, 0, 255), 3)
            
            # Classify behavior
            label = classifier.classify(features, landmarks, has_phone)
            
            # Set UI text color based on the posture label
            if label == "Focused":
                color = (0, 255, 0)       # Green
            elif label == 'Looking Away':
                color = (255, 0, 0)       # Blue
            elif label in ["Slouching", "Leaning on Desk"]:
                color = (0, 165, 255)     # Orange
            elif label == 'Using Phone':
                color = (0, 0, 255)       # Red
            else: 
                color = (255, 255, 255)   # White
                
            # Render status text
            cv2.putText(frame, f"Status: {label}", (20, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2, cv2.LINE_AA)
            
            # Render debugging metrics
            debug_text = f"Pitch(X): {features['pose_x']:.1f} | Yaw(Y): {features['pose_y']:.1f} | Dir: {features['head_direction']}"
            cv2.putText(frame, debug_text, (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 200, 0), 1, cv2.LINE_AA)
            
            # Show calibration prompt if needed
            if classifier.baseline_features is None:
                cv2.putText(frame, "Sit straight and press 'C' to Calibrate", (20, 110), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        cv2.imshow('Student Focus AI - Week 2', frame)

        # Handle keyboard inputs
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('c'):
            if 'features' in locals():
                classifier.set_baseline(features)

    # Clean up resources
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()