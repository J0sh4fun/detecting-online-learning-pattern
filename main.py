import cv2
import mediapipe as mp
import time
from src.posture_classifier import PostureClassifier

def main():
    print("Đang khởi tạo Hệ thống AI...")
    classifier = PostureClassifier()
    
    # --- ĐÃ SỬA LỖI Ở ĐÂY: KHỞI TẠO CỖ MÁY MEDIAPIPE ---
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5)
    
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, min_detection_confidence=0.5)
    # ----------------------------------------------------

    cap = cv2.VideoCapture(0)
    
    attention_score = 100.0
    last_time = time.time()
    
    print("=== HỆ THỐNG ĐÃ SẴN SÀNG ===")
    print("Nhấn 'q' để thoát.")

    while True:
        ret, frame = cap.read()
        if not ret: break
            
        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        current_time = time.time()
        dt = current_time - last_time
        last_time = current_time

        # --- ĐÃ SỬA LỖI Ở ĐÂY: GỌI TRỰC TIẾP TỪ INSTANCE ---
        results = pose.process(image_rgb)
        mesh_results = face_mesh.process(image_rgb)
        has_phone = classifier.detect_phone(frame)
        
        label = "Absence" 
        features = None
        
        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            face_landmarks = mesh_results.multi_face_landmarks[0] if mesh_results.multi_face_landmarks else None
            
            features = classifier.extract_features(landmarks, face_landmarks, w, h)
            label = classifier.classify(features, landmarks, has_phone)
            
            # --- ĐÃ SỬA LỖI Ở ĐÂY: mp_pose.POSE_CONNECTIONS ---
            mp.solutions.drawing_utils.draw_landmarks(
                frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                mp.solutions.drawing_utils.DrawingSpec(color=(200, 200, 200), thickness=1, circle_radius=1),
                mp.solutions.drawing_utils.DrawingSpec(color=(200, 200, 200), thickness=1)
            )

        # --- THUẬT TOÁN TÍNH ĐIỂM ---
        if label == "Focused":
            attention_score += 5.0 * dt   
        elif label in ["Slouching", "Leaning on Desk"]:
            attention_score -= 5.0 * dt   
        elif label == "Looking Away":
            attention_score -= 10.0 * dt  
        elif label in ["Using Phone", "Absence"]:
            attention_score -= 20.0 * dt  
            
        attention_score = max(0.0, min(100.0, attention_score))

        # --- GIAO DIỆN (UI DASHBOARD) ---
        if attention_score > 80:
            ui_color = (0, 255, 0)      
        elif attention_score > 50:
            ui_color = (0, 255, 255)    
        else:
            ui_color = (0, 0, 255)      
            if int(current_time * 5) % 2 == 0:
                cv2.rectangle(frame, (0, 0), (w, h), (0, 0, 255), 10)

        bar_width = int((attention_score / 100.0) * 300)
        cv2.rectangle(frame, (20, 80), (320, 100), (50, 50, 50), -1) 
        cv2.rectangle(frame, (20, 80), (20 + bar_width, 100), ui_color, -1) 
        cv2.putText(frame, f"Score: {int(attention_score)}/100", (330, 95), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, ui_color, 2)

        cv2.putText(frame, f"Status: {label}", (20, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, ui_color, 3, cv2.LINE_AA)

        if features:
            debug_txt = f"Phone: {has_phone} | Pitch: {features['pose_x']:.1f} | Yaw: {features['pose_y']:.1f}"
            cv2.putText(frame, debug_txt, (20, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        cv2.imshow('Student Attention AI Dashboard', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
