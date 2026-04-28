from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
import json
import joblib
import numpy as np
import os
import sys

app = FastAPI()

# Enable CORS for the frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load AI Model (from model_pipeline folder)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODEL_DIR = os.path.join(BASE_DIR, 'model_pipeline', 'models')
sys.path.append(os.path.join(BASE_DIR, 'model_pipeline'))

scaler = joblib.load(os.path.join(MODEL_DIR, 'scaler.pkl'))
model = joblib.load(os.path.join(MODEL_DIR, 'best_posture_model.pkl'))

class ConnectionManager:
    def __init__(self):
        # Maps room_code -> { "teacher": WebSocket, "students": {student_id: WebSocket} }
        self.active_rooms = {}
        # Stores scores per student per room: room_code -> {student_id: current_score}
        self.room_scores = {}

    async def connect_teacher(self, websocket: WebSocket, room_code: str):
        await websocket.accept()
        if room_code not in self.active_rooms:
            self.active_rooms[room_code] = {"teacher": websocket, "students": {}}
            self.room_scores[room_code] = {}
        else:
            # Teacher rejoining
            self.active_rooms[room_code]["teacher"] = websocket
            
        # Broadcast the current known scores immediately to sync upon reload
        payload = {"type": "scores_update", "scores": self._serialize_scores(room_code)}
        try:
            await websocket.send_json(payload)
        except:
            self.active_rooms[room_code]["teacher"] = None
            
    async def connect_student(self, websocket: WebSocket, room_code: str, student_id: str):
        if room_code not in self.active_rooms:
            # Create room lazily so students are not rejected (e.g. backend restarted,
            # teacher reconnecting, or student joins slightly before teacher socket).
            self.active_rooms[room_code] = {"teacher": None, "students": {}}
            self.room_scores[room_code] = {}
            
        if len(self.active_rooms[room_code]["students"]) >= 50:
            # Reject gracefully: accept first, then close with policy code.
            await websocket.accept()
            await websocket.close(code=1008)
            return False # Max capacity reached
            
        await websocket.accept()
        self.active_rooms[room_code]["students"][student_id] = websocket
        
        # Initialize score
        if student_id not in self.room_scores[room_code]:
            self.room_scores[room_code][student_id] = {"score": 100.0, "label": "Waiting"}
        
        # Broadcast initial state to teacher
        await self.broadcast_to_teacher(room_code)
        return True

    def disconnect_student(self, room_code: str, student_id: str):
        if room_code in self.active_rooms and student_id in self.active_rooms[room_code]["students"]:
            del self.active_rooms[room_code]["students"][student_id]
            
    def disconnect_teacher(self, room_code: str):
        # Teacher disconnected, optionally end room but for now just clear reference
        if room_code in self.active_rooms:
            self.active_rooms[room_code]["teacher"] = None

    async def broadcast_to_teacher(self, room_code: str):
        teacher_ws = self.active_rooms.get(room_code, {}).get("teacher")
        if teacher_ws:
            payload = {"type": "scores_update", "scores": self._serialize_scores(room_code)}
            try:
                await teacher_ws.send_json(payload)
            except:
                self.active_rooms[room_code]["teacher"] = None

    def _serialize_scores(self, room_code: str):
        room_data = self.room_scores.get(room_code, {})
        return {
            student_id: {
                "score": float(student_score.get("score", 0.0)),
                "label": str(student_score.get("label", "Waiting")),
            }
            for student_id, student_score in room_data.items()
        }


def normalize_prediction_label(prediction_value):
    # Ensure model output is always JSON-serializable plain string.
    return str(prediction_value)

manager = ConnectionManager()

@app.get("/")
def read_root():
    return {"message": "Focus Application Backend API"}

@app.websocket("/ws/teacher/{room_code}")
async def websocket_teacher(websocket: WebSocket, room_code: str):
    await manager.connect_teacher(websocket, room_code)
    try:
        while True:
            data = await websocket.receive_text()
            # Handle instructions from teacher (e.g. end_session)
    except WebSocketDisconnect:
        manager.disconnect_teacher(room_code)

@app.websocket("/ws/student/{room_code}/{student_id}")
async def websocket_student(websocket: WebSocket, room_code: str, student_id: str):
    success = await manager.connect_student(websocket, room_code, student_id)
    if not success:
        return
        
    try:
        while True:
            try:
                data = await websocket.receive_json()
                # Anti-tampering checking here:
                # - Check timing frequency
                # - Verify signature (nonce logic to be added)
                
                features = data.get("features")
                has_phone = data.get("has_phone", False)
                dt_raw = data.get("dt", 0.1) # delta time
                try:
                    dt = float(dt_raw)
                except (TypeError, ValueError):
                    dt = 0.1
                dt = max(0.01, min(dt, 1.0))
                
                valid_features = (
                    isinstance(features, list)
                    and len(features) == 9
                    and all(isinstance(value, (int, float)) for value in features)
                )

                if valid_features:
                    # 1. Check heuristics (Absence / Phone)
                    raw_label = "Focused"
                    
                    # We expect the client to send the engineered features as a 9-element array
                    feature_vector = np.array([features], dtype=np.float64)
                    scaled_vector = scaler.transform(feature_vector)
                    
                    # Predict
                    prediction = model.predict(scaled_vector)
                    raw_label = normalize_prediction_label(prediction[0])
                    
                    # Override if phone or absent
                    if has_phone:
                        raw_label = "Using Phone"
                    if data.get("is_absent", False):
                        raw_label = "Absence"
                    
                    # Score tracking
                    current_score = manager.room_scores[room_code][student_id]["score"]
                    if raw_label == "Focused":
                        current_score += 2.0 * dt
                    elif raw_label == "Slouching":
                        current_score -= 0.2 * dt
                    elif raw_label == "Leaning on Desk":
                        current_score -= 1.5 * dt
                    elif raw_label == "Looking Away":
                        current_score -= 0.5 * dt
                    elif raw_label in ["Using Phone", "Absence"]:
                        current_score -= 8.0 * dt
                        
                    current_score = max(0.0, min(100.0, current_score))
                    manager.room_scores[room_code][student_id] = {"score": float(current_score), "label": str(raw_label)}
                    
                    await manager.broadcast_to_teacher(room_code)
                    
                    # Send feedback back to student client
                    await websocket.send_json({"label": str(raw_label), "score": float(current_score)})
                else:
                    # No features detected (likely absent)
                    current_score = manager.room_scores[room_code][student_id]["score"]
                    current_score -= 8.0 * dt
                    current_score = max(0.0, min(100.0, current_score))
                    manager.room_scores[room_code][student_id] = {"score": float(current_score), "label": "Absence"}
                    
                    await manager.broadcast_to_teacher(room_code)
                    await websocket.send_json({"label": "Absence", "score": float(current_score)})
            except WebSocketDisconnect:
                raise
            except Exception as exc:
                # Keep the websocket alive if a single payload/inference fails.
                print(f"[student_ws_error] room={room_code} student={student_id} error={exc}")

    except WebSocketDisconnect:
        manager.disconnect_student(room_code, student_id)
        await manager.broadcast_to_teacher(room_code)
