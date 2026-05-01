# AI Classroom Monitor

A real-time classroom monitoring system that uses a hybrid AI pipeline to track student focus, posture, and engagement. It features browser-based inference using Mediapipe and ONNX Runtime to ensure scalability and privacy.

## Features
- **Real-time Monitoring**: Teachers can view student status (Focused, Slouching, Absence, etc.) and live camera feeds.
- **Hybrid AI Pipeline**: Mediapipe detection runs on the main thread for stability, while ONNX inference (Posture & Phone detection) runs in a dedicated Web Worker.
- **YOLOv8 Integration**: Detects mobile phone usage in the browser.
- **Visual Dashboard**: Premium teacher interface with real-time progress bars and status badges.

## 🎙️ LiveKit Server Setup (Host Machine)

Since LiveKit handles heavy media processing, it runs natively on your Windows host instead of inside Docker.

1. **Download LiveKit**:
   Run the following script to download the latest LiveKit binary:
   ```powershell
   powershell -ExecutionPolicy Bypass -File .\download-livekit.ps1
   ```
2. **Start LiveKit**:
   Use the start script to launch the server in the background:
   ```powershell
   powershell -ExecutionPolicy Bypass -File .\start-livekit.ps1
   ```
   *Note: Logs will be created in the `web_app/logs` directory.*

---

## 🚀 Quick Start (Application)

The fastest way to run the entire stack (Frontend, Backend, and LiveKit) is using Docker Compose.

1. **Prerequisites**: Install [Docker Desktop](https://www.docker.com/products/docker-desktop).
2. **Launch**:
   ```bash
   docker-compose up --build
   ```
3. **External Dependencies**:
   - **LiveKit Server**: Ensure LiveKit is running locally on your host machine (port 7880).
   - **Environment**: Your `.env` should point to `host.docker.internal` for backend-to-LiveKit communication.

4. **Access**:
   - **Frontend**: `http://localhost:5173`
   - **Backend API**: `http://localhost:8000`
   - **LiveKit Server**: `http://localhost:7880` (Running on host)

---

## 🛠️ Manual Development Setup

If you want to run the components individually for development:

### 1. Backend (FastAPI)
1. Navigate to `web_app/backend`.
2. Create a virtual environment and install dependencies:
   ```bash
   python -m venv venv
   source venv/bin/activate  # Windows: .\venv\Scripts\activate
   pip install -r requirements.txt
   ```
3. Run the server:
   ```bash
   uvicorn app.main:app --reload --port 8000
   ```

### 2. Frontend (Vite + React)
1. Navigate to `web_app/frontend`.
2. Install dependencies:
   ```bash
   npm install
   ```
3. Copy `.env.example` to `.env` and configure your API URLs.
4. Run the development server:
   ```bash
   npm run dev
   ```

### 3. Model Files
Ensure your AI models are placed in the `frontend/public/models` directory:
- `best_posture_model.onnx`: The posture classification model.
- `yolo26s.onnx`: The phone detection model.

---

## 🧠 Architecture Notes

### Hybrid Inference
To bypass browser security restrictions regarding WebAssembly (WASM) and SharedArrayBuffer:
- **Main Thread**: Initializes Mediapipe Pose and FaceMesh. It captures frames and extracts landmarks.
- **Web Worker (`aiWorker.js`)**: Receives landmarks and image fragments to run ONNX inference for posture classification and YOLO object detection.
- **WebSocket**: The student client sends smoothed scores and status updates to the FastAPI backend, which broadcasts them to the Teacher Dashboard.

### Stability & Smoothing
- **5 FPS Processing**: The system samples the camera 5 times per second.
- **Label History (Mode Filter)**: Uses a 5-second window (25 frames) to stabilize the status and prevent "score jumping."
- **EMA Scoring**: Focus scores use an Exponential Moving Average to provide smooth visual transitions on the teacher's dashboard.
