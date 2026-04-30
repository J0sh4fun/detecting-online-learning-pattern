# AI Focus Classroom (FastAPI + React + LiveKit + ONNX Worker)

## Updated directory structure

```text
online_study_detech/
├── model_pipeline/
├── web_app/
│   ├── backend/
│   │   ├── app/
│   │   │   ├── core/
│   │   │   │   ├── config.py
│   │   │   │   └── security.py
│   │   │   ├── models/
│   │   │   │   └── schemas.py
│   │   │   ├── services/
│   │   │   │   ├── livekit_auth.py
│   │   │   │   ├── ml_scoring.py
│   │   │   │   └── room_store.py
│   │   │   ├── ws/
│   │   │   │   └── manager.py
│   │   │   └── main.py
│   │   ├── main.py
│   │   └── requirements.txt
│   └── frontend/
│       ├── src/
│       │   ├── lib/
│       │   │   ├── api.js
│       │   │   └── sessionStore.js
│       │   ├── pages/
│       │   │   ├── StudentRoom.jsx
│       │   │   └── TeacherDashboard.jsx
│       │   ├── workers/
│       │   │   └── aiWorker.js
│       │   ├── App.jsx
│       │   └── index.css
│       └── package.json
└── yolo26s.pt
```

## Core implementation highlights

- **FastAPI Auth + Room APIs**: room create/join returns **LiveKit token + session JWT**.
- **Teacher WebSocket**: receives real-time student snapshots with warning states.
- **Student WebSocket ingest**: accepts throttled aggregate scores every few seconds (JWT required in each payload).
- **Anti-cheat verification API**: receives random frame, reruns server-side verification model, and flags large discrepancies.
- **Teacher dashboard**: LiveKit camera smart-grid with pagination and red warning highlights.
- **Student room**: LiveKit classroom view with hidden AI pipeline (no score/box overlays).
- **aiWorker.js**: ONNX worker loop at 1 FPS + 4s score batching + random 5-10 minute silent verification frame emission.

## Run steps (Windows local)

1. Start LiveKit (Docker):
   - `cd web_app`
   - `docker compose -f docker-compose.livekit.yml up -d`
2. Backend env setup:
   - `copy web_app\backend\.env.example web_app\backend\.env`
   - Set these in terminal before running backend (or your shell profile):
     - `LIVEKIT_URL=ws://localhost:7880`
     - `LIVEKIT_API_KEY=devkey`
     - `LIVEKIT_API_SECRET=secret`
3. Backend:
   - `cd web_app\backend`
   - `pip install -r requirements.txt`
   - `uvicorn main:app --reload --port 8000`
4. Frontend:
   - `copy web_app\frontend\.env.example web_app\frontend\.env`
   - `cd web_app\frontend`
   - `npm install`
   - `npm run dev`
5. Ensure ONNX model exists at `web_app\frontend\public\models\best_posture_model.onnx`.
6. Optional helper script:
   - `powershell -ExecutionPolicy Bypass -File web_app\start-local.ps1`
7. One-command run script (opens backend + frontend terminals):
   - `powershell -ExecutionPolicy Bypass -File web_app\run-web_app.ps1`

### Docker error troubleshooting

If you see:
`open //./pipe/dockerDesktopLinuxEngine: The system cannot find the file specified`

Docker Desktop is installed but its engine is not running.

1. Start Docker Desktop and wait for **Engine running**.
2. Retry:
   - `cd web_app`
   - `docker compose -f docker-compose.livekit.yml up -d`
3. If you cannot use Docker, point backend to a hosted LiveKit by updating:
   - `LIVEKIT_URL`
   - `LIVEKIT_API_KEY`
   - `LIVEKIT_API_SECRET`

