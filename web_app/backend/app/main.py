from __future__ import annotations

import base64
from datetime import datetime, timezone

import cv2
import numpy as np
from fastapi import FastAPI, HTTPException, Query, Response, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

from app.core.config import settings
from app.core.security import create_session_token, decode_session_token
from app.models.schemas import (
    CreateRoomRequest,
    JoinRoomRequest,
    RoomConnectionResponse,
    RoomReportResponse,
    ScoreFrameRequest,
    ScoreFrameResponse,
    StudentReport,
    StudentScoreIngest,
    StudentTimelinePoint,
    VerifyFrameRequest,
    VerifyFrameResponse,
)
from app.services.livekit_auth import build_livekit_token
from app.services.ml_scoring import get_verification_scorer
from app.services.room_store import store
from app.ws.manager import socket_manager

app = FastAPI(title=settings.api_title)

app.add_middleware(
    CORSMiddleware,
    allow_origins=list(settings.cors_origins) or ["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _room_not_found(room_code: str) -> HTTPException:
    return HTTPException(status_code=404, detail=f"Room '{room_code}' not found")


def _build_invitation_link(room_code: str) -> str:
    room = room_code.strip().upper()
    return f"http://localhost:5173/?join={room}"


def _decode_image(frame_base64: str) -> np.ndarray:
    if "," in frame_base64:
        frame_base64 = frame_base64.split(",", 1)[1]
    image_bytes = base64.b64decode(frame_base64)
    np_buffer = np.frombuffer(image_bytes, dtype=np.uint8)
    frame = cv2.imdecode(np_buffer, cv2.IMREAD_COLOR)
    if frame is None:
        raise ValueError("Unable to decode frame")
    return frame


def _validate_socket_identity(*, token: str, room_code: str, role: str, participant_id: str | None = None) -> None:
    claims = decode_session_token(token)
    if claims.get("room_code") != room_code:
        raise ValueError("Token room mismatch")
    if claims.get("role") != role:
        raise ValueError("Token role mismatch")
    if participant_id and claims.get("participant_id") != participant_id:
        raise ValueError("Token participant mismatch")


@app.get("/health")
def healthcheck() -> dict:
    return {"status": "ok", "time": datetime.now(timezone.utc).isoformat()}


@app.get("/")
def root() -> dict:
    return {
        "service": settings.api_title,
        "status": "running",
        "health": "/health",
    }


@app.get("/favicon.ico", include_in_schema=False)
def favicon() -> Response:
    return Response(status_code=204)


@app.get("/.well-known/appspecific/com.chrome.devtools.json", include_in_schema=False)
def chrome_devtools_probe() -> Response:
    return Response(status_code=204)


@app.post("/api/rooms", response_model=RoomConnectionResponse)
def create_room(payload: CreateRoomRequest) -> RoomConnectionResponse:
    room = store.create_room(teacher_id=payload.teacher_id, room_name=payload.room_name)
    participant_id = f"teacher-{payload.teacher_id}"
    return RoomConnectionResponse(
        room_code=room.room_code,
        role="teacher",
        livekit_url=settings.livekit_url,
        livekit_token=build_livekit_token(
            room_code=room.room_code,
            participant_id=participant_id,
            is_teacher=True,
        ),
        session_token=create_session_token(
            room_code=room.room_code,
            participant_id=participant_id,
            role="teacher",
        ),
        score_ws_url=f"ws://localhost:8000/ws/teacher/{room.room_code}",
        invitation_link=_build_invitation_link(room.room_code),
        room_name=room.room_name,
        teacher_id=room.teacher_id,
    )


@app.post("/api/rooms/join", response_model=RoomConnectionResponse)
def join_room(payload: JoinRoomRequest) -> RoomConnectionResponse:
    room_code = payload.room_code.strip().upper()
    room = store.get_room(room_code)
    if not room:
        raise _room_not_found(room_code)

    student_id = payload.student_id.strip()
    try:
        store.ensure_student(room_code=room_code, student_id=student_id)
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    participant_id = f"student-{student_id}"
    return RoomConnectionResponse(
        room_code=room_code,
        role="student",
        livekit_url=settings.livekit_url,
        livekit_token=build_livekit_token(
            room_code=room_code,
            participant_id=participant_id,
            is_teacher=False,
        ),
        session_token=create_session_token(
            room_code=room_code,
            participant_id=participant_id,
            role="student",
        ),
        score_ws_url=f"ws://localhost:8000/ws/student/{room_code}/{student_id}",
        room_name=room.room_name,
        teacher_id=room.teacher_id,
    )


@app.websocket("/ws/teacher/{room_code}")
async def teacher_scores_socket(
    websocket: WebSocket,
    room_code: str,
    token: str = Query(...),
) -> None:
    room_code = room_code.strip().upper()
    if not store.get_room(room_code):
        await websocket.close(code=4404, reason="Room not found")
        return

    try:
        _validate_socket_identity(token=token, room_code=room_code, role="teacher")
    except Exception:
        await websocket.close(code=4401, reason="Invalid token")
        return

    await socket_manager.connect_teacher(room_code, websocket)
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        socket_manager.disconnect_teacher(room_code, websocket)


@app.websocket("/ws/student/{room_code}/{student_id}")
async def student_scores_socket(websocket: WebSocket, room_code: str, student_id: str) -> None:
    room_code = room_code.strip().upper()
    student_id = student_id.strip()

    if not store.get_room(room_code):
        await websocket.close(code=4404, reason="Room not found")
        return

    try:
        store.ensure_student(room_code=room_code, student_id=student_id)
    except ValueError:
        await websocket.close(code=4409, reason="Room is full")
        return

    await websocket.accept()
    await socket_manager.connect_student(room_code, student_id, websocket)
    await socket_manager.broadcast_snapshot(room_code)

    try:
        while True:
            data = StudentScoreIngest.model_validate(await websocket.receive_json())
            _validate_socket_identity(
                token=data.token,
                room_code=room_code,
                role="student",
                participant_id=f"student-{student_id}",
            )
            store.update_student_score(
                room_code=room_code,
                student_id=student_id,
                average_score=data.average_score,
                status=data.status,
                camera_on=data.camera_on,
                client_sent_at=data.client_sent_at,
            )
            await socket_manager.broadcast_snapshot(room_code)
    except WebSocketDisconnect:
        socket_manager.disconnect_student(room_code, student_id)
        student = store.ensure_student(room_code=room_code, student_id=student_id)
        student.camera_on = False
        student.status = "Camera Off"
        student.last_update = datetime.now(timezone.utc)
        await socket_manager.broadcast_snapshot(room_code)
    except Exception:
        socket_manager.disconnect_student(room_code, student_id)
        await websocket.close(code=4400, reason="Malformed payload")


@app.post("/api/verify/frame", response_model=VerifyFrameResponse)
def verify_frame(payload: VerifyFrameRequest) -> VerifyFrameResponse:
    claims = decode_session_token(payload.token)
    if claims.get("role") != "student":
        raise HTTPException(status_code=403, detail="Only student tokens can verify frame")
    if claims.get("room_code") != payload.room_code.strip().upper():
        raise HTTPException(status_code=403, detail="Token room mismatch")

    try:
        frame = _decode_image(payload.frame_base64)
    except Exception as exc:
        raise HTTPException(status_code=400, detail="Invalid frame payload") from exc

    try:
        server = get_verification_scorer().score_frame(frame)
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Scoring model unavailable: {exc}") from exc
    discrepancy = abs(server.score - payload.client_score)
    is_flagged = discrepancy >= settings.verify_discrepancy_threshold

    if is_flagged:
        store.add_verification_flag(
            payload.room_code.strip().upper(),
            {
                "student_id": payload.student_id,
                "client_score": payload.client_score,
                "server_score": server.score,
                "server_status": server.status,
                "discrepancy": discrepancy,
                "created_at": datetime.now(timezone.utc).isoformat(),
            },
        )

    return VerifyFrameResponse(
        is_flagged=is_flagged,
        discrepancy=discrepancy,
        server_score=server.score,
        server_status=server.status,
        reason="Large score discrepancy detected" if is_flagged else "Within threshold",
    )


@app.post("/api/score/frame", response_model=ScoreFrameResponse)
def score_frame(payload: ScoreFrameRequest) -> ScoreFrameResponse:
    claims = decode_session_token(payload.token)
    room_code = payload.room_code.strip().upper()
    if claims.get("role") != "student":
        raise HTTPException(status_code=403, detail="Only student tokens can score frame")
    if claims.get("room_code") != room_code:
        raise HTTPException(status_code=403, detail="Token room mismatch")
    if claims.get("participant_id") != f"student-{payload.student_id.strip()}":
        raise HTTPException(status_code=403, detail="Token participant mismatch")

    try:
        frame = _decode_image(payload.frame_base64)
    except Exception as exc:
        raise HTTPException(status_code=400, detail="Invalid frame payload") from exc

    try:
        result = get_verification_scorer().score_frame(frame)
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Scoring model unavailable: {exc}") from exc
    return ScoreFrameResponse(score=result.score, status=result.status)


@app.post("/api/rooms/{room_code}/end", response_model=RoomReportResponse)
async def end_room(room_code: str, token: str = Query(...)) -> RoomReportResponse:
    room_code = room_code.strip().upper()
    room = store.get_room(room_code)
    if not room:
        raise _room_not_found(room_code)

    claims = decode_session_token(token)
    if claims.get("role") != "teacher" or claims.get("room_code") != room_code:
        raise HTTPException(status_code=403, detail="Invalid teacher token")

    store.end_room(room_code)
    await socket_manager.broadcast_to_students(
        room_code, {"type": "room_closed", "teacher_id": room.teacher_id}
    )
    return get_room_report(room_code, token)


@app.get("/api/rooms/{room_code}/report", response_model=RoomReportResponse)
def get_room_report(room_code: str, token: str = Query(...)) -> RoomReportResponse:
    room_code = room_code.strip().upper()
    room = store.get_room(room_code)
    if not room:
        raise _room_not_found(room_code)

    claims = decode_session_token(token)
    if claims.get("role") != "teacher" or claims.get("room_code") != room_code:
        raise HTTPException(status_code=403, detail="Invalid teacher token")

    reports: list[StudentReport] = []
    room_students = store.snapshot_students(room_code)
    class_total = 0.0
    for student in room_students:
        samples = list(store.timeline[room_code][student.student_id])
        avg = (
            sum(sample.score for sample in samples) / len(samples)
            if samples
            else student.score
        )
        class_total += avg
        reports.append(
            StudentReport(
                student_id=student.student_id,
                average_score=round(avg, 2),
                timeline=[
                    StudentTimelinePoint(
                        timestamp=sample.timestamp,
                        score=sample.score,
                        status=sample.status,
                        camera_on=sample.camera_on,
                    )
                    for sample in samples
                ],
            )
        )

    class_avg = round(class_total / len(room_students), 2) if room_students else 0.0
    return RoomReportResponse(
        room_code=room_code,
        room_name=room.room_name,
        teacher_id=room.teacher_id,
        generated_at=datetime.now(timezone.utc),
        class_average_score=class_avg,
        students=reports,
    )
