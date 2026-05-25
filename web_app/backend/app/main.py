from __future__ import annotations
from contextlib import asynccontextmanager

import base64
from datetime import datetime, timezone

import cv2
import numpy as np
from fastapi import FastAPI, HTTPException, Query, Response, WebSocket, WebSocketDisconnect, Depends
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.ext.asyncio import AsyncSession

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
from app.core.database import Base, engine, get_db
from app.routes import auth_routes, history_routes
from app.services.livekit_auth import build_livekit_token
from app.services.ml_scoring import get_verification_scorer
from app.services.room_store import store
from app.services.auth import get_current_user
from app.models.models import User, RoomReport
from app.ws.manager import socket_manager

@asynccontextmanager
async def lifespan(app: FastAPI):
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    yield

app = FastAPI(title=settings.api_title, lifespan=lifespan)
app.include_router(auth_routes.router)
app.include_router(history_routes.router)

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
async def create_room(
    payload: CreateRoomRequest,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
) -> RoomConnectionResponse:
    if current_user.role.value != "teacher":
        raise HTTPException(status_code=403, detail="Only teachers can create rooms")

    room = await store.create_room(db, teacher_id=current_user.id, room_name=payload.room_name)
    participant_id = f"teacher-{current_user.username}"
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
        score_ws_url=f"{settings.livekit_url.replace('http', 'ws')}/ws/teacher/{room.room_code}", # not used by UI
        invitation_link=_build_invitation_link(room.room_code),
        room_name=room.room_name,
        teacher_id=current_user.username,
    )


@app.post("/api/rooms/join", response_model=RoomConnectionResponse)
async def join_room(
    payload: JoinRoomRequest,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
) -> RoomConnectionResponse:
    room_code = payload.room_code.strip().upper()
    room = await store.get_room(db, room_code)
    if not room:
        raise _room_not_found(room_code)

    if current_user.role.value != "student":
        raise HTTPException(status_code=403, detail="Only students can join rooms")

    try:
        await store.ensure_student(db, room_id=room.id, user_id=current_user.id, display_id=current_user.username)
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    participant_id = f"student-{current_user.username}"
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
        score_ws_url=f"ws://localhost:8000/ws/student/{room_code}/{current_user.username}",
        room_name=room.room_name,
        teacher_id=room.teacher.username,
    )


@app.websocket("/ws/teacher/{room_code}")
async def teacher_scores_socket(
    websocket: WebSocket,
    room_code: str,
    token: str = Query(...),
    db: AsyncSession = Depends(get_db),
) -> None:
    room_code = room_code.strip().upper()
    room = await store.get_room(db, room_code)
    if not room:
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
async def student_scores_socket(
    websocket: WebSocket, 
    room_code: str, 
    student_id: str,
    token: str = Query(None), # Need token on connection query or message
    db: AsyncSession = Depends(get_db),
) -> None:
    room_code = room_code.strip().upper()
    student_id = student_id.strip()

    room = await store.get_room(db, room_code)
    if not room:
        await websocket.close(code=4404, reason="Room not found")
        return

    await websocket.accept()
    await socket_manager.connect_student(room_code, student_id, websocket)
    await socket_manager.broadcast_snapshot(room_code)

    user_id = None
    try:
        while True:
            data_json = await websocket.receive_json()
            data = StudentScoreIngest.model_validate(data_json)
            
            claims = decode_session_token(data.token)
            if claims.get("role") != "student" or claims.get("room_code") != room_code:
                raise ValueError("Invalid student token")
                
            from sqlalchemy import select
            res = await db.execute(select(User).filter(User.username == student_id))
            user = res.scalar_one_or_none()
            if not user:
                raise ValueError("User not found")

            await store.update_student_score(
                db,
                room_code=room_code,
                user_id=user.id,
                display_id=student_id,
                average_score=data.average_score,
                status=data.status,
                camera_on=data.camera_on,
                client_sent_at=data.client_sent_at,
            )
            await socket_manager.broadcast_snapshot(room_code)
    except WebSocketDisconnect:
        socket_manager.disconnect_student(room_code, student_id)
        if user_id:
            res = await db.execute(select(User).filter(User.username == student_id))
            user = res.scalar_one_or_none()
            if user:
                # Update status to offline
                await store.update_student_score(
                    db, room_code=room_code, user_id=user.id, display_id=student_id,
                    average_score=100.0, status="Camera Off", camera_on=False, client_sent_at=datetime.now(timezone.utc).timestamp()
                )
        await socket_manager.broadcast_snapshot(room_code)
    except Exception:
        socket_manager.disconnect_student(room_code, student_id)
        await websocket.close(code=4400, reason="Malformed payload")


@app.post("/api/verify/frame", response_model=VerifyFrameResponse)
async def verify_frame(payload: VerifyFrameRequest, db: AsyncSession = Depends(get_db)) -> VerifyFrameResponse:
    claims = decode_session_token(payload.token)
    if claims.get("role") != "student":
        raise HTTPException(status_code=403, detail="Only student tokens can verify frame")
    room_code = payload.room_code.strip().upper()
    if claims.get("room_code") != room_code:
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
        from sqlalchemy import select
        res = await db.execute(select(User).filter(User.username == payload.student_id))
        user = res.scalar_one_or_none()
        room = await store.get_room(db, room_code)
        if user and room:
            await store.add_verification_flag(
                db,
                room_id=room.id,
                user_id=user.id,
                payload={
                    "client_score": payload.client_score,
                    "server_score": server.score,
                    "server_status": server.status,
                    "discrepancy": discrepancy,
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
async def end_room(
    room_code: str, 
    token: str = Query(...), 
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
) -> RoomReportResponse:
    room_code = room_code.strip().upper()
    room = await store.get_room(db, room_code)
    if not room:
        raise _room_not_found(room_code)

    if current_user.id != room.teacher_id:
        raise HTTPException(status_code=403, detail="Only the room teacher can end it")

    claims = decode_session_token(token)
    if claims.get("role") != "teacher" or claims.get("room_code") != room_code:
        raise HTTPException(status_code=403, detail="Invalid teacher token")

    # Generate report before ending
    report = await get_room_report(room_code, token, db, current_user)

    await store.end_room(db, room_code)
    await socket_manager.broadcast_to_students(
        room_code, {"type": "room_closed", "teacher_id": current_user.username}
    )

    # Save report to DB
    db_report = RoomReport(
        room_id=room.id,
        class_average_score=report.class_average_score,
        total_students=len(report.students),
        student_summaries=report.model_dump(mode="json")["students"],
    )
    db.add(db_report)
    await db.commit()

    return report


@app.get("/api/rooms/{room_code}/report", response_model=RoomReportResponse)
async def get_room_report(
    room_code: str, 
    token: str = Query(...), 
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
) -> RoomReportResponse:
    room_code = room_code.strip().upper()
    room = await store.get_room(db, room_code)
    if not room:
        raise _room_not_found(room_code)

    if current_user.id != room.teacher_id:
        raise HTTPException(status_code=403, detail="Only the room teacher can view report")

    claims = decode_session_token(token)
    if claims.get("role") != "teacher" or claims.get("room_code") != room_code:
        raise HTTPException(status_code=403, detail="Invalid teacher token")

    # Check if report already generated
    from sqlalchemy import select
    res = await db.execute(select(RoomReport).filter(RoomReport.room_id == room.id))
    saved_report = res.scalar_one_or_none()
    if saved_report:
        return RoomReportResponse(
            room_code=room_code,
            room_name=room.room_name,
            teacher_id=current_user.username,
            generated_at=saved_report.generated_at.replace(tzinfo=timezone.utc),
            class_average_score=saved_report.class_average_score,
            students=saved_report.student_summaries,
        )

    # Generate live report from DB
    from app.models.models import RoomParticipant, FocusScore
    from sqlalchemy.orm import selectinload

    part_res = await db.execute(
        select(RoomParticipant).filter(RoomParticipant.room_id == room.id, RoomParticipant.role == "student").options(selectinload(RoomParticipant.scores))
    )
    participants = part_res.scalars().all()

    reports: list[StudentReport] = []
    class_total = 0.0
    for participant in participants:
        samples = participant.scores
        samples.sort(key=lambda s: s.recorded_at)
        
        avg = (
            sum(sample.score for sample in samples) / len(samples)
            if samples
            else participant.current_score
        )
        class_total += avg
        reports.append(
            StudentReport(
                student_id=participant.display_id,
                average_score=round(avg, 2),
                timeline=[
                    StudentTimelinePoint(
                        timestamp=sample.recorded_at.replace(tzinfo=timezone.utc),
                        score=sample.score,
                        status=sample.status_label,
                        camera_on=sample.camera_on,
                    )
                    for sample in samples
                ],
            )
        )

    class_avg = round(class_total / len(participants), 2) if participants else 0.0
    return RoomReportResponse(
        room_code=room_code,
        room_name=room.room_name,
        teacher_id=current_user.username,
        generated_at=datetime.now(timezone.utc),
        class_average_score=class_avg,
        students=reports,
    )
