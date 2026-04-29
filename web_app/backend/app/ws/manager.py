from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timezone

from fastapi import WebSocket

from app.models.schemas import StudentSummary, TeacherScoresSnapshot
from app.services.room_store import store


class SocketManager:
    def __init__(self) -> None:
        self.teacher_sockets: dict[str, set[WebSocket]] = defaultdict(set)

    async def connect_teacher(self, room_code: str, websocket: WebSocket) -> None:
        await websocket.accept()
        self.teacher_sockets[room_code].add(websocket)
        await self.broadcast_snapshot(room_code)

    def disconnect_teacher(self, room_code: str, websocket: WebSocket) -> None:
        room_set = self.teacher_sockets.get(room_code)
        if not room_set:
            return
        room_set.discard(websocket)
        if not room_set:
            self.teacher_sockets.pop(room_code, None)

    async def broadcast_snapshot(self, room_code: str) -> None:
        sockets = list(self.teacher_sockets.get(room_code, set()))
        if not sockets:
            return

        students = [
            StudentSummary(
                student_id=student.student_id,
                score=student.score,
                status=student.status,
                camera_on=student.camera_on,
                is_warning=student.is_warning,
                last_update=student.last_update,
            )
            for student in store.snapshot_students(room_code)
        ]

        payload = TeacherScoresSnapshot(
            room_code=room_code,
            students=students,
            updated_at=datetime.now(timezone.utc),
        )

        disconnected: list[WebSocket] = []
        for socket in sockets:
            try:
                await socket.send_json(payload.model_dump(mode="json"))
            except Exception:
                disconnected.append(socket)

        for socket in disconnected:
            self.disconnect_teacher(room_code, socket)


socket_manager = SocketManager()

