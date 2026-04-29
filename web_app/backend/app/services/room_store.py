from __future__ import annotations

import secrets
import string
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timezone

from app.core.config import settings


@dataclass
class RoomMeta:
    room_code: str
    room_name: str
    teacher_id: str
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    ended_at: datetime | None = None


@dataclass
class StudentState:
    student_id: str
    score: float = 100.0
    status: str = "Waiting"
    camera_on: bool = True
    last_update: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    last_ingest_epoch: float = 0.0

    @property
    def is_warning(self) -> bool:
        return (not self.camera_on) or self.score <= settings.score_low_threshold


@dataclass
class TimelineSample:
    timestamp: datetime
    score: float
    status: str


class RoomStore:
    def __init__(self) -> None:
        self.rooms: dict[str, RoomMeta] = {}
        self.students: dict[str, dict[str, StudentState]] = defaultdict(dict)
        self.timeline: dict[str, dict[str, deque[TimelineSample]]] = defaultdict(
            lambda: defaultdict(lambda: deque(maxlen=2400))
        )
        self.verification_flags: dict[str, list[dict]] = defaultdict(list)

    @staticmethod
    def generate_room_code(length: int = 6) -> str:
        alphabet = string.ascii_uppercase + string.digits
        return "".join(secrets.choice(alphabet) for _ in range(length))

    def create_room(self, *, teacher_id: str, room_name: str) -> RoomMeta:
        room_code = self.generate_room_code()
        while room_code in self.rooms:
            room_code = self.generate_room_code()
        meta = RoomMeta(room_code=room_code, room_name=room_name, teacher_id=teacher_id)
        self.rooms[room_code] = meta
        return meta

    def get_room(self, room_code: str) -> RoomMeta | None:
        return self.rooms.get(room_code)

    def ensure_student(self, *, room_code: str, student_id: str) -> StudentState:
        students = self.students[room_code]
        if student_id not in students:
            if len(students) >= settings.max_students_per_room:
                raise ValueError("Room is full")
            students[student_id] = StudentState(student_id=student_id)
        return students[student_id]

    def update_student_score(
        self,
        *,
        room_code: str,
        student_id: str,
        average_score: float,
        status: str,
        camera_on: bool,
        client_sent_at: float,
    ) -> StudentState:
        student = self.ensure_student(room_code=room_code, student_id=student_id)
        now = datetime.now(timezone.utc)

        if client_sent_at - student.last_ingest_epoch < settings.min_ingest_interval_sec:
            return student

        student.score = average_score
        student.status = status
        student.camera_on = camera_on
        student.last_update = now
        student.last_ingest_epoch = client_sent_at

        self.timeline[room_code][student_id].append(
            TimelineSample(timestamp=now, score=average_score, status=status)
        )
        return student

    def snapshot_students(self, room_code: str) -> list[StudentState]:
        students = self.students.get(room_code, {})
        return sorted(students.values(), key=lambda item: item.student_id.lower())

    def end_room(self, room_code: str) -> None:
        room = self.get_room(room_code)
        if room:
            room.ended_at = datetime.now(timezone.utc)

    def add_verification_flag(self, room_code: str, payload: dict) -> None:
        self.verification_flags[room_code].append(payload)


store = RoomStore()

