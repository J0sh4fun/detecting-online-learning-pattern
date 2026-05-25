from __future__ import annotations

import secrets
import string
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone

from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.core.config import settings
from app.models.models import FocusScore, RoleEnum, Room, RoomParticipant, RoomStatusEnum, VerificationFlag


@dataclass
class StudentLiveStatus:
    student_id: str
    score: float
    status: str
    camera_on: bool
    last_update: datetime
    is_warning: bool


class RoomStore:
    def __init__(self) -> None:
        # In-memory cache for live broadcast performance
        # room_code -> { student_id_str: StudentLiveStatus }
        self._live_cache: dict[str, dict[str, StudentLiveStatus]] = defaultdict(dict)

    @staticmethod
    def generate_room_code(length: int = 6) -> str:
        alphabet = string.ascii_uppercase + string.digits
        return "".join(secrets.choice(alphabet) for _ in range(length))

    async def create_room(self, db: AsyncSession, *, teacher_id: int, room_name: str) -> Room:
        while True:
            code = self.generate_room_code()
            res = await db.execute(select(Room).filter(Room.room_code == code))
            if res.scalar_one_or_none() is None:
                break

        room = Room(room_code=code, room_name=room_name, teacher_id=teacher_id)
        db.add(room)
        await db.commit()
        await db.refresh(room)
        return room

    async def get_room(self, db: AsyncSession, room_code: str) -> Room | None:
        res = await db.execute(
            select(Room)
            .options(selectinload(Room.teacher))
            .filter(Room.room_code == room_code, Room.status == RoomStatusEnum.active)
        )
        return res.scalar_one_or_none()

    async def ensure_student(
        self, db: AsyncSession, *, room_id: int, user_id: int, display_id: str
    ) -> RoomParticipant:
        res = await db.execute(
            select(RoomParticipant).filter(
                RoomParticipant.room_id == room_id, RoomParticipant.user_id == user_id
            )
        )
        participant = res.scalar_one_or_none()
        if not participant:
            # Check room capacity
            count_res = await db.execute(
                select(RoomParticipant).filter(RoomParticipant.room_id == room_id)
            )
            if len(count_res.scalars().all()) >= settings.max_students_per_room:
                raise ValueError("Room is full")

            participant = RoomParticipant(
                room_id=room_id,
                user_id=user_id,
                display_id=display_id,
                role=RoleEnum.student,
            )
            db.add(participant)
            await db.commit()
            await db.refresh(participant)

            # Init cache
            room_res = await db.execute(select(Room).filter(Room.id == room_id))
            room = room_res.scalar_one()
            self._live_cache[room.room_code][display_id] = StudentLiveStatus(
                student_id=display_id,
                score=100.0,
                status="Waiting",
                camera_on=True,
                last_update=datetime.now(timezone.utc),
                is_warning=False,
            )

        return participant

    async def update_student_score(
        self,
        db: AsyncSession,
        *,
        room_code: str,
        user_id: int,
        display_id: str,
        average_score: float,
        status: str,
        camera_on: bool,
        client_sent_at: float,
    ) -> RoomParticipant | None:
        room = await self.get_room(db, room_code)
        if not room:
            return None

        participant = await self.ensure_student(
            db, room_id=room.id, user_id=user_id, display_id=display_id
        )

        if client_sent_at - participant.last_ingest_epoch < settings.min_ingest_interval_sec:
            return participant

        now = datetime.now(timezone.utc)
        participant.current_score = average_score
        participant.current_status = status
        participant.camera_on = camera_on
        participant.last_score_update = now
        participant.last_ingest_epoch = client_sent_at

        score_entry = FocusScore(
            participant_id=participant.id,
            room_id=room.id,
            score=average_score,
            status_label=status,
            camera_on=camera_on,
            recorded_at=now,
        )
        db.add(score_entry)
        await db.commit()
        await db.refresh(participant)

        # Update cache
        is_warning = (not camera_on) or average_score <= settings.score_low_threshold
        self._live_cache[room_code][display_id] = StudentLiveStatus(
            student_id=display_id,
            score=average_score,
            status=status,
            camera_on=camera_on,
            last_update=now,
            is_warning=is_warning,
        )

        return participant

    def snapshot_students(self, room_code: str) -> list[StudentLiveStatus]:
        students = self._live_cache.get(room_code, {})
        return sorted(students.values(), key=lambda item: item.student_id.lower())

    async def end_room(self, db: AsyncSession, room_code: str) -> Room | None:
        room = await self.get_room(db, room_code)
        if room:
            room.status = RoomStatusEnum.ended
            room.ended_at = datetime.now(timezone.utc)
            await db.commit()
            
            # Clean up cache
            self._live_cache.pop(room_code, None)
        return room

    async def add_verification_flag(
        self, db: AsyncSession, *, room_id: int, user_id: int, payload: dict
    ) -> None:
        res = await db.execute(
            select(RoomParticipant).filter(
                RoomParticipant.room_id == room_id, RoomParticipant.user_id == user_id
            )
        )
        participant = res.scalar_one_or_none()
        if participant:
            flag = VerificationFlag(
                participant_id=participant.id,
                room_id=room_id,
                client_score=payload["client_score"],
                server_score=payload["server_score"],
                server_status=payload["server_status"],
                discrepancy=payload["discrepancy"],
            )
            db.add(flag)
            await db.commit()


store = RoomStore()
