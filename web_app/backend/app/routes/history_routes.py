from __future__ import annotations

from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.core.database import get_db
from app.models.models import Room, RoomParticipant, RoomReport, RoomStatusEnum, User
from app.models.schemas import RoomHistoryItem, RoomHistoryListResponse, RoomReportResponse, StudentReport, StudentTimelinePoint
from app.services.auth import get_current_user

router = APIRouter(prefix="/api/history", tags=["history"])


def as_utc(value: datetime) -> datetime:
    if value.tzinfo:
        return value.astimezone(timezone.utc)
    return value.replace(tzinfo=timezone.utc)


def build_report_response(room: Room, teacher_name: str, generated_at: datetime) -> RoomReportResponse:
    reports: list[StudentReport] = []
    class_total = 0.0

    # All participants are students — role column has been removed
    participants = list(room.participants)

    for participant in sorted(participants, key=lambda item: item.display_id.lower()):
        samples = sorted(participant.scores, key=lambda sample: sample.recorded_at)
        avg = (
            sum(sample.score for sample in samples) / len(samples)
            if samples
            else 0.0  # current_score removed from DB — no samples means student never sent data
        )
        class_total += avg
        reports.append(
            StudentReport(
                student_id=participant.display_id,
                average_score=round(avg, 2),
                timeline=[
                    StudentTimelinePoint(
                        timestamp=as_utc(sample.recorded_at),
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
        room_code=room.room_code,
        room_name=room.room_name,
        teacher_id=teacher_name,
        generated_at=as_utc(generated_at),
        class_average_score=class_avg,
        students=reports,
    )


async def ensure_saved_report(db: AsyncSession, room: Room, teacher_name: str) -> RoomReportResponse:
    if room.report:
        return RoomReportResponse(
            room_code=room.room_code,
            room_name=room.room_name,
            teacher_id=teacher_name,
            generated_at=as_utc(room.report.generated_at),
            class_average_score=room.report.class_average_score,
            students=room.report.student_summaries,
        )

    report = build_report_response(room, teacher_name, datetime.now(timezone.utc))
    if room.status == RoomStatusEnum.ended:
        db_report = RoomReport(
            room_id=room.id,
            class_average_score=report.class_average_score,
            total_students=len(report.students),
            student_summaries=report.model_dump(mode="json")["students"],
        )
        db.add(db_report)
        await db.commit()
    return report

@router.get("/rooms", response_model=RoomHistoryListResponse)
async def get_rooms_history(
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    if current_user.role.value != "teacher":
        raise HTTPException(status_code=403, detail="Only teachers can view room history")

    res = await db.execute(
        select(Room)
        .options(selectinload(Room.report))
        .filter(Room.teacher_id == current_user.id)
        .order_by(Room.created_at.desc())
    )
    rooms = res.scalars().all()

    items = []
    for room in rooms:
        # A room might not be ended yet
        student_count = 0
        class_average = None
        if room.report:
            student_count = room.report.total_students
            class_average = room.report.class_average_score
        
        items.append(
            RoomHistoryItem(
                room_code=room.room_code,
                room_name=room.room_name,
                status=room.status.value,
                created_at=room.created_at,
                ended_at=room.ended_at,
                student_count=student_count,
                class_average=class_average,
            )
        )
    return RoomHistoryListResponse(rooms=items)


@router.get("/rooms/{room_code}/report", response_model=RoomReportResponse)
async def get_history_report(
    room_code: str,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    if current_user.role.value != "teacher":
        raise HTTPException(status_code=403, detail="Only teachers can view room reports")

    room_code = room_code.strip().upper()
    res = await db.execute(
        select(Room)
        .options(
            selectinload(Room.report),
            selectinload(Room.participants).selectinload(RoomParticipant.scores),
        )
        .filter(Room.room_code == room_code)
    )
    room = res.scalar_one_or_none()
    if not room:
        raise HTTPException(status_code=404, detail="Room not found")
    
    if room.teacher_id != current_user.id:
        raise HTTPException(status_code=403, detail="Not authorized to view this room's report")
    
    return await ensure_saved_report(db, room, current_user.username)
