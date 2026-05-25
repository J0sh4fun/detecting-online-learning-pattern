from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.core.database import get_db
from app.models.models import Room, RoomStatusEnum, User
from app.models.schemas import RoomHistoryItem, RoomHistoryListResponse, RoomReportResponse
from app.services.auth import get_current_user
from app.services.room_store import store

router = APIRouter(prefix="/api/history", tags=["history"])

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
        select(Room).options(selectinload(Room.report)).filter(Room.room_code == room_code)
    )
    room = res.scalar_one_or_none()
    if not room:
        raise HTTPException(status_code=404, detail="Room not found")
    
    if room.teacher_id != current_user.id:
        raise HTTPException(status_code=403, detail="Not authorized to view this room's report")
    
    if not room.report:
        raise HTTPException(status_code=404, detail="Report not generated for this room yet")

    # In auth context, timezone handling is slightly tricky. The DB stores naïve UTC times by default depending on driver.
    # RoomReportResponse expects generated_at as datetime (Pydantic will serialize to ISO).
    from datetime import timezone
    
    return RoomReportResponse(
        room_code=room.room_code,
        room_name=room.room_name,
        teacher_id=current_user.username,
        generated_at=room.report.generated_at.replace(tzinfo=timezone.utc),
        class_average_score=room.report.class_average_score,
        students=room.report.student_summaries,
    )
