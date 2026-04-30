from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field


RoleLiteral = Literal["teacher", "student"]


class CreateRoomRequest(BaseModel):
    teacher_id: str = Field(min_length=2, max_length=64)
    room_name: str = Field(min_length=2, max_length=120)


class JoinRoomRequest(BaseModel):
    room_code: str = Field(min_length=4, max_length=12)
    student_id: str = Field(min_length=2, max_length=64)


class RoomConnectionResponse(BaseModel):
    room_code: str
    role: RoleLiteral
    livekit_url: str
    livekit_token: str
    session_token: str
    score_ws_url: str
    invitation_link: str | None = None


class StudentScoreIngest(BaseModel):
    token: str
    average_score: float = Field(ge=0, le=100)
    status: str = Field(min_length=2, max_length=80)
    camera_on: bool
    sampled_fps: float = Field(ge=0.1, le=60)
    sample_count: int = Field(ge=1, le=50)
    client_sent_at: float


class StudentSummary(BaseModel):
    student_id: str
    score: float
    status: str
    camera_on: bool
    is_warning: bool
    last_update: datetime


class TeacherScoresSnapshot(BaseModel):
    type: Literal["scores_snapshot"] = "scores_snapshot"
    room_code: str
    students: list[StudentSummary]
    updated_at: datetime


class VerifyFrameRequest(BaseModel):
    token: str
    room_code: str
    student_id: str
    client_score: float = Field(ge=0, le=100)
    frame_base64: str = Field(min_length=64)


class VerifyFrameResponse(BaseModel):
    is_flagged: bool
    discrepancy: float
    server_score: float
    server_status: str
    reason: str


class ScoreFrameRequest(BaseModel):
    token: str
    room_code: str
    student_id: str
    frame_base64: str = Field(min_length=64)


class ScoreFrameResponse(BaseModel):
    score: float
    status: str


class StudentTimelinePoint(BaseModel):
    timestamp: datetime
    score: float
    status: str


class StudentReport(BaseModel):
    student_id: str
    average_score: float
    timeline: list[StudentTimelinePoint]


class RoomReportResponse(BaseModel):
    room_code: str
    generated_at: datetime
    class_average_score: float
    students: list[StudentReport]

