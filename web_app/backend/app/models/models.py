from __future__ import annotations

import enum
from datetime import datetime
from typing import Any

from sqlalchemy import Boolean, Column, DateTime, Enum, Float, ForeignKey, Integer, String
from sqlalchemy.dialects.mysql import JSON
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func

from app.core.database import Base


class RoleEnum(str, enum.Enum):
    teacher = "teacher"
    student = "student"


class RoomStatusEnum(str, enum.Enum):
    active = "active"
    ended = "ended"


class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, autoincrement=True)
    username = Column(String(64), unique=True, nullable=False, index=True)
    password_hash = Column(String(255), nullable=False)
    role = Column(Enum(RoleEnum), nullable=False)
    created_at = Column(DateTime, default=func.now(), nullable=False)

    rooms = relationship("Room", back_populates="teacher", cascade="all, delete")
    participations = relationship("RoomParticipant", back_populates="user", cascade="all, delete")


class Room(Base):
    __tablename__ = "rooms"

    id = Column(Integer, primary_key=True, autoincrement=True)
    room_code = Column(String(12), unique=True, nullable=False, index=True)
    room_name = Column(String(120), nullable=False)
    teacher_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
    max_students = Column(Integer, default=20, nullable=False)
    status = Column(Enum(RoomStatusEnum), default=RoomStatusEnum.active, nullable=False)
    created_at = Column(DateTime, default=func.now(), nullable=False)
    ended_at = Column(DateTime, nullable=True)

    teacher = relationship("User", back_populates="rooms")
    participants = relationship("RoomParticipant", back_populates="room", cascade="all, delete")
    verification_flags = relationship("VerificationFlag", back_populates="room", cascade="all, delete")
    report = relationship("RoomReport", back_populates="room", uselist=False, cascade="all, delete")


class RoomParticipant(Base):
    __tablename__ = "room_participants"

    id = Column(Integer, primary_key=True, autoincrement=True)
    room_id = Column(Integer, ForeignKey("rooms.id", ondelete="CASCADE"), nullable=False, index=True)
    user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
    display_id = Column(String(64), nullable=False)
    # NOTE: role removed — all participants are students by design
    # NOTE: camera_on / current_score / current_status removed — live state lives in RoomStore._live_cache
    joined_at = Column(DateTime, default=func.now(), nullable=False)
    left_at = Column(DateTime, nullable=True)
    last_score_update = Column(DateTime, nullable=True)
    last_ingest_epoch = Column(Float, default=0.0, nullable=False)

    room = relationship("Room", back_populates="participants")
    user = relationship("User", back_populates="participations")
    verification_flags = relationship("VerificationFlag", back_populates="participant", cascade="all, delete")


class VerificationFlag(Base):
    __tablename__ = "verification_flags"

    id = Column(Integer, primary_key=True, autoincrement=True)
    participant_id = Column(Integer, ForeignKey("room_participants.id", ondelete="CASCADE"), nullable=False, index=True)
    room_id = Column(Integer, ForeignKey("rooms.id", ondelete="CASCADE"), nullable=False, index=True)
    client_score = Column(Float, nullable=False)
    server_score = Column(Float, nullable=False)
    server_status = Column(String(80), nullable=False)
    discrepancy = Column(Float, nullable=False)
    resolved = Column(Boolean, default=False, nullable=False)
    created_at = Column(DateTime, default=func.now(), nullable=False)

    participant = relationship("RoomParticipant", back_populates="verification_flags")
    room = relationship("Room", back_populates="verification_flags")


class RoomReport(Base):
    __tablename__ = "room_reports"

    id = Column(Integer, primary_key=True, autoincrement=True)
    room_id = Column(Integer, ForeignKey("rooms.id", ondelete="CASCADE"), unique=True, nullable=False, index=True)
    class_average_score = Column(Float, nullable=False)
    total_students = Column(Integer, nullable=False)
    student_summaries = Column(JSON, nullable=False)
    generated_at = Column(DateTime, default=func.now(), nullable=False)

    room = relationship("Room", back_populates="report")
