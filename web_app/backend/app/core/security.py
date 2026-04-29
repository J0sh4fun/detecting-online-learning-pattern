from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any

from jose import JWTError, jwt

from app.core.config import settings


def create_session_token(*, room_code: str, participant_id: str, role: str) -> str:
    expire = datetime.now(tz=timezone.utc) + timedelta(minutes=settings.app_jwt_exp_minutes)
    payload = {
        "room_code": room_code,
        "participant_id": participant_id,
        "role": role,
        "exp": expire,
    }
    return jwt.encode(payload, settings.app_jwt_secret, algorithm=settings.app_jwt_algo)


def decode_session_token(token: str) -> dict[str, Any]:
    try:
        return jwt.decode(token, settings.app_jwt_secret, algorithms=[settings.app_jwt_algo])
    except JWTError as exc:
        raise ValueError("Invalid session token") from exc

