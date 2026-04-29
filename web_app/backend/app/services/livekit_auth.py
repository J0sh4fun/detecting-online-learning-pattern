from __future__ import annotations

from livekit import api

from app.core.config import settings


def build_livekit_token(*, room_code: str, participant_id: str, is_teacher: bool) -> str:
    grants = api.VideoGrants(
        room=room_code,
        room_join=True,
        can_publish=True,
        can_subscribe=True,
        can_publish_data=True,
    )
    if is_teacher:
        grants.room_admin = True

    token = (
        api.AccessToken(settings.livekit_api_key, settings.livekit_api_secret)
        .with_identity(participant_id)
        .with_name(participant_id)
        .with_grants(grants)
    )
    return token.to_jwt()

