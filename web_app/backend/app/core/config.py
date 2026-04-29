from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


_backend_root = Path(__file__).resolve().parents[2]
load_dotenv(_backend_root / ".env")


@dataclass(frozen=True)
class Settings:
    api_title: str = "Focus Classroom API"
    app_jwt_secret: str = os.getenv("APP_JWT_SECRET", "dev-secret-change-me")
    app_jwt_algo: str = os.getenv("APP_JWT_ALGO", "HS256")
    app_jwt_exp_minutes: int = int(os.getenv("APP_JWT_EXP_MINUTES", "240"))
    cors_origins: tuple[str, ...] = tuple(
        item.strip()
        for item in os.getenv("CORS_ORIGINS", "http://localhost:5173").split(",")
        if item.strip()
    )
    livekit_url: str = os.getenv("LIVEKIT_URL", "ws://localhost:7880")
    livekit_api_key: str = os.getenv("LIVEKIT_API_KEY", "devkey")
    livekit_api_secret: str = os.getenv("LIVEKIT_API_SECRET", "secret")
    max_students_per_room: int = int(os.getenv("MAX_STUDENTS_PER_ROOM", "50"))
    min_ingest_interval_sec: float = float(os.getenv("MIN_INGEST_INTERVAL_SEC", "2.0"))
    score_low_threshold: float = float(os.getenv("SCORE_LOW_THRESHOLD", "55"))
    verify_discrepancy_threshold: float = float(os.getenv("VERIFY_DISCREPANCY_THRESHOLD", "25"))

    @property
    def project_root(self) -> Path:
        return Path(__file__).resolve().parents[4]

    @property
    def model_pipeline_dir(self) -> Path:
        return self.project_root / "model_pipeline"

    @property
    def scaler_path(self) -> Path:
        return self.model_pipeline_dir / "models" / "scaler.pkl"

    @property
    def classifier_path(self) -> Path:
        return self.model_pipeline_dir / "models" / "best_posture_model.pkl"


settings = Settings()

