"""Application configuration loaded from environment variables and .env."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=PROJECT_ROOT / ".env", env_file_encoding="utf-8", extra="ignore")

    app_name: str = "管道泄漏多目标智能检测系统"
    backend_host: str = "127.0.0.1"
    backend_port: int = 8000
    frontend_port: int = 5173
    yolo_model_path: str = "models/deployed/best.pt"
    yolo_device: str = "0"
    default_confidence: float = 0.25
    default_iou: float = 0.50
    max_image_size_mb: int = 20
    max_video_size_mb: int = 500
    database_url: str = "sqlite:///storage/database/app.db"
    enable_llm: bool = False
    ollama_base_url: str = "http://127.0.0.1:11434"
    ollama_model: str = "qwen3:4b"

    @property
    def model_path(self) -> Path:
        path = Path(self.yolo_model_path)
        return path if path.is_absolute() else PROJECT_ROOT / path


@lru_cache
def get_settings() -> Settings:
    return Settings()
