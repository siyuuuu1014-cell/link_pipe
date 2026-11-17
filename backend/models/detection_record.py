"""Persistent detection history model."""

from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy import DateTime, Float, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from backend.database import Base


class DetectionRecord(Base):
    __tablename__ = "detection_records"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), index=True)
    source_type: Mapped[str] = mapped_column(String(16), default="image", index=True)
    original_filename: Mapped[str] = mapped_column(String(255))
    original_path: Mapped[str] = mapped_column(Text)
    result_path: Mapped[str | None] = mapped_column(Text, nullable=True)
    status: Mapped[str] = mapped_column(String(32), index=True)
    risk_level: Mapped[str] = mapped_column(String(32), index=True)
    confidence_threshold: Mapped[float] = mapped_column(Float)
    iou_threshold: Mapped[float] = mapped_column(Float)
    detections_json: Mapped[str] = mapped_column(Text, default="[]")
    summary: Mapped[str] = mapped_column(Text)
    advice: Mapped[str] = mapped_column(Text)
    model_version: Mapped[str] = mapped_column(String(255), default="unavailable")
    inference_ms: Mapped[float] = mapped_column(Float, default=0)
    video_result_path: Mapped[str | None] = mapped_column(Text, nullable=True)
    duration: Mapped[float | None] = mapped_column(Float, nullable=True)
    frame_count: Mapped[int | None] = mapped_column(Integer, nullable=True)
    leak_time_ranges_json: Mapped[str | None] = mapped_column(Text, nullable=True)
