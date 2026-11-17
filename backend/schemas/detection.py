"""API response schemas."""

from __future__ import annotations

from pydantic import BaseModel


class DetectionItem(BaseModel):
    class_id: int
    class_name: str
    class_name_zh: str
    confidence: float
    bbox_xyxy: list[float]
    bbox_normalized: list[float]
    area_ratio: float
    location_text: str


class DetectionResponse(BaseModel):
    record_id: int
    status: str
    status_zh: str
    risk_level: str
    risk_level_zh: str
    counts: dict[str, int]
    detections: list[DetectionItem]
    summary: str
    advice: list[str]
    original_image_url: str
    result_image_url: str
    report_url: str
    inference_ms: float
    disclaimer: str
