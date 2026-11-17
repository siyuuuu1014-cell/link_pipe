"""Reusable FastAPI dependencies."""

from backend.services.yolo_service import YoloService, get_yolo_service

__all__ = ["YoloService", "get_yolo_service"]
