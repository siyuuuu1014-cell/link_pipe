from fastapi import APIRouter

from backend.config import get_settings
from backend.services.yolo_service import get_yolo_service


router = APIRouter(tags=["health"])


@router.get("/health")
def health() -> dict:
    service = get_yolo_service()
    settings = get_settings()
    return {
        "status": "ok",
        "app_name": settings.app_name,
        "model_available": service.available,
        "model_path": str(service.model_path or settings.model_path),
        "model_error": service.load_error,
        "device": service.device,
    }
