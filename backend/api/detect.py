"""Image detection endpoint."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from sqlalchemy.orm import Session

from backend.config import get_settings
from backend.database import get_db
from backend.models.detection_record import DetectionRecord
from backend.schemas.detection import DetectionResponse
from backend.services.advice_service import DISCLAIMER, fixed_advice
from backend.services.image_service import save_valid_image
from backend.services.llm_service import polish_advice
from backend.services.risk_service import assess
from backend.services.yolo_service import YoloService, get_yolo_service


router = APIRouter(tags=["detection"])


@router.post("/detect/image", response_model=DetectionResponse)
async def detect_image(
    file: UploadFile = File(...),
    confidence: float = Form(0.25, ge=0.01, le=1.0),
    iou: float = Form(0.50, ge=0.01, le=1.0),
    use_llm: bool = Form(False),
    db: Session = Depends(get_db),
    service: YoloService = Depends(get_yolo_service),
) -> dict:
    image_path, _ = await save_valid_image(file)
    try:
        detections, annotated, inference_ms = service.predict(image_path, confidence, iou)
    except RuntimeError as exc:
        image_path.unlink(missing_ok=True)
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    result_path = image_path.parents[1] / "results" / f"{image_path.stem}_detected.jpg"
    service.save_annotated(annotated, result_path)
    assessment = assess(detections)
    advice = await polish_advice({**assessment, "detections": detections}, fixed_advice(detections), use_llm)
    summary = assessment["status_zh"]
    record = DetectionRecord(
        source_type="image",
        original_filename=file.filename or image_path.name,
        original_path=image_path.as_posix(),
        result_path=result_path.as_posix(),
        status=assessment["status"],
        risk_level=assessment["risk_level"],
        confidence_threshold=confidence,
        iou_threshold=iou,
        detections_json=json.dumps(detections, ensure_ascii=False),
        summary=summary,
        advice=json.dumps(advice, ensure_ascii=False),
        model_version=str(service.model_path or "unavailable"),
        inference_ms=inference_ms,
    )
    db.add(record)
    db.commit()
    db.refresh(record)
    return {
        "record_id": record.id,
        **assessment,
        "detections": detections,
        "summary": summary,
        "advice": advice,
        "original_image_url": f"/storage/uploads/{image_path.name}",
        "result_image_url": f"/storage/results/{result_path.name}",
        "report_url": f"/api/v1/reports/{record.id}",
        "inference_ms": inference_ms,
        "disclaimer": DISCLAIMER,
    }
