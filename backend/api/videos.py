"""Video detection endpoint."""

from __future__ import annotations

import json

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from sqlalchemy.orm import Session

from backend.database import get_db
from backend.models.detection_record import DetectionRecord
from backend.services.video_service import process_video, save_valid_video
from backend.services.yolo_service import YoloService, get_yolo_service


router = APIRouter(tags=["video"])


@router.post("/detect/video")
async def detect_video(
    file: UploadFile = File(...),
    confidence: float = Form(0.25, ge=0.01, le=1.0),
    iou: float = Form(0.50, ge=0.01, le=1.0),
    frame_interval: int = Form(1, ge=1, le=120),
    db: Session = Depends(get_db),
    service: YoloService = Depends(get_yolo_service),
) -> dict:
    source = await save_valid_video(file)
    try:
        result = process_video(source, service, confidence, iou, frame_interval)
    except RuntimeError as exc:
        source.unlink(missing_ok=True)
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    leak_count = result["counts"]["leak"]
    anomaly_count = sum(result["counts"].values())
    status = "suspected_leak" if leak_count else ("related_anomaly" if anomaly_count else "no_visible_anomaly")
    summary = "检测到疑似泄漏" if leak_count else ("未检测到明确泄漏，但发现相关异常" if anomaly_count else "当前图像中未检测到明显异常")
    record = DetectionRecord(
        source_type="video",
        original_filename=file.filename or source.name,
        original_path=source.as_posix(),
        result_path=None,
        status=status,
        risk_level="high" if leak_count >= 2 else ("medium" if anomaly_count else "none"),
        confidence_threshold=confidence,
        iou_threshold=iou,
        detections_json="[]",
        summary=summary,
        advice="[]",
        model_version=str(service.model_path or "unavailable"),
        inference_ms=result["inference_ms"],
        video_result_path=result["result_path"].as_posix(),
        duration=result["duration"],
        frame_count=result["frame_count"],
        leak_time_ranges_json=json.dumps(result["leak_time_ranges"]),
    )
    db.add(record)
    db.commit()
    db.refresh(record)
    return {
        "record_id": record.id,
        "status": status,
        "summary": summary,
        "counts": result["counts"],
        "leak_time_ranges": result["leak_time_ranges"],
        "duration": result["duration"],
        "frame_count": result["frame_count"],
        "result_video_url": f"/storage/videos/{result['result_path'].name}",
    }
