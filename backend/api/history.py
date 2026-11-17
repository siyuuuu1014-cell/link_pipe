from __future__ import annotations

import json
from datetime import date, datetime, time, timezone

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import select
from sqlalchemy.orm import Session

from backend.database import get_db
from backend.models.detection_record import DetectionRecord


router = APIRouter(tags=["history"])


def serialize(record: DetectionRecord) -> dict:
    return {
        "id": record.id,
        "created_at": record.created_at,
        "source_type": record.source_type,
        "original_filename": record.original_filename,
        "status": record.status,
        "risk_level": record.risk_level,
        "summary": record.summary,
        "detections": json.loads(record.detections_json or "[]"),
        "advice": json.loads(record.advice or "[]"),
        "model_version": record.model_version,
        "inference_ms": record.inference_ms,
        "original_image_url": f"/storage/uploads/{record.original_path.split('/')[-1]}" if record.source_type == "image" else None,
        "result_image_url": f"/storage/results/{record.result_path.split('/')[-1]}" if record.result_path else None,
        "video_result_url": f"/storage/videos/{record.video_result_path.split('/')[-1]}" if record.video_result_path else None,
        "report_url": f"/api/v1/reports/{record.id}",
    }


@router.get("/history")
def list_history(
    status: str | None = None,
    risk_level: str | None = None,
    day: date | None = Query(None),
    db: Session = Depends(get_db),
) -> dict:
    query = select(DetectionRecord).order_by(DetectionRecord.created_at.desc())
    if status:
        query = query.where(DetectionRecord.status == status)
    if risk_level:
        query = query.where(DetectionRecord.risk_level == risk_level)
    if day:
        start = datetime.combine(day, time.min, tzinfo=timezone.utc)
        end = datetime.combine(day, time.max, tzinfo=timezone.utc)
        query = query.where(DetectionRecord.created_at.between(start, end))
    records = db.scalars(query.limit(200)).all()
    return {"items": [serialize(record) for record in records], "total": len(records)}


@router.get("/history/{record_id}")
def history_detail(record_id: int, db: Session = Depends(get_db)) -> dict:
    record = db.get(DetectionRecord, record_id)
    if not record:
        raise HTTPException(status_code=404, detail="检测记录不存在")
    return serialize(record)


@router.delete("/history/{record_id}")
def delete_history(record_id: int, db: Session = Depends(get_db)) -> dict:
    record = db.get(DetectionRecord, record_id)
    if not record:
        raise HTTPException(status_code=404, detail="检测记录不存在")
    db.delete(record)
    db.commit()
    return {"deleted": True, "record_id": record_id}
