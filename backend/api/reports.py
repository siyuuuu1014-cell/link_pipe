from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import HTMLResponse
from sqlalchemy.orm import Session

from backend.database import get_db
from backend.models.detection_record import DetectionRecord
from backend.services.report_service import render_report


router = APIRouter(tags=["reports"])


@router.get("/reports/{record_id}", response_class=HTMLResponse)
def report(record_id: int, db: Session = Depends(get_db)) -> HTMLResponse:
    record = db.get(DetectionRecord, record_id)
    if not record:
        raise HTTPException(status_code=404, detail="检测记录不存在")
    return HTMLResponse(render_report(record), headers={"Content-Disposition": f'inline; filename="report-{record_id}.html"'})
