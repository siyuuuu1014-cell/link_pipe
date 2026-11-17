"""Render printable detection reports from persisted records."""

from __future__ import annotations

import json

from jinja2 import Environment, FileSystemLoader, select_autoescape

from backend.config import PROJECT_ROOT
from backend.models.detection_record import DetectionRecord
from backend.services.advice_service import DISCLAIMER


_environment = Environment(
    loader=FileSystemLoader(PROJECT_ROOT / "backend/templates"),
    autoescape=select_autoescape(["html", "xml"]),
)


def render_report(record: DetectionRecord) -> str:
    template = _environment.get_template("detection_report.html")
    detections = json.loads(record.detections_json or "[]")
    try:
        advice = json.loads(record.advice)
    except json.JSONDecodeError:
        advice = [record.advice]
    return template.render(record=record, detections=detections, advice=advice, disclaimer=DISCLAIMER)
