"""Deterministic status and risk classification rules."""

from __future__ import annotations

from collections import Counter
from functools import lru_cache
from pathlib import Path

import yaml

from backend.config import PROJECT_ROOT


RISK_ZH = {"high": "高风险", "medium": "中风险", "none": "未发现明显异常"}
STATUS_ZH = {
    "suspected_leak": "检测到疑似泄漏",
    "related_anomaly": "未检测到明确泄漏，但发现相关异常",
    "no_visible_anomaly": "当前图像中未检测到明显异常",
}


@lru_cache
def load_rules() -> dict:
    path = PROJECT_ROOT / "backend/config/risk_rules.yaml"
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def bbox_iou(left: list[float], right: list[float]) -> float:
    x1 = max(left[0], right[0])
    y1 = max(left[1], right[1])
    x2 = min(left[2], right[2])
    y2 = min(left[3], right[3])
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    left_area = max(0.0, left[2] - left[0]) * max(0.0, left[3] - left[1])
    right_area = max(0.0, right[2] - right[0]) * max(0.0, right[3] - right[1])
    union = left_area + right_area - intersection
    return intersection / union if union else 0.0


def assess(detections: list[dict]) -> dict:
    rules = load_rules()
    counts = Counter(item["class_name"] for item in detections)
    leaks = [item for item in detections if item["class_name"] == "leak"]
    cracks = [item for item in detections if item["class_name"] == "crack"]
    if leaks:
        status = "suspected_leak"
    elif counts["crack"] or counts["water"]:
        status = "related_anomaly"
    else:
        status = "no_visible_anomaly"

    high = rules["high"]
    leak_crack_overlap = any(bbox_iou(leak["bbox_xyxy"], crack["bbox_xyxy"]) >= high["leak_crack_iou"] for leak in leaks for crack in cracks)
    is_high = any(item["confidence"] >= high["leak_confidence"] or item["area_ratio"] >= high["leak_area_ratio"] for item in leaks)
    is_high = is_high or len(leaks) >= high["leak_count"] or leak_crack_overlap
    risk = "high" if is_high else ("medium" if detections else "none")
    normalized_counts = {name: counts[name] for name in ("crack", "water", "leak")}
    return {
        "status": status,
        "status_zh": STATUS_ZH[status],
        "risk_level": risk,
        "risk_level_zh": RISK_ZH[risk],
        "counts": normalized_counts,
    }
