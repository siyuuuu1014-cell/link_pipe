from backend.services.advice_service import DISCLAIMER, fixed_advice
from backend.services.image_service import describe_location
from backend.services.risk_service import assess


def detection(name: str, confidence: float = 0.5, area: float = 0.01, box: list[float] | None = None) -> dict:
    return {"class_name": name, "confidence": confidence, "area_ratio": area, "bbox_xyxy": box or [0, 0, 10, 10]}


def test_no_target_wording_is_bounded() -> None:
    result = assess([])
    assert result["status_zh"] == "当前图像中未检测到明显异常"
    assert "确认无泄漏" not in result["status_zh"]
    assert fixed_advice([])
    assert "可见光" in DISCLAIMER


def test_high_risk_rules() -> None:
    assert assess([detection("leak", confidence=0.8)])["risk_level"] == "high"
    assert assess([detection("leak"), detection("leak")])["risk_level"] == "high"


def test_related_anomaly_is_medium() -> None:
    result = assess([detection("water")])
    assert result["status"] == "related_anomaly"
    assert result["risk_level"] == "medium"


def test_location_grid() -> None:
    assert describe_location([0.0, 0.0, 0.2, 0.2]) == "左上"
    assert describe_location([0.4, 0.4, 0.6, 0.6]) == "图像中央"
