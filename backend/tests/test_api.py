from fastapi.testclient import TestClient
import cv2
import numpy as np

from backend.main import app
from backend.services.yolo_service import get_yolo_service


class FakeYoloService:
    available = True
    load_error = None
    device = "cpu"
    model_path = "fake-smoke.pt"

    def __init__(self, detections: list[dict] | None = None):
        self.detections = detections or []

    def predict(self, source, confidence, iou):
        return self.detections, np.zeros((32, 32, 3), dtype=np.uint8), 1.25

    def save_annotated(self, image, destination):
        destination.parent.mkdir(parents=True, exist_ok=True)
        assert cv2.imwrite(str(destination), image)


def png_bytes() -> bytes:
    ok, encoded = cv2.imencode(".png", np.zeros((32, 32, 3), dtype=np.uint8))
    assert ok
    return encoded.tobytes()


def test_health_reports_missing_model_without_fake_success() -> None:
    with TestClient(app) as client:
        response = client.get("/api/v1/health")
    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "ok"
    assert isinstance(payload["model_available"], bool)


def test_invalid_image_is_rejected() -> None:
    with TestClient(app) as client:
        response = client.post(
            "/api/v1/detect/image",
            files={"file": ("bad.txt", b"not an image", "text/plain")},
            data={"confidence": "0.25", "iou": "0.5", "use_llm": "false"},
        )
    assert response.status_code == 415


def test_empty_file_is_rejected() -> None:
    with TestClient(app) as client:
        response = client.post("/api/v1/detect/image", files={"file": ("empty.png", b"", "image/png")})
    assert response.status_code == 400


def test_no_target_image_history_and_report() -> None:
    app.dependency_overrides[get_yolo_service] = lambda: FakeYoloService()
    try:
        with TestClient(app) as client:
            response = client.post("/api/v1/detect/image", files={"file": ("normal.png", png_bytes(), "image/png")})
            assert response.status_code == 200
            payload = response.json()
            assert payload["status"] == "no_visible_anomaly"
            assert payload["detections"] == []
            assert client.get(f"/api/v1/history/{payload['record_id']}").status_code == 200
            report = client.get(payload["report_url"])
            assert report.status_code == 200
            assert "不能替代压力" in report.text
            assert client.delete(f"/api/v1/history/{payload['record_id']}").status_code == 200
    finally:
        app.dependency_overrides.clear()


def test_multi_target_image_returns_structured_risk() -> None:
    detections = [
        {"class_id": 2, "class_name": "leak", "class_name_zh": "疑似泄漏", "confidence": 0.8, "bbox_xyxy": [1, 1, 20, 20], "bbox_normalized": [0.03, 0.03, 0.62, 0.62], "area_ratio": 0.35, "location_text": "左上"},
        {"class_id": 0, "class_name": "crack", "class_name_zh": "管道裂纹", "confidence": 0.6, "bbox_xyxy": [2, 2, 18, 18], "bbox_normalized": [0.06, 0.06, 0.56, 0.56], "area_ratio": 0.25, "location_text": "左上"},
    ]
    app.dependency_overrides[get_yolo_service] = lambda: FakeYoloService(detections)
    try:
        with TestClient(app) as client:
            payload = client.post("/api/v1/detect/image", files={"file": ("risk.png", png_bytes(), "image/png")}).json()
            assert payload["risk_level"] == "high"
            assert payload["counts"]["leak"] == 1
            client.delete(f"/api/v1/history/{payload['record_id']}")
    finally:
        app.dependency_overrides.clear()


def test_metrics_never_invents_values() -> None:
    with TestClient(app) as client:
        payload = client.get("/api/v1/metrics").json()
    assert "evaluation_complete" in payload
    if not payload["evaluation_complete"]:
        assert payload["selected"] is None
