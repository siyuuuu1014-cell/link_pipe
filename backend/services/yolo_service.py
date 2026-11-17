"""Process-wide YOLO model singleton and structured inference adapter."""

from __future__ import annotations

import threading
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from ultralytics import YOLO

from backend.config import get_settings
from backend.services.image_service import describe_location


CLASS_NAME_ZH = {"crack": "管道裂纹", "water": "可见水体/积水", "leak": "疑似泄漏"}


class YoloService:
    def __init__(self) -> None:
        self.model: YOLO | None = None
        self.model_path: Path | None = None
        self.device = "cpu"
        self.load_error: str | None = None
        self._lock = threading.Lock()

    @property
    def available(self) -> bool:
        return self.model is not None

    def load(self) -> None:
        if self.model is not None:
            return
        with self._lock:
            if self.model is not None:
                return
            settings = get_settings()
            self.model_path = settings.model_path.resolve()
            self.device = settings.yolo_device if torch.cuda.is_available() and settings.yolo_device.lower() != "cpu" else "cpu"
            if not self.model_path.is_file():
                self.load_error = f"模型文件不存在：{self.model_path}"
                return
            try:
                self.model = YOLO(str(self.model_path))
                self.load_error = None
            except Exception as exc:
                self.load_error = f"模型加载失败：{exc}"

    def predict(self, source: str | Path | np.ndarray, confidence: float, iou: float) -> tuple[list[dict[str, Any]], np.ndarray, float]:
        if self.model is None:
            self.load()
        if self.model is None:
            raise RuntimeError(self.load_error or "模型尚未加载")
        started = time.perf_counter()
        result = self.model.predict(source=source, conf=confidence, iou=iou, device=self.device, verbose=False)[0]
        inference_ms = (time.perf_counter() - started) * 1000
        height, width = result.orig_shape
        detections: list[dict[str, Any]] = []
        if result.boxes is not None:
            for box in result.boxes:
                class_id = int(box.cls.item())
                class_name = str(result.names[class_id])
                if class_name not in CLASS_NAME_ZH:
                    continue
                xyxy = [float(value) for value in box.xyxy[0].tolist()]
                normalized = [xyxy[0] / width, xyxy[1] / height, xyxy[2] / width, xyxy[3] / height]
                area_ratio = max(0.0, normalized[2] - normalized[0]) * max(0.0, normalized[3] - normalized[1])
                detections.append({
                    "class_id": class_id,
                    "class_name": class_name,
                    "class_name_zh": CLASS_NAME_ZH[class_name],
                    "confidence": float(box.conf.item()),
                    "bbox_xyxy": xyxy,
                    "bbox_normalized": normalized,
                    "area_ratio": area_ratio,
                    "location_text": describe_location(normalized),
                })
        return detections, result.plot(), inference_ms

    def save_annotated(self, image: np.ndarray, destination: Path) -> None:
        destination.parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(destination), image):
            raise OSError(f"Unable to write result image: {destination}")


_service = YoloService()


def get_yolo_service() -> YoloService:
    return _service
