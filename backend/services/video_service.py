"""Synchronous MP4 inference used by the POC video endpoint."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from uuid import uuid4

import cv2
from fastapi import HTTPException, UploadFile

from backend.config import PROJECT_ROOT, get_settings
from backend.services.yolo_service import YoloService


async def save_valid_video(upload: UploadFile) -> Path:
    settings = get_settings()
    if not (upload.filename or "").lower().endswith(".mp4"):
        raise HTTPException(status_code=415, detail="仅支持 MP4 视频")
    content = await upload.read()
    if not content:
        raise HTTPException(status_code=400, detail="上传文件为空")
    if len(content) > settings.max_video_size_mb * 1024 * 1024:
        raise HTTPException(status_code=413, detail=f"视频不能超过 {settings.max_video_size_mb} MB")
    path = PROJECT_ROOT / "storage/videos" / f"{uuid4().hex}.mp4"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    capture = cv2.VideoCapture(str(path))
    valid = capture.isOpened() and int(capture.get(cv2.CAP_PROP_FRAME_COUNT)) > 0
    capture.release()
    if not valid:
        path.unlink(missing_ok=True)
        raise HTTPException(status_code=415, detail="视频损坏或无法解码")
    return path


def process_video(source: Path, service: YoloService, confidence: float, iou: float, frame_interval: int = 1) -> dict:
    capture = cv2.VideoCapture(str(source))
    fps = capture.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    result_path = PROJECT_ROOT / "storage/videos" / f"{source.stem}_result.mp4"
    writer = cv2.VideoWriter(str(result_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    if not writer.isOpened():
        capture.release()
        raise RuntimeError("无法创建结果视频")
    counts: Counter[str] = Counter()
    leak_seconds: list[float] = []
    frame_index = 0
    inference_ms = 0.0
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            annotated = frame
            if frame_index % max(frame_interval, 1) == 0:
                detections, annotated, elapsed = service.predict(frame, confidence, iou)
                inference_ms += elapsed
                counts.update(item["class_name"] for item in detections)
                if any(item["class_name"] == "leak" for item in detections):
                    leak_seconds.append(frame_index / fps)
            writer.write(annotated)
            frame_index += 1
    finally:
        capture.release()
        writer.release()

    ranges: list[list[float]] = []
    step = max(frame_interval, 1) / fps
    for timestamp in leak_seconds:
        if not ranges or timestamp - ranges[-1][1] > step * 1.5:
            ranges.append([timestamp, timestamp + step])
        else:
            ranges[-1][1] = timestamp + step
    return {
        "result_path": result_path,
        "counts": {name: counts[name] for name in ("crack", "water", "leak")},
        "leak_time_ranges": [[round(start, 2), round(end, 2)] for start, end in ranges],
        "duration": total_frames / fps,
        "frame_count": total_frames,
        "inference_ms": inference_ms,
    }
