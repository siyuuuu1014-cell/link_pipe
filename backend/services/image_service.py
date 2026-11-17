"""Image validation, storage, and location description helpers."""

from __future__ import annotations

from pathlib import Path
from uuid import uuid4

import cv2
import numpy as np
from fastapi import HTTPException, UploadFile

from backend.config import PROJECT_ROOT, get_settings


def detect_suffix(content: bytes) -> str | None:
    if content.startswith(b"\xff\xd8\xff"):
        return ".jpg"
    if content.startswith(b"\x89PNG\r\n\x1a\n"):
        return ".png"
    if len(content) >= 12 and content[:4] == b"RIFF" and content[8:12] == b"WEBP":
        return ".webp"
    return None


async def save_valid_image(upload: UploadFile) -> tuple[Path, bytes]:
    settings = get_settings()
    content = await upload.read()
    if not content:
        raise HTTPException(status_code=400, detail="上传文件为空")
    if len(content) > settings.max_image_size_mb * 1024 * 1024:
        raise HTTPException(status_code=413, detail=f"图片不能超过 {settings.max_image_size_mb} MB")
    suffix = detect_suffix(content)
    decoded = cv2.imdecode(np.frombuffer(content, dtype=np.uint8), cv2.IMREAD_COLOR) if suffix else None
    if suffix is None or decoded is None:
        raise HTTPException(status_code=415, detail="仅支持 JPG、JPEG、PNG、WEBP 图片")
    destination = PROJECT_ROOT / "storage/uploads" / f"{uuid4().hex}{suffix}"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(content)
    return destination, content


def describe_location(bbox_normalized: list[float]) -> str:
    x1, y1, x2, y2 = bbox_normalized
    x = (x1 + x2) / 2
    y = (y1 + y2) / 2
    columns = ("左", "中央", "右")
    rows = ("上", "中央", "下")
    column = min(int(x * 3), 2)
    row = min(int(y * 3), 2)
    labels = {
        (0, 0): "左上",
        (0, 1): "上部中央",
        (0, 2): "右上",
        (1, 0): "左侧中央",
        (1, 1): "图像中央",
        (1, 2): "右侧中央",
        (2, 0): "左下",
        (2, 1): "下部中央",
        (2, 2): "右下",
    }
    return labels[(row, column)]
