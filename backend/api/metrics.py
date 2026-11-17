from __future__ import annotations

import json
from pathlib import Path

from fastapi import APIRouter

from backend.config import PROJECT_ROOT


router = APIRouter(tags=["metrics"])


def read_optional(path: Path) -> dict | None:
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None


@router.get("/metrics")
def metrics() -> dict:
    base = PROJECT_ROOT / "models/metrics"
    full = read_optional(base / "full_internal_metrics.json")
    lora = read_optional(base / "lora_internal_metrics.json")
    selected = read_optional(base / "selected_model.json")
    smoke = read_optional(base / "smoke_model_comparison.json")
    return {
        "evaluation_scope": "内部测试集",
        "evaluation_complete": full is not None and lora is not None and selected is not None,
        "message": None if full or lora else "尚未完成模型评估",
        "full": full,
        "lora": lora,
        "selected": selected,
        "smoke_validation": smoke,
    }
