"""Helpers shared by full and LoRA training entry points."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from training.audit_dataset import audit
from training.common import PROJECT_ROOT, project_path, read_yaml


LORA_KEYS = ("lora_r", "lora_alpha", "lora_dropout", "lora_gradient_checkpointing")


def load_training_config(config_path: Path, overrides: dict[str, Any], include_lora: bool) -> dict[str, Any]:
    config = read_yaml(config_path)
    for key, value in overrides.items():
        if value is not None:
            config[key] = value
    if not include_lora:
        for key in LORA_KEYS:
            config.pop(key, None)
    config["data"] = str(project_path(config["data"]))
    return config


def require_audited_dataset(data_yaml: str | Path) -> None:
    report = audit(Path(data_yaml).resolve().parent)
    if not report["passed"]:
        preview = "\n".join(report["errors"][:10])
        raise RuntimeError(f"Dataset audit failed; training is blocked:\n{preview}")


def training_destination(kind: str, smoke: bool) -> tuple[Path, str]:
    if smoke:
        return PROJECT_ROOT / "models/trained/smoke", f"yolo11s_{kind}_smoke"
    return PROJECT_ROOT / "models/trained", f"yolo11s_{kind}"
