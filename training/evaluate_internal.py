"""Evaluate a trained checkpoint on the held-out internal test split."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ultralytics import YOLO

from training.common import PROJECT_ROOT, TARGET_NAMES, project_path, read_yaml, write_json


def evaluate(model_path: Path, output_path: Path, data_path: Path, run_name: str) -> dict:
    if not model_path.is_file():
        raise FileNotFoundError(f"Model checkpoint does not exist: {model_path}")
    model = YOLO(str(model_path))
    metrics = model.val(
        data=str(data_path),
        split="test",
        device=0,
        imgsz=640,
        batch=8,
        workers=0,
        plots=True,
        project=str(PROJECT_ROOT / "models/metrics/evaluation_runs"),
        name=run_name,
        exist_ok=True,
    )
    per_class_maps = getattr(metrics.box, "maps", [])
    speed = dict(getattr(metrics, "speed", {}) or {})
    audit_path = PROJECT_ROOT / "models/metrics/dataset_audit.json"
    audit = json.loads(audit_path.read_text(encoding="utf-8")) if audit_path.is_file() else {}
    payload = {
        "model_path": str(model_path),
        "dataset": str(data_path),
        "split": "test",
        "precision": float(metrics.box.mp),
        "recall": float(metrics.box.mr),
        "map50": float(metrics.box.map50),
        "map50_95": float(metrics.box.map),
        "per_class_map50_95": {
            name: float(per_class_maps[index]) if index < len(per_class_maps) else None
            for index, name in enumerate(TARGET_NAMES)
        },
        "per_class_object_count": audit.get("splits", {}).get("test", {}).get("class_objects", {}),
        "speed_ms": {key: float(value) for key, value in speed.items()},
        "model_size_mb": model_path.stat().st_size / (1024 * 1024),
        "plots_directory": str(getattr(metrics, "save_dir", "")),
    }
    write_json(output_path, payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--kind", choices=("full", "lora"), required=True)
    parser.add_argument("--data", type=Path, default=PROJECT_ROOT / "data/processed/data.yaml")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    prefix = "smoke_" if args.smoke else ""
    output = PROJECT_ROOT / f"models/metrics/{prefix}{args.kind}_internal_metrics.json"
    payload = evaluate(args.model.resolve(), output, args.data.resolve(), f"{prefix}{args.kind}_internal")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
