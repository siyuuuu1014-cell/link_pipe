"""Promote an eligible selected model and attempt optional deployment exports."""

from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

from ultralytics import YOLO

from training.common import PROJECT_ROOT, write_json


def export(source: Path, destination: Path, *, smoke: bool = False, export_onnx: bool = True, export_engine: bool = False) -> dict:
    if not source.is_file():
        raise FileNotFoundError(f"Selected model does not exist: {source}")
    destination.mkdir(parents=True, exist_ok=True)
    pt_path = destination / "best.pt"
    shutil.copy2(source, pt_path)
    exports: dict[str, dict] = {"pt": {"success": True, "path": str(pt_path)}}
    model = YOLO(str(pt_path))
    for format_name, enabled in (("onnx", export_onnx), ("engine", export_engine)):
        if not enabled:
            exports[format_name] = {"success": False, "skipped": True}
            continue
        try:
            exported = model.export(format=format_name, imgsz=640, device=0)
            exports[format_name] = {"success": True, "path": str(exported)}
        except Exception as exc:  # export dependencies vary by machine
            exports[format_name] = {"success": False, "error": str(exc)}
    payload = {
        "source": str(source),
        "deployed_pt": str(pt_path),
        "smoke_only": smoke,
        "customer_demo_eligible": not smoke,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "exports": exports,
    }
    write_json(destination / "model_info.json", payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--no-onnx", action="store_true")
    parser.add_argument("--engine", action="store_true")
    args = parser.parse_args()
    if args.model:
        source = args.model.resolve()
    else:
        selected_path = PROJECT_ROOT / "models/metrics/selected_model.json"
        if not selected_path.is_file():
            raise FileNotFoundError("No selected_model.json; train and compare both final models first.")
        selected = json.loads(selected_path.read_text(encoding="utf-8"))
        if not selected.get("deployment_eligible"):
            raise RuntimeError("Selected model is not deployment eligible.")
        source = Path(selected["selected_model_path"])
    destination = PROJECT_ROOT / ("models/deployed/smoke" if args.smoke else "models/deployed")
    payload = export(source, destination, smoke=args.smoke, export_onnx=not args.no_onnx, export_engine=args.engine)
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
