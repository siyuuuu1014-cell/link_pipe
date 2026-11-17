"""Evaluate labeled external data or create qualitative predictions when labels are absent."""

from __future__ import annotations

import argparse
from pathlib import Path

from ultralytics import YOLO

from training.common import PROJECT_ROOT, IMAGE_SUFFIXES, write_json


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--data", type=Path, default=PROJECT_ROOT / "data/external_test")
    args = parser.parse_args()
    model = YOLO(str(args.model.resolve()))
    yaml_path = args.data / "data.yaml"
    if yaml_path.is_file():
        metrics = model.val(data=str(yaml_path), split="test", workers=0, device=0)
        payload = {"mode": "quantitative", "precision": float(metrics.box.mp), "recall": float(metrics.box.mr), "map50": float(metrics.box.map50), "map50_95": float(metrics.box.map)}
    else:
        images = [path for path in args.data.rglob("*") if path.suffix.lower() in IMAGE_SUFFIXES]
        if not images:
            raise FileNotFoundError(f"No external images or data.yaml found under {args.data}")
        output = PROJECT_ROOT / "models/metrics/external_predictions"
        model.predict(source=[str(path) for path in images], save=True, project=str(output.parent), name=output.name, device=0)
        payload = {"mode": "qualitative", "label": "外部图片定性测试", "images": len(images), "results": str(output)}
    write_json(PROJECT_ROOT / "models/metrics/external_evaluation.json", payload)
    print(payload)


if __name__ == "__main__":
    main()
