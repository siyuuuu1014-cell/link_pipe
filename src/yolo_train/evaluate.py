"""Evaluate a trained YOLO checkpoint."""

import argparse

from ultralytics import YOLO

from .config import resolve_project_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="Path to a trained .pt checkpoint")
    parser.add_argument("--data", default="datasets/data.yaml", help="Dataset YAML path")
    parser.add_argument("--split", default="val", choices=("val", "test"))
    parser.add_argument("--device", default="0")
    parser.add_argument("--imgsz", type=int, default=640)
    args = parser.parse_args()

    model = YOLO(resolve_project_path(args.model))
    model.val(
        data=resolve_project_path(args.data),
        split=args.split,
        device=args.device,
        imgsz=args.imgsz,
    )


if __name__ == "__main__":
    main()
