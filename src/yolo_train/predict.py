"""Run prediction using a trained YOLO checkpoint."""

import argparse

from ultralytics import YOLO

from .config import resolve_project_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="Path to a trained .pt checkpoint")
    parser.add_argument("--source", required=True, help="Image, video, directory, or stream URL")
    parser.add_argument("--device", default="0")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--save", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    model = YOLO(resolve_project_path(args.model))
    model.predict(
        source=args.source,
        device=args.device,
        conf=args.conf,
        save=args.save,
        project=resolve_project_path("runs/predict"),
    )


if __name__ == "__main__":
    main()
