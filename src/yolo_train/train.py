"""Train an Ultralytics YOLO model from a YAML configuration."""

import argparse
from typing import Any

from ultralytics import YOLO

from .config import load_yaml, resolve_project_path


def train(config_path: str, overrides: dict[str, Any] | None = None) -> None:
    config = load_yaml(config_path)
    config.update(overrides or {})

    model_source = config.pop("model", "yolo11n.pt")
    if "data" in config:
        config["data"] = resolve_project_path(config["data"])
    if "project" in config:
        config["project"] = resolve_project_path(config["project"])

    model = YOLO(model_source)
    model.train(**config)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/train.yaml", help="Training YAML path")
    parser.add_argument("--model", help="Override the model checkpoint or model YAML")
    parser.add_argument("--data", help="Override the dataset YAML path")
    parser.add_argument("--epochs", type=int, help="Override the epoch count")
    parser.add_argument("--batch", type=int, help="Override the batch size")
    parser.add_argument("--device", help="Override the device, e.g. 0, cpu, or 0,1")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    overrides = {
        key: value
        for key, value in vars(args).items()
        if key != "config" and value is not None
    }
    train(args.config, overrides)


if __name__ == "__main__":
    main()
