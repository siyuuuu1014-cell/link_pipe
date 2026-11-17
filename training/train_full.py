"""Run a full YOLO11s fine-tuning job after mandatory dataset audit."""

from __future__ import annotations

import argparse
from pathlib import Path

from ultralytics import YOLO

from training.common import PROJECT_ROOT
from training.train_utils import load_training_config, require_audited_dataset, training_destination


def train_full(config_path: Path, *, epochs: int | None = None, batch: int | None = None, device: str | None = None, smoke: bool = False) -> Path:
    config = load_training_config(config_path, {"epochs": epochs, "batch": batch, "device": device}, include_lora=False)
    require_audited_dataset(config["data"])
    model_source = config.pop("model")
    project, name = training_destination("full", smoke)
    config.update(project=str(project), name=name, exist_ok=True)
    model = YOLO(model_source)
    model.train(**config)
    best = project / name / "weights/best.pt"
    if not best.is_file():
        raise RuntimeError(f"Training finished without best.pt: {best}")
    return best


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "training/configs/train.yaml")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch", type=int)
    parser.add_argument("--device")
    parser.add_argument("--smoke", action="store_true", help="Write weights under models/trained/smoke")
    args = parser.parse_args()
    best = train_full(args.config, epochs=args.epochs, batch=args.batch, device=args.device, smoke=args.smoke)
    print(f"Best checkpoint: {best}")
    if args.smoke:
        print("SMOKE ONLY: this checkpoint is not eligible for customer deployment.")


if __name__ == "__main__":
    main()
