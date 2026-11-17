"""Run YOLO11s LoRA tuning using Tencent YOLO-Master's Ultralytics fork."""

from __future__ import annotations

import argparse
from pathlib import Path

from ultralytics import YOLO
from ultralytics.cfg import DEFAULT_CFG_DICT

from training.common import PROJECT_ROOT
from training.train_utils import LORA_KEYS, load_training_config, require_audited_dataset, training_destination


def require_lora_capability() -> None:
    missing = [key for key in LORA_KEYS if key not in DEFAULT_CFG_DICT]
    if missing:
        raise RuntimeError(
            "The active ultralytics package does not expose YOLO-Master LoRA options: "
            f"{missing}. Install Tencent/YOLO-Master in the active environment before LoRA training."
        )


def train_lora(config_path: Path, *, epochs: int | None = None, batch: int | None = None, device: str | None = None, smoke: bool = False) -> Path:
    require_lora_capability()
    config = load_training_config(config_path, {"epochs": epochs, "batch": batch, "device": device}, include_lora=True)
    require_audited_dataset(config["data"])
    # The PEFT Conv2d path was empirically unstable on this dataset with AMP,
    # unfrozen BN, RS-LoRA scaling, and alpha warmup. Keep the formal recipe
    # aligned with the validated notebook precheck.
    config.update(
        amp=False,
        warmup_epochs=5,
        warmup_bias_lr=0.01,
        cos_lr=True,
    )
    model_source = config.pop("model")
    project, name = training_destination("lora", smoke)
    config.update(project=str(project), name=name, exist_ok=True)
    model = YOLO(model_source)
    model.train(**config)
    best = project / name / "weights/best.pt"
    if not best.is_file():
        raise RuntimeError(f"LoRA training finished without a loadable best.pt: {best}")
    adapter_dir = project / name / "lora_adapter"
    if hasattr(model, "save_lora_only"):
        model.save_lora_only(str(adapter_dir))
    return best


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "training/configs/train.yaml")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch", type=int)
    parser.add_argument("--device")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    best = train_lora(args.config, epochs=args.epochs, batch=args.batch, device=args.device, smoke=args.smoke)
    print(f"Best checkpoint: {best}")
    if args.smoke:
        print("SMOKE ONLY: this checkpoint is not eligible for customer deployment.")


if __name__ == "__main__":
    main()
