"""Run a short, isolated stability check for the formal LoRA notebook recipe."""

from __future__ import annotations

import csv
from pathlib import Path

from ultralytics import YOLO

from training.common import PROJECT_ROOT
from training.train_lora import require_lora_capability
from training.train_utils import require_audited_dataset


DATA_YAML = PROJECT_ROOT / "data/processed/data.yaml"
OUTPUT_PROJECT = PROJECT_ROOT / "models/trained/notebook_precheck"
RUN_NAME = "yolo11s_lora_conservative_precheck"


def main() -> None:
    require_lora_capability()
    require_audited_dataset(DATA_YAML)
    model = YOLO("yolo11s.pt")
    model.train(
        data=str(DATA_YAML),
        epochs=3,
        imgsz=640,
        batch=8,
        device=0,
        workers=0,
        optimizer="AdamW",
        lr0=0.001,
        lrf=0.01,
        weight_decay=0.0005,
        patience=3,
        seed=42,
        deterministic=True,
        amp=False,
        pretrained=True,
        cos_lr=True,
        warmup_epochs=5,
        warmup_bias_lr=0.01,
        lora_r=8,
        lora_alpha=8,
        lora_dropout=0.05,
        lora_freeze_bn=True,
        lora_use_rslora=False,
        lora_init_lora_weights="gaussian",
        lora_lr_mult=0.5,
        lora_alpha_warmup=0,
        lora_layer_decay=1.0,
        lora_ortho_weight=0.0,
        lora_dropout_end=0.05,
        lora_dropout_start_ratio=1.0,
        lora_gradient_checkpointing=False,
        lora_include_head=False,
        project=str(OUTPUT_PROJECT),
        name=RUN_NAME,
        exist_ok=True,
        plots=True,
        save=True,
    )

    results_csv = OUTPUT_PROJECT / RUN_NAME / "results.csv"
    with results_csv.open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    map50 = [float(row["metrics/mAP50(B)"]) for row in rows]
    print(f"mAP50 by epoch: {map50}")
    if not map50 or max(map50) < 0.05 or map50[-1] == 0.0:
        raise RuntimeError("Stable LoRA precheck failed: validation mAP collapsed or remained too low.")
    print(f"Stable LoRA precheck passed: best mAP50={max(map50):.5f}, final mAP50={map50[-1]:.5f}")


if __name__ == "__main__":
    main()
