"""Run the fair YOLO26s full-finetune versus LoRA comparison branch.

Unlike the incremental LoRA experiment, this branch starts from the same official ``yolo26s.pt``
checkpoint as the completed full-parameter run. The new three-class detection head is trainable.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from training.common import PROJECT_ROOT
from training.train_yolo26_lora import train


OFFICIAL_BASE = PROJECT_ROOT / "yolo26s.pt"
EXPERIMENT_ROOT = PROJECT_ROOT / "models/experiments/yolo26s_full_vs_lora/lora_runs"


def run_comparison_lora(
    *,
    smoke: bool,
    epochs: int,
    imgsz: int,
    batch: int,
    run_name: str | None = None,
    resume: bool = False,
    export_onnx: bool = True,
) -> Path:
    name = run_name or (
        "yolo26s_lora_from_pretrained_smoke"
        if smoke
        else "yolo26s_lora_from_pretrained_formal"
    )
    return train(
        smoke=smoke,
        epochs=epochs,
        batch=batch,
        imgsz=imgsz,
        rank=8,
        alpha=8.0,
        dropout=0.05,
        run_name=name,
        resume=resume,
        export_onnx=export_onnx,
        base_model=OFFICIAL_BASE,
        output_root=EXPERIMENT_ROOT,
        train_head=True,
        lr0=1e-3,
        adapter_lr_mult=0.5,
        source_name=(
            "yolo26s_lora_from_pretrained_smoke"
            if smoke
            else "yolo26s_lora_from_pretrained_formal"
        ),
        patience=3 if smoke else 25,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--imgsz", type=int, default=768)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--run-name")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--no-export", action="store_true")
    args = parser.parse_args()
    best = run_comparison_lora(
        smoke=args.smoke,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        run_name=args.run_name,
        resume=args.resume,
        export_onnx=not args.no_export,
    )
    print(f"Comparison LoRA merged best: {best}")
    if args.smoke:
        print("SMOKE ONLY: this checkpoint is not part of the formal comparison.")


if __name__ == "__main__":
    main()
