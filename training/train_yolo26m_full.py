"""Train official YOLO26m without overwriting any YOLO26s experiment."""

from __future__ import annotations

import argparse
from pathlib import Path

from ultralytics import YOLO

from training.common import PROJECT_ROOT
from training.train_utils import require_audited_dataset


DATA_YAML = PROJECT_ROOT / "data/processed/data.yaml"
MODEL_PATH = PROJECT_ROOT / "yolo26m.pt"
OUTPUT_ROOT = PROJECT_ROOT / "models/trained/yolo26m"


def train(*, smoke: bool, epochs: int, batch: int, imgsz: int, resume: bool = False) -> Path:
    require_audited_dataset(DATA_YAML)
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(f"Official YOLO26m checkpoint is missing: {MODEL_PATH}")

    name = f"yolo26m_full_{imgsz}_smoke" if smoke else f"yolo26m_full_{imgsz}_formal"
    project = OUTPUT_ROOT / ("smoke" if smoke else "formal")
    run_dir = project / name
    last_pt = run_dir / "weights/last.pt"
    if resume and not last_pt.is_file():
        raise FileNotFoundError(f"Resume checkpoint is missing: {last_pt}")

    model = YOLO(str(last_pt if resume else MODEL_PATH))
    head = model.model.model[-1]
    if not getattr(head, "end2end", False) or getattr(head, "reg_max", None) != 1:
        raise RuntimeError("The active runtime did not construct a genuine YOLO26 end-to-end head.")
    if not hasattr(head, "one2one_cv2") or not hasattr(head, "one2one_cv3"):
        raise RuntimeError("The YOLO26m checkpoint is missing its one-to-one detection branches.")

    model.train(
        data=str(DATA_YAML),
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        device=0,
        workers=0,
        optimizer="MuSGD",
        patience=3 if smoke else 25,
        seed=42,
        deterministic=True,
        amp=True,
        pretrained=True,
        close_mosaic=10 if smoke else 15,
        project=str(project),
        name=name,
        exist_ok=True,
        plots=True,
        save=True,
        save_period=-1 if smoke else 10,
        resume=resume,
    )
    best_pt = run_dir / "weights/best.pt"
    if not best_pt.is_file():
        raise RuntimeError(f"YOLO26m training finished without best.pt: {best_pt}")
    return best_pt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--imgsz", type=int, default=768)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    best = train(smoke=args.smoke, epochs=args.epochs, batch=args.batch, imgsz=args.imgsz, resume=args.resume)
    print(f"Best checkpoint: {best}")
    if args.smoke:
        print("SMOKE ONLY: do not deploy this checkpoint.")


if __name__ == "__main__":
    main()
