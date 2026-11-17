"""Train and evaluate the isolated YOLO11s LoRA v3 experiment.

The v3 recipe keeps the same model, dataset, image size and seed as v2, but
uses a lower learning rate, a larger adapter rank and an augmentation schedule
that can finish before early stopping.  It never writes into the v2 run.
"""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ultralytics import YOLO

from training.audit_dataset import audit
from training.common import PROJECT_ROOT, TARGET_NAMES, write_json
from training.train_lora import require_lora_capability


MODEL_PATH = PROJECT_ROOT / "yolo11s.pt"
DATA_YAML = PROJECT_ROOT / "data/processed/data.yaml"
FORMAL_PROJECT = PROJECT_ROOT / "models/trained/notebook"
SMOKE_PROJECT = PROJECT_ROOT / "models/trained/notebook_smoke"
FORMAL_NAME = "yolo11s_lora_formal_v3"
SMOKE_NAME = "yolo11s_lora_v3_smoke"
V2_METRICS = PROJECT_ROOT / "models/metrics/notebook_lora_v2_internal_metrics.json"


def v3_recipe(*, smoke: bool = False) -> dict[str, Any]:
    """Return the reviewed v3 parameters without output-path side effects."""
    return {
        "data": str(DATA_YAML),
        "epochs": 3 if smoke else 100,
        "imgsz": 640,
        "batch": 8,
        "device": 0,
        "workers": 0,
        "optimizer": "AdamW",
        "lr0": 3e-4,
        "lrf": 0.05,
        "weight_decay": 5e-4,
        "patience": 3 if smoke else 30,
        "seed": 42,
        "deterministic": True,
        "amp": False,
        "pretrained": True,
        "cos_lr": True,
        "warmup_epochs": 3,
        "warmup_bias_lr": 0.01,
        "mosaic": 0.5,
        "close_mosaic": 1 if smoke else 40,
        "save_period": -1 if smoke else 10,
        "lora_r": 16,
        "lora_alpha": 16,
        "lora_dropout": 0.02,
        "lora_freeze_bn": True,
        "lora_use_rslora": False,
        "lora_init_lora_weights": "gaussian",
        "lora_lr_mult": 0.5,
        "lora_alpha_warmup": 3,
        "lora_layer_decay": 1.0,
        "lora_ortho_weight": 0.0,
        "lora_dropout_end": 0.02,
        "lora_dropout_start_ratio": 1.0,
        "lora_gradient_checkpointing": False,
        "lora_include_head": False,
        "plots": True,
        "save": True,
        "verbose": True,
    }


def _check_preconditions() -> None:
    require_lora_capability()
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(f"Missing pretrained checkpoint: {MODEL_PATH}")
    report = audit(DATA_YAML.parent)
    if not report["passed"]:
        raise RuntimeError("Dataset audit failed:\n" + "\n".join(report["errors"][:20]))


def _read_rows(results_csv: Path) -> list[dict[str, str]]:
    with results_csv.open(encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _evaluate(best_pt: Path, *, smoke: bool, batch: int) -> dict[str, Any]:
    run_tag = "smoke" if smoke else "formal"
    model = YOLO(str(best_pt))
    metrics = model.val(
        data=str(DATA_YAML),
        split="test",
        imgsz=640,
        batch=batch,
        workers=0,
        device=0,
        plots=True,
        project=str(PROJECT_ROOT / "models/metrics/evaluation_runs"),
        name=f"yolo11s_lora_v3_{run_tag}_internal",
        exist_ok=True,
    )
    maps = metrics.box.maps
    return {
        "source": f"yolo11s_lora_v3_{run_tag}",
        "model_path": str(best_pt),
        "dataset": str(DATA_YAML),
        "split": "test",
        "precision": float(metrics.box.mp),
        "recall": float(metrics.box.mr),
        "map50": float(metrics.box.map50),
        "map50_95": float(metrics.box.map),
        "per_class_map50_95": {
            name: float(maps[index]) for index, name in enumerate(TARGET_NAMES)
        },
        "speed_ms": {key: float(value) for key, value in metrics.speed.items()},
        "model_size_mb": best_pt.stat().st_size / (1024 * 1024),
        "created_at": datetime.now(timezone.utc).isoformat(),
    }


def run_v3(
    *,
    smoke: bool = False,
    resume: bool = False,
    export_onnx: bool = False,
    batch: int = 8,
) -> dict[str, Any]:
    """Run v3 and return paths plus fixed-test metrics."""
    _check_preconditions()
    project = SMOKE_PROJECT if smoke else FORMAL_PROJECT
    name = SMOKE_NAME if smoke else FORMAL_NAME
    run_dir = project / name
    last_pt = run_dir / "weights/last.pt"
    best_pt = run_dir / "weights/best.pt"

    if resume and not last_pt.is_file():
        raise FileNotFoundError(f"Cannot resume because last.pt is missing: {last_pt}")
    if not resume and run_dir.exists():
        raise FileExistsError(
            f"Refusing to overwrite an existing v3 run: {run_dir}. "
            "Use --resume or choose a fresh output name in the source."
        )

    recipe = v3_recipe(smoke=smoke)
    recipe["batch"] = batch
    recipe.update(project=str(project), name=name, exist_ok=True)
    model = YOLO(str(last_pt) if resume else str(MODEL_PATH))
    if resume:
        recipe["resume"] = True
    model.train(**recipe)

    if not best_pt.is_file():
        raise RuntimeError(f"Training finished without best.pt: {best_pt}")
    results_csv = run_dir / "results.csv"
    rows = _read_rows(results_csv)
    if not rows:
        raise RuntimeError(f"Training produced no metric rows: {results_csv}")

    map50_values = [float(row["metrics/mAP50(B)"]) for row in rows]
    if max(map50_values) < 0.05:
        raise RuntimeError("LoRA v3 stability check failed: mAP50 did not rise above 0.05")

    best_model = YOLO(str(best_pt))
    adapter_dir = run_dir / "lora_adapter"
    if hasattr(best_model, "save_lora_only"):
        best_model.save_lora_only(str(adapter_dir))

    metrics = _evaluate(best_pt, smoke=smoke, batch=batch)
    metrics.update(
        configured_epochs=recipe["epochs"],
        completed_epochs=len(rows),
        best_validation_map50=max(map50_values),
        recipe=v3_recipe(smoke=smoke),
    )
    suffix = "smoke" if smoke else "formal"
    metrics_path = PROJECT_ROOT / f"models/metrics/notebook_lora_v3_{suffix}_internal_metrics.json"
    write_json(metrics_path, metrics)

    exported_path = None
    if export_onnx:
        exported_path = best_model.export(format="onnx", imgsz=640, device=0)

    comparison = None
    if not smoke and V2_METRICS.is_file():
        v2 = json.loads(V2_METRICS.read_text(encoding="utf-8"))
        comparison = {
            key: metrics[key] - float(v2[key])
            for key in ("precision", "recall", "map50", "map50_95")
        }
        write_json(
            PROJECT_ROOT / "models/metrics/notebook_lora_v3_vs_v2.json",
            {"v2": v2, "v3": metrics, "v3_minus_v2": comparison},
        )

    manifest = {
        "training_line": f"notebook_yolo11s_lora_v3_{suffix}",
        "mainline_isolated": True,
        "automatically_deployed": False,
        "best_pt": str(best_pt),
        "adapter_dir": str(adapter_dir),
        "metrics_json": str(metrics_path),
        "onnx": str(exported_path) if exported_path else None,
        "v3_minus_v2": comparison,
    }
    write_json(run_dir / "notebook_model_manifest.json", manifest)
    return manifest | {"metrics": metrics}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--formal", action="store_true", help="Run the 100-epoch formal recipe")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--export-onnx", action="store_true")
    parser.add_argument("--batch", type=int, default=8)
    args = parser.parse_args()
    result = run_v3(
        smoke=not args.formal,
        resume=args.resume,
        export_onnx=args.export_onnx,
        batch=args.batch,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if not args.formal:
        print("SMOKE ONLY: this checkpoint is not eligible for deployment.")


if __name__ == "__main__":
    main()
