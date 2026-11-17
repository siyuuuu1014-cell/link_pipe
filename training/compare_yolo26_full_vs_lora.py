"""Build a reproducible report for the fair YOLO26s full-vs-LoRA experiment."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
from ultralytics import YOLO

from training.common import PROJECT_ROOT


FULL_RUN = PROJECT_ROOT / "models/trained/yolo26/formal/yolo26s_full_768_formal"
FULL_METRICS = PROJECT_ROOT / "models/metrics/yolo26s_full_768_internal_metrics.json"
DEFAULT_LORA_RUN = (
    PROJECT_ROOT
    / "models/experiments/yolo26s_full_vs_lora/lora_runs/formal"
    / "yolo26s_lora_from_pretrained_formal"
)
DEFAULT_LORA_METRICS = (
    PROJECT_ROOT / "models/metrics/yolo26s_lora_from_pretrained_formal_internal_metrics.json"
)
REPORT_DIR = PROJECT_ROOT / "models/experiments/yolo26s_full_vs_lora/comparison"


def _load_rows(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(f"Training results are missing: {path}")
    with path.open(encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _training_summary(run_dir: Path) -> dict[str, float | int]:
    rows = _load_rows(run_dir / "results.csv")
    best = max(rows, key=lambda row: float(row["metrics/mAP50-95(B)"]))
    return {
        "epochs": len(rows),
        "best_epoch": int(best["epoch"]),
        "training_seconds": float(rows[-1]["time"]),
        "best_val_map50_95": float(best["metrics/mAP50-95(B)"]),
    }


def _metric_view(payload: dict) -> dict:
    return {
        "precision": float(payload["precision"]),
        "recall": float(payload["recall"]),
        "map50": float(payload["map50"]),
        "map50_95": float(payload["map50_95"]),
        "per_class_map50_95": payload.get("per_class_map50_95", {}),
        "inference_ms": float(payload.get("speed_ms", {}).get("inference", 0.0)),
    }


def build_report(
    *,
    lora_run: Path = DEFAULT_LORA_RUN,
    lora_metrics: Path = DEFAULT_LORA_METRICS,
    output_dir: Path = REPORT_DIR,
) -> Path:
    required = [FULL_METRICS, FULL_RUN / "weights/best.pt", lora_metrics, lora_run / "lora_manifest.json"]
    missing = [path for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("Formal comparison is incomplete; missing: " + ", ".join(map(str, missing)))

    full_raw = json.loads(FULL_METRICS.read_text(encoding="utf-8"))
    lora_raw = json.loads(lora_metrics.read_text(encoding="utf-8"))
    manifest = json.loads((lora_run / "lora_manifest.json").read_text(encoding="utf-8"))
    full_metrics = _metric_view(full_raw)
    lora_metrics_view = _metric_view(lora_raw)
    full_training = _training_summary(FULL_RUN)
    lora_training = _training_summary(lora_run)

    full_model = YOLO(FULL_RUN / "weights/best.pt")
    full_parameters = sum(parameter.numel() for parameter in full_model.model.parameters())
    trainable = manifest["trainable_parameters"]
    lora_trainable = int(trainable["trainable"])
    adapter_path = Path(manifest["adapter"])
    merged_path = Path(manifest["merged_best"])

    report = {
        "experiment": "yolo26s_full_vs_lora_from_same_pretrained_checkpoint",
        "fairness": {
            "base_checkpoint": str((PROJECT_ROOT / "yolo26s.pt").resolve()),
            "dataset": str((PROJECT_ROOT / "data/processed/data.yaml").resolve()),
            "split": "fixed_test",
            "imgsz": 768,
            "seed": 42,
            "note": "LoRA trains adapters plus the newly initialized three-class detection head.",
        },
        "full": {
            "model": str((FULL_RUN / "weights/best.pt").resolve()),
            "trainable_parameters": full_parameters,
            "trainable_ratio": 1.0,
            "merged_model_mb": (FULL_RUN / "weights/best.pt").stat().st_size / 1024**2,
            "metrics": full_metrics,
            "training": full_training,
        },
        "lora": {
            "model": str(merged_path.resolve()),
            "adapter": str(adapter_path.resolve()),
            "trainable_parameters": lora_trainable,
            "adapter_parameters": int(trainable["adapter"]),
            "head_parameters": int(trainable["head"]),
            "trainable_ratio": lora_trainable / full_parameters,
            "merged_model_mb": merged_path.stat().st_size / 1024**2,
            "adapter_mb": adapter_path.stat().st_size / 1024**2,
            "metrics": lora_metrics_view,
            "training": lora_training,
        },
        "delta_lora_minus_full": {
            key: lora_metrics_view[key] - full_metrics[key]
            for key in ("precision", "recall", "map50", "map50_95", "inference_ms")
        },
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "comparison.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    metric_names = ["precision", "recall", "map50", "map50_95"]
    x = range(len(metric_names))
    width = 0.36
    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.bar([index - width / 2 for index in x], [full_metrics[k] for k in metric_names], width, label="Full")
    ax.bar([index + width / 2 for index in x], [lora_metrics_view[k] for k in metric_names], width, label="LoRA")
    ax.set_xticks(list(x), ["Precision", "Recall", "mAP50", "mAP50-95"])
    ax.set_ylim(0, 1)
    ax.set_title("YOLO26s fixed-test metrics: full fine-tune vs LoRA")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "metrics_comparison.png", dpi=180)
    plt.close(fig)

    full_rows = _load_rows(FULL_RUN / "results.csv")
    lora_rows = _load_rows(lora_run / "results.csv")
    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.plot(
        [int(row["epoch"]) for row in full_rows],
        [float(row["metrics/mAP50-95(B)"]) for row in full_rows],
        label="Full fine-tune",
    )
    ax.plot(
        [int(row["epoch"]) for row in lora_rows],
        [float(row["metrics/mAP50-95(B)"]) for row in lora_rows],
        label="LoRA",
    )
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Validation mAP50-95")
    ax.set_title("Training dynamics")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "training_dynamics.png", dpi=180)
    plt.close(fig)
    return report_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lora-run", type=Path, default=DEFAULT_LORA_RUN)
    parser.add_argument("--lora-metrics", type=Path, default=DEFAULT_LORA_METRICS)
    parser.add_argument("--output-dir", type=Path, default=REPORT_DIR)
    args = parser.parse_args()
    print(build_report(lora_run=args.lora_run, lora_metrics=args.lora_metrics, output_dir=args.output_dir))


if __name__ == "__main__":
    main()
