"""Train a YOLO26s LoRA adapter without modifying the completed full-finetune baseline."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from dataclasses import asdict
from pathlib import Path

import torch
from torch import nn
from ultralytics import YOLO, __version__ as ultralytics_version
from ultralytics.models.yolo.detect import DetectionTrainer
from ultralytics.nn.tasks import DetectionModel
from ultralytics.utils import LOGGER, RANK
from ultralytics.utils.torch_utils import unwrap_model

from training.common import PROJECT_ROOT
from training.train_utils import require_audited_dataset
from training.yolo26_lora import (
    LoRAConv2d,
    YOLO26LoRAConfig,
    adapter_state_dict,
    inject_lora,
    mark_only_lora_trainable,
    merge_lora,
    save_adapter,
)


DATA_YAML = PROJECT_ROOT / "data/processed/data.yaml"
BASE_MODEL = (
    PROJECT_ROOT
    / "models/trained/yolo26/formal/yolo26s_full_768_formal/weights/best.pt"
)
OUTPUT_ROOT = PROJECT_ROOT / "models/trained/yolo26_lora"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class YOLO26LoRATrainer(DetectionTrainer):
    """Official detection trainer with adapter injection and strict parameter freezing."""

    def __init__(self, *args, overrides=None, **kwargs) -> None:
        custom = dict(overrides or {})
        self.lora_config = YOLO26LoRAConfig(
            rank=int(custom.pop("yolo26_lora_rank", 8)),
            alpha=float(custom.pop("yolo26_lora_alpha", 16.0)),
            dropout=float(custom.pop("yolo26_lora_dropout", 0.05)),
        )
        self.train_head = bool(custom.pop("yolo26_lora_train_head", False))
        self.adapter_lr_mult = float(custom.pop("yolo26_lora_adapter_lr_mult", 1.0))
        if not 0 < self.adapter_lr_mult <= 1:
            raise ValueError("YOLO26 LoRA adapter LR multiplier must be in (0, 1]")
        self.lora_targets: list[str] = []
        self.trainable_stats: dict[str, int] = {}
        super().__init__(*args, overrides=custom, **kwargs)

    def get_model(self, cfg=None, weights=None, verbose=True):
        model = DetectionModel(
            cfg,
            nc=self.data["nc"],
            ch=self.data["channels"],
            verbose=verbose and RANK == -1,
        )
        self.lora_targets = inject_lora(model, self.lora_config)
        if weights:
            model.load(weights)
        head = model.model[-1]
        if not getattr(head, "end2end", False) or getattr(head, "reg_max", None) != 1:
            raise RuntimeError("LoRA trainer did not construct a genuine YOLO26 end-to-end head")
        LOGGER.info(f"YOLO26 LoRA injected into {len(self.lora_targets)} convolution layers")
        return model

    def build_optimizer(self, model, name="AdamW", lr=1e-4, momentum=0.9, decay=1e-4, iterations=1e5):
        stats = mark_only_lora_trainable(unwrap_model(model), include_head=self.train_head)
        self.trainable_stats = stats
        LOGGER.info(
            "YOLO26 LoRA trainable parameters: "
            f"{stats['trainable']:,}/{stats['total']:,} ({stats['trainable'] / stats['total']:.2%}); "
            f"adapter={stats['adapter']:,}, head={stats['head']:,}"
        )
        if name.lower() != "adamw":
            raise ValueError("YOLO26 LoRA currently supports optimizer=AdamW only")
        named = list(model.named_parameters())
        adapter_parameters = [
            parameter
            for param_name, parameter in named
            if parameter.requires_grad and (".lora_A." in param_name or ".lora_B." in param_name)
        ]
        head_parameters = [
            parameter
            for param_name, parameter in named
            if parameter.requires_grad and param_name.startswith("model.23.")
        ]
        groups = [
            {
                "params": adapter_parameters,
                "lr": lr * self.adapter_lr_mult,
                "weight_decay": decay,
                "param_group": "lora_adapter",
            }
        ]
        if head_parameters:
            groups.append(
                {
                    "params": head_parameters,
                    "lr": lr,
                    "weight_decay": decay,
                    "param_group": "detection_head",
                }
            )
        return torch.optim.AdamW(groups, lr=lr, betas=(momentum, 0.999), weight_decay=decay)

    def _model_train(self):
        super()._model_train()
        for name, module in unwrap_model(self.model).named_modules():
            if isinstance(module, nn.BatchNorm2d) and not (
                self.train_head and name.startswith("model.23.")
            ):
                module.eval()


def _assert_clean_run(run_dir: Path, *, resume: bool) -> None:
    last_pt = run_dir / "weights/last.pt"
    if resume:
        if not last_pt.is_file():
            raise FileNotFoundError(f"Resume checkpoint is missing: {last_pt}")
    elif run_dir.exists() and any(run_dir.iterdir()):
        raise FileExistsError(
            f"Refusing to overwrite an existing YOLO26 LoRA run: {run_dir}. "
            "Use --resume or choose a new --run-name."
        )


def _verify_merged(lora_model: nn.Module, merged_model: nn.Module) -> float:
    lora_model.eval()
    merged_model.eval()
    device = next(lora_model.parameters()).device
    merged_model.to(device)
    sample = torch.randn(1, 3, 128, 128, device=device)
    with torch.inference_mode():
        left = lora_model(sample)
        right = merged_model(sample)

    def tensors(value):
        if torch.is_tensor(value):
            return [value]
        if isinstance(value, (list, tuple)):
            result = []
            for item in value:
                result.extend(tensors(item))
            return result
        if isinstance(value, dict):
            result = []
            for item in value.values():
                result.extend(tensors(item))
            return result
        return []

    pairs = list(zip(tensors(left), tensors(right), strict=True))
    maximum = max((a.float() - b.float()).abs().max().item() for a, b in pairs)
    if maximum > 2e-4:
        raise RuntimeError(f"Merged LoRA output mismatch is too large: {maximum}")
    return maximum


def train(
    *,
    smoke: bool,
    epochs: int,
    batch: int,
    imgsz: int,
    rank: int,
    alpha: float,
    dropout: float,
    run_name: str | None = None,
    resume: bool = False,
    export_onnx: bool = True,
    base_model: Path | None = None,
    output_root: Path | None = None,
    train_head: bool = False,
    lr0: float = 1e-4,
    adapter_lr_mult: float = 1.0,
    source_name: str | None = None,
    patience: int | None = None,
) -> Path:
    require_audited_dataset(DATA_YAML)
    selected_base = (base_model or BASE_MODEL).resolve()
    selected_output = (output_root or OUTPUT_ROOT).resolve()
    if not selected_base.is_file():
        raise FileNotFoundError(f"YOLO26s base checkpoint is missing: {selected_base}")
    if tuple(map(int, ultralytics_version.split(".")[:2])) < (8, 4):
        raise RuntimeError(f"Official YOLO26 runtime is required, found Ultralytics {ultralytics_version}")

    config = YOLO26LoRAConfig(rank=rank, alpha=alpha, dropout=dropout)
    config.validate()
    base_hash_before = sha256(selected_base)
    mode = "smoke" if smoke else "formal"
    name = run_name or ("yolo26s_lora_smoke" if smoke else "yolo26s_lora_formal")
    project = selected_output / mode
    run_dir = project / name
    _assert_clean_run(run_dir, resume=resume)
    last_pt = run_dir / "weights/last.pt"

    overrides = {
        "model": str(last_pt if resume else selected_base),
        "data": str(DATA_YAML),
        "epochs": epochs,
        "imgsz": imgsz,
        "batch": batch,
        "device": 0,
        "workers": 0,
        "optimizer": "AdamW",
        "lr0": lr0,
        "lrf": 0.05,
        "weight_decay": 1e-4,
        "warmup_epochs": 1.0,
        "patience": patience if patience is not None else (3 if smoke else 20),
        "seed": 42,
        "deterministic": True,
        # Keep the formal LoRA line deterministic and independent of Ultralytics' auxiliary
        # yolo26n.pt AMP probe. The local probe checkpoint may be absent/corrupt even when the
        # protected YOLO26s baseline and CUDA runtime are healthy.
        "amp": False,
        "pretrained": True,
        "close_mosaic": 3 if smoke else 10,
        "project": str(project),
        "name": name,
        "exist_ok": resume,
        "plots": True,
        "save": True,
        "save_period": -1 if smoke else 10,
        "resume": str(last_pt) if resume else False,
        "freeze": list(range(23)),
        "yolo26_lora_rank": rank,
        "yolo26_lora_alpha": alpha,
        "yolo26_lora_dropout": dropout,
        "yolo26_lora_train_head": train_head,
        "yolo26_lora_adapter_lr_mult": adapter_lr_mult,
    }
    trainer = YOLO26LoRATrainer(overrides=overrides)
    trainer.train()

    raw_best = run_dir / "weights/best.pt"
    if not raw_best.is_file():
        raise RuntimeError(f"YOLO26 LoRA training finished without best.pt: {raw_best}")
    lora_best = run_dir / "weights/lora_best.pt"
    shutil.copy2(raw_best, lora_best)

    wrapped = YOLO(str(lora_best))
    state = adapter_state_dict(wrapped.model)
    if not state or not any(tensor.abs().max().item() > 0 for name, tensor in state.items() if ".lora_B." in name):
        raise RuntimeError("LoRA training produced no non-zero B adapter weights")
    adapter_path = run_dir / "adapters/adapter.pt"
    save_adapter(
        wrapped.model,
        adapter_path,
        config=config,
        base_model=selected_base,
        base_sha256=base_hash_before,
    )

    merged_model = YOLO(str(lora_best))
    merged_names = merge_lora(merged_model.model)
    if any(isinstance(module, LoRAConv2d) for module in merged_model.model.modules()):
        raise RuntimeError("Some LoRA layers remained after merge")
    max_merge_error = _verify_merged(wrapped.model, merged_model.model)
    merged_model.save(raw_best)

    reloaded = YOLO(str(raw_best))
    head = reloaded.model.model[-1]
    if not getattr(head, "end2end", False) or getattr(head, "reg_max", None) != 1:
        raise RuntimeError("Merged checkpoint lost the YOLO26 end-to-end detection head")
    test_result = reloaded.val(
        data=str(DATA_YAML),
        split="test",
        imgsz=imgsz,
        batch=batch,
        workers=0,
        device=0,
        plots=True,
        project=str(PROJECT_ROOT / "models/metrics/evaluation_runs"),
        name=f"{name}_fixed_test",
        exist_ok=True,
    )
    fixed_test = {
        "precision": float(test_result.box.mp),
        "recall": float(test_result.box.mr),
        "map50": float(test_result.box.map50),
        "map50_95": float(test_result.box.map),
        "per_class_map50_95": {
            str(test_result.names[index]): float(value)
            for index, value in enumerate(test_result.box.maps)
        },
        "speed_ms": {key: float(value) for key, value in test_result.speed.items()},
    }
    if export_onnx:
        reloaded.export(format="onnx", imgsz=imgsz, device=0)

    base_hash_after = sha256(selected_base)
    if base_hash_after != base_hash_before:
        raise RuntimeError("The protected YOLO26s baseline changed during LoRA training")
    manifest = {
        "format": "yolo26-conv-lora-v1",
        "smoke_only": smoke,
        "base_model": str(selected_base),
        "base_sha256": base_hash_before,
        "merged_best": str(raw_best.resolve()),
        "lora_best": str(lora_best.resolve()),
        "adapter": str(adapter_path.resolve()),
        "config": asdict(config),
        "train_head": train_head,
        "trainable_parameters": trainer.trainable_stats,
        "lr0": lr0,
        "adapter_lr_mult": adapter_lr_mult,
        "target_count": len(merged_names),
        "merge_max_abs_error": max_merge_error,
        "fixed_test": fixed_test,
        "ultralytics": ultralytics_version,
    }
    (run_dir / "lora_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    metrics_path = PROJECT_ROOT / "models/metrics" / f"{name}_internal_metrics.json"
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(
        json.dumps(
            {
                "source": source_name
                or ("yolo26s_lora_smoke" if smoke else "yolo26s_lora_formal"),
                "smoke_only": smoke,
                "model_path": str(raw_best.resolve()),
                "dataset": str(DATA_YAML.resolve()),
                "split": "test",
                **fixed_test,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return raw_best


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--imgsz", type=int, default=768)
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--alpha", type=float, default=16.0)
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--run-name")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--no-export", action="store_true")
    parser.add_argument("--base-model", type=Path)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--train-head", action="store_true")
    parser.add_argument("--lr0", type=float, default=1e-4)
    parser.add_argument("--adapter-lr-mult", type=float, default=1.0)
    parser.add_argument("--patience", type=int)
    args = parser.parse_args()
    best = train(
        smoke=args.smoke,
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        rank=args.rank,
        alpha=args.alpha,
        dropout=args.dropout,
        run_name=args.run_name,
        resume=args.resume,
        export_onnx=not args.no_export,
        base_model=args.base_model,
        output_root=args.output_root,
        train_head=args.train_head,
        lr0=args.lr0,
        adapter_lr_mult=args.adapter_lr_mult,
        patience=args.patience,
    )
    print(f"Merged best checkpoint: {best}")
    if args.smoke:
        print("SMOKE ONLY: do not deploy this checkpoint.")


if __name__ == "__main__":
    main()
