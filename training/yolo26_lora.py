"""Minimal, mergeable convolutional LoRA support for official YOLO26 models.

This module intentionally does not depend on the YOLO-Master fork.  Its checkpoints must be
created and consumed with the official YOLO26 runtime used by ``.venv-yolo26``.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn


@dataclass(frozen=True)
class YOLO26LoRAConfig:
    rank: int = 8
    alpha: float = 16.0
    dropout: float = 0.05
    min_layer: int = 10
    max_layer: int = 22
    min_channels: int = 32
    exclude_attention: bool = True

    def validate(self) -> None:
        if self.rank <= 0:
            raise ValueError("LoRA rank must be positive")
        if self.alpha <= 0:
            raise ValueError("LoRA alpha must be positive")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("LoRA dropout must be in [0, 1)")
        if self.min_layer > self.max_layer:
            raise ValueError("min_layer cannot be greater than max_layer")


class LoRAConv2d(nn.Conv2d):
    """A Conv2d with a zero-initialized low-rank residual branch.

    Subclassing ``nn.Conv2d`` preserves the original ``weight``/``bias`` state-dict names, so
    official YOLO26 weights can be loaded before or after adapter injection.
    """

    def __init__(self, base: nn.Conv2d, *, rank: int, alpha: float, dropout: float) -> None:
        if base.groups != 1:
            raise ValueError("LoRAConv2d only supports groups=1")
        super().__init__(
            base.in_channels,
            base.out_channels,
            base.kernel_size,
            base.stride,
            base.padding,
            base.dilation,
            base.groups,
            base.bias is not None,
            base.padding_mode,
            device=base.weight.device,
            dtype=base.weight.dtype,
        )
        self.weight.data.copy_(base.weight.data)
        if base.bias is not None and self.bias is not None:
            self.bias.data.copy_(base.bias.data)

        self.rank = int(rank)
        self.alpha = float(alpha)
        self.scaling = self.alpha / self.rank
        self.lora_dropout = nn.Dropout2d(dropout) if dropout else nn.Identity()
        self.lora_A = nn.Conv2d(
            base.in_channels,
            self.rank,
            kernel_size=1,
            bias=False,
            device=base.weight.device,
            dtype=base.weight.dtype,
        )
        self.lora_B = nn.Conv2d(
            self.rank,
            base.out_channels,
            kernel_size=base.kernel_size,
            stride=base.stride,
            padding=base.padding,
            dilation=base.dilation,
            bias=False,
            padding_mode=base.padding_mode,
            device=base.weight.device,
            dtype=base.weight.dtype,
        )
        nn.init.kaiming_uniform_(self.lora_A.weight, a=5**0.5)
        nn.init.zeros_(self.lora_B.weight)
        self.weight.requires_grad_(False)
        if self.bias is not None:
            self.bias.requires_grad_(False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base = self._conv_forward(x, self.weight, self.bias)
        update = self.lora_B(self.lora_A(self.lora_dropout(x)))
        return base + update * self.scaling

    def merged_conv(self) -> nn.Conv2d:
        """Return an ordinary Conv2d with the adapter delta folded into its weight."""
        merged = nn.Conv2d(
            self.in_channels,
            self.out_channels,
            self.kernel_size,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
            self.bias is not None,
            self.padding_mode,
            device=self.weight.device,
            dtype=self.weight.dtype,
        )
        a = self.lora_A.weight[:, :, 0, 0]
        delta = torch.einsum("orhw,ri->oihw", self.lora_B.weight, a)
        merged.weight.data.copy_(self.weight.data + delta * self.scaling)
        if self.bias is not None and merged.bias is not None:
            merged.bias.data.copy_(self.bias.data)
        return merged


def _top_level_index(name: str) -> int | None:
    parts = name.split(".")
    if len(parts) < 2 or parts[0] != "model" or not parts[1].isdigit():
        return None
    return int(parts[1])


def _resolve_parent(root: nn.Module, name: str) -> tuple[nn.Module, str]:
    parts = name.split(".")
    parent = root
    for part in parts[:-1]:
        parent = parent[int(part)] if part.isdigit() else getattr(parent, part)
    return parent, parts[-1]


def _set_child(parent: nn.Module, key: str, child: nn.Module) -> None:
    if key.isdigit():
        parent[int(key)] = child  # type: ignore[index]
    else:
        setattr(parent, key, child)


def is_lora_target(name: str, module: nn.Module, config: YOLO26LoRAConfig) -> bool:
    if not isinstance(module, nn.Conv2d) or isinstance(module, LoRAConv2d):
        return False
    layer = _top_level_index(name)
    if layer is None or not config.min_layer <= layer <= config.max_layer:
        return False
    if module.groups != 1 or min(module.in_channels, module.out_channels) < config.min_channels:
        return False
    if config.exclude_attention and ".attn." in name:
        return False
    return True


def inject_lora(model: nn.Module, config: YOLO26LoRAConfig) -> list[str]:
    """Inject LoRA branches and return their stable module names."""
    config.validate()
    targets = [
        name
        for name, module in model.named_modules()
        if name and is_lora_target(name, module, config)
    ]
    if not targets:
        raise RuntimeError("No compatible YOLO26 LoRA target layers were found")
    for name in targets:
        parent, key = _resolve_parent(model, name)
        base = parent[int(key)] if key.isdigit() else getattr(parent, key)
        _set_child(
            parent,
            key,
            LoRAConv2d(base, rank=config.rank, alpha=config.alpha, dropout=config.dropout),
        )
    return targets


def mark_only_lora_trainable(model: nn.Module, *, include_head: bool = False) -> dict[str, int]:
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    for module in model.modules():
        if isinstance(module, LoRAConv2d):
            module.lora_A.weight.requires_grad_(True)
            module.lora_B.weight.requires_grad_(True)
    if include_head:
        for name, parameter in model.named_parameters():
            if name.startswith("model.23."):
                parameter.requires_grad_(True)
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    adapters = sum(
        parameter.numel()
        for name, parameter in model.named_parameters()
        if ".lora_A." in name or ".lora_B." in name
    )
    head = sum(
        parameter.numel()
        for name, parameter in model.named_parameters()
        if name.startswith("model.23.") and parameter.requires_grad
    )
    expected = adapters + head
    if trainable == 0 or trainable != expected:
        raise RuntimeError(
            f"Invalid LoRA freeze state: trainable={trainable}, adapters={adapters}, head={head}"
        )
    return {"total": total, "trainable": trainable, "adapter": adapters, "head": head}


def merge_lora(model: nn.Module) -> list[str]:
    names = [name for name, module in model.named_modules() if isinstance(module, LoRAConv2d)]
    for name in names:
        parent, key = _resolve_parent(model, name)
        module = parent[int(key)] if key.isdigit() else getattr(parent, key)
        _set_child(parent, key, module.merged_conv())
    return names


def adapter_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: tensor.detach().cpu()
        for name, tensor in model.state_dict().items()
        if ".lora_A." in name or ".lora_B." in name
    }


def save_adapter(
    model: nn.Module,
    path: Path,
    *,
    config: YOLO26LoRAConfig,
    base_model: Path,
    base_sha256: str,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    state = adapter_state_dict(model)
    if not state:
        raise RuntimeError("Refusing to save an empty LoRA adapter")
    payload: dict[str, Any] = {
        "format": "yolo26-conv-lora-v1",
        "config": asdict(config),
        "base_model": str(base_model.resolve()),
        "base_sha256": base_sha256,
        "state_dict": state,
    }
    torch.save(payload, path)
