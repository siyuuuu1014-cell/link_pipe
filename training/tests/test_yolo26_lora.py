from __future__ import annotations

import torch
from torch import nn

from training.yolo26_lora import (
    LoRAConv2d,
    YOLO26LoRAConfig,
    adapter_state_dict,
    inject_lora,
    mark_only_lora_trainable,
    merge_lora,
)


class TinyYOLO26(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.model = nn.ModuleList([nn.Identity() for _ in range(24)])
        self.model[10] = nn.Sequential(nn.Conv2d(32, 64, 3, padding=1, bias=False))
        self.model[11] = nn.Sequential(nn.Conv2d(64, 64, 3, padding=1, groups=64))
        self.model[23] = nn.Sequential(nn.Conv2d(64, 3, 1))

    def forward(self, x):
        return self.model[10](x)


def test_injection_starts_equivalent_and_excludes_head_and_grouped_conv() -> None:
    torch.manual_seed(0)
    model = TinyYOLO26().eval()
    sample = torch.randn(2, 32, 12, 12)
    expected = model(sample)
    targets = inject_lora(model, YOLO26LoRAConfig(rank=4, alpha=8, dropout=0))
    actual = model(sample)
    assert targets == ["model.10.0"]
    assert torch.equal(expected, actual)


def test_trainable_state_and_merge_equivalence() -> None:
    torch.manual_seed(1)
    model = TinyYOLO26().eval()
    inject_lora(model, YOLO26LoRAConfig(rank=4, alpha=8, dropout=0))
    layer = model.model[10][0]
    assert isinstance(layer, LoRAConv2d)
    layer.lora_B.weight.data.normal_(std=0.01)
    stats = mark_only_lora_trainable(model)
    assert stats["trainable"] == stats["adapter"] > 0
    sample = torch.randn(2, 32, 12, 12)
    expected = model(sample)
    assert adapter_state_dict(model)
    assert merge_lora(model) == ["model.10.0"]
    assert not any(isinstance(module, LoRAConv2d) for module in model.modules())
    actual = model(sample)
    assert torch.allclose(expected, actual, atol=1e-5, rtol=1e-5)


def test_optional_detection_head_is_trainable_for_fair_pretrained_comparison() -> None:
    model = TinyYOLO26()
    inject_lora(model, YOLO26LoRAConfig(rank=4, alpha=4, dropout=0))
    stats = mark_only_lora_trainable(model, include_head=True)
    assert stats["adapter"] > 0
    assert stats["head"] > 0
    assert stats["trainable"] == stats["adapter"] + stats["head"]
    assert model.model[23][0].weight.requires_grad
