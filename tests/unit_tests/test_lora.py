# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import pytest
import torch

from torchtitan.components.lora import _get_lora_cls, LoRAConverter
from torchtitan.components.quantization import Float8LinearConverter
from torchtitan.models.common.linear import Linear
from torchtitan.models.llama3 import model_registry
from torchtitan.models.utils import validate_converter_order


def test_lora_model_builds():
    """LoRA debug model builds, has trainable adapters and frozen base."""
    model_spec = model_registry(
        "debugmodel",
        converters=[
            LoRAConverter.Config(
                rank=8, alpha=16.0, target_modules=["wq", "wkv", "wo"]
            ),
        ],
    )
    model = model_spec.model.build()
    model.init_states()

    lora_params = {
        n for n, p in model.named_parameters() if "lora_a" in n or "lora_b" in n
    }
    frozen_linears = {n for n, p in model.named_parameters() if not p.requires_grad}

    assert len(lora_params) > 0, "No LoRA parameters found"
    assert len(frozen_linears) > 0, "No frozen parameters found"
    for name in lora_params:
        assert model.get_parameter(
            name
        ).requires_grad, f"LoRA param '{name}' should be trainable"
    for name in frozen_linears:
        assert (
            "lora_a" not in name and "lora_b" not in name
        ), f"Frozen param '{name}' looks like a LoRA adapter"


def test_lora_forward():
    """LoRA model forward produces correct output shape."""
    model_spec = model_registry(
        "debugmodel",
        converters=[
            LoRAConverter.Config(
                rank=8, alpha=16.0, target_modules=["wq", "wkv", "wo"]
            ),
        ],
    )
    model = model_spec.model.build()
    model.init_states()

    vocab_size = model_spec.model.vocab_size
    batch_size, seq_len = 2, 16
    tokens = torch.randint(0, vocab_size, (batch_size, seq_len))
    output = model(tokens)
    assert output.shape == (batch_size, seq_len, vocab_size)


def test_validate_converter_order():
    """Quantization before LoRA is valid; LoRA before quantization is not."""
    lora_cfg = LoRAConverter.Config(rank=8, alpha=16.0)

    # Valid order: no error
    validate_converter_order([lora_cfg])

    # Invalid order: quantization after LoRA
    float8_cfg = Float8LinearConverter.Config(emulate=True)
    with pytest.raises(ValueError, match="must be applied before"):
        validate_converter_order([lora_cfg, float8_cfg])

    # Valid order: quantization before LoRA
    validate_converter_order([float8_cfg, lora_cfg])


def test_lora_cls_cache():
    """Dynamic LoRA class creation is cached per parent class."""
    cls1 = _get_lora_cls(Linear)
    cls2 = _get_lora_cls(Linear)
    assert cls1 is cls2
    assert cls1.__name__ == "LoRALinear"
    assert issubclass(cls1, Linear)


def test_lora_rank_validation():
    """LoRA rank must be positive."""
    with pytest.raises(ValueError, match="rank must be positive"):
        LoRAConverter(LoRAConverter.Config(rank=0))
    with pytest.raises(ValueError, match="rank must be positive"):
        LoRAConverter(LoRAConverter.Config(rank=-1))
