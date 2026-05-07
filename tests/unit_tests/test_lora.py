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
from torchtitan.protocols.model_spec import validate_converter_order


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

    trainable = {n for n, p in model.named_parameters() if p.requires_grad}
    frozen = {n for n, p in model.named_parameters() if not p.requires_grad}

    assert len(trainable) > 0, "No trainable parameters found"
    assert len(frozen) > 0, "No frozen parameters found"
    for name in trainable:
        assert (
            "lora_a" in name or "lora_b" in name
        ), f"Trainable param '{name}' is not a LoRA adapter"
    for name in frozen:
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
    dim = model_spec.model.dim
    batch_size, seq_len = 2, 16
    tokens = torch.randint(0, vocab_size, (batch_size, seq_len))
    output = model(tokens)
    assert output.shape == (batch_size, seq_len, vocab_size)


def test_validate_converter_order():
    """Quantization before LoRA is valid; LoRA before quantization is not."""
    lora = LoRAConverter(LoRAConverter.Config(rank=8, alpha=16.0))

    # Valid order: no error
    validate_converter_order([lora])

    # Invalid order: quantization after LoRA
    pytest.importorskip("torchao")
    float8 = Float8LinearConverter(Float8LinearConverter.Config(emulate=True))
    with pytest.raises(ValueError, match="must be applied before"):
        validate_converter_order([lora, float8])

    # Valid order: quantization before LoRA
    validate_converter_order([float8, lora])


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
