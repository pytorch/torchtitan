# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
from collections import OrderedDict

import pytest
import torch
import torch.nn as nn

from torchtitan.components.lora import LoRAConverter
from torchtitan.components.quantization.float8 import Float8LinearConverter
from torchtitan.config import ConfigManager
from torchtitan.distributed import ParallelDims
from torchtitan.protocols.model_converter import ModelConvertersContainer


def build_parallel_dims(trainer_config, world_size):
    parallelism_config = trainer_config.parallelism
    parallel_dims = ParallelDims(
        dp_shard=parallelism_config.data_parallel_shard_degree,
        dp_replicate=parallelism_config.data_parallel_replicate_degree,
        cp=parallelism_config.context_parallel_degree,
        tp=parallelism_config.tensor_parallel_degree,
        pp=parallelism_config.pipeline_parallel_degree,
        ep=parallelism_config.expert_parallel_degree,
        etp=parallelism_config.expert_tensor_parallel_degree,
        world_size=world_size,
    )
    return parallel_dims


def test_build_model_converters_empty_list():
    config_manager = ConfigManager()
    config = config_manager.parse_args(
        ["--module", "llama3", "--config", "llama3_debugmodel"]
    )
    parallel_dims = build_parallel_dims(config, 1)

    model_compile_enabled = (
        config.compile.enable and "model" in config.compile.components
    )
    model_converters = config.model_converters.build(
        parallel_dims=parallel_dims,
        model_compile_enabled=model_compile_enabled,
    )
    assert isinstance(model_converters, ModelConvertersContainer)
    assert model_converters.converters == []


def test_build_model_converters_float8_converter():
    pytest.importorskip("torchao")
    config_manager = ConfigManager()
    config = config_manager.parse_args(
        ["--module", "llama3", "--config", "llama3_debugmodel"]
    )
    # Set converter config directly (not via CLI)
    config.model_converters = ModelConvertersContainer.Config(
        converters=[Float8LinearConverter.Config(emulate=True)],
    )
    parallel_dims = build_parallel_dims(config, 1)

    model_compile_enabled = (
        config.compile.enable and "model" in config.compile.components
    )
    model_converters = config.model_converters.build(
        parallel_dims=parallel_dims,
        model_compile_enabled=model_compile_enabled,
    )
    assert isinstance(model_converters, ModelConvertersContainer)
    assert len(model_converters.converters) == 1
    assert isinstance(model_converters.converters[0], Float8LinearConverter)


def test_lora_before_quantization_raises():
    """LoRA must come after quantization converters."""
    with pytest.raises(ValueError, match="LoRA converter must come after"):
        ModelConvertersContainer.Config(
            converters=[
                LoRAConverter.Config(rank=8, alpha=16.0),
                Float8LinearConverter.Config(emulate=True),
            ],
        )


def test_lora_freeze_and_training():
    """LoRA adapters on all linears: correct freeze, trainability, and training."""
    torch.manual_seed(42)
    model = nn.Sequential(
        OrderedDict(
            [
                ("fc1", nn.Linear(64, 64)),
                ("relu", nn.ReLU()),
                ("fc2", nn.Linear(64, 64)),
            ]
        )
    )
    converter = LoRAConverter(LoRAConverter.Config(rank=4, alpha=8.0))
    converter.convert(model)

    # LoRA adapters should be added to all linears
    assert hasattr(model.fc1, "lora_a")
    assert hasattr(model.fc2, "lora_a")

    # Base params frozen, LoRA params trainable
    for name, param in model.named_parameters():
        if "lora_a" in name or "lora_b" in name:
            assert param.requires_grad, f"LoRA param '{name}' should be trainable"
        else:
            assert not param.requires_grad, f"Base param '{name}' should be frozen"

    # Train for 5 steps: only LoRA params should change
    base_before = {
        name: param.data.clone()
        for name, param in model.named_parameters()
        if "lora_a" not in name and "lora_b" not in name
    }
    optimizer = torch.optim.SGD(
        [p for p in model.parameters() if p.requires_grad], lr=0.1
    )
    for _ in range(5):
        x = torch.randn(4, 64)
        loss = model(x).sum()
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    for name, param in model.named_parameters():
        if name in base_before:
            assert torch.equal(
                param.data, base_before[name]
            ), f"Base param '{name}' changed during training"


def test_lora_target_modules():
    """target_modules selectively applies LoRA in nested hierarchies."""

    class Attention(nn.Module):
        def __init__(self):
            super().__init__()
            self.wq = nn.Linear(64, 64)
            self.wk = nn.Linear(64, 64)
            self.wv = nn.Linear(64, 64)
            self.wo = nn.Linear(64, 64)

        def forward(self, x):
            return self.wo(self.wq(x) + self.wk(x) + self.wv(x))

    class Block(nn.Module):
        def __init__(self):
            super().__init__()
            self.attention = Attention()
            self.w1 = nn.Linear(64, 64)
            self.w2 = nn.Linear(64, 64)

        def forward(self, x):
            return self.w2(self.w1(self.attention(x)))

    model = Block()
    converter = LoRAConverter(
        LoRAConverter.Config(rank=4, alpha=8.0, target_modules=["wq", "wv"])
    )
    converter.convert(model)

    # Only targeted layers get LoRA, others untouched
    assert hasattr(model.attention.wq, "lora_a")
    assert hasattr(model.attention.wv, "lora_a")
    assert not hasattr(model.attention.wk, "lora_a")
    assert not hasattr(model.attention.wo, "lora_a")
    assert not hasattr(model.w1, "lora_a")
    assert not hasattr(model.w2, "lora_a")


def test_lora_key_remap_roundtrip():
    """Remap torchtitan LoRA keys to HF and back, verify roundtrip."""
    from torchtitan.components.lora import (
        remap_lora_keys_from_hf,
        remap_lora_keys_to_hf,
    )

    from_hf_map = {
        "model.layers.{}.self_attn.q_proj.weight": "layers.{}.attention.wq.weight",
        "model.layers.{}.mlp.gate_proj.weight": "layers.{}.feed_forward.w1.weight",
    }

    tt_sd = {
        "layers.0.attention.wq.lora_a.weight": torch.randn(8, 64),
        "layers.0.attention.wq.lora_b.weight": torch.randn(64, 8),
        "layers.2.feed_forward.w1.lora_a.weight": torch.randn(8, 64),
    }

    hf_sd = remap_lora_keys_to_hf(tt_sd, from_hf_map)
    assert "base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight" in hf_sd
    assert "base_model.model.model.layers.0.self_attn.q_proj.lora_B.weight" in hf_sd
    assert "base_model.model.model.layers.2.mlp.gate_proj.lora_A.weight" in hf_sd

    rt_sd = remap_lora_keys_from_hf(hf_sd, from_hf_map)
    assert set(rt_sd.keys()) == set(tt_sd.keys())
    for k in tt_sd:
        assert torch.equal(rt_sd[k], tt_sd[k])
