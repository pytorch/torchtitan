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
from torchtitan.components.quantization.qat import QATConverter
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


def test_lora_freeze_and_trainability():
    """After convert: base params frozen, LoRA adapters present and trainable."""
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
    assert hasattr(model.fc1, "lora_b")
    assert hasattr(model.fc2, "lora_a")
    assert hasattr(model.fc2, "lora_b")

    # Check every parameter
    lora_param_names = []
    base_param_names = []
    for name, param in model.named_parameters():
        if "lora_a" in name or "lora_b" in name:
            lora_param_names.append(name)
            assert param.requires_grad, f"LoRA param '{name}' should be trainable"
        else:
            base_param_names.append(name)
            assert not param.requires_grad, f"Base param '{name}' should be frozen"

    assert len(lora_param_names) > 0, "No LoRA params found"
    assert len(base_param_names) > 0, "No base params found"


def test_lora_trains_base_frozen():
    """Train for several steps: LoRA params should change, base params should not."""
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

    # Snapshot all params before training
    base_before = {
        name: param.data.clone()
        for name, param in model.named_parameters()
        if "lora_a" not in name and "lora_b" not in name
    }
    lora_before = {
        name: param.data.clone()
        for name, param in model.named_parameters()
        if "lora_a" in name or "lora_b" in name
    }

    # Only LoRA params go to optimizer
    optimizer = torch.optim.SGD(
        [p for p in model.parameters() if p.requires_grad], lr=0.1
    )

    # Train for 5 steps
    for _ in range(5):
        x = torch.randn(4, 64)
        loss = model(x).sum()
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    # Base params must not change
    for name, param in model.named_parameters():
        if name in base_before:
            assert torch.equal(
                param.data, base_before[name]
            ), f"Base param '{name}' changed during training"

    # At least some LoRA params must change
    any_lora_changed = any(
        not torch.equal(param.data, lora_before[name])
        for name, param in model.named_parameters()
        if name in lora_before
    )
    assert any_lora_changed, "No LoRA param changed after 5 training steps"


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


@pytest.mark.parametrize(
    "scheme, group_size, expected_linear_cls",
    [
        ("int4_weight_only", 64, "FakeQuantizedLinear"),
        ("intx_weight_only", 64, "FakeQuantizedLinear"),
        ("int8_dynamic_act_intx_weight", 64, "FakeQuantizedLinear"),
        ("float8_dynamic_act_float8_weight", None, "FakeQuantizedLinear"),
        ("float8_dynamic_act_int4_weight", None, "FakeQuantizedLinear"),
        ("nvfp4", None, "NVFP4FakeQuantizedLinear"),
        ("mx", None, "MXFakeQuantizedLinear"),
    ],
)
def test_qat_all_schemes(scheme, group_size, expected_linear_cls):
    """Each QAT scheme should replace nn.Linear with the correct fake-quantized
    class and preserve weight dtype (fake quantization happens in forward)."""
    pytest.importorskip("torchao")

    model = nn.Sequential(
        OrderedDict(
            [
                ("fc1", nn.Linear(64, 64)),
                ("relu", nn.ReLU()),
                ("fc2", nn.Linear(64, 64)),
            ]
        )
    )
    original_dtypes = {name: param.dtype for name, param in model.named_parameters()}

    config_kwargs = {"scheme": scheme}
    if group_size is not None:
        config_kwargs["group_size"] = group_size
    converter = QATConverter(QATConverter.Config(**config_kwargs))
    converter.convert(model)

    # Linear layers should be replaced with the expected class
    assert (
        type(model.fc1).__name__ == expected_linear_cls
    ), f"scheme={scheme}: expected {expected_linear_cls}, got {type(model.fc1).__name__}"
    assert (
        type(model.fc2).__name__ == expected_linear_cls
    ), f"scheme={scheme}: expected {expected_linear_cls}, got {type(model.fc2).__name__}"

    # Weight dtype should be preserved
    for name, param in model.named_parameters():
        assert (
            param.dtype == original_dtypes[name]
        ), f"'{name}' dtype changed from {original_dtypes[name]} to {param.dtype}"


def test_qat_unknown_scheme_raises():
    """QATConverter should raise ValueError for unknown schemes."""
    with pytest.raises(ValueError, match="Unknown QAT scheme"):
        QATConverter(QATConverter.Config(scheme="not_a_real_scheme"))


def test_qat_group_size_warning_for_unsupported_scheme(caplog):
    """QATConverter should warn when group_size is set for a scheme that ignores it."""
    pytest.importorskip("torchao")
    import logging

    with caplog.at_level(logging.WARNING):
        QATConverter(
            QATConverter.Config(
                scheme="float8_dynamic_act_float8_weight", group_size=64
            )
        )
    assert "does not use group_size" in caplog.text


def test_qat_lora_adapter_qat():
    """QAT + LoRA: base and adapter weights are both fake-quantized.
    Also tests that group_size > rank is clamped to rank."""
    pytest.importorskip("torchao")
    from torchao.quantization.qat.linear import FakeQuantizedLinear

    # --- group_size > rank: clamped to rank so adapters still work ---
    model = nn.Sequential(
        OrderedDict(
            [
                ("fc1", nn.Linear(128, 128)),
                ("relu", nn.ReLU()),
                ("fc2", nn.Linear(128, 128)),
            ]
        )
    )
    qat_converter = QATConverter(
        QATConverter.Config(scheme="intx_weight_only", group_size=128)
    )
    qat_converter.convert(model)
    lora_converter = LoRAConverter(LoRAConverter.Config(rank=8, alpha=16.0))
    lora_converter.convert(model)
    # Adapter linears should be FakeQuantizedLinear with clamped group_size
    assert isinstance(model.fc1.lora_a, FakeQuantizedLinear)
    assert isinstance(model.fc1.lora_b, FakeQuantizedLinear)
    # Forward pass should succeed
    x = torch.randn(4, 128)
    out = model(x)
    assert out.shape == (4, 128)

    # --- Compatible group_size should work ---
    model = nn.Sequential(
        OrderedDict(
            [
                ("fc1", nn.Linear(64, 64)),
                ("relu", nn.ReLU()),
                ("fc2", nn.Linear(64, 64)),
            ]
        )
    )
    qat_converter = QATConverter(
        QATConverter.Config(scheme="intx_weight_only", group_size=8)
    )
    qat_converter.convert(model)

    assert isinstance(model.fc1, FakeQuantizedLinear)

    lora_converter = LoRAConverter(LoRAConverter.Config(rank=8, alpha=16.0))
    lora_converter.convert(model)

    # Base linears are LoRA-wrapped FakeQuantizedLinear
    assert isinstance(model.fc1, FakeQuantizedLinear)
    # Adapter linears are also FakeQuantizedLinear
    assert isinstance(model.fc1.lora_a, FakeQuantizedLinear)
    assert isinstance(model.fc1.lora_b, FakeQuantizedLinear)

    # Forward pass should succeed
    x = torch.randn(4, 64)
    out = model(x)
    assert out.shape == (4, 64)
