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


def test_qlora_base_weights_quantized_adapters_full_precision():
    """After first forward: base weights are NF4, LoRA adapters remain full precision."""
    torchao = pytest.importorskip("torchao")
    from torchao.dtypes.nf4tensor import NF4Tensor

    model = nn.Sequential(
        OrderedDict(
            [
                ("fc1", nn.Linear(64, 64)),
                ("relu", nn.ReLU()),
                ("fc2", nn.Linear(64, 64)),
            ]
        )
    )
    converter = LoRAConverter(
        LoRAConverter.Config(
            rank=4, alpha=8.0, quantize_base="nf4", nf4_scaler_block_size=1
        )
    )
    converter.convert(model)

    # Before first forward: base weights are regular tensors
    assert not isinstance(model.fc1.weight.data, NF4Tensor)

    # Trigger first forward to fire the quantization hook
    model(torch.randn(2, 64))

    # After first forward: base weights should be NF4, adapters stay float32
    for name in ("fc1", "fc2"):
        layer = getattr(model, name)
        assert isinstance(
            layer.weight.data, NF4Tensor
        ), f"{name}.weight should be NF4 after first forward"
        assert (
            layer.lora_a.weight.dtype == torch.float32
        ), f"{name}.lora_a.weight should be float32"
        assert (
            layer.lora_b.weight.dtype == torch.float32
        ), f"{name}.lora_b.weight should be float32"


def test_qat_preserves_weight_dtype():
    """QAT converter should not change weight dtype (fake quantization happens in forward)."""
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

    converter = QATConverter(QATConverter.Config(group_size=64))
    converter.convert(model)

    for name, param in model.named_parameters():
        assert (
            param.dtype == original_dtypes[name]
        ), f"'{name}' dtype changed from {original_dtypes[name]} to {param.dtype}"


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
    """Each QAT scheme should replace nn.Linear with the correct fake-quantized class."""
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
