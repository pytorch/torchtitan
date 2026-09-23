# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
import inspect
from dataclasses import dataclass

import pytest
import spmd_types as spmd
import torch
import torch.distributed.checkpoint as dcp
import torchtitan.config.transform.quantization as quantization_transform
from spmd_types import SpmdType

from torchtitan.components.data import (
    FirstFitPackingConfig,
    GrainDataLoader,
    SingleDatasetConfig,
)
from torchtitan.components.data.sources import HuggingFaceRandomAccessSource
from torchtitan.config import ConfigManager
from torchtitan.config.transform import (
    Float8LinearConverter,
    MXFP8LinearConverter,
    NVFP4LinearConverter,
)
from torchtitan.models.common.activation import Sigmoid
from torchtitan.models.common.attention import QKVLinear
from torchtitan.models.common.config_utils import make_router_config
from torchtitan.models.common.decoder_sharding import (
    colwise_config,
    dense_sequence_parallel_placement,
    rowwise_config,
)
from torchtitan.models.common.feed_forward import FeedForward
from torchtitan.models.common.linear import (
    CastLinear,
    ColumnParallelLinear,
    Linear,
    RouterGateLinear,
    RowParallelLinear,
)
from torchtitan.models.common.moe import GroupedExperts
from torchtitan.models.common.vision_encoder import InvariantRowParallelLinear
from torchtitan.models.gpt_oss.moe import GptOssGroupedExperts
from torchtitan.quantization import Float8Linear, MXFP8Linear, NVFP4Linear
from torchtitan.quantization.float8 import _get_float8_grouped_experts_cls
from torchtitan.quantization.mxfp8.experts import _get_mxfp8_grouped_experts_cls
from torchtitan.quantization.utils import get_quantized_linear, has_quantization


class _ScaledLinear(Linear):
    @dataclass(kw_only=True, slots=True)
    class Config(Linear.Config):
        scale: float = 2.0

    def __init__(self, config: Config):
        super().__init__(config)
        self.scale = config.scale

    def _linear(
        self,
        input: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor | None,
    ) -> torch.Tensor:
        return self.scale * super()._linear(input, weight, bias)


def test_no_float8_by_default():
    config_manager = ConfigManager()
    config = config_manager.parse_args(
        ["--module", "llama3", "--config", "llama3_debugmodel"]
    )
    model_config = config.model
    assert not has_quantization(model_config)
    # All Linear.Config instances should remain Linear.Config
    if Float8Linear is not None:
        for _fqn, lc, _parent, _attr in model_config.traverse(Linear.Config):
            assert not isinstance(lc, Float8Linear.Config)


def _router_config_for_quantization(dim: int):
    return make_router_config(
        dim=dim,
        num_experts=dim,
        score_func=Sigmoid.Config(),
        gate_param_init={"weight": torch.nn.init.zeros_},
    )


def test_quantization_preserves_invariant_row_parallel_linear():
    config_cls = get_quantized_linear(_ScaledLinear, InvariantRowParallelLinear).Config
    converted = config_cls(in_features=16, out_features=16, bias=True, scale=3.0)

    assert converted._owner is not None
    assert issubclass(converted._owner, InvariantRowParallelLinear)
    assert issubclass(converted._owner, _ScaledLinear)

    linear = converted.build()
    input = torch.randn(2, 16)
    expected = 3.0 * torch.nn.functional.linear(input, linear.weight, linear.bias)
    torch.testing.assert_close(linear(input), expected)


@pytest.mark.parametrize("config_cls", [CastLinear.Config, RouterGateLinear.Config])
def test_quantization_rejects_unsupported_linear_wrapper(config_cls):
    config = config_cls(in_features=16, out_features=16)

    with pytest.raises(ValueError, match=f"does not support {config._owner.__name__}"):
        quantization_transform._validate_quantizable_linear(config, "projection")


@pytest.mark.parametrize("parallel_cls", [ColumnParallelLinear, RowParallelLinear])
def test_get_quantized_linear_preserves_compute_and_tp_role(parallel_cls):
    quantized_cls = get_quantized_linear(_ScaledLinear, parallel_cls)
    config = quantized_cls.Config(
        in_features=4, out_features=2, num_linears=2, scale=3.0
    )
    linear = config.build()

    assert quantized_cls is get_quantized_linear(_ScaledLinear, parallel_cls)
    assert issubclass(quantized_cls, parallel_cls)
    assert issubclass(quantized_cls, _ScaledLinear)
    assert issubclass(quantized_cls.Config, _ScaledLinear.Config)

    input = torch.randn(3, 4)
    expected = 3.0 * torch.nn.functional.linear(
        input, linear.weight.flatten(0, -2), linear.bias
    ).unflatten(-1, linear.weight.shape[:-1])
    torch.testing.assert_close(linear(input), expected)


def test_float8_converter_rejects_router_gate():
    pytest.importorskip("torchao")
    if Float8Linear is None:
        pytest.skip("torchao Float8Linear is unavailable")
    converter = Float8LinearConverter(
        Float8LinearConverter.Config(emulate=True, model_compile_enabled=False)
    )
    with pytest.raises(ValueError, match="does not support RouterGateLinear"):
        converter.convert(_router_config_for_quantization(16))


def test_float8_converter_preserves_recipe_when_emulating(monkeypatch):
    pytest.importorskip("torchao")
    if Float8Linear is None:
        pytest.skip("torchao Float8Linear kernels are unavailable")
    monkeypatch.setattr(quantization_transform, "has_cuda_capability", lambda *_: True)
    converter = Float8LinearConverter(
        Float8LinearConverter.Config(
            recipe_name="rowwise_with_gw_hp",
            emulate=True,
        )
    )

    converted = converter.convert(
        Linear.Config(in_features=128, out_features=128, bias=False)
    )

    assert isinstance(converted, Float8Linear.Config)
    assert converted.recipe_name == "rowwise_with_gw_hp"
    assert converted.emulate


def test_float8_auto_filter_uses_config_dimensions(monkeypatch):
    pytest.importorskip("torchao")
    if Float8Linear is None:
        pytest.skip("torchao Float8Linear kernels are unavailable")
    monkeypatch.setattr(quantization_transform, "has_cuda_capability", lambda *_: True)
    converter = Float8LinearConverter(
        Float8LinearConverter.Config(filter_fqns=["auto_filter_small_kn"])
    )

    large = converter.convert(Linear.Config(in_features=4096, out_features=4096))
    small = converter.convert(Linear.Config(in_features=1024, out_features=4096))

    assert isinstance(large, Float8Linear.Config)
    assert type(small) is Linear.Config


@pytest.mark.parametrize(
    ("config_cls", "parallel_cls"),
    [
        (ColumnParallelLinear.Config, ColumnParallelLinear),
        (RowParallelLinear.Config, RowParallelLinear),
    ],
)
def test_float8_converter_preserves_tensor_parallel_role(config_cls, parallel_cls):
    pytest.importorskip("torchao")
    if Float8Linear is None:
        pytest.skip("torchao Float8Linear is unavailable")
    converter = Float8LinearConverter(
        Float8LinearConverter.Config(emulate=True, model_compile_enabled=False)
    )
    converted = converter.convert(config_cls(in_features=16, out_features=16))

    assert converted._owner is not None
    assert issubclass(converted._owner, Float8Linear)
    assert issubclass(converted._owner, parallel_cls)


def test_mxfp8_converter_rejects_router_gate(monkeypatch):
    pytest.importorskip("torchao")
    if MXFP8Linear is None:
        pytest.skip("torchao MXFP8Linear is unavailable")
    monkeypatch.setattr(quantization_transform, "has_cuda_capability", lambda *_: True)
    converter = MXFP8LinearConverter(MXFP8LinearConverter.Config())
    with pytest.raises(ValueError, match="does not support RouterGateLinear"):
        converter.convert(_router_config_for_quantization(128))


@pytest.mark.parametrize(
    ("config_cls", "parallel_cls"),
    [
        (ColumnParallelLinear.Config, ColumnParallelLinear),
        (RowParallelLinear.Config, RowParallelLinear),
    ],
)
def test_mxfp8_converter_preserves_tensor_parallel_role(
    monkeypatch, config_cls, parallel_cls
):
    if MXFP8Linear is None:
        pytest.skip("torchao MXFP8Linear is unavailable")
    monkeypatch.setattr(quantization_transform, "has_cuda_capability", lambda *_: True)
    converter = MXFP8LinearConverter(
        MXFP8LinearConverter.Config(model_compile_enabled=True)
    )
    converted = converter.convert(config_cls(in_features=128, out_features=128))

    assert converted._owner is not None
    assert issubclass(converted._owner, MXFP8Linear)
    assert issubclass(converted._owner, parallel_cls)


def test_nvfp4_converter_rejects_router_gate(monkeypatch):
    pytest.importorskip("torchao")
    if NVFP4Linear is None:
        pytest.skip("torchao NVFP4 training prototype not available")
    monkeypatch.setattr(quantization_transform, "has_cuda_capability", lambda *_: True)
    converter = NVFP4LinearConverter(NVFP4LinearConverter.Config())
    with pytest.raises(ValueError, match="does not support RouterGateLinear"):
        converter.convert(_router_config_for_quantization(128))


@pytest.mark.parametrize(
    ("config_cls", "parallel_cls"),
    [
        (ColumnParallelLinear.Config, ColumnParallelLinear),
        (RowParallelLinear.Config, RowParallelLinear),
    ],
)
def test_nvfp4_converter_preserves_tensor_parallel_role(
    monkeypatch, config_cls, parallel_cls
):
    if NVFP4Linear is None:
        pytest.skip("torchao NVFP4Linear is unavailable")
    monkeypatch.setattr(quantization_transform, "has_cuda_capability", lambda *_: True)
    converter = NVFP4LinearConverter(
        NVFP4LinearConverter.Config(model_compile_enabled=True)
    )
    converted = converter.convert(config_cls(in_features=128, out_features=128))

    assert converted._owner is not None
    assert issubclass(converted._owner, NVFP4Linear)
    assert issubclass(converted._owner, parallel_cls)


def test_float8_applied_by_model_registry():
    pytest.importorskip("torchao")
    config_manager = ConfigManager()
    config = config_manager.parse_args(
        ["--module", "llama3", "--config", "llama3_debugmodel_float8_emulate_lora"]
    )
    model_config = config.model
    assert has_quantization(model_config)
    # Some Linear.Config instances should be swapped to Float8Linear
    converted = [
        fqn
        for fqn, lc, _parent, _attr in model_config.traverse(Linear.Config)
        if isinstance(lc, Float8Linear.Config)
    ]
    assert len(converted) > 0
    lora_converted = {
        fqn
        for fqn, lc, _parent, _attr in model_config.traverse(Linear.Config)
        if hasattr(lc, "rank") and hasattr(lc, "alpha")
    }
    assert lora_converted == {
        f"layers.{layer}.attention.{projection}"
        for layer in range(6)
        for projection in ("qkv_linear.wqkv", "wo")
    }


@pytest.mark.parametrize(
    "module, recipe, expected_num_layers",
    [
        ("llama3", "llama3_debugmodel_nvfp4", 6),
        ("qwen3", "qwen3_debugmodel_nvfp4", 8),
    ],
)
def test_nvfp4_converter_targets_layers_not_lm_head(
    monkeypatch, module, recipe, expected_num_layers
):
    pytest.importorskip("torchao")
    from torchtitan.quantization import NVFP4Linear

    if NVFP4Linear is None:
        pytest.skip("torchao NVFP4 training prototype not available")
    # Exercise convert() targeting independent of GPU: bypass the sm100 gate
    # that NVFP4LinearConverter.__init__ enforces (hardware is irrelevant to the
    # config-tree transform under test).
    monkeypatch.setattr(quantization_transform, "has_cuda_capability", lambda *_: True)

    config_manager = ConfigManager()
    config = config_manager.parse_args(["--module", module, "--config", recipe])
    model_config = config.model
    assert has_quantization(model_config)

    converted, stock = [], []
    for fqn, lc, _parent, _attr in model_config.traverse(Linear.Config):
        (converted if isinstance(lc, NVFP4Linear.Config) else stock).append(fqn)

    # Every in-layer linear is swapped; the lm_head stays stock (NVFP4 requires
    # each GEMM dim divisible by 128, which the vocab projection violates).
    assert converted and all("layers" in fqn for fqn in converted)
    assert {int(fqn.split(".")[1]) for fqn in converted} == set(
        range(expected_num_layers)
    )
    assert stock == ["lm_head"]


def test_nvfp4_bf16_tail_fqns():
    from torchtitan.quantization.nvfp4 import nvfp4_bf16_tail_fqns

    # 32 layers, 15% tail -> ceil(4.8)=5 bf16, convert layers 0..26.
    fqns = nvfp4_bf16_tail_fqns(32, 0.15)
    assert fqns == [f"layers.{i}." for i in range(27)]
    # Every fqn is trailing-dot anchored so "layers.2." matches layer 2 only,
    # not "layers.20".."layers.29" (the converter substring-matches).
    assert all(f.startswith("layers.") and f.endswith(".") for f in fqns)
    # Fraction 0 keeps nothing in bf16 -> every layer converted.
    assert nvfp4_bf16_tail_fqns(4, 0.0) == [
        "layers.0.",
        "layers.1.",
        "layers.2.",
        "layers.3.",
    ]
    # A fraction that rounds up to all layers leaves nothing to convert -> raise
    # (an empty fqns list would instead convert *all* Linears).
    with pytest.raises(ValueError, match="nothing to convert"):
        nvfp4_bf16_tail_fqns(4, 1.0)


@pytest.mark.parametrize(
    "module, recipe, expected_cutoff",
    [
        ("llama3", "llama3_debugmodel_first_85_pct_layers_nvfp4", 5),
        ("llama3", "llama3_8b_first_85_pct_layers_nvfp4", 27),
        ("qwen3", "qwen3_debugmodel_first_85_pct_layers_nvfp4", 6),
        ("qwen3", "qwen3_8b_first_85_pct_layers_nvfp4", 30),
    ],
)
def test_nvfp4_first_85_pct_layers_converts_only_leading_layers(
    monkeypatch, module, recipe, expected_cutoff
):
    pytest.importorskip("torchao")
    from torchtitan.quantization import NVFP4Linear

    if NVFP4Linear is None:
        pytest.skip("torchao NVFP4 training prototype not available")
    import math

    monkeypatch.setattr(quantization_transform, "has_cuda_capability", lambda *_: True)

    config = ConfigManager().parse_args(["--module", module, "--config", recipe])
    model_config = config.model
    n_layers = len(model_config.layers)
    cutoff = n_layers - math.ceil(n_layers * 0.15)
    assert cutoff == expected_cutoff
    assert 0 < cutoff < n_layers  # a real split: some NVFP4, some bf16

    converted_layers, stock = set(), []
    for fqn, lc, _parent, _attr in model_config.traverse(Linear.Config):
        if isinstance(lc, NVFP4Linear.Config):
            converted_layers.add(int(fqn.split(".")[1]))
        else:
            stock.append(fqn)

    # Only the leading layers are NVFP4; the bf16 tail + lm_head stay stock.
    assert converted_layers == set(range(cutoff))
    assert "lm_head" in stock
    assert all(
        not fqn.startswith("layers.") or int(fqn.split(".")[1]) >= cutoff
        for fqn in stock
    )


def _nvfp4_linear_cls():
    pytest.importorskip("torchao")
    from torchtitan.quantization import NVFP4Linear

    if NVFP4Linear is None:
        pytest.skip("torchao NVFP4 training prototype not available")
    return NVFP4Linear


@pytest.mark.parametrize("in_features, out_features", [(512, 300), (300, 512)])
def test_nvfp4_config_rejects_non_128_dims(in_features, out_features):
    # The model dims are known at config-build time, so a non-128 in/out_features
    # (e.g. the LM head) is rejected in Config.__post_init__ before any TP.
    NVFP4Linear = _nvfp4_linear_cls()
    with pytest.raises(ValueError, match="divisible by 128"):
        NVFP4Linear.Config(in_features=in_features, out_features=out_features)


@pytest.mark.parametrize(
    "sharding_config_factory, input_tp",
    [
        pytest.param(
            lambda: colwise_config(input_layout=dense_sequence_parallel_placement()),
            spmd.R,
            id="colwise",
        ),
        pytest.param(
            lambda: rowwise_config(output_layout=dense_sequence_parallel_placement()),
            spmd.S(-1),
            id="rowwise",
        ),
    ],
)
def test_nvfp4_build_configures_local_spmd_sharding(sharding_config_factory, input_tp):
    # Config.build() folds the stock colwise/rowwise sharding into the local
    # SPMD region for the opaque NVFP4 GEMM.
    NVFP4Linear = _nvfp4_linear_cls()
    from torchtitan.distributed.parallel_dims import MeshAxisName
    from torchtitan.models.common.decoder_sharding import dense_activation_placement

    module = NVFP4Linear.Config(
        in_features=512,
        out_features=1024,
        sharding_config=sharding_config_factory(),
    ).build()
    sc = module._sharding_config
    assert sc.local_spmd
    input_layout = dense_activation_placement(tp=input_tp, cp=spmd.S(0))
    assert sc.in_src_shardings == {"input": input_layout}
    assert sc.in_dst_shardings == {"input": input_layout}
    assert list(inspect.signature(module.forward).parameters) == ["input"]
    assert "weight" in sc.state_shardings
    assert sc.state_shardings["_sr_seed"] == SpmdType(
        {
            MeshAxisName.DP: spmd.V,
            MeshAxisName.CP: spmd.V,
            MeshAxisName.TP: spmd.V,
        }
    )


@pytest.mark.parametrize("parallel_cls", [ColumnParallelLinear, RowParallelLinear])
def test_nvfp4_parallel_build_preserves_collective_boundary(parallel_cls):
    if NVFP4Linear is None:
        pytest.skip("torchao NVFP4 training prototype not available")

    boundary_layout = dense_sequence_parallel_placement()
    linear_cls = get_quantized_linear(NVFP4Linear, parallel_cls)
    sharding_config = (
        colwise_config(input_layout=boundary_layout)
        if parallel_cls is ColumnParallelLinear
        else rowwise_config(output_layout=boundary_layout)
    )
    module = linear_cls.Config(
        in_features=512,
        out_features=1024,
        sharding_config=sharding_config,
    ).build()

    assert not module._sharding_config.local_spmd
    assert module._sharding_config.in_src_shardings == (
        sharding_config.in_src_shardings
    )
    assert module._sharding_config.out_src_shardings == (
        sharding_config.out_src_shardings
    )
    assert "_sr_seed" in module._sharding_config.state_shardings


@pytest.mark.parametrize(
    "module, recipe",
    [
        ("llama3", "llama3_debugmodel_nvfp4"),
        ("llama3", "llama3_debugmodel_first_85_pct_layers_nvfp4"),
        ("llama3", "llama3_8b_first_85_pct_layers_nvfp4"),
        ("qwen3", "qwen3_debugmodel_nvfp4"),
        ("qwen3", "qwen3_debugmodel_first_85_pct_layers_nvfp4"),
        ("qwen3", "qwen3_8b_first_85_pct_layers_nvfp4"),
    ],
)
def test_nvfp4_recipes_parse(monkeypatch, module, recipe):
    _nvfp4_linear_cls()
    monkeypatch.setattr(quantization_transform, "has_cuda_capability", lambda *_: True)
    base_args = ["--module", module, "--config", recipe]

    ConfigManager().parse_args(base_args)


@pytest.mark.parametrize(
    "recipe",
    [
        "qwen3_debugmodel_nvfp4",
        "qwen3_debugmodel_first_85_pct_layers_nvfp4",
        "qwen3_8b_first_85_pct_layers_nvfp4",
    ],
)
def test_qwen3_recipes_resolve(monkeypatch, recipe):
    _nvfp4_linear_cls()
    monkeypatch.setattr(quantization_transform, "has_cuda_capability", lambda *_: True)
    config = ConfigManager().parse_args(["--module", "qwen3", "--config", recipe])
    assert type(config.model).__qualname__ == "Qwen3Model.Config"
    if recipe == "qwen3_8b_first_85_pct_layers_nvfp4":
        assert isinstance(config.dataloader, GrainDataLoader.Config)
        packed_dataset = config.dataloader.dataset
        assert isinstance(packed_dataset, FirstFitPackingConfig)
        dataset = packed_dataset.dataset
        assert isinstance(dataset, SingleDatasetConfig)
        assert isinstance(dataset.source, HuggingFaceRandomAccessSource.Config)
        assert dataset.source.path == "openai/gsm8k"
        assert config.checkpointer.initial_load_in_hf
        assert config.compile is not None
        assert "model" in config.compile.components


def test_nvfp4_module_buffers_and_native_checkpoint():
    """Built module has the stock weight param plus the two NVFP4 runtime
    buffers, and both buffers are non-persistent -- the RHT vector is a fixed
    constant and the SR seed is per-rank -- so a native checkpoint carries only
    the stock weight."""
    NVFP4Linear = _nvfp4_linear_cls()
    from torchtitan.quantization.nvfp4 import _HARDCODED_SIGN_VECTOR

    module = NVFP4Linear.Config(in_features=512, out_features=1024).build()
    assert {name for name, _ in module.named_parameters()} == {"weight"}
    module.init_states()
    buffers = dict(module.named_buffers())
    assert set(buffers) == {"_sr_seed", "_rht_sign_vector"}
    assert buffers["_sr_seed"].dtype == torch.int64
    assert tuple(buffers["_rht_sign_vector"].shape) == (16,)
    # The RHT vector is the fixed v1-recipe constant, identical on every rank.
    assert tuple(int(v) for v in buffers["_rht_sign_vector"]) == _HARDCODED_SIGN_VECTOR
    # Both runtime buffers are non-persistent, so a native checkpoint carries
    # only the stock weight.
    assert set(module.state_dict()) == {"weight"}


def test_nvfp4_stock_checkpoint_loads_before_init_states():
    """A stock bf16 checkpoint (no NVFP4 buffers) loads; buffers stay unmaterialized
    until init_states creates them."""
    NVFP4Linear = _nvfp4_linear_cls()
    stock = Linear.Config(in_features=512, out_features=1024).build()
    nvfp4 = NVFP4Linear.Config(in_features=512, out_features=1024).build()

    nvfp4.load_state_dict(stock.state_dict(), strict=False)
    assert nvfp4._rht_sign_vector is None
    assert nvfp4._rht_sign_vector_tuple is None

    nvfp4.init_states()
    assert nvfp4._rht_sign_vector is not None
    assert nvfp4._rht_sign_vector_tuple is not None


def test_nvfp4_hf_export_strips_buffers(monkeypatch):
    """The HF export boundary contains only stock keys -- no NVFP4 runtime buffers."""
    NVFP4Linear = _nvfp4_linear_cls()
    monkeypatch.setattr(quantization_transform, "has_cuda_capability", lambda *_: True)
    from torchtitan.models.llama3.state_dict_adapter import Llama3StateDictAdapter

    config = ConfigManager().parse_args(
        ["--module", "llama3", "--config", "llama3_debugmodel_nvfp4"]
    )
    model_config = config.model
    model = model_config.build()
    model.init_states()
    assert isinstance(model.get_submodule("layers.0.feed_forward.w13"), NVFP4Linear)

    sd = model.state_dict()
    # Both NVFP4 runtime buffers are non-persistent, so neither the RHT vector
    # nor the per-rank SR seed appears in the native state dict.
    assert not any("_rht_sign_vector" in k for k in sd)
    assert not any("_sr_seed" in k for k in sd)

    hf_sd = Llama3StateDictAdapter(model_config, hf_assets_path=None).to_hf(sd)
    assert "model.layers.0.mlp.gate_proj.weight" in hf_sd
    assert not any("_rht_sign_vector" in k for k in hf_sd)


def test_quantized_grouped_experts():
    """Quantized GroupedExperts: _owner, subclass handling, extra config fields."""
    # Base case
    MXFP8GroupedExperts = _get_mxfp8_grouped_experts_cls(GroupedExperts)
    Float8GroupedExperts = _get_float8_grouped_experts_cls(GroupedExperts)

    assert MXFP8GroupedExperts.Config._owner is MXFP8GroupedExperts
    assert Float8GroupedExperts.Config._owner is Float8GroupedExperts

    # Subclass case (GptOssGroupedExperts has extra swiglu_limit field)
    mxfp8_cls = _get_mxfp8_grouped_experts_cls(GptOssGroupedExperts)
    float8_cls = _get_float8_grouped_experts_cls(GptOssGroupedExperts)

    assert mxfp8_cls.Config._owner is mxfp8_cls
    assert float8_cls.Config._owner is float8_cls
    assert issubclass(mxfp8_cls, GptOssGroupedExperts)
    assert issubclass(float8_cls, GptOssGroupedExperts)
    assert hasattr(mxfp8_cls.Config, "swiglu_limit")
    assert hasattr(float8_cls.Config, "swiglu_limit")

    from torchtitan.quantization.float8.tensor import (
        _GroupedExpertsShardedTensorWithFloat8Compute,
    )

    for parent_cls in (GroupedExperts, GptOssGroupedExperts):
        quantized_cls = _get_float8_grouped_experts_cls(parent_cls)
        module = quantized_cls.Config(
            dim=128,
            hidden_dim=128,
            num_experts=4,
        ).build()
        grouped_weights = [
            parameter
            for parameter in module.parameters(recurse=False)
            if parameter.ndim == 3
        ]
        assert grouped_weights
        assert all(
            isinstance(weight, _GroupedExpertsShardedTensorWithFloat8Compute)
            for weight in grouped_weights
        )


@pytest.mark.parametrize("parent_cls", [GroupedExperts, GptOssGroupedExperts])
@pytest.mark.parametrize(
    "make_quantized_cls",
    [_get_mxfp8_grouped_experts_cls, _get_float8_grouped_experts_cls],
    ids=["mxfp8", "float8"],
)
def test_grouped_mm_overrides_keep_the_seam_signature(make_quantized_cls, parent_cls):
    """Every ``_grouped_mm`` override must accept the base class's keywords.

    ``MoE.forward`` calls the seam by keyword, so an override whose parameter
    names drift raises TypeError at the first expert GEMM rather than at import
    time -- and only in a MoE training run, which no other unit test reaches.
    That is how the ``B_t`` -> ``weight_EOI`` rename left the MXFP8 override
    behind while the float8 one was updated.
    """
    base = inspect.signature(parent_cls._grouped_mm)
    override = inspect.signature(make_quantized_cls(parent_cls)._grouped_mm)

    assert list(override.parameters) == list(base.parameters)
    for name, parameter in base.parameters.items():
        assert override.parameters[name].kind == parameter.kind


@pytest.mark.parametrize("parent_cls", [GroupedExperts, GptOssGroupedExperts])
def test_float8_grouped_experts_checkpoint_state_uses_plain_tensors(parent_cls):
    pytest.importorskip("torchao")
    from torchtitan.quantization.float8.tensor import (
        _GroupedExpertsShardedTensorWithFloat8Compute,
    )

    stock = parent_cls.Config(dim=16, hidden_dim=32, num_experts=2).build()
    float8_cls = _get_float8_grouped_experts_cls(parent_cls)
    module = float8_cls.Config(dim=16, hidden_dim=32, num_experts=2).build()

    assert all(
        isinstance(param, _GroupedExpertsShardedTensorWithFloat8Compute)
        if param.ndim == 3
        else type(param) is torch.nn.Parameter
        for param in module.parameters()
    )
    stock_state = stock.state_dict()
    float8_state = module.state_dict()
    assert float8_state.keys() == stock_state.keys()
    for key, value in float8_state.items():
        assert type(value) is torch.Tensor
        assert value.shape == stock_state[key].shape
        assert value.dtype == stock_state[key].dtype


@pytest.mark.filterwarnings("ignore:torch.distributed is disabled")
def test_float8_grouped_experts_dcp_round_trip_needs_no_safe_globals(tmp_path):
    pytest.importorskip("torchao")
    float8_cls = _get_float8_grouped_experts_cls(GroupedExperts)
    config = float8_cls.Config(dim=16, hidden_dim=32, num_experts=2)
    source = config.build()
    target = config.build()

    with torch.no_grad():
        for value, parameter in enumerate(source.parameters(), start=1):
            parameter.fill_(value)
        for parameter in target.parameters():
            parameter.zero_()

    saved_safe_globals = torch.serialization.get_safe_globals()
    try:
        torch.serialization.clear_safe_globals()
        dcp.save(source.state_dict(), checkpoint_id=tmp_path, no_dist=True)
        dcp.load(target.state_dict(), checkpoint_id=tmp_path, no_dist=True)
    finally:
        torch.serialization.clear_safe_globals()
        torch.serialization.add_safe_globals(saved_safe_globals)

    for source_parameter, target_parameter in zip(
        source.parameters(), target.parameters(), strict=True
    ):
        torch.testing.assert_close(target_parameter, source_parameter)


def test_mxfp8_linear_validates_config_and_installs_weight_wrapper():
    pytest.importorskip("torchao")
    if MXFP8Linear is None:
        pytest.skip("torchao MXFP8Linear is unavailable")
    from torchtitan.quantization._fsdp_tensor import _UnshardedFSDPTensor
    from torchtitan.quantization.mxfp8.tensor import (
        _LinearShardedTensorWithMXFP8Compute,
    )

    with pytest.raises(ValueError, match="in_features divisible by 32"):
        MXFP8Linear.Config(in_features=127, out_features=128)
    with pytest.raises(ValueError, match="out_features divisible by 32"):
        MXFP8Linear.Config(in_features=128, out_features=127)
    with pytest.raises(
        ValueError,
        match="input_activation_format_for_backward must be one of",
    ):
        MXFP8Linear.Config(
            in_features=128,
            out_features=128,
            input_activation_format_for_backward="missing",
        )
    with pytest.raises(ValueError, match="out_features divisible by 32"):
        MXFP8Linear.Config(
            in_features=128,
            out_features=127,
            num_linears=2,
        )

    local_stacked_weight = _LinearShardedTensorWithMXFP8Compute(
        torch.empty(3, 16, 128, dtype=torch.bfloat16)
    )
    with pytest.raises(ValueError, match="local matrix out_features divisible by 32"):
        local_stacked_weight._build_operands(local_stacked_weight._tensor)

    for sharding_config in (
        colwise_config(input_layout=dense_sequence_parallel_placement()),
        rowwise_config(output_layout=dense_sequence_parallel_placement()),
    ):
        linear = MXFP8Linear.Config(
            in_features=128,
            out_features=128,
            bias=False,
            sharding_config=sharding_config,
        ).build()
        assert linear._sharding_config is not None
        # The wrapper is installed at construction, so no caller has to opt
        # in. Until a data parallel implementation drives its lifecycle it is
        # the sharded state, which holds the BF16 weight; the unsharded tensor
        # is a separate type the post-all-gather hook produces.
        assert isinstance(linear.weight, _LinearShardedTensorWithMXFP8Compute)
        assert not isinstance(linear.weight, _UnshardedFSDPTensor)


def test_mxfp8_converter_replaces_a_root_linear_config(monkeypatch):
    """A Linear.Config with no parent is returned, not mutated in place.

    ``convert`` writes into ``parent`` for nested configs, so the root case is
    the one branch that has to return the replacement. Not covered by the FQN
    test below, which passes a FeedForward and so always has a parent.
    """
    monkeypatch.setattr(quantization_transform, "has_cuda_capability", lambda *_: True)
    converter = MXFP8LinearConverter(
        MXFP8LinearConverter.Config(
            model_compile_enabled=True,
        )
    )

    converted = converter.convert(
        Linear.Config(in_features=128, out_features=128, bias=False)
    )

    assert isinstance(converted, MXFP8Linear.Config)
    assert converted.input_activation_format_for_backward == "bf16"


def test_mxfp8_converter_rejects_unaligned_fused_qkv_head_dim(monkeypatch):
    if MXFP8Linear is None:
        pytest.skip("torchao MXFP8Linear is unavailable")
    monkeypatch.setattr(quantization_transform, "has_cuda_capability", lambda *_: True)
    converter = MXFP8LinearConverter(
        MXFP8LinearConverter.Config(model_compile_enabled=True)
    )
    head_dim = 48
    n_heads = 4
    n_kv_heads = 2
    qkv_config = QKVLinear.Config(
        head_dim=head_dim,
        n_heads=n_heads,
        n_kv_heads=n_kv_heads,
        wqkv=Linear.Config(
            in_features=128,
            out_features=(n_heads + 2 * n_kv_heads) * head_dim,
        ),
    )

    with pytest.raises(ValueError, match="head_dim divisible by 32"):
        converter.convert(qkv_config)


def test_mxfp8_converter_applies_mxfp8_saved_input_fqns(monkeypatch):
    monkeypatch.setattr(quantization_transform, "has_cuda_capability", lambda *_: True)
    converter = MXFP8LinearConverter(
        MXFP8LinearConverter.Config(
            model_compile_enabled=True,
            linears_saving_inputs_for_backward_in_mxfp8=["w2"],
        )
    )
    converted = converter.convert(
        FeedForward.Config(
            w13=Linear.Config(in_features=128, out_features=128, num_linears=2),
            w2=Linear.Config(in_features=128, out_features=128),
        )
    )

    assert isinstance(converted.w13, MXFP8Linear.Config)
    assert isinstance(converted.w2, MXFP8Linear.Config)
    assert converted.w13.input_activation_format_for_backward == "bf16"
    assert converted.w2.input_activation_format_for_backward == "mxfp8"


def test_mxfp8_converter_rejects_unmatched_saved_input_fqns(monkeypatch):
    monkeypatch.setattr(quantization_transform, "has_cuda_capability", lambda *_: True)
    converter = MXFP8LinearConverter(
        MXFP8LinearConverter.Config(
            model_compile_enabled=True,
            linears_saving_inputs_for_backward_in_mxfp8=["missing"],
        )
    )
    model_config = FeedForward.Config(
        w13=Linear.Config(in_features=128, out_features=128, num_linears=2),
        w2=Linear.Config(in_features=128, out_features=128),
    )

    with pytest.raises(
        ValueError,
        match="selectors did not match any converted Linear.Config",
    ):
        converter.convert(model_config)


def test_mxfp8_converter_rejects_empty_saved_input_fqn():
    with pytest.raises(ValueError, match="cannot contain an empty FQN selector"):
        MXFP8LinearConverter.Config(
            model_compile_enabled=True,
            linears_saving_inputs_for_backward_in_mxfp8=[""],
        )


@pytest.mark.parametrize(
    "config_factory, mxfp8_fqns",
    [
        (
            "llama3",
            ("attention.qkv_linear.wqkv", "feed_forward.w2"),
        ),
        (
            "llama3_graph",
            ("attention.qkv_linear.wqkv", "feed_forward.w2"),
        ),
        (
            "deepseek_v3",
            ("attention.wkv_b", "feed_forward.w2", "shared_experts.w2"),
        ),
        (
            "deepseek_v3_graph",
            ("attention.wkv_b", "feed_forward.w2", "shared_experts.w2"),
        ),
    ],
)
def test_builtin_mxfp8_configs_assign_input_activation_format_for_backward(
    monkeypatch, config_factory, mxfp8_fqns
):
    if MXFP8Linear is None:
        pytest.skip("torchao MXFP8Linear is unavailable")
    monkeypatch.setattr(quantization_transform, "has_cuda_capability", lambda *_: True)
    if config_factory == "llama3":
        from torchtitan.models.llama3.config_registry import (
            llama3_debugmodel_mxfp8 as build_config,
        )
    elif config_factory == "llama3_graph":
        from torchtitan.experiments.graph_trainer.llama3.config_registry import (
            graph_trainer_llama3_debugmodel_mxfp8 as build_config,
        )
    elif config_factory == "deepseek_v3":
        from torchtitan.models.deepseek_v3.config_registry import (
            deepseek_v3_debugmodel_mxfp8 as build_config,
        )
    else:
        from torchtitan.experiments.graph_trainer.deepseek_v3.config_registry import (
            graph_trainer_deepseek_v3_debugmodel_mxfp8 as build_config,
        )

    trainer_config = build_config()
    model_config = trainer_config.model
    assignments = {
        fqn: config.input_activation_format_for_backward
        for fqn, config, _parent, _attr in model_config.traverse(MXFP8Linear.Config)
    }
    assert assignments
    assert "bf16" in assignments.values()
    assert "mxfp8" in assignments.values()
    for fqn, save_format in assignments.items():
        expected = (
            "mxfp8" if any(selector in fqn for selector in mxfp8_fqns) else "bf16"
        )
        assert save_format == expected, f"Unexpected policy for {fqn}"


def test_mxfp8_linear_loads_stock_checkpoint():
    pytest.importorskip("torchao")
    if MXFP8Linear is None:
        pytest.skip("torchao MXFP8Linear is unavailable")
    from torchtitan.quantization.mxfp8.tensor import (
        _LinearShardedTensorWithMXFP8Compute,
    )

    stock = Linear.Config(in_features=128, out_features=96).build()
    mxfp8 = MXFP8Linear.Config(in_features=128, out_features=96).build()
    with torch.no_grad():
        stock.weight.normal_()

    mxfp8.load_state_dict(stock.state_dict())
    assert isinstance(mxfp8.weight, _LinearShardedTensorWithMXFP8Compute)
    assert torch.equal(mxfp8.weight._tensor, stock.weight)
