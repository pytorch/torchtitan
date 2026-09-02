# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
import pytest
import spmd_types as spmd
import torch
import torch.distributed.checkpoint as dcp
from spmd_types import SpmdType

from torchtitan.components.data import (
    FirstFitPackingConfig,
    GrainDataLoader,
    SingleDatasetConfig,
)
from torchtitan.components.data.sources import HuggingFaceRandomAccessSource
from torchtitan.components.quantization import Float8Linear
from torchtitan.components.quantization.float8 import _get_float8_grouped_experts_cls
from torchtitan.components.quantization.mxfp8.converter import (
    get_mxfp8_grouped_experts_cls,
    MXFP8GroupedExpertsConverter,
    MXFP8Linear,
    MXFP8LinearConverter,
)
from torchtitan.components.quantization.utils import has_quantization
from torchtitan.config import ConfigManager
from torchtitan.models.common.decoder_sharding import colwise_config, rowwise_config
from torchtitan.models.common.feed_forward import FeedForward
from torchtitan.models.common.linear import Linear
from torchtitan.models.common.moe import GroupedExperts
from torchtitan.models.gpt_oss.moe import GptOssGroupedExperts


def test_no_float8_by_default():
    config_manager = ConfigManager()
    config = config_manager.parse_args(
        ["--module", "llama3", "--config", "llama3_debugmodel"]
    )
    model_config = config.model_spec.model
    assert not has_quantization(model_config)
    # All Linear.Config instances should remain Linear.Config
    if Float8Linear is not None:
        for _fqn, lc, _parent, _attr in model_config.traverse(Linear.Config):
            assert not isinstance(lc, Float8Linear.Config)


def test_float8_applied_by_model_registry():
    pytest.importorskip("torchao")
    config_manager = ConfigManager()
    config = config_manager.parse_args(
        ["--module", "llama3", "--config", "llama3_debugmodel_float8_emulate_lora"]
    )
    model_config = config.model_spec.model
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
    from torchtitan.components.quantization import NVFP4Linear

    if NVFP4Linear is None:
        pytest.skip("torchao NVFP4 training prototype not available")
    # Exercise convert() targeting independent of GPU: bypass the sm100 gate
    # that NVFP4LinearConverter.__init__ enforces (hardware is irrelevant to the
    # config-tree transform under test).
    import torchtitan.components.quantization.nvfp4 as nvfp4_mod

    monkeypatch.setattr(nvfp4_mod, "has_cuda_capability", lambda *_: True)

    config_manager = ConfigManager()
    config = config_manager.parse_args(["--module", module, "--config", recipe])
    model_config = config.model_spec.model
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
    from torchtitan.components.quantization.nvfp4 import nvfp4_bf16_tail_fqns

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
    from torchtitan.components.quantization import NVFP4Linear

    if NVFP4Linear is None:
        pytest.skip("torchao NVFP4 training prototype not available")
    import math

    import torchtitan.components.quantization.nvfp4 as nvfp4_mod

    monkeypatch.setattr(nvfp4_mod, "has_cuda_capability", lambda *_: True)

    config = ConfigManager().parse_args(["--module", module, "--config", recipe])
    model_config = config.model_spec.model
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
    from torchtitan.components.quantization import NVFP4Linear

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
    "sharding_config_factory, input_tp, input_grad_tp",
    [
        pytest.param(lambda: colwise_config(), spmd.R, spmd.P, id="colwise"),
        pytest.param(
            lambda: rowwise_config(output_sp=True),
            spmd.S(-1),
            spmd.S(-1),
            id="rowwise",
        ),
    ],
)
def test_nvfp4_build_configures_local_spmd_sharding(
    sharding_config_factory, input_tp, input_grad_tp
):
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
    assert sc.local_map is not None
    input_layout = dense_activation_placement(tp=input_tp, cp=spmd.S(0))
    assert sc.in_src_shardings == {"x": input_layout}
    assert sc.in_dst_shardings == {"x": input_layout}
    assert sc.local_map.in_grad_placements == (
        dense_activation_placement(tp=input_grad_tp, cp=spmd.S(0)),
    )
    assert "weight" in sc.state_shardings
    assert sc.state_shardings["_sr_seed"] == SpmdType(
        {
            MeshAxisName.DP: spmd.V,
            MeshAxisName.CP: spmd.V,
            MeshAxisName.TP: spmd.V,
        }
    )


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
def test_nvfp4_recipes_default_to_spmd_types_and_allow_cli_override(
    monkeypatch, module, recipe
):
    _nvfp4_linear_cls()
    import torchtitan.components.quantization.nvfp4 as nvfp4_mod

    monkeypatch.setattr(nvfp4_mod, "has_cuda_capability", lambda *_: True)
    base_args = ["--module", module, "--config", recipe]

    config = ConfigManager().parse_args(base_args)
    assert config.parallelism.spmd_backend == "spmd_types"

    overridden = ConfigManager().parse_args(
        [*base_args, "--parallelism.spmd_backend", "partial_dtensor"]
    )
    assert overridden.parallelism.spmd_backend == "partial_dtensor"


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
    import torchtitan.components.quantization.nvfp4 as nvfp4_mod

    monkeypatch.setattr(nvfp4_mod, "has_cuda_capability", lambda *_: True)
    config = ConfigManager().parse_args(["--module", "qwen3", "--config", recipe])
    assert config.model_spec.name == "qwen3"
    if recipe == "qwen3_8b_first_85_pct_layers_nvfp4":
        assert isinstance(config.dataloader, GrainDataLoader.Config)
        packed_dataset = config.dataloader.dataset
        assert isinstance(packed_dataset, FirstFitPackingConfig)
        dataset = packed_dataset.dataset
        assert isinstance(dataset, SingleDatasetConfig)
        assert isinstance(dataset.source, HuggingFaceRandomAccessSource.Config)
        assert dataset.source.path == "openai/gsm8k"
        assert config.checkpoint.initial_load_in_hf
        assert config.compile.enable
        assert "model" in config.compile.components


def test_nvfp4_module_buffers_and_native_checkpoint():
    """Built module has the stock weight param plus the two NVFP4 runtime
    buffers, and both buffers are non-persistent -- the RHT vector is a fixed
    constant and the SR seed is per-rank -- so a native checkpoint carries only
    the stock weight."""
    NVFP4Linear = _nvfp4_linear_cls()
    from torchtitan.components.quantization.nvfp4 import _HARDCODED_SIGN_VECTOR

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
    import torchtitan.components.quantization.nvfp4 as nvfp4_mod

    monkeypatch.setattr(nvfp4_mod, "has_cuda_capability", lambda *_: True)
    from torchtitan.models.llama3.state_dict_adapter import Llama3StateDictAdapter

    config = ConfigManager().parse_args(
        ["--module", "llama3", "--config", "llama3_debugmodel_nvfp4"]
    )
    model_config = config.model_spec.model
    model = model_config.build()
    model.init_states()
    assert isinstance(model.get_submodule("layers.0.feed_forward.w1"), NVFP4Linear)

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
    MXFP8GroupedExperts = get_mxfp8_grouped_experts_cls(GroupedExperts)
    Float8GroupedExperts = _get_float8_grouped_experts_cls(GroupedExperts)

    assert MXFP8GroupedExperts.Config._owner is MXFP8GroupedExperts
    assert Float8GroupedExperts.Config._owner is Float8GroupedExperts

    # Subclass case (GptOssGroupedExperts has extra swiglu_limit field)
    mxfp8_cls = get_mxfp8_grouped_experts_cls(GptOssGroupedExperts)
    float8_cls = _get_float8_grouped_experts_cls(GptOssGroupedExperts)

    assert mxfp8_cls.Config._owner is mxfp8_cls
    assert float8_cls.Config._owner is float8_cls
    assert issubclass(mxfp8_cls, GptOssGroupedExperts)
    assert issubclass(float8_cls, GptOssGroupedExperts)
    assert hasattr(mxfp8_cls.Config, "swiglu_limit")
    assert hasattr(float8_cls.Config, "swiglu_limit")


def test_mxfp8_grouped_experts_config_validation():
    """The grouped-expert config rejects unusable padding and activation formats."""
    experts_cls = get_mxfp8_grouped_experts_cls(GroupedExperts)
    experts_cls.Config(dim=16, hidden_dim=32, num_experts=2)

    with pytest.raises(ValueError, match="input_activation_format_for_backward"):
        experts_cls.Config(
            dim=16,
            hidden_dim=32,
            num_experts=2,
            input_activation_format_for_backward="fp8",
        )

    # Token groups must land on the 128-row boundary the blocked grouped-GEMM
    # scale layout assumes.
    MXFP8GroupedExpertsConverter.Config(model_compile_enabled=True, pad_multiple=128)
    MXFP8GroupedExpertsConverter.Config(model_compile_enabled=True, pad_multiple=256)
    with pytest.raises(ValueError, match="multiple of 128"):
        MXFP8GroupedExpertsConverter.Config(model_compile_enabled=True, pad_multiple=32)


@pytest.mark.parametrize("parent_cls", [GroupedExperts, GptOssGroupedExperts])
def test_float8_grouped_experts_checkpoint_state_uses_plain_tensors(parent_cls):
    pytest.importorskip("torchao")
    stock = parent_cls.Config(dim=16, hidden_dim=32, num_experts=2).build()
    float8_cls = _get_float8_grouped_experts_cls(parent_cls)
    module = float8_cls.Config(dim=16, hidden_dim=32, num_experts=2).build()

    assert all(type(param) is torch.nn.Parameter for param in module.parameters())
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
    from torchtitan.components.quantization._fsdp_tensor import _UnshardedFSDPTensor
    from torchtitan.components.quantization.mxfp8.tensor import (
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

    for sharding_config in (colwise_config(), rowwise_config()):
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


def test_mxfp8_linear_rejects_the_partial_dtensor_backend():
    """MXFP8 needs the spmd_types backend to survive tensor parallelism.

    The matmul is an opaque autograd function, so DTensor has no sharding
    strategy for it and propagation fails on the storage-free unsharded tensor.
    spmd_types annotates the function instead.
    """
    pytest.importorskip("torchao")
    if MXFP8Linear is None:
        pytest.skip("torchao MXFP8Linear is unavailable")
    from torchtitan.distributed.utils import get_spmd_backend, set_spmd_backend

    previous_backend = get_spmd_backend()
    set_spmd_backend("partial_dtensor")
    try:
        with pytest.raises(ValueError, match="spmd_backend"):
            MXFP8Linear.Config(in_features=128, out_features=128).build()
    finally:
        set_spmd_backend(previous_backend)


def test_mxfp8_converter_sets_default_input_activation_format_for_backward(
    monkeypatch,
):
    import torchtitan.components.quantization.mxfp8.converter as converter_mod

    monkeypatch.setattr(converter_mod, "has_cuda_capability", lambda *_: True)
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


def test_mxfp8_converter_applies_mxfp8_saved_input_fqns(monkeypatch):
    import torchtitan.components.quantization.mxfp8.converter as converter_mod

    monkeypatch.setattr(converter_mod, "has_cuda_capability", lambda *_: True)
    converter = MXFP8LinearConverter(
        MXFP8LinearConverter.Config(
            model_compile_enabled=True,
            linears_saving_inputs_for_backward_in_mxfp8=["w2"],
        )
    )
    converted = converter.convert(
        FeedForward.Config(
            w1=Linear.Config(in_features=128, out_features=128),
            w2=Linear.Config(in_features=128, out_features=128),
            w3=Linear.Config(in_features=128, out_features=128),
        )
    )

    assert isinstance(converted.w1, MXFP8Linear.Config)
    assert isinstance(converted.w2, MXFP8Linear.Config)
    assert isinstance(converted.w3, MXFP8Linear.Config)
    assert converted.w1.input_activation_format_for_backward == "bf16"
    assert converted.w2.input_activation_format_for_backward == "mxfp8"
    assert converted.w3.input_activation_format_for_backward == "bf16"


def test_mxfp8_converter_rejects_unmatched_saved_input_fqns(monkeypatch):
    import torchtitan.components.quantization.mxfp8.converter as converter_mod

    monkeypatch.setattr(converter_mod, "has_cuda_capability", lambda *_: True)
    converter = MXFP8LinearConverter(
        MXFP8LinearConverter.Config(
            model_compile_enabled=True,
            linears_saving_inputs_for_backward_in_mxfp8=["missing"],
        )
    )
    model_config = FeedForward.Config(
        w1=Linear.Config(in_features=128, out_features=128),
        w2=Linear.Config(in_features=128, out_features=128),
        w3=Linear.Config(in_features=128, out_features=128),
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
    import torchtitan.components.quantization.mxfp8.converter as converter_mod

    monkeypatch.setattr(converter_mod, "has_cuda_capability", lambda *_: True)
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
    assert trainer_config.model_spec is not None
    model_config = trainer_config.model_spec.model
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
    from torchtitan.components.quantization.mxfp8.tensor import (
        _LinearShardedTensorWithMXFP8Compute,
    )

    stock = Linear.Config(in_features=128, out_features=96).build()
    mxfp8 = MXFP8Linear.Config(in_features=128, out_features=96).build()
    with torch.no_grad():
        stock.weight.normal_()

    mxfp8.load_state_dict(stock.state_dict())
    assert isinstance(mxfp8.weight, _LinearShardedTensorWithMXFP8Compute)
    assert torch.equal(mxfp8.weight._tensor, stock.weight)
