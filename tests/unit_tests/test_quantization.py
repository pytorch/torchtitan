# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
import pytest
import torch

from torchtitan.components.quantization import Float8Linear
from torchtitan.components.quantization.float8 import _get_float8_grouped_experts_cls
from torchtitan.components.quantization.mx import _get_mxfp8_grouped_experts_cls
from torchtitan.components.quantization.utils import has_quantization
from torchtitan.config import ConfigManager
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


def test_nvfp4_converter_targets_layers_not_lm_head(monkeypatch):
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
    config = config_manager.parse_args(
        ["--module", "llama3", "--config", "llama3_debugmodel_nvfp4"]
    )
    model_config = config.model_spec.model
    assert has_quantization(model_config)

    converted, stock = [], []
    for fqn, lc, _parent, _attr in model_config.traverse(Linear.Config):
        (converted if isinstance(lc, NVFP4Linear.Config) else stock).append(fqn)

    # Every in-layer linear is swapped; the lm_head stays stock (NVFP4 requires
    # each GEMM dim divisible by 128, which the vocab projection violates).
    assert converted and all("layers" in fqn for fqn in converted)
    assert stock == ["lm_head"]


def _nvfp4_linear_cls():
    pytest.importorskip("torchao")
    from torchtitan.components.quantization import NVFP4Linear

    if NVFP4Linear is None:
        pytest.skip("torchao NVFP4 training prototype not available")
    return NVFP4Linear


def test_nvfp4_infer_tp_style():
    """colwise/rowwise is read back from the weight's declared TP placement."""
    _nvfp4_linear_cls()
    from torchtitan.components.quantization.nvfp4 import _infer_tp_style
    from torchtitan.models.common.decoder_sharding import colwise_config, rowwise_config

    assert _infer_tp_style(colwise_config()) == "colwise"
    assert _infer_tp_style(rowwise_config(output_sp=True)) == "rowwise"
    assert _infer_tp_style(None) is None


@pytest.mark.parametrize("in_features, out_features", [(512, 300), (300, 512)])
def test_nvfp4_config_rejects_non_128_dims(in_features, out_features):
    # The model dims are known at config-build time, so a non-128 in/out_features
    # (e.g. the LM head) is rejected in Config.__post_init__ before any TP.
    NVFP4Linear = _nvfp4_linear_cls()
    with pytest.raises(ValueError, match="divisible by 128"):
        NVFP4Linear.Config(in_features=in_features, out_features=out_features)


def test_nvfp4_build_augments_sharding_config_with_local_map():
    # The parallelize override is gone: Config.build() folds the stock colwise
    # sharding into a local_map region for the opaque nvfp4 GEMM, which base
    # Module.parallelize then consumes directly.
    NVFP4Linear = _nvfp4_linear_cls()
    from torchtitan.models.common.decoder_sharding import colwise_config

    module = NVFP4Linear.Config(
        in_features=512, out_features=1024, sharding_config=colwise_config()
    ).build()
    sc = module._sharding_config
    assert sc.local_map is not None
    assert "x" in (sc.in_dst_shardings or {})
    # The weight placement (colwise) is left untouched by the augmentation.
    assert "weight" in sc.state_shardings


def test_nvfp4_module_exposes_weight_and_two_buffers():
    """Built module has the stock weight param and the two NVFP4 runtime buffers."""
    NVFP4Linear = _nvfp4_linear_cls()
    module = NVFP4Linear.Config(in_features=512, out_features=1024).build()
    assert {name for name, _ in module.named_parameters()} == {"weight"}
    module.init_states()
    buffers = dict(module.named_buffers())
    assert set(buffers) == {"_sr_seed", "_rht_sign_vector"}
    assert buffers["_sr_seed"].dtype == torch.int64
    assert tuple(buffers["_rht_sign_vector"].shape) == (16,)
    # The RHT vector is the fixed v1-recipe constant, identical on every rank.
    from torchtitan.components.quantization.nvfp4 import _HARDCODED_SIGN_VECTOR

    assert tuple(int(v) for v in buffers["_rht_sign_vector"]) == _HARDCODED_SIGN_VECTOR


def test_nvfp4_native_checkpoint_excludes_runtime_buffers():
    """Both NVFP4 runtime buffers are non-persistent -- the RHT vector is a fixed
    constant (recomputed on init) and the SR seed is per-rank -- so a native
    checkpoint carries only the stock weight."""
    NVFP4Linear = _nvfp4_linear_cls()

    def _built():
        module = NVFP4Linear.Config(in_features=512, out_features=1024).build()
        module.init_states()
        return module

    src, dst = _built(), _built()
    sd = src.state_dict()
    assert set(sd) == {"weight"}
    dst.load_state_dict(sd)
    # The fixed RHT constant is identical on both modules regardless of the load.
    assert torch.equal(dst._rht_sign_vector, src._rht_sign_vector)


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
