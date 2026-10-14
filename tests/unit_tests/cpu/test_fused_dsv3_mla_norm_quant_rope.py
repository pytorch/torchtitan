# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from unittest.mock import patch

import pytest
import torch
from torch import nn
from torch._subclasses.fake_tensor import FakeTensorMode

from torchtitan.config import apply_overrides, OverrideConfig
from torchtitan.config.override import _REGISTRY
from torchtitan.models.common.linear import Linear
from torchtitan.models.common.nn_modules import RMSNorm
from torchtitan.models.common.rope import ComplexRoPE
from torchtitan.models.deepseek_v3 import build_model_config
from torchtitan.models.deepseek_v3.model import Attention
from torchtitan_recipes.overrides import fused_dsv3_mla_norm_quant_rope as fusion


_OVERRIDE = "torchtitan_recipes.overrides.fused_dsv3_mla_norm_quant_rope.fused_mla_norm_quant_rope"
_REGISTERED_OVERRIDES = {
    key: value
    for key, value in _REGISTRY.items()
    if key.startswith("torchtitan_recipes.overrides.fused_dsv3_")
}


@pytest.fixture(autouse=True)
def _registered_overrides():
    with patch.dict(_REGISTRY, _REGISTERED_OVERRIDES):
        yield


def test_model_override_keeps_production_shapes_and_checkpoint_config():
    model = build_model_config("671B", seq_len=4096)
    native = model.layers[3].attention
    apply_overrides(
        OverrideConfig(imports=[(_OVERRIDE, {"experimental": True})]), model
    )
    candidate = model.layers[3].attention
    assert type(candidate) is fusion.FusedDSv3MLANormQuantRoPE.Config
    assert candidate.mla_nqr_stages == fusion._STAGES
    for field in (
        "dim",
        "n_heads",
        "q_lora_rank",
        "kv_lora_rank",
        "qk_nope_head_dim",
        "qk_rope_head_dim",
        "v_head_dim",
        "wq_a",
        "wq_b",
        "wkv_a",
        "wkv_b",
        "wo",
        "q_norm",
        "kv_norm",
        "rope",
        "inner_attention",
    ):
        assert getattr(candidate, field) == getattr(native, field)
    assert (
        candidate.dim,
        candidate.n_heads,
        candidate.q_lora_rank,
        candidate.kv_lora_rank,
    ) == (7168, 128, 1536, 512)


def test_stage_options_are_local_to_each_config_and_acceptance_is_explicit(caplog):
    cfg = build_model_config("671B", seq_len=4096).layers[3].attention
    default = fusion.fused_mla_norm_quant_rope(cfg)
    first = fusion.fused_mla_norm_quant_rope(
        cfg, stages=("norm_quant",), experimental=True
    )
    second = fusion.fused_mla_norm_quant_rope(
        cfg, stages=("q_bwd", "kv_bwd"), experimental=True
    )
    assert not fusion.ACCEPTED
    assert default.mla_nqr_stages == ()
    assert first.mla_nqr_stages == ("norm_quant",)
    assert second.mla_nqr_stages == ("q_bwd", "kv_bwd")
    assert "has not met all roofline gates" in caplog.text
    with pytest.raises(ValueError, match="Unknown MLA fusion stages"):
        fusion.fused_mla_norm_quant_rope(cfg, stages=("invalid",), experimental=True)


class _AttentionProduct(nn.Module):
    def forward(self, q, k, v, **kwargs):
        return v * (q[..., :4] + k[..., :4])


def test_cpu_fallback_preserves_forward_backward_and_state():
    torch.manual_seed(42)
    cfg = Attention.Config(
        dim=16,
        n_heads=2,
        q_lora_rank=8,
        kv_lora_rank=4,
        qk_nope_head_dim=4,
        qk_rope_head_dim=4,
        v_head_dim=4,
        wq_a=Linear.Config(in_features=16, out_features=8),
        wq_b=Linear.Config(in_features=8, out_features=16),
        wkv_a=Linear.Config(in_features=16, out_features=8),
        wkv_b=Linear.Config(in_features=4, out_features=16),
        wo=Linear.Config(in_features=8, out_features=16),
        q_norm=RMSNorm.Config(normalized_shape=8),
        kv_norm=RMSNorm.Config(normalized_shape=4),
        rope=ComplexRoPE.Config(dim=4, max_context_length=8),
    )
    modules = [
        cfg.build(),
        fusion.fused_mla_norm_quant_rope(cfg, experimental=True).build(),
    ]
    for module in modules:
        module.init_states(buffer_device=torch.device("cpu"))
        module.inner_attention = _AttentionProduct()
    modules[1].load_state_dict(modules[0].state_dict())
    x = torch.randn(8, 16)
    dy = torch.randn_like(x)
    results = []
    with patch.object(
        fusion.FusedMLAQChainFunction,
        "apply",
        side_effect=AssertionError("CPU requires native attention"),
    ):
        for module in modules:
            value = x.clone().requires_grad_()
            output = module(value, None)
            gradients = torch.autograd.grad(output, (value, *module.parameters()), dy)
            results.append((output, *gradients))
    assert modules[0].state_dict().keys() == modules[1].state_dict().keys()
    for native, fused in zip(*results):
        assert torch.equal(
            native.detach().contiguous().view(torch.uint8),
            fused.detach().contiguous().view(torch.uint8),
        )


def test_fake_operator_metadata_preserves_payloads_scales_and_packed_views():
    with FakeTensorMode():
        latent = torch.empty(1, 4096, 1536, dtype=torch.bfloat16, device="cuda")
        weight = torch.empty(1536, dtype=torch.bfloat16, device="cuda")
        rstd, row, column, row_scale, column_scale = fusion.norm_quant_op(
            latent, weight, 1e-5, True
        )
        for input_grad, weight_grad in ((True, True), (False, True), (True, False)):
            dx, dw = fusion.norm_backward_op(
                latent, latent, rstd, weight, input_grad, weight_grad
            )
            assert (dx is not None) == input_grad
            assert (dw is not None) == weight_grad
            if dx is not None:
                assert dx.shape == latent.shape and dx.dtype == latent.dtype
            if dw is not None:
                assert dw.shape == weight.shape and dw.dtype == weight.dtype
        assert (rstd.shape, rstd.dtype) == ((1, 4096, 1), torch.float32)
        assert (row.shape, row.stride(), row.dtype) == (
            (4096, 1536),
            (1536, 1),
            torch.float8_e4m3fn,
        )
        assert column.stride() == (1, 4096)
        assert row_scale.shape == column_scale.shape == (4096 * 1536 // 32,)
        assert row_scale.dtype == column_scale.dtype == torch.float8_e8m0fnu
        cache = torch.empty(4096, 32, 2, device="cuda")
        positions = torch.empty(1, 4096, dtype=torch.int32, device="cuda")
        q_weight = torch.empty(24576, 1536, dtype=torch.float8_e4m3fn, device="cuda")
        q = fusion.q_up_rope_op(row, row_scale, q_weight, row_scale, cache, positions)
        assert q.shape == (1, 4096, 24576)
        q_grad = fusion.q_backward_quant_op(q.view(1, 4096, 128, 192), cache, positions)
        assert q_grad[0].shape == q_grad[1].shape == (4096, 24576)
        assert q_grad[1].stride() == (1, 4096)
        kv_weight = torch.empty(32768, 512, dtype=torch.float8_e4m3fn, device="cuda")
        k_pe = torch.empty(1, 4096, 64, dtype=torch.bfloat16, device="cuda")
        kv, k = fusion.kv_up_rope_op(
            row[:, :512], row_scale, kv_weight, row_scale, k_pe, cache, positions
        )
        assert kv.shape == (1, 4096, 128, 256)
        assert k.shape == (1, 4096, 128, 192)
        grad = fusion.kv_backward_quant_op(k, kv[..., 128:], cache, positions)
        assert grad[0].shape == grad[1].shape == (4096, 32768)
        assert grad[-1].shape == (1, 4096, 64)
