# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import io
from copy import deepcopy
from dataclasses import dataclass
from functools import partial

import pytest
import torch
import torch.nn.functional as F

from torchtitan.config import ConfigLoader
from torchtitan.config.parallelism import ParallelismConfig
from torchtitan.distributed.activation_remat import (
    DefaultActivationRematConfig,
    FullActivationRematConfig,
)
from torchtitan.experiments.peri_ln.model import (
    apply_peri_ln,
    PeriLNModel,
    PeriLNTransformerBlock,
)
from torchtitan.models.common import ComplexRoPE, Embedding, Linear, RMSNorm
from torchtitan.models.common.attention import InnerAttention
from torchtitan.models.common.config_utils import make_ffn_config, make_gqa_config
from torchtitan.models.llama3 import build_model_config
from torchtitan.models.llama3.model import Llama3Model, Llama3TransformerBlock


class CPUAttention(InnerAttention):
    """Use PyTorch SDPA for small, single-document CPU numerical tests."""

    @dataclass(kw_only=True, slots=True)
    class Config(InnerAttention.Config):
        pass

    def __init__(self, config: Config):
        super().__init__()

    def forward(
        self,
        q_THK,
        k_THK,
        v_THV,
        *,
        attention_metadata,
        scale=None,
        enable_gqa=False,
    ):
        assert attention_metadata is None
        return F.scaled_dot_product_attention(
            q_THK.transpose(0, 1).unsqueeze(0),
            k_THK.transpose(0, 1).unsqueeze(0),
            v_THV.transpose(0, 1).unsqueeze(0),
            is_causal=True,
            scale=scale,
            enable_gqa=enable_gqa,
        ).squeeze(0).transpose(0, 1)


def _model_config() -> Llama3Model.Config:
    dim, vocab_size, seq_len = 16, 32, 8
    linear_init = {"weight": partial(torch.nn.init.normal_, std=0.1)}
    norm = RMSNorm.Config(
        normalized_shape=dim,
        eps=1e-5,
        param_init={"weight": torch.nn.init.ones_},
    )
    layer = Llama3TransformerBlock.Config(
        attention=make_gqa_config(
            dim=dim,
            n_heads=2,
            n_kv_heads=1,
            wqkv_param_init=linear_init,
            wo_param_init=linear_init,
            inner_attention=CPUAttention.Config(),
            rope=ComplexRoPE.Config(dim=8, max_context_length=seq_len),
            enable_sp=False,
        ),
        feed_forward=make_ffn_config(
            dim=dim,
            hidden_dim=32,
            w13_param_init=linear_init,
            w2_param_init=linear_init,
            enable_sp=False,
        ),
        attention_norm=deepcopy(norm),
        ffn_norm=deepcopy(norm),
    )
    return Llama3Model.Config(
        dim=dim,
        vocab_size=vocab_size,
        max_context_length=seq_len,
        tok_embeddings=Embedding.Config(
            num_embeddings=vocab_size, embedding_dim=dim
        ),
        lm_head=Linear.Config(
            in_features=dim, out_features=vocab_size, param_init=linear_init
        ),
        norm=norm,
        layers=[layer, deepcopy(layer)],
        local_compile_regions=[],
    )


def _rms_norm(x, norm):
    return x * torch.rsqrt(x.square().mean(dim=-1, keepdim=True) + norm.eps) * norm.weight


def test_forward_and_backward_match_peri_ln_equation():
    torch.manual_seed(42)
    block = apply_peri_ln(_model_config()).layers[0].build()
    block.init_states()
    for norm in (
        block.attention_norm,
        block.attention_output_norm,
        block.ffn_norm,
        block.ffn_output_norm,
    ):
        torch.nn.init.uniform_(norm.weight, 0.5, 1.5)
    reference = deepcopy(block)
    x = torch.randn(8, 16, requires_grad=True)
    reference_x = x.detach().clone().requires_grad_()
    positions = torch.arange(8)

    actual = block(x, None, positions)
    attention = reference.attention(
        _rms_norm(reference_x, reference.attention_norm), None, positions
    )
    hidden = reference_x + _rms_norm(attention, reference.attention_output_norm)
    ffn = reference.feed_forward(_rms_norm(hidden, reference.ffn_norm))
    expected = hidden + _rms_norm(ffn, reference.ffn_output_norm)
    torch.testing.assert_close(actual, expected)

    grad_output = torch.randn_like(actual)
    actual.backward(grad_output)
    expected.backward(grad_output)
    torch.testing.assert_close(x.grad, reference_x.grad)
    for (name, parameter), (reference_name, reference_parameter) in zip(
        block.named_parameters(), reference.named_parameters(), strict=True
    ):
        assert name == reference_name
        assert parameter.grad is not None, name
        assert torch.isfinite(parameter.grad).all(), name
        torch.testing.assert_close(parameter.grad, reference_parameter.grad)


def test_zero_branches_preserve_residual_and_identity_gradient():
    block = apply_peri_ln(_model_config()).layers[0].build()
    block.init_states()
    torch.nn.init.zeros_(block.attention.wo.weight)
    torch.nn.init.zeros_(block.feed_forward.w2.weight)
    x = (10 * torch.randn(8, 16)).requires_grad_()
    actual = block(x, None, torch.arange(8))
    torch.testing.assert_close(actual, x, rtol=0, atol=0)
    actual.sum().backward()
    torch.testing.assert_close(x.grad, torch.ones_like(x), rtol=0, atol=0)


@pytest.mark.parametrize("enable_sp", [False, True])
def test_conversion_preserves_sharding_without_aliasing(enable_sp):
    base = build_model_config(
        "debugmodel",
        seq_len=8,
        attn_backend="flex",
        parallelism=ParallelismConfig(enable_sequence_parallel=enable_sp),
    )
    converted = apply_peri_ln(base)
    assert isinstance(converted, PeriLNModel.Config)
    assert type(base) is Llama3Model.Config
    for original, layer in zip(base.layers, converted.layers, strict=True):
        assert type(original) is Llama3TransformerBlock.Config
        assert isinstance(layer, PeriLNTransformerBlock.Config)
        for input_name, output_name in (
            ("attention_norm", "attention_output_norm"),
            ("ffn_norm", "ffn_output_norm"),
        ):
            original_norm = getattr(original, input_name)
            input_norm = getattr(layer, input_name)
            output_norm = getattr(layer, output_name)
            assert output_norm == original_norm
            assert output_norm is not input_norm
            assert output_norm is not original_norm
            assert output_norm.sharding_config is not input_norm.sharding_config
            output_norm.eps = 0.01
            assert original_norm.eps == input_norm.eps == 1e-5
    converted.layers[0].attention.dim = 99
    assert base.layers[0].attention.dim == 256


@pytest.mark.parametrize("enable_weight_tying", [False, True])
def test_meta_initialization_and_native_checkpoint_roundtrip(enable_weight_tying):
    base = _model_config()
    base.enable_weight_tying = enable_weight_tying
    config = apply_peri_ln(base)
    with torch.device("meta"):
        model = config.build()
    model.to_empty(device="cpu")
    model.init_states(buffer_device=torch.device("cpu"))
    assert isinstance(model, PeriLNModel)
    assert model.state_dict_adapter_cls is None
    if enable_weight_tying:
        assert model.tok_embeddings.weight is model.lm_head.weight

    added_keys = {
        f"layers.{index}.{norm}.weight"
        for index in range(len(config.layers))
        for norm in ("attention_output_norm", "ffn_output_norm")
    }
    for key in added_keys:
        torch.testing.assert_close(model.state_dict()[key], torch.ones(base.dim))

    original = base.build()
    original.init_states()
    assert set(model.state_dict()) - set(original.state_dict()) == added_keys
    assert set(original.state_dict()) <= set(model.state_dict())
    with pytest.raises(RuntimeError, match="Missing key"):
        model.load_state_dict(original.state_dict(), strict=True)

    for key in added_keys:
        torch.nn.init.uniform_(model.get_parameter(key), 0.5, 1.5)
    checkpoint = io.BytesIO()
    torch.save(model.state_dict(), checkpoint)
    checkpoint.seek(0)
    restored = config.build()
    restored.init_states()
    restored.load_state_dict(torch.load(checkpoint, weights_only=True), strict=True)
    tokens = torch.arange(8)
    positions = torch.arange(8)
    torch.testing.assert_close(
        restored(tokens, positions), model(tokens, positions), rtol=0, atol=0
    )


@pytest.mark.parametrize(
    "policy", [DefaultActivationRematConfig, FullActivationRematConfig]
)
def test_remat_preserves_outputs_and_gradients(policy):
    torch.manual_seed(42)
    reference = apply_peri_ln(_model_config()).build()
    reference.init_states()
    checkpointed = deepcopy(reference)
    policy().apply(checkpointed)
    tokens = torch.arange(8)
    positions = torch.arange(8)
    expected = reference(tokens, positions)
    actual = checkpointed(tokens, positions)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    expected.square().mean().backward()
    actual.square().mean().backward()
    for (name, parameter), (expected_name, expected_parameter) in zip(
        checkpointed.named_parameters(), reference.named_parameters(), strict=True
    ):
        assert name == expected_name
        assert parameter.grad is not None, name
        torch.testing.assert_close(
            parameter.grad, expected_parameter.grad, rtol=0, atol=0
        )


def test_cli_selects_peri_ln_debug_recipe():
    config = ConfigLoader().load(
        [
            "--module",
            "torchtitan.experiments.peri_ln.recipes",
            "--config",
            "llama3_debugmodel",
        ]
    )
    assert isinstance(config.model, PeriLNModel.Config)
    assert all(
        isinstance(layer, PeriLNTransformerBlock.Config)
        for layer in config.model.layers
    )
    assert config.training.steps == 10
    assert config.debug.seed == 42
    assert config.debug.deterministic
    assert not config.debug.deterministic_warn_only


def test_gpu_recipes_select_peri_ln_and_requested_parallelism():
    from .gpu_configs import single_gpu, tensor_parallel

    for recipe, degree in ((single_gpu, 1), (tensor_parallel, 2)):
        config = recipe()
        assert isinstance(config.model, PeriLNModel.Config)
        assert all(
            isinstance(layer, PeriLNTransformerBlock.Config)
            for layer in config.model.layers
        )
        assert config.parallelism.tensor_parallel_degree == degree
        assert config.parallelism.enable_sequence_parallel
        assert config.training.steps == 10
    assert isinstance(tensor_parallel().activation_remat, FullActivationRematConfig)
