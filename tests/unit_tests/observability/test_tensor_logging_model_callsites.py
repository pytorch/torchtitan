# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from collections.abc import Callable, Iterable
from unittest import mock

import pytest
import torch
from torch import nn

from torchtitan.models.common.linear import Linear
from torchtitan.models.common.nn_modules import LayerNorm
from torchtitan.models.common.vision_encoder import (
    VisionAttention,
    VisionMLP,
    VisionTransformerBlock,
)
from torchtitan.models.deepseek_v3.config_registry import (
    model_registry as deepseek_model_registry,
)
from torchtitan.models.deepseek_v3.model import Attention as DeepSeekAttention
from torchtitan.models.flux import model_registry as flux_model_registry
from torchtitan.models.gpt_oss.config_registry import (
    model_registry as gpt_oss_model_registry,
)
from torchtitan.models.gpt_oss.model import (
    Attention as GptOssAttention,
    GptOssTransformerBlock,
)
from torchtitan.models.muse_glimmer import model_registry as muse_model_registry
from torchtitan.models.muse_glimmer.model import MuseGlimmerTransformerBlock
from torchtitan.models.qwen3_5.config_registry import (
    model_registry as qwen35_model_registry,
)
from torchtitan.models.qwen3_5.model import GatedDeltaNet, Qwen35Attention
from torchtitan.observability.tensor_logging import init, set_enabled


class _IdentityRope(nn.Module):
    def forward(self, q, k, *_args):
        return q, k


class _AllInputsAttention(nn.Module):
    """Return the value shape while preserving gradients from Q, K, and V."""

    def forward(self, q, k, v, **kwargs):
        output_heads = q.shape[-2]

        def match_heads(value):
            if value.shape[-2] == output_heads:
                return value
            return value.repeat_interleave(output_heads // value.shape[-2], dim=-2)

        value_dim = v.shape[-1]
        output = q[..., :value_dim] + match_heads(k)[..., :value_dim] + match_heads(v)
        if out_transform := kwargs.get("out_transform"):
            output = out_transform(output, torch.zeros_like(output[..., 0]))
        return output


class _DeltaKernel(nn.Module):
    def forward(self, q, k, v, decay, update, **_kwargs):
        value_heads = v.shape[-2]
        q = q.repeat_interleave(value_heads // q.shape[-2], dim=-2)
        k = k.repeat_interleave(value_heads // k.shape[-2], dim=-2)
        return v + q + k + decay.unsqueeze(-1) + update.unsqueeze(-1)


class _Scale(nn.Module):
    def __init__(self, factor: float):
        super().__init__()
        self.factor = factor

    def forward(self, value, *_args, **_kwargs):
        return value * self.factor


def _initialize_parameters(module: nn.Module) -> None:
    for parameter in module.parameters():
        nn.init.uniform_(parameter, -0.02, 0.02)


def _assert_forward_backward_counts(
    module: nn.Module,
    forward: Callable[[], torch.Tensor | tuple[torch.Tensor, ...]],
    expected_base_names: Iterable[str],
) -> None:
    runtime = init(module, device=torch.device("cpu"))
    try:
        with set_enabled(True):
            outputs = forward()
            if isinstance(outputs, torch.Tensor):
                outputs = (outputs,)
            sum(output.float().sum() for output in outputs).backward()

        snapshot = runtime.snapshot_unreduced_statistics()
        expected_names = {
            f"{base_name}.{suffix}"
            for base_name in expected_base_names
            for suffix in ("x", "dx")
        }
        assert set(snapshot) == expected_names
        for name in expected_names:
            assert snapshot[name]["counts"][3].item() == 1
            assert snapshot[name]["counts"][1].item() == 0
    finally:
        runtime.close()


@pytest.mark.parametrize("family", ["qwen35", "deepseek", "gpt_oss"])
def test_attention_families_record_forward_and_backward(family: str) -> None:
    if family == "qwen35":
        config = qwen35_model_registry("debugmodel_moe").model.layers[3].attention
        assert isinstance(config, Qwen35Attention.Config)
        attention = config.build()
        expected_names = (
            "xq",
            "xk",
            "xv",
            "xq_normed",
            "xk_normed",
            "output_gate",
            "head_out_pre_gate",
            "head_out",
        )
    elif family == "deepseek":
        config = deepseek_model_registry("debugmodel").model.layers[0].attention
        assert isinstance(config, DeepSeekAttention.Config)
        attention = config.build()
        expected_names = ("xq", "xk", "xv", "head_out")
    else:
        config = (
            gpt_oss_model_registry("debugmodel", attn_backend="flex")
            .model.layers[0]
            .attention
        )
        assert isinstance(config, GptOssAttention.Config)
        attention = config.build()
        expected_names = ("xq", "xk", "xv", "head_out")

    attention.rope = _IdentityRope()
    attention.inner_attention = _AllInputsAttention()
    _initialize_parameters(attention)
    hidden = torch.randn(1, 4, 256, requires_grad=True)
    _assert_forward_backward_counts(
        attention,
        lambda: attention(hidden, None, None),
        expected_names,
    )


def test_gated_delta_net_records_final_head_output() -> None:
    config = qwen35_model_registry("debugmodel_moe").model.layers[0].delta_net
    assert isinstance(config, GatedDeltaNet.Config)
    delta_net = config.build()
    delta_net.kernel = _DeltaKernel()
    _initialize_parameters(delta_net)
    hidden = torch.randn(1, 4, 256, requires_grad=True)
    _assert_forward_backward_counts(
        delta_net,
        lambda: delta_net(hidden),
        ("head_out",),
    )


@pytest.mark.parametrize("family", ["gpt_oss", "muse"])
def test_text_blocks_record_stream_and_branch_boundaries(family: str) -> None:
    if family == "gpt_oss":
        config = gpt_oss_model_registry("debugmodel", attn_backend="flex").model.layers[
            0
        ]
        block = config.build()
        assert isinstance(block, GptOssTransformerBlock)
        block.attention = _Scale(0.5)
        block.attention_norm = nn.Identity()
        block.ffn_norm = nn.Identity()
        block.moe = _Scale(0.25)
    else:
        config = muse_model_registry("debugmodel_mm").model.layers[0]
        block = config.build()
        assert isinstance(block, MuseGlimmerTransformerBlock)
        block.attention = _Scale(0.5)
        block.attention_norm = nn.Identity()
        block.post_attention_norm = nn.Identity()
        block.ffn_norm = nn.Identity()
        block.feed_forward = _Scale(0.25)
        block.post_ffn_norm = nn.Identity()

    hidden = torch.randn(1, 4, 256, requires_grad=True)
    _assert_forward_backward_counts(
        block,
        lambda: block(hidden, None, None),
        ("attn_stream", "attn_out", "ffn_stream", "ffn_out"),
    )


def test_shared_vision_block_records_attention_mlp_and_residuals() -> None:
    dim = 8
    linear = lambda in_features, out_features: Linear.Config(
        in_features=in_features,
        out_features=out_features,
        bias=False,
    )
    block = VisionTransformerBlock.Config(
        norm1=LayerNorm.Config(normalized_shape=dim),
        norm2=LayerNorm.Config(normalized_shape=dim),
        attn=VisionAttention.Config(
            dim=dim,
            num_heads=2,
            wq=linear(dim, dim),
            wk=linear(dim, dim),
            wv=linear(dim, dim),
            proj=linear(dim, dim),
        ),
        mlp=VisionMLP.Config(
            fc1=linear(dim, 2 * dim),
            fc2=linear(2 * dim, dim),
        ),
    ).build()
    block.attn.flex_attention = _AllInputsAttention()
    _initialize_parameters(block)
    hidden = torch.randn(2, 3, dim, requires_grad=True)
    _assert_forward_backward_counts(
        block,
        lambda: block(
            hidden,
            rope_cache=torch.empty(0),
            rope_apply=lambda q, k, _cache: (q, k),
            attention_mask=None,
        ),
        (
            "attn.xq",
            "attn.xk",
            "attn.xv",
            "attn.head_out",
            "mlp.act_out",
            "post_ln1",
            "post_attn",
            "post_attn_residual",
            "post_ln2",
            "post_mlp",
            "post_mlp_residual",
        ),
    )


@pytest.mark.parametrize("stream", ["double", "single"])
def test_flux_blocks_record_architecture_honest_boundaries(stream: str) -> None:
    model_config = flux_model_registry("flux-debug").model
    if stream == "double":
        block = model_config.double_blocks[0].build()
        expected_names = (
            "img_attn_stream",
            "img_attn_out",
            "img_ffn_stream",
            "img_ffn_out",
            "txt_attn_stream",
            "txt_attn_out",
            "txt_ffn_stream",
            "txt_ffn_out",
        )
    else:
        block = model_config.single_blocks[0].build()
        expected_names = ("stream", "parallel_branch_out")

    block.inner_attention = _AllInputsAttention()
    _initialize_parameters(block)
    hidden_size = block.hidden_size
    vector_size = (
        block.modulation.lin.in_features
        if stream == "single"
        else block.img_mod.lin.in_features
    )
    vector = torch.randn(1, vector_size, requires_grad=True)

    with mock.patch(
        "torchtitan.models.flux.model.layers.apply_rope",
        side_effect=lambda q, k, _pe: (q, k),
    ):
        if stream == "double":
            image = torch.randn(1, 2, hidden_size, requires_grad=True)
            text = torch.randn(1, 3, hidden_size, requires_grad=True)
            forward = lambda: block(image, text, vector, torch.empty(0))
        else:
            hidden = torch.randn(1, 5, hidden_size, requires_grad=True)
            forward = lambda: block(hidden, vector, torch.empty(0))
        _assert_forward_backward_counts(block, forward, expected_names)
