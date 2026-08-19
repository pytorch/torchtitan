# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from collections.abc import Callable, Iterable
from types import SimpleNamespace
from unittest import mock

import pytest
import torch
from torch import nn

from torchtitan.models.common.decoder import Decoder
from torchtitan.models.common.dist_gemm import AllGatherFusedFeedForward
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
from torchtitan.models.deepseek_v3.model import (
    Attention as DeepSeekAttention,
    DeepSeekV3Model,
)
from torchtitan.models.deepseek_v3.mtp import MTPDecoder
from torchtitan.models.flux import model_registry as flux_model_registry
from torchtitan.models.gpt_oss.config_registry import (
    model_registry as gpt_oss_model_registry,
)
from torchtitan.models.gpt_oss.model import (
    Attention as GptOssAttention,
    GptOssTransformerBlock,
)
from torchtitan.models.kimi_k2_7.model import KimiK25Model
from torchtitan.models.muse_glimmer import model_registry as muse_model_registry
from torchtitan.models.muse_glimmer.model import (
    MuseGlimmerModel,
    MuseGlimmerTransformerBlock,
)
from torchtitan.models.qwen3_5.config_registry import (
    model_registry as qwen35_model_registry,
)
from torchtitan.models.qwen3_5.model import GatedDeltaNet, Qwen35Attention, Qwen35Model
from torchtitan.observability.tensor_logging import init, register_fwd_bwd, set_enabled
from torchtitan.overrides.fused_swiglu import FusedSwiGLU


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


class _Builds:
    def __init__(self, module: nn.Module, **attributes) -> None:
        self.module = module
        for name, value in attributes.items():
            setattr(self, name, value)

    def build(self) -> nn.Module:
        return self.module


class _VisionEncoderStub(nn.Module):
    """Preserve padded patch shape while exposing model-required attributes."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.patch_embed = nn.Linear(dim, dim, bias=False)
        self.spatial_merge_unit = 1
        self.merge_kernel_size = (1, 1)

    def forward(self, pixel_values, *, grid_thw):
        return pixel_values


class _DecoderLayerStub(nn.Module):
    def forward(self, hidden, *_args):
        return hidden * 1.1


class _MTPLayerStub(nn.Module):
    def forward(self, input_embed, previous_hidden, *_args):
        return input_embed + previous_hidden


def _initialize_parameters(module: nn.Module) -> None:
    for parameter in module.parameters():
        nn.init.uniform_(parameter, -0.02, 0.02)


def _assert_forward_backward_counts(
    module: nn.Module,
    forward: Callable[[], torch.Tensor | tuple[torch.Tensor, ...]],
    expected_base_names: Iterable[str],
    *,
    exact_names: bool = True,
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
        if exact_names:
            assert set(snapshot) == expected_names
        else:
            assert expected_names <= set(snapshot)
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


@pytest.mark.parametrize("family", ["qwen35", "kimi", "muse"])
def test_multimodal_models_record_projected_vision_boundary(family: str) -> None:
    dim = 8
    vision_encoder = _VisionEncoderStub(dim)

    if family == "qwen35":
        config = SimpleNamespace(
            vision_encoder=_Builds(
                vision_encoder,
                spatial_merge_size=1,
            )
        )
        with mock.patch.object(
            Decoder,
            "__init__",
            lambda self, _config: nn.Module.__init__(self),
        ):
            model = Qwen35Model(config)
        pixels = torch.randn(1, 2, dim, requires_grad=True)
        grid = torch.tensor([[1, 1, 2]])
        forward = lambda: model._get_vision_embeds(pixels, grid_thw=grid)[0]
    elif family == "kimi":
        config = SimpleNamespace(vision_encoder=_Builds(vision_encoder))

        def initialize_kimi_base(self, _config) -> None:
            nn.Module.__init__(self)
            # Preserve the inherited input metric emitted by this real method.
            register_fwd_bwd(self, ["input"])

        with mock.patch.object(DeepSeekV3Model, "__init__", initialize_kimi_base):
            model = KimiK25Model(config)
        model.tok_embeddings = nn.Embedding(16, dim)
        pixels = torch.randn(1, 1, dim, requires_grad=True)
        grid = torch.tensor([[1, 1, 1]])
        tokens = torch.tensor([[9, 1]])
        forward = lambda: model._prepare_multimodal_embeds(
            tokens,
            pixel_values=pixels,
            grid_thw=grid,
            special_tokens={"image_id": 9, "video_id": 9},
        )
    else:
        config = SimpleNamespace(
            vision_projection=_Builds(nn.Linear(dim, dim, bias=False)),
            perception_emb_norm=_Builds(nn.Identity()),
            vision_encoder=None,
            vision_adapter=None,
        )
        with mock.patch.object(
            Decoder,
            "__init__",
            lambda self, _config: nn.Module.__init__(self),
        ):
            model = MuseGlimmerModel(config)
        hidden = torch.zeros(1, 2, dim)
        vision_features = torch.randn(1, 1, dim, requires_grad=True)
        vision_mask = torch.tensor([[True, False]])
        forward = lambda: model._inject_vision(
            hidden,
            vision_features,
            vision_mask,
        )

    _assert_forward_backward_counts(
        model,
        forward,
        ("vision_embeddings_after_projection",),
        exact_names=False,
    )


def test_shared_vision_padding_is_included_in_raw_statistics() -> None:
    dim = 8
    linear = lambda: Linear.Config(in_features=dim, out_features=dim, bias=False)
    attention = VisionAttention.Config(
        dim=dim,
        num_heads=2,
        wq=linear(),
        wk=linear(),
        wv=linear(),
        proj=linear(),
    ).build()
    attention.flex_attention = _AllInputsAttention()
    _initialize_parameters(attention)

    # Item 0 has three patches; item 1 has two plus one stored padding row.
    hidden = torch.randn(2, 3, dim)
    hidden[1, 2].zero_()
    hidden.requires_grad_()
    runtime = init(attention, device=torch.device("cpu"))
    try:
        with set_enabled(True):
            output = attention(
                hidden,
                rope_cache=torch.empty(0),
                rope_apply=lambda q, k, _cache: (q, k),
                attention_mask=None,
            )
            output.sum().backward()

        snapshot = runtime.snapshot_unreduced_statistics()
        for name in ("xq", "xk", "xv", "head_out"):
            assert snapshot[f"{name}.x"]["counts"][2].item() == dim
    finally:
        runtime.close()


@pytest.mark.parametrize("family", ["qwen35", "kimi", "muse"])
@pytest.mark.parametrize("skip_lm_head", [False, True])
def test_custom_decoders_record_inherited_input_and_head_metrics(
    family: str,
    skip_lm_head: bool,
) -> None:
    dim = 8

    def initialize_decoder_base(self, _config) -> None:
        nn.Module.__init__(self)
        register_fwd_bwd(self, ["input"])

    if family == "qwen35":
        config = SimpleNamespace(
            vision_encoder=_Builds(
                _VisionEncoderStub(dim),
                spatial_merge_size=1,
            )
        )
        with mock.patch.object(Decoder, "__init__", initialize_decoder_base):
            model = Qwen35Model(config)
        forward = lambda tokens: model(
            tokens,
            positions=torch.arange(tokens.shape[1]).unsqueeze(0),
            special_tokens={"image_id": 9, "video_id": 10},
        )
    elif family == "kimi":
        config = SimpleNamespace(vision_encoder=None)
        with mock.patch.object(
            DeepSeekV3Model,
            "__init__",
            initialize_decoder_base,
        ):
            model = KimiK25Model(config)
        forward = lambda tokens: model(tokens, positions=None, special_tokens={})
    else:
        config = SimpleNamespace(
            vision_projection=None,
            perception_emb_norm=None,
            vision_encoder=None,
            vision_adapter=None,
        )
        with mock.patch.object(Decoder, "__init__", initialize_decoder_base):
            model = MuseGlimmerModel(config)
        forward = lambda tokens: model(tokens)

    model.tok_embeddings = nn.Embedding(16, dim)
    model.layers = nn.ModuleDict()
    model.norm = nn.Identity()
    model.lm_head = nn.Linear(dim, 16, bias=False)
    register_fwd_bwd(model.lm_head, ["output"])
    model._skip_lm_head = skip_lm_head

    runtime = init(model, device=torch.device("cpu"))
    try:
        with set_enabled(True):
            output = forward(torch.tensor([[1, 2, 3]]))
            output.float().sum().backward()

        snapshot = runtime.snapshot_unreduced_statistics()
        assert snapshot["input.x"]["counts"][3].item() == 1
        assert snapshot["input.dx"]["counts"][3].item() == 1
        expected_head_count = 0 if skip_lm_head else 1
        assert snapshot["lm_head.output.x"]["counts"][3].item() == expected_head_count
        assert snapshot["lm_head.output.dx"]["counts"][3].item() == expected_head_count
    finally:
        runtime.close()


def test_mtp_decoder_records_every_head_prediction() -> None:
    dim = 8
    model = MTPDecoder.__new__(MTPDecoder)
    nn.Module.__init__(model)
    register_fwd_bwd(model, ["input"])
    model.tok_embeddings = nn.Embedding(16, dim)
    model.layers = nn.ModuleDict({"0": _DecoderLayerStub()})
    model.norm = nn.Identity()
    model.mtp_layers = nn.ModuleList([_MTPLayerStub(), _MTPLayerStub()])
    model.lm_head = nn.Linear(dim, 16, bias=False)
    register_fwd_bwd(model.lm_head, ["output"])
    model._skip_lm_head = False

    runtime = init(model, device=torch.device("cpu"))
    try:
        with set_enabled(True):
            outputs = model(torch.tensor([[1, 2, 3, 4]]))
            sum(output.float().sum() for output in outputs).backward()

        snapshot = runtime.snapshot_unreduced_statistics()
        assert snapshot["input.x"]["counts"][3].item() == 1
        assert snapshot["input.dx"]["counts"][3].item() == 1
        assert snapshot["lm_head.output.x"]["counts"][3].item() == 3
        assert snapshot["lm_head.output.dx"]["counts"][3].item() == 3
    finally:
        runtime.close()


@pytest.mark.parametrize("implementation", ["dist_gemm", "fused_swiglu"])
def test_feed_forward_overrides_record_activation_forward_and_backward(
    implementation: str,
) -> None:
    dim = 8
    hidden_dim = 16
    config_type = (
        AllGatherFusedFeedForward.Config
        if implementation == "dist_gemm"
        else FusedSwiGLU.Config
    )
    module = config_type(
        w1=Linear.Config(in_features=dim, out_features=hidden_dim, bias=False),
        w2=Linear.Config(in_features=hidden_dim, out_features=dim, bias=False),
        w3=Linear.Config(in_features=dim, out_features=hidden_dim, bias=False),
    ).build()
    _initialize_parameters(module)
    hidden = torch.randn(2, 3, dim, requires_grad=True)

    if implementation == "dist_gemm":
        tp_group = SimpleNamespace(group_name="tp")

        def gather(input_value, gate_weight, up_weight, *_args):
            return input_value @ gate_weight.T, input_value @ up_weight.T

        def reduce_scatter(input_value, weight, bias, *_args):
            output = input_value @ weight.T
            return output if bias is None else output + bias

        contexts = (
            mock.patch(
                "torchtitan.models.common.dist_gemm._tp_group_from_context",
                return_value=tp_group,
            ),
            mock.patch(
                "torchtitan.models.common.dist_gemm.AllGatherLinearMulti.apply",
                side_effect=gather,
            ),
            mock.patch(
                "torchtitan.models.common.dist_gemm.LinearReduceScatter.apply",
                side_effect=reduce_scatter,
            ),
        )
    else:
        contexts = (
            mock.patch(
                "torchtitan.overrides.fused_swiglu._fused_silu_and_mul",
                side_effect=lambda gate, up: torch.nn.functional.silu(gate) * up,
            ),
        )

    with contexts[0]:
        if len(contexts) == 1:
            _assert_forward_backward_counts(
                module, lambda: module(hidden), ("act_out",)
            )
        else:
            with contexts[1], contexts[2]:
                _assert_forward_backward_counts(
                    module,
                    lambda: module(hidden),
                    ("act_out",),
                )


@pytest.mark.parametrize("stream", ["double", "single"])
def test_flux_blocks_record_architecture_honest_boundaries(stream: str) -> None:
    model_config = flux_model_registry("flux-debug").model
    if stream == "double":
        block = model_config.double_blocks[0].build()
        expected_names = (
            "img_attn_stream",
            "img_attn_branch",
            "img_ffn_stream",
            "img_ffn_branch",
            "txt_attn_stream",
            "txt_attn_branch",
            "txt_ffn_stream",
            "txt_ffn_branch",
        )
    else:
        block = model_config.single_blocks[0].build()
        expected_names = ("stream", "parallel_branch")

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
