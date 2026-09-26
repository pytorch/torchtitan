# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest

from collections.abc import Callable
from copy import deepcopy
from dataclasses import fields, MISSING
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

import torch
import torch_remat as remat

from torchtitan.config.transform import AsyncTensorParallelTransform
from torchtitan.distributed.activation_checkpoint import RegionAC
from torchtitan.models.common.activation import BinaryActivationFn, Sigmoid, SwiGLU
from torchtitan.models.common.attention import GQAttention
from torchtitan.models.common.feed_forward import FeedForward
from torchtitan.models.common.linear import (
    ColumnParallelLinear,
    GroupedLinear,
    Linear,
    RouterGateLinear,
    RowParallelLinear,
)
from torchtitan.models.common.moe import (
    MoE,
    QuantileBalancedTopKRouter,
    RoutedExperts,
    TokenChoiceTopKRouter,
)
from torchtitan.models.common.token_dispatcher import LocalTokenDispatcher
from torchtitan.models.common.vision_encoder import (
    VisionAttention,
    VisionMLP,
    VisionTransformerBlock,
)
from torchtitan.models.deepseek_v3.model import Attention as DeepSeekV3Attention
from torchtitan.models.gpt_oss.moe import GptOssGroupedLinear, GptOssSwiGLU
from torchtitan.models.kimi_k3.kda import KDA
from torchtitan.models.kimi_k3.model import KimiMLAAttention
from torchtitan.models.kimi_k3.moe import KimiLatentMoE
from torchtitan.models.muse_glimmer.model import Attention as MuseGlimmerAttention
from torchtitan.models.qwen3_5.gdn import GatedDeltaNet
from torchtitan.models.qwen3_5.model import Qwen35Attention
from torchtitan.overrides.fused_swiglu import fused_swiglu, FusedSwiGLU
from torchtitan.protocols.module import Module, ModuleDict


class _CountingOp(Module):
    def __init__(self, operation: Callable[..., Any]):
        super().__init__()
        self.operation = operation
        self.num_forwards = 0

    def forward(self, *args, **kwargs):
        self.num_forwards += 1
        return self.operation(*args, **kwargs)


def _qkv_projection(
    x_TD: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    x_TNH = x_TD.unsqueeze(1)
    return x_TNH, x_TNH, x_TNH


def _inner_attention(
    q_TNH: torch.Tensor,
    k_TNH: torch.Tensor,
    v_TNH: torch.Tensor,
    **kwargs,
) -> torch.Tensor:
    return q_TNH + k_TNH + v_TNH


def _identity_rope(
    q_TNH: torch.Tensor,
    k_TNH: torch.Tensor,
    positions: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    return q_TNH, k_TNH


class _CountingGQAttention(GQAttention):
    """Use ``GQAttention.forward`` unchanged with counted test submodules."""

    def __init__(self):
        Module.__init__(self)
        self.n_heads = 1
        self.n_kv_heads = 1
        self.head_dim = 4
        self.enable_gqa = False
        self.rope = _CountingOp(_identity_rope)  # pyrefly: ignore [bad-assignment]
        self.qkv_linear = _CountingOp(_qkv_projection)
        self.wo = _CountingOp(Linear(Linear.Config(in_features=4, out_features=4)))
        self.inner_attention = _CountingOp(_inner_attention)
        self.q_norm = None
        self.k_norm = None
        self.scaling = None


class _CountingModelSpecificAttention:
    projection_forwards: int
    inner_compute: _CountingOp
    output_projection: _CountingOp

    def region_counts(self) -> tuple[int, ...]:
        return (
            self.projection_forwards,
            self.inner_compute.num_forwards,
            self.output_projection.num_forwards,
        )


class _CountingDeepSeekV3Attention(
    DeepSeekV3Attention, _CountingModelSpecificAttention
):
    def __init__(self):
        Module.__init__(self)
        self.projection_forwards = 0
        self.softmax_scale = 1.0
        self.inner_compute = _CountingOp(_inner_attention)
        self.inner_attention = self.inner_compute
        self.output_projection = _CountingOp(
            Linear(Linear.Config(in_features=4, out_features=4))
        )
        self.wo = self.output_projection

    def _project_latents(self, x_TD):
        self.projection_forwards += 1
        return x_TD * 1.0, x_TD + 0.0

    def _project_qkv(self, x_TD, q_latent_TC, compressed_kv_TC, positions):
        del positions
        x_T1D = (x_TD + q_latent_TC + compressed_kv_TC).unsqueeze(1)
        return x_T1D, x_T1D, x_T1D


class _CountingQwen35Attention(Qwen35Attention, _CountingModelSpecificAttention):
    def __init__(self):
        Module.__init__(self)
        self.projection_forwards = 0
        self.head_dim = 4
        self.rotary_dim = 2
        self.scaling = 1.0
        self.enable_gqa = False
        self.q_norm = torch.nn.Identity()
        self.k_norm = torch.nn.Identity()
        self.rope = _CountingOp(_identity_rope)
        self.inner_compute = _CountingOp(_inner_attention)
        self.inner_attention = self.inner_compute
        self.output_projection = _CountingOp(
            Linear(Linear.Config(in_features=4, out_features=4))
        )
        self.wo = self.output_projection

    def _project_qkv(self, x_TD):
        self.projection_forwards += 1
        x_T1D = x_TD.unsqueeze(1)
        return x_T1D, x_T1D, x_T1D, x_T1D


class _CountingKimiMLAAttention(KimiMLAAttention, _CountingModelSpecificAttention):
    def __init__(self):
        Module.__init__(self)
        self.projection_forwards = 0
        self.scale = 1.0
        self.inner_compute = _CountingOp(_inner_attention)
        self.inner_attention = self.inner_compute
        self.output_projection = _CountingOp(
            Linear(Linear.Config(in_features=4, out_features=4))
        )
        self.wo = self.output_projection
        self.gate_projection = _CountingOp(
            Linear(Linear.Config(in_features=4, out_features=4))
        )
        self.gate = self.gate_projection
        self.q_head_dim = 4
        self.kv_lora_rank = 2
        self.qk_rope_head_dim = 2
        self.qk_nope_head_dim = 2
        self.v_head_dim = 4
        self.q_norm = torch.nn.Identity()
        self.wq_b = torch.nn.Identity()
        self.kv_norm = torch.nn.Identity()
        self.wkv_b = _CountingOp(lambda x_TC: x_TC.repeat(1, 3))

    def _project_latents(self, x_TD):
        self.projection_forwards += 1
        return x_TD * 1.0, x_TD + 0.0

    def region_counts(self) -> tuple[int, ...]:
        return (
            self.projection_forwards,
            self.gate_projection.num_forwards,
            self.inner_compute.num_forwards,
            self.output_projection.num_forwards,
        )


class _CountingMuseGlimmerAttention(
    MuseGlimmerAttention, _CountingModelSpecificAttention
):
    def __init__(self):
        Module.__init__(self)
        self.projection_forwards = 0
        self.q_norm = None
        self.k_norm = None
        self.rope = None
        self.window_size = None
        self.scaling = 1.0
        self.enable_gqa = False
        self.inner_compute = _CountingOp(_inner_attention)
        self.inner_attention = self.inner_compute
        self.output_projection = _CountingOp(
            Linear(Linear.Config(in_features=4, out_features=4))
        )
        self.wo = self.output_projection
        self.gate_projection = _CountingOp(lambda x_TD: x_TD * 1.0)
        self.o_gate = self.gate_projection

    def _project_qkv(self, x_TD):
        self.projection_forwards += 1
        projected_TD = x_TD * 1.0
        projected_T1D = projected_TD.unsqueeze(1)
        return projected_T1D, projected_T1D, projected_T1D

    def region_counts(self) -> tuple[int, ...]:
        return (
            self.projection_forwards,
            self.gate_projection.num_forwards,
            self.inner_compute.num_forwards,
            self.output_projection.num_forwards,
        )


def _recurrent_inner(query_TC, key_TC, value_TC, *_args, **_kwargs):
    return (query_TC + key_TC + value_TC).unsqueeze(1)


def _gated_norm(x_T1D: torch.Tensor, gate_T1D: torch.Tensor) -> torch.Tensor:
    return x_T1D * torch.sigmoid(gate_T1D)


class _CountingGatedDeltaNet(GatedDeltaNet, _CountingModelSpecificAttention):
    def __init__(self):
        Module.__init__(self)
        self.projection_forwards = 0
        self.key_head_dim = 4
        self.value_head_dim = 4
        self.inner_compute = _CountingOp(_recurrent_inner)
        self.inner_gated_delta_net = self.inner_compute
        self.norm = _CountingOp(_gated_norm)
        self.output_projection = _CountingOp(
            Linear(Linear.Config(in_features=4, out_features=4))
        )
        self.out_proj = self.output_projection
        self.gate_projection = _CountingOp(lambda x_TD: x_TD * 1.0)
        self.in_proj_z = self.gate_projection
        self.in_proj_a = torch.nn.Identity()
        self.in_proj_b = torch.nn.Identity()
        self.conv_q = SimpleNamespace(weight=torch.empty(0))
        self.conv_k = SimpleNamespace(weight=torch.empty(0))
        self.conv_v = SimpleNamespace(weight=torch.empty(0))
        self.A_log = torch.empty(0)
        self.dt_bias = torch.empty(0)

    def _project_qkv(self, x_TD):
        self.projection_forwards += 1
        projected_TD = x_TD * 1.0
        return projected_TD, projected_TD, projected_TD

    def region_counts(self) -> tuple[int, ...]:
        return (
            self.projection_forwards,
            self.gate_projection.num_forwards,
            self.inner_compute.num_forwards,
            self.output_projection.num_forwards,
        )


class _CountingKDA(KDA, _CountingModelSpecificAttention):
    def __init__(self):
        Module.__init__(self)
        self.projection_forwards = 0
        self.inner_compute = _CountingOp(_recurrent_inner)
        self.inner_kda = self.inner_compute
        self.output_norm = _CountingOp(_gated_norm)
        self.output_projection = _CountingOp(
            Linear(Linear.Config(in_features=4, out_features=4))
        )
        self.output_proj = self.output_projection
        self.gate_projection = _CountingOp(lambda x_TD: x_TD * 1.0)
        self.output_gate = self.gate_projection
        self.forget_a = torch.nn.Identity()
        self.forget_b = torch.nn.Identity()
        self.beta = torch.nn.Identity()
        self.head_dim = 4
        self.q_conv = SimpleNamespace(weight=torch.empty(0))
        self.k_conv = SimpleNamespace(weight=torch.empty(0))
        self.v_conv = SimpleNamespace(weight=torch.empty(0))
        self.A_log = torch.empty(0)
        self.dt_bias = torch.empty(0)

    def _project_qkv(self, x_TD):
        self.projection_forwards += 1
        projected_TD = x_TD * 1.0
        return projected_TD, projected_TD, projected_TD

    def region_counts(self) -> tuple[int, ...]:
        return (
            self.projection_forwards,
            self.gate_projection.num_forwards,
            self.inner_compute.num_forwards,
            self.output_projection.num_forwards,
        )


class _AttentionBlock(Module):
    def __init__(self):
        super().__init__()
        self.attention = _CountingGQAttention()

    def forward(self, x_TD: torch.Tensor) -> torch.Tensor:
        return self.attention(x_TD, attention_masks=None).sum()


class _ModelSpecificAttentionBlock(Module):
    def __init__(self, attention: Module):
        super().__init__()
        self.attention = attention

    def forward(self, x_TD: torch.Tensor) -> torch.Tensor:
        return self.attention(x_TD, None).sum()


class _IdentityRouter(Module):
    def forward(self, x_TD, expert_bias_E, **kwargs):
        del expert_bias_E, kwargs
        num_tokens = x_TD.shape[0]
        weights_T1 = torch.ones(num_tokens, 1, device=x_TD.device)
        expert_ids_T1 = torch.zeros(num_tokens, 1, device=x_TD.device, dtype=torch.long)
        routing_map_T1 = torch.ones(num_tokens, 1, device=x_TD.device, dtype=torch.bool)
        return weights_T1, expert_ids_T1, routing_map_T1


class _IdentityRoutedExperts(Module):
    def forward(
        self,
        x_TD,
        weights_TK,
        expert_ids_TK,
        num_tokens_per_expert_E,
    ):
        del weights_TK, expert_ids_TK, num_tokens_per_expert_E
        return x_TD


class _CountingKimiLatentMoE(KimiLatentMoE):
    def __init__(self):
        Module.__init__(self)
        self.router = _IdentityRouter()
        self.expert_bias_E = torch.empty(1)
        self.routed_down = _CountingOp(
            Linear(Linear.Config(in_features=4, out_features=4))
        )
        self.routed_norm = torch.nn.Identity()
        self.routed_experts = _IdentityRoutedExperts()
        self.routed_up = _CountingOp(
            Linear(Linear.Config(in_features=4, out_features=4))
        )
        self.shared_experts = None

    def _maybe_shard_routed_branch_inputs_across_tp(self, x_TD, padding_mask_T):
        return x_TD, padding_mask_T

    def _maybe_zero_fill_routed_output_to_tp_partial(self, out_TD):
        return out_TD

    def _maybe_all_reduce_moe_output_across_tp(self, out_TD):
        return out_TD


class _KimiLatentMoEBlock(Module):
    def __init__(self):
        super().__init__()
        self.moe = _CountingKimiLatentMoE()

    def forward(self, x_TD: torch.Tensor) -> torch.Tensor:
        return self.moe(x_TD).sum()


class _FeedForwardBlock(Module):
    def __init__(self, feed_forward: Module):
        super().__init__()
        self.feed_forward = feed_forward

    def forward(self, x_TD: torch.Tensor) -> torch.Tensor:
        return self.feed_forward(x_TD).sum()


class _RoutedExpertsBlock(Module):
    def __init__(self, routed_experts: RoutedExperts):
        super().__init__()
        self.routed_experts = routed_experts

    def forward(self, x_TD: torch.Tensor) -> torch.Tensor:
        num_tokens = x_TD.shape[0]
        topk_scores_T1 = torch.ones(num_tokens, 1, device=x_TD.device)
        topk_expert_ids_T1 = torch.zeros(
            num_tokens, 1, device=x_TD.device, dtype=torch.long
        )
        num_tokens_per_expert_1 = torch.tensor([num_tokens], device=x_TD.device)
        return self.routed_experts(
            x_TD,
            topk_scores_T1,
            topk_expert_ids_T1,
            num_tokens_per_expert_1,
        ).sum()


class _MoEOutputReductionBlock(Module):
    def __init__(self):
        super().__init__()
        self.moe = MoE.__new__(MoE)
        Module.__init__(self.moe)

    def forward(self, x_TD: torch.Tensor) -> torch.Tensor:
        out_TD = self.moe._maybe_all_reduce_moe_output_across_tp(x_TD)
        return out_TD.square().sum()


class _CountingGroupedLinear(GroupedLinear):
    """GroupedLinear with CPU reference compute and a forward counter."""

    def __init__(self, config: GroupedLinear.Config):
        super().__init__(config)
        self.num_forwards = 0

    def forward(self, input_RI: torch.Tensor, offsets_E: torch.Tensor) -> torch.Tensor:
        self.num_forwards += 1
        return super().forward(input_RI, offsets_E)

    def _grouped_mm(
        self,
        *,
        input_RI: torch.Tensor,
        weight_EOI: torch.Tensor,
        offsets_E: torch.Tensor,
    ) -> torch.Tensor:
        del offsets_E
        return input_RI.float() @ weight_EOI[0].float().T


def _routed_experts_config(
    *,
    grouped_linear_cls: type[GroupedLinear] = GroupedLinear,
    activation_fn: BinaryActivationFn.Config | None = None,
) -> RoutedExperts.Config:
    return RoutedExperts.Config(
        w13=grouped_linear_cls.Config(
            group_size=1,
            in_features=4,
            out_features=8,
            num_linears=2,
        ),
        w2=grouped_linear_cls.Config(
            group_size=1,
            in_features=8,
            out_features=4,
        ),
        token_dispatcher=LocalTokenDispatcher.Config(num_experts=1, top_k=1),
        activation_fn=activation_fn or SwiGLU.Config(),
    )


def _vision_inner_attention(
    q_THDh: torch.Tensor,
    k_THDh: torch.Tensor,
    v_THDh: torch.Tensor,
    **kwargs,
) -> torch.Tensor:
    return q_THDh + k_THDh + v_THDh


def _vision_identity_rope(
    q_THDh: torch.Tensor,
    k_THDh: torch.Tensor,
    rope_cache: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    return q_THDh, k_THDh


class _CountingVisionAttention(VisionAttention):
    """Use VisionAttention.forward unchanged with counted region bodies."""

    def __init__(self):
        Module.__init__(self)
        self.head_dim = 4
        self.wq = _CountingOp(Linear(Linear.Config(in_features=4, out_features=4)))
        self.wk = _CountingOp(Linear(Linear.Config(in_features=4, out_features=4)))
        self.wv = _CountingOp(Linear(Linear.Config(in_features=4, out_features=4)))
        self.proj = _CountingOp(Linear(Linear.Config(in_features=4, out_features=4)))
        self.flex_attention = _CountingOp(_vision_inner_attention)


class _CountingVisionBlock(VisionTransformerBlock):
    """Use the common vision block with counted attention and MLP projections."""

    def __init__(self):
        Module.__init__(self)
        self.norm1 = torch.nn.Identity()
        self.norm2 = torch.nn.Identity()
        self.attn = _CountingVisionAttention()
        self.mlp = VisionMLP.Config(
            fc1=_linear_config(4, 8),
            fc2=_linear_config(8, 4),
        ).build()
        self.mlp.linear_fc1 = _CountingOp(self.mlp.linear_fc1)
        self.mlp.linear_fc2 = _CountingOp(self.mlp.linear_fc2)


class _VisionRematModel(Module):
    def __init__(self):
        super().__init__()
        self.layers = ModuleDict({"0": _CountingVisionBlock()})

    def forward(self, x_TD: torch.Tensor) -> torch.Tensor:
        return self.layers["0"](
            x_TD,
            rope_cache=x_TD.new_empty(0),
            rope_apply=_vision_identity_rope,
            attention_mask=None,  # pyrefly: ignore [bad-argument-type]
        ).sum()


class _RematModel(Module):
    def __init__(self, block: Module):
        super().__init__()
        self.layers = ModuleDict({"0": block})

    def forward(self, x_BD: torch.Tensor) -> torch.Tensor:
        return self.layers["0"](x_BD)


def _run_forward_backward(
    model: Module, x_BD: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, list[torch.Tensor]]:
    model.zero_grad(set_to_none=True)
    input_BD = x_BD.detach().clone().requires_grad_(True)
    output = model(input_BD)
    output.backward()
    assert input_BD.grad is not None
    parameter_grads = []
    for parameter in model.parameters():
        assert parameter.grad is not None
        parameter_grads.append(parameter.grad.detach().clone())
    return output.detach(), input_BD.grad.detach().clone(), parameter_grads


def _trace_region_names(forward: Callable[[], Any]) -> list[str]:
    with remat.collect_trace() as trace:
        forward()
    return [entry.name for entry in trace.entries]


def _linear_config(in_features: int, out_features: int) -> Linear.Config:
    return Linear.Config(in_features=in_features, out_features=out_features)


def _feed_forward_config() -> FeedForward.Config:
    return FeedForward.Config(
        w13=ColumnParallelLinear.Config(in_features=4, out_features=8, num_linears=2),
        w2=RowParallelLinear.Config(in_features=8, out_features=4),
    )


class TestRematRegions(unittest.TestCase):
    def test_save_regions_config_is_required(self):
        save_regions_field = next(
            field for field in fields(RegionAC.Config) if field.name == "save_regions"
        )
        self.assertIs(save_regions_field.default, MISSING)
        self.assertIs(save_regions_field.default_factory, MISSING)

    def test_unsupported_config_options_error(self):
        for config_factory, message in (
            (
                lambda: RegionAC.Config(save_regions=[], preserve_rng_state=True),
                "preserve_rng_state=True",
            ),
            (lambda: RegionAC.Config(save_regions=[], debug=True), "debug option"),
        ):
            with self.subTest(message=message), self.assertRaisesRegex(
                ValueError, message
            ):
                config_factory()

    def test_llama_attention_policy_applies_without_changing_state_dict(self):
        from torchtitan.models.llama3 import model_registry

        with torch.device("meta"):
            model = model_registry("debugmodel").build()
        state_keys = list(model.state_dict())

        RegionAC.Config(save_regions=["attention.*"]).build().apply(model)

        self.assertEqual(list(model.state_dict()), state_keys)

    def test_attention_save_regions_control_recomputation(self):
        for save_regions, expected_counts in (
            ([], (2, 2, 2)),
            (["attention.*"], (1, 1, 1)),
            (["attention.qkv"], (1, 2, 2)),
            (["attention.inner_attention"], (2, 1, 2)),
            (["attention.wo"], (2, 2, 1)),
        ):
            with self.subTest(save_regions=save_regions):
                torch.manual_seed(42)
                baseline = _RematModel(_AttentionBlock())
                remat_model = deepcopy(baseline)
                RegionAC.Config(save_regions=save_regions).build().apply(remat_model)

                x_TD = torch.randn(3, 4)
                expected = _run_forward_backward(baseline, x_TD)
                actual = _run_forward_backward(remat_model, x_TD)

                torch.testing.assert_close(actual[0], expected[0], rtol=0, atol=0)
                torch.testing.assert_close(actual[1], expected[1], rtol=0, atol=0)
                for actual_grad, expected_grad in zip(actual[2], expected[2]):
                    torch.testing.assert_close(
                        actual_grad, expected_grad, rtol=0, atol=0
                    )

                block = remat_model.layers["0"]
                assert isinstance(block, _AttentionBlock)
                self.assertEqual(
                    (
                        block.attention.qkv_linear.num_forwards,
                        block.attention.inner_attention.num_forwards,
                        block.attention.wo.num_forwards,
                    ),
                    expected_counts,
                )

    def test_model_specific_attention_regions_control_recomputation(self):
        attention_cases = (
            (
                _CountingDeepSeekV3Attention,
                ("latent_projections", "inner_attention", "wo"),
            ),
            (
                _CountingQwen35Attention,
                ("qkv", "inner_attention", "wo"),
            ),
            (
                _CountingKimiMLAAttention,
                ("latent_projections", "gate", "inner_attention", "wo"),
            ),
            (
                _CountingMuseGlimmerAttention,
                ("qkv", "gate", "inner_attention", "wo"),
            ),
            (
                _CountingGatedDeltaNet,
                ("qkv", "gate", "inner_attention", "wo"),
            ),
            (
                _CountingKDA,
                ("qkv", "gate", "inner_attention", "wo"),
            ),
        )
        for attention_factory, region_names in attention_cases:
            policies = [
                ([], tuple(2 for _ in region_names)),
                *[
                    (
                        [f"attention.{region_name}"],
                        tuple(
                            1 if index == saved_index else 2
                            for index in range(len(region_names))
                        ),
                    )
                    for saved_index, region_name in enumerate(region_names)
                ],
            ]
            for save_regions, expected_counts in policies:
                with self.subTest(
                    attention=attention_factory.__name__,
                    save_regions=save_regions,
                ):
                    torch.manual_seed(42)
                    baseline = _RematModel(
                        _ModelSpecificAttentionBlock(attention_factory())
                    )
                    remat_model = deepcopy(baseline)
                    RegionAC.Config(save_regions=save_regions).build().apply(
                        remat_model
                    )

                    x_TD = torch.randn(3, 4)
                    expected = _run_forward_backward(baseline, x_TD)
                    actual = _run_forward_backward(remat_model, x_TD)

                    torch.testing.assert_close(actual[0], expected[0], rtol=0, atol=0)
                    torch.testing.assert_close(actual[1], expected[1], rtol=0, atol=0)
                    for actual_grad, expected_grad in zip(actual[2], expected[2]):
                        torch.testing.assert_close(
                            actual_grad, expected_grad, rtol=0, atol=0
                        )

                    block = remat_model.layers["0"]
                    assert isinstance(block, _ModelSpecificAttentionBlock)
                    attention = block.attention
                    assert isinstance(attention, _CountingModelSpecificAttention)
                    self.assertEqual(attention.region_counts(), expected_counts)

    def test_kimi_latent_moe_regions_control_recomputation(self):
        for save_regions, expected_counts in (
            ([], (2, 2)),
            (["moe.routed_down"], (1, 2)),
            (["moe.routed_up"], (2, 1)),
            (["moe.routed_*"], (1, 1)),
        ):
            with self.subTest(save_regions=save_regions):
                torch.manual_seed(42)
                baseline = _RematModel(_KimiLatentMoEBlock())
                remat_model = deepcopy(baseline)
                RegionAC.Config(save_regions=save_regions).build().apply(remat_model)

                x_TD = torch.randn(3, 4)
                expected = _run_forward_backward(baseline, x_TD)
                actual = _run_forward_backward(remat_model, x_TD)

                torch.testing.assert_close(actual[0], expected[0], rtol=0, atol=0)
                torch.testing.assert_close(actual[1], expected[1], rtol=0, atol=0)
                for actual_grad, expected_grad in zip(actual[2], expected[2]):
                    torch.testing.assert_close(
                        actual_grad, expected_grad, rtol=0, atol=0
                    )

                block = remat_model.layers["0"]
                assert isinstance(block, _KimiLatentMoEBlock)
                self.assertEqual(
                    (
                        block.moe.routed_down.num_forwards,
                        block.moe.routed_up.num_forwards,
                    ),
                    expected_counts,
                )

    def test_feed_forward_save_regions_control_recomputation(self):
        for save_regions, expected_counts in (
            ([], (2, 2)),
            (["feed_forward.*"], (1, 1)),
            (["feed_forward.w13"], (1, 2)),
            (["feed_forward.w2"], (2, 1)),
        ):
            with self.subTest(save_regions=save_regions):
                torch.manual_seed(42)
                feed_forward = _feed_forward_config().build()
                feed_forward.w13 = _CountingOp(feed_forward.w13)
                feed_forward.w2 = _CountingOp(feed_forward.w2)
                baseline = _RematModel(_FeedForwardBlock(feed_forward))
                remat_model = deepcopy(baseline)
                RegionAC.Config(save_regions=save_regions).build().apply(remat_model)

                x_TD = torch.randn(3, 4)
                expected = _run_forward_backward(baseline, x_TD)
                actual = _run_forward_backward(remat_model, x_TD)

                torch.testing.assert_close(actual[0], expected[0], rtol=0, atol=0)
                torch.testing.assert_close(actual[1], expected[1], rtol=0, atol=0)
                for actual_grad, expected_grad in zip(actual[2], expected[2]):
                    torch.testing.assert_close(
                        actual_grad, expected_grad, rtol=0, atol=0
                    )

                block = remat_model.layers["0"]
                assert isinstance(block, _FeedForwardBlock)
                feed_forward = block.feed_forward
                assert isinstance(feed_forward, FeedForward)
                self.assertEqual(
                    (
                        feed_forward.w13.num_forwards,
                        feed_forward.w2.num_forwards,
                    ),
                    expected_counts,
                )

    def test_feed_forward_variants_use_expected_region_boundaries(self):
        feed_forward_config = _feed_forward_config()

        def silu_and_mul(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
            return torch.nn.functional.silu(gate) * up

        with patch(
            "torchtitan.overrides.fused_swiglu.silu_and_mul_op",
            side_effect=silu_and_mul,
        ):
            async_config = AsyncTensorParallelTransform(
                enable_sequence_parallel=True
            ).transform(deepcopy(feed_forward_config))
            fused_config = deepcopy(feed_forward_config)
            fused_config.activation_fn = fused_swiglu(fused_config.activation_fn)
            fused_async_config = deepcopy(async_config)
            fused_async_config.activation_fn = fused_swiglu(
                fused_async_config.activation_fn
            )
            variants = (
                (async_config.build(), ["w13", "w2"]),
                (fused_config.build(), ["w13", "w2"]),
                (fused_async_config.build(), ["w13", "w2"]),
            )
            for feed_forward, expected_names in variants:
                with self.subTest(feed_forward=type(feed_forward).__name__):
                    model = _RematModel(_FeedForwardBlock(feed_forward))
                    RegionAC.Config(save_regions=[]).build().apply(model)
                    self.assertEqual(
                        _trace_region_names(lambda: model(torch.randn(3, 4))),
                        [f"feed_forward.{name}" for name in expected_names],
                    )

    def test_grouped_linear_save_regions_control_recomputation(self):
        for save_regions, expected_counts in (
            ([], (2, 2)),
            (["routed_experts.*"], (1, 1)),
            (["routed_experts.w13"], (1, 2)),
            (["routed_experts.w2"], (2, 1)),
        ):
            with self.subTest(save_regions=save_regions):
                torch.manual_seed(42)
                config = _routed_experts_config()
                routed_experts = config.build()
                routed_experts.w13 = _CountingGroupedLinear(config.w13)
                routed_experts.w2 = _CountingGroupedLinear(config.w2)
                for parameter in routed_experts.parameters():
                    torch.nn.init.normal_(parameter)
                baseline = _RematModel(_RoutedExpertsBlock(routed_experts))
                remat_model = deepcopy(baseline)
                RegionAC.Config(save_regions=save_regions).build().apply(remat_model)

                x_TD = torch.randn(3, 4)
                expected = _run_forward_backward(baseline, x_TD)
                actual = _run_forward_backward(remat_model, x_TD)

                torch.testing.assert_close(actual[0], expected[0], rtol=0, atol=0)
                torch.testing.assert_close(actual[1], expected[1], rtol=0, atol=0)
                for actual_grad, expected_grad in zip(actual[2], expected[2]):
                    torch.testing.assert_close(
                        actual_grad, expected_grad, rtol=0, atol=0
                    )

                block = remat_model.layers["0"]
                assert isinstance(block, _RoutedExpertsBlock)
                w13 = block.routed_experts.w13
                w2 = block.routed_experts.w2
                assert isinstance(w13, _CountingGroupedLinear)
                assert isinstance(w2, _CountingGroupedLinear)
                self.assertEqual(
                    (w13.num_forwards, w2.num_forwards),
                    expected_counts,
                )

    def test_moe_tp_output_reduction_region_controls_recomputation(self):
        for save_regions, expected_reductions in (
            ([], 2),
            (["moe.tp_output_reduction"], 1),
        ):
            with self.subTest(save_regions=save_regions):
                model = _RematModel(_MoEOutputReductionBlock())
                RegionAC.Config(save_regions=save_regions).build().apply(model)
                num_reductions = 0

                def counted_redistribute(tensor, *_args, **_kwargs):
                    nonlocal num_reductions
                    num_reductions += 1
                    return tensor * 2

                with (
                    patch(
                        "torchtitan.models.common.moe.spmd_dense_sp_enabled",
                        return_value=False,
                    ),
                    patch(
                        "torchtitan.models.common.moe.spmd_mesh_group",
                        return_value=object(),
                    ),
                    patch(
                        "torchtitan.models.common.moe.spmd.redistribute",
                        new=counted_redistribute,
                    ),
                ):
                    x_TD = torch.randn(3, 4, requires_grad=True)
                    model(x_TD).backward()

                self.assertEqual(num_reductions, expected_reductions)
                self.assertIsNotNone(x_TD.grad)

    def test_shared_w2_region_controls_reduce_scatter_recomputation(self):
        for save_regions, expected_reductions in (
            ([], 2),
            (["feed_forward.w2"], 1),
        ):
            with self.subTest(save_regions=save_regions):
                shared_expert = FeedForward.Config(
                    w13=Linear.Config(
                        in_features=4,
                        out_features=8,
                        num_linears=2,
                    ),
                    w2=RowParallelLinear.Config(
                        in_features=8,
                        out_features=4,
                    ),
                ).build()
                model = _RematModel(_FeedForwardBlock(shared_expert))
                RegionAC.Config(save_regions=save_regions).build().apply(model)
                num_reductions = 0

                def counted_redistribute(tensor, *_args, **_kwargs):
                    nonlocal num_reductions
                    num_reductions += 1
                    return tensor * 2

                with (
                    patch(
                        "torchtitan.models.common.linear.spmd_dense_sp_enabled",
                        return_value=True,
                    ),
                    patch(
                        "torchtitan.models.common.linear.spmd_mesh_group",
                        return_value=object(),
                    ),
                    patch(
                        "torchtitan.models.common.linear.spmd.redistribute",
                        new=counted_redistribute,
                    ),
                ):
                    x_TD = torch.randn(3, 4, requires_grad=True)
                    model(x_TD).backward()

                self.assertEqual(num_reductions, expected_reductions)
                self.assertIsNotNone(x_TD.grad)

    def test_grouped_linear_variants_use_expected_region_boundaries(self):
        configs = (
            _routed_experts_config(),
            _routed_experts_config(
                grouped_linear_cls=GptOssGroupedLinear,
                activation_fn=GptOssSwiGLU.Config(),
            ),
            _routed_experts_config(activation_fn=FusedSwiGLU.Config()),
        )

        def grouped_mm(
            *,
            input_RI: torch.Tensor,
            weight_EOI: torch.Tensor,
            offsets_E: torch.Tensor,
        ) -> torch.Tensor:
            del offsets_E
            return input_RI.float() @ weight_EOI[0].float().T

        def silu_and_mul(
            gate_RF: torch.Tensor,
            up_RF: torch.Tensor,
            offsets_E: torch.Tensor,
        ) -> torch.Tensor:
            del offsets_E
            return torch.nn.functional.silu(gate_RF) * up_RF

        for config in configs:
            routed_experts = config.build()
            with self.subTest(
                grouped_linear=type(routed_experts.w13).__name__,
                activation=type(routed_experts.activation_fn).__name__,
            ):
                for parameter in routed_experts.parameters():
                    torch.nn.init.normal_(parameter)
                model = _RematModel(_RoutedExpertsBlock(routed_experts))
                RegionAC.Config(save_regions=[]).build().apply(model)

                with (
                    patch.object(
                        routed_experts.w13, "_grouped_mm", side_effect=grouped_mm
                    ),
                    patch.object(
                        routed_experts.w2, "_grouped_mm", side_effect=grouped_mm
                    ),
                    patch(
                        "torchtitan.overrides.fused_swiglu.silu_and_mul_op",
                        side_effect=silu_and_mul,
                    ),
                ):
                    x_TD = torch.randn(3, 4, requires_grad=True)
                    with remat.collect_trace() as trace:
                        output = model(x_TD)
                    output.backward()

                self.assertEqual(
                    [entry.name for entry in trace.entries],
                    ["routed_experts.w13", "routed_experts.w2"],
                )
                self.assertIsNotNone(x_TD.grad)

    def test_vision_save_regions_control_recomputation(self):
        for save_regions, expected_counts in (
            ([], (2, 2, 2, 2, 2, 2, 2)),
            (["attn.qkv"], (1, 1, 1, 2, 2, 2, 2)),
            (["attn.inner_attention"], (2, 2, 2, 1, 2, 2, 2)),
            (["attn.wo"], (2, 2, 2, 2, 1, 2, 2)),
            (["mlp.w1"], (2, 2, 2, 2, 2, 1, 2)),
            (["mlp.w2"], (2, 2, 2, 2, 2, 2, 1)),
            (["attn.*", "mlp.*"], (1, 1, 1, 1, 1, 1, 1)),
        ):
            with self.subTest(save_regions=save_regions):
                torch.manual_seed(42)
                baseline = _VisionRematModel()
                remat_model = deepcopy(baseline)
                RegionAC.Config(save_regions=save_regions).build().apply(remat_model)

                x_TD = torch.randn(3, 4)
                expected = _run_forward_backward(baseline, x_TD)
                actual = _run_forward_backward(remat_model, x_TD)

                torch.testing.assert_close(actual[0], expected[0], rtol=0, atol=0)
                torch.testing.assert_close(actual[1], expected[1], rtol=0, atol=0)
                for actual_grad, expected_grad in zip(actual[2], expected[2]):
                    torch.testing.assert_close(
                        actual_grad, expected_grad, rtol=0, atol=0
                    )

                block = remat_model.layers["0"]
                assert isinstance(block, _CountingVisionBlock)
                self.assertEqual(
                    (
                        block.attn.wq.num_forwards,
                        block.attn.wk.num_forwards,
                        block.attn.wv.num_forwards,
                        block.attn.flex_attention.num_forwards,
                        block.attn.proj.num_forwards,
                        block.mlp.linear_fc1.num_forwards,
                        block.mlp.linear_fc2.num_forwards,
                    ),
                    expected_counts,
                )

    def test_router_decision_is_always_saved(self):
        router = TokenChoiceTopKRouter.Config(
            num_experts=4,
            gate=RouterGateLinear.Config(in_features=4, out_features=4),
            score_func=Sigmoid.Config(),
            top_k=1,
        ).build()

        def forward(x_TD: torch.Tensor) -> torch.Tensor:
            topk_scores_TK, _, routing_map_TE = router(x_TD)
            return topk_scores_TK.sum() + routing_map_TE.sum()

        with patch.object(
            router,
            "_select_experts",
            wraps=router._select_experts,
        ) as select_experts:
            checkpointed_forward = remat.checkpoint(
                region_name="transformer_block", preserve_rng_state=False
            )(forward)
            output = checkpointed_forward(torch.randn(3, 4, requires_grad=True))
            output.backward()

        self.assertEqual(select_experts.call_count, 1)
        self.assertEqual(router.tokens_per_expert_E.sum().item(), 3)

    def test_quantile_router_statistics_are_recorded_once(self):
        router = QuantileBalancedTopKRouter.Config(
            num_experts=4,
            gate=RouterGateLinear.Config(in_features=4, out_features=4),
            score_func=Sigmoid.Config(),
            top_k=1,
            num_bins=8,
        ).build()
        router.train()
        expert_bias_E = torch.zeros(4)

        def forward(x_TD: torch.Tensor) -> torch.Tensor:
            topk_scores_TK, _, _ = router(x_TD, expert_bias_E)
            return topk_scores_TK.sum()

        checkpointed_forward = remat.checkpoint(
            region_name="transformer_block", preserve_rng_state=False
        )(forward)
        x_TD = torch.randn(3, 4, requires_grad=True)
        checkpointed_forward(x_TD).backward()

        self.assertEqual(router.tokens_per_expert_E.sum().item(), 3)
        self.assertEqual(
            router.quantile_balancer.required_bias_histogram_EB.sum().item(),
            12,
        )
        self.assertIsNotNone(x_TD.grad)

    def test_forced_router_statistics_are_recorded_once(self):
        router = TokenChoiceTopKRouter.Config(
            num_experts=4,
            gate=RouterGateLinear.Config(in_features=4, out_features=4),
            score_func=Sigmoid.Config(),
            top_k=1,
            _debug_force_load_balance=True,
        ).build()
        router.train()

        def forward(x_TD: torch.Tensor) -> torch.Tensor:
            topk_scores_TK, _, _ = router(x_TD)
            return topk_scores_TK.sum()

        checkpointed_forward = remat.checkpoint(
            region_name="transformer_block", preserve_rng_state=False
        )(forward)
        x_TD = torch.randn(3, 4, requires_grad=True)
        checkpointed_forward(x_TD).backward()

        torch.testing.assert_close(
            router.tokens_per_expert_E,
            torch.tensor([1.0, 1.0, 1.0, 0.0]),
        )
        self.assertIsNotNone(x_TD.grad)


if __name__ == "__main__":
    unittest.main()
