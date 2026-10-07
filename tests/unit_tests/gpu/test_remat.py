# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import gc
import unittest

from collections.abc import Callable
from copy import deepcopy
from dataclasses import fields, MISSING
from typing import Any
from unittest.mock import patch

import torch
import torch_remat as remat
from torch.multiprocessing.reductions import StorageWeakRef

from torchtitan.config.transform import AsyncTensorParallelTransform
from torchtitan.distributed.activation_checkpoint import RegionAC
from torchtitan.models.common.activation import BinaryActivationFn, Sigmoid, SwiGLU
from torchtitan.models.common.attention import GQAttention
from torchtitan.models.common.feed_forward import FeedForward
from torchtitan.models.common.hi_mid_lo_linear import HiMidLoLinear
from torchtitan.models.common.linear import (
    ColumnParallelLinear,
    GroupedLinear,
    Linear,
    maybe_gather_tp_input,
    RowParallelLinear,
)
from torchtitan.models.common.moe import (
    MoE,
    QuantileBalancedTopKRouter,
    RoundRobinTokenChoiceTopKRouter,
    RoutedExperts,
    TokenChoiceTopKRouter,
)
from torchtitan.models.common.token_dispatcher import LocalTokenDispatcher
from torchtitan.models.common.vision_encoder import (
    VisionAttention,
    VisionMLP,
    VisionTransformerBlock,
)
from torchtitan.models.gpt_oss.moe import GptOssGroupedLinear, GptOssSwiGLU
from torchtitan.protocols.module import Module, ModuleDict
from torchtitan_recipes.overrides.fused_swiglu import fused_swiglu, FusedSwiGLU


class _CountingOp(Module):
    def __init__(self, operation: Callable[..., Any]):
        super().__init__()
        self.operation = operation
        self.num_forwards = 0

    def forward(self, *args, **kwargs):
        self.num_forwards += 1
        return self.operation(*args, **kwargs)


class _CountingProjection:
    """Count local projection calls, which run inside the linear's own region."""

    num_forwards = 0

    def _linear(self, input, weight, bias):
        self.num_forwards += 1
        return super()._linear(
            input, weight, bias
        )  # pyrefly: ignore [missing-attribute]


class _CountingLinear(_CountingProjection, Linear):
    pass


class _CountingColumnParallelLinear(_CountingProjection, ColumnParallelLinear):
    pass


class _CountingRowParallelLinear(_CountingProjection, RowParallelLinear):
    pass


class _CountingQKVProjection(Module):
    def __init__(self):
        super().__init__()
        self.wqkv = _CountingColumnParallelLinear(
            ColumnParallelLinear.Config(in_features=4, out_features=4)
        )

    def forward(
        self, x_TD: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        qkv_TD = self.wqkv(x_TD)
        remat.recompute_needs_tensor(qkv_TD)
        x_TNH = qkv_TD.unsqueeze(1)
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
        self.qkv_linear = _CountingQKVProjection()
        self.wo = _CountingLinear(Linear.Config(in_features=4, out_features=4))
        self.inner_attention = _CountingOp(_inner_attention)
        self.q_norm = None
        self.k_norm = None
        self.scaling = None


class _AttentionBlock(Module):
    def __init__(self):
        super().__init__()
        self.attention = _CountingGQAttention()

    def forward(self, x_TD: torch.Tensor) -> torch.Tensor:
        out_TD = self.attention(x_TD, attention_metadata=None)
        # The sum is a bare consumer of the attention output.
        remat.recompute_needs_tensor(out_TD)
        return out_TD.sum()


class _SharedInputProjections(Module):
    """Two plain projections that share one TP input, gathered once."""

    def __init__(self):
        super().__init__()
        self.wa = _CountingLinear(_linear_config(4, 4))
        self.wb = _CountingLinear(_linear_config(4, 4))

    def forward(self, x_TD: torch.Tensor) -> torch.Tensor:
        x_TD = maybe_gather_tp_input(self, x_TD)
        a_TD, b_TD = self.wa(x_TD), self.wb(x_TD)
        remat.recompute_needs_tensor(a_TD, b_TD)
        return (a_TD * b_TD).sum()


class _SharedInputBlock(Module):
    def __init__(self):
        super().__init__()
        self.attention = _SharedInputProjections()

    def forward(self, x_TD: torch.Tensor) -> torch.Tensor:
        return self.attention(x_TD)


class _FeedForwardBlock(Module):
    def __init__(self, feed_forward: Module):
        super().__init__()
        self.feed_forward = feed_forward

    def forward(self, x_TD: torch.Tensor) -> torch.Tensor:
        out_TD = self.feed_forward(x_TD)
        # The sum is a bare consumer of the feed-forward output.
        remat.recompute_needs_tensor(out_TD)
        return out_TD.sum()


class _RoutedExpertsBlock(Module):
    def __init__(self, routed_experts: RoutedExperts, *, learned_scores: bool = False):
        super().__init__()
        self.routed_experts = routed_experts
        self.learned_scores = learned_scores

    def forward(self, x_TD: torch.Tensor) -> torch.Tensor:
        num_tokens = x_TD.shape[0]
        if self.learned_scores:
            # Score gradients need the expert outputs in combine backward.
            topk_scores_T1 = torch.sigmoid(x_TD.sum(dim=-1, keepdim=True))
        else:
            topk_scores_T1 = torch.ones(num_tokens, 1, device=x_TD.device)
        topk_expert_ids_T1 = torch.zeros(
            num_tokens, 1, device=x_TD.device, dtype=torch.long
        )
        num_tokens_per_expert_1 = torch.tensor([num_tokens], device=x_TD.device)
        out_TD = self.routed_experts(
            x_TD,
            topk_scores_T1,
            topk_expert_ids_T1,
            num_tokens_per_expert_1,
        )
        # The sum is a bare consumer of the routed-expert output.
        remat.recompute_needs_tensor(out_TD)
        return out_TD.sum()


class _MoEOutputReductionBlock(Module):
    def __init__(self):
        super().__init__()
        self.moe = MoE.__new__(MoE)
        Module.__init__(self.moe)

    def forward(self, x_TD: torch.Tensor) -> torch.Tensor:
        out_TD = self.moe._maybe_all_reduce_moe_output_across_tp(x_TD)
        # The square is a bare consumer of the reduced output.
        remat.recompute_needs_tensor(out_TD)
        return out_TD.square().sum()


class _CountingGroupedLinear(GroupedLinear):
    """GroupedLinear with CPU reference compute and a grouped-matmul counter.

    The counter runs inside the grouped_mm region, so a saved region is not
    counted again during replay.
    """

    def __init__(self, config: GroupedLinear.Config):
        super().__init__(config)
        self.num_forwards = 0

    def _grouped_mm(
        self,
        *,
        input_RI: torch.Tensor,
        weight_EOI: torch.Tensor,
        offsets_E: torch.Tensor,
    ) -> torch.Tensor:
        del offsets_E
        self.num_forwards += 1
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
        self.wq = _CountingLinear(_linear_config(4, 4))
        self.wk = _CountingLinear(_linear_config(4, 4))
        self.wv = _CountingLinear(_linear_config(4, 4))
        self.proj = _CountingLinear(_linear_config(4, 4))
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
        self.mlp.linear_fc1 = _CountingLinear(_linear_config(4, 8))
        self.mlp.linear_fc2 = _CountingLinear(_linear_config(8, 4))


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
        from torchtitan.models.llama3 import build_model_config

        with torch.device("meta"):
            model = build_model_config("debugmodel").build()
        state_keys = list(model.state_dict())

        RegionAC.Config(save_regions=["attention.*"]).build().apply(model)

        self.assertEqual(list(model.state_dict()), state_keys)

    def test_attention_save_regions_control_recomputation(self):
        for save_regions, expected_counts in (
            ([], (2, 2, 2)),
            (["attention.*"], (1, 1, 1)),
            (["attention.qkv_linear.wqkv.linear"], (1, 2, 2)),
            (["attention.inner_attention"], (2, 1, 2)),
            (["attention.wo.linear"], (2, 2, 1)),
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
                        block.attention.qkv_linear.wqkv.num_forwards,
                        block.attention.inner_attention.num_forwards,
                        block.attention.wo.num_forwards,
                    ),
                    expected_counts,
                )

    def test_feed_forward_save_regions_control_recomputation(self):
        for save_regions, expected_counts in (
            ([], (2, 2)),
            (["feed_forward.*"], (1, 1)),
            (["feed_forward.w13.linear"], (1, 2)),
            (["feed_forward.w2.linear"], (2, 1)),
        ):
            with self.subTest(save_regions=save_regions):
                torch.manual_seed(42)
                feed_forward = _feed_forward_config().build()
                feed_forward.w13 = _CountingColumnParallelLinear(
                    _feed_forward_config().w13
                )
                feed_forward.w2 = _CountingRowParallelLinear(_feed_forward_config().w2)
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

        def silu_and_mul(gate_up: torch.Tensor) -> torch.Tensor:
            return torch.nn.functional.silu(gate_up[:, 0]) * gate_up[:, 1]

        with patch(
            "torchtitan_recipes.overrides.fused_swiglu.silu_and_mul_op",
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
                (async_config.build(), ["w13.linear", "w2.linear"]),
                (fused_config.build(), ["w13.linear", "w2.linear"]),
                (
                    fused_async_config.build(),
                    ["w13.linear", "w2.linear"],
                ),
            )
            for feed_forward, expected_names in variants:
                with self.subTest(feed_forward=type(feed_forward).__name__):
                    model = _RematModel(_FeedForwardBlock(feed_forward))
                    RegionAC.Config(save_regions=[]).build().apply(model)
                    self.assertEqual(
                        _trace_region_names(lambda: model(torch.randn(3, 4))),
                        [f"feed_forward.{name}" for name in expected_names],
                    )

    def test_lora_adapters_run_inside_the_base_projection_region(self):
        # Adapters declaring their own regions would nest a recomputed region in
        # a saved one when only the base projection matches the save pattern.
        from torchtitan.models.common.lora import get_lora_linear

        lora_cls = get_lora_linear(Linear)
        for save_regions in ([], ["feed_forward.w2.linear"], ["*"]):
            with self.subTest(save_regions=save_regions):
                torch.manual_seed(42)
                feed_forward = FeedForward.Config(
                    w13=Linear.Config(in_features=4, out_features=8, num_linears=2),
                    w2=Linear.Config(in_features=8, out_features=4),
                ).build()
                feed_forward.w2 = lora_cls(
                    lora_cls.Config(in_features=8, out_features=4, rank=2, alpha=4.0)
                )
                for parameter in feed_forward.parameters():
                    # LoRA freezes the base weight; train everything so the
                    # comparison covers every gradient.
                    parameter.requires_grad_(True)
                    torch.nn.init.normal_(parameter)
                baseline = _RematModel(_FeedForwardBlock(feed_forward))
                remat_model = deepcopy(baseline)
                RegionAC.Config(save_regions=save_regions).build().apply(remat_model)

                x_TD = torch.randn(3, 4)
                expected = _run_forward_backward(baseline, x_TD)
                with remat.collect_trace() as trace:
                    actual = _run_forward_backward(remat_model, x_TD)

                names = [entry.name for entry in trace.entries]
                self.assertNotIn("feed_forward.w2.lora_a.linear", names)
                torch.testing.assert_close(actual[0], expected[0], rtol=0, atol=0)
                for actual_grad, expected_grad in zip(actual[2], expected[2]):
                    torch.testing.assert_close(
                        actual_grad, expected_grad, rtol=0, atol=0
                    )

    def test_grouped_linear_save_regions_control_recomputation(self):
        for save_regions, expected_counts in (
            ([], (2, 2)),
            (["routed_experts.*"], (1, 1)),
            (["routed_experts.w13.grouped_mm"], (1, 2)),
            (["routed_experts.w2.grouped_mm"], (2, 1)),
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

    def test_routed_output_feeds_combine_region_without_pin(self):
        # Learned scores make combine backward read the w2 output, so replay
        # must rebuild or keep it without a pin between w2 and combine.
        for save_regions, expected_w2_forwards in (
            ([], 2),
            (["routed_experts.w2.grouped_mm"], 1),
            (["routed_experts.token_dispatcher.combine"], 2),
            (["routed_experts.*"], 1),
        ):
            with self.subTest(save_regions=save_regions):
                torch.manual_seed(42)
                config = _routed_experts_config()
                routed_experts = config.build()
                routed_experts.w13 = _CountingGroupedLinear(config.w13)
                routed_experts.w2 = _CountingGroupedLinear(config.w2)
                for parameter in routed_experts.parameters():
                    torch.nn.init.normal_(parameter)
                baseline = _RematModel(
                    _RoutedExpertsBlock(routed_experts, learned_scores=True)
                )
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
                w2 = block.routed_experts.w2
                assert isinstance(w2, _CountingGroupedLinear)
                self.assertEqual(w2.num_forwards, expected_w2_forwards)

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

    def test_column_parallel_tp_gather_region_controls_regather(self):
        # A saved projection whose input gather is recomputed must not retain
        # the gathered input: replay re-gathers it for the weight gradient.
        for save_regions, expected_gathers, expected_projections, expect_retained in (
            ([], 2, 2, False),
            (["feed_forward.w13.linear"], 2, 1, False),
            (["feed_forward.w13.*"], 1, 1, True),
        ):
            with self.subTest(save_regions=save_regions):
                torch.manual_seed(42)
                feed_forward = FeedForward.Config(
                    w13=ColumnParallelLinear.Config(
                        in_features=4, out_features=8, num_linears=2
                    ),
                    w2=Linear.Config(in_features=8, out_features=4),
                ).build()
                feed_forward.w13 = _CountingColumnParallelLinear(
                    ColumnParallelLinear.Config(
                        in_features=4, out_features=8, num_linears=2
                    )
                )
                baseline = _RematModel(_FeedForwardBlock(feed_forward))
                remat_model = deepcopy(baseline)
                RegionAC.Config(save_regions=save_regions).build().apply(remat_model)
                num_gathers = 0
                gathered_refs = []

                def counted_redistribute(tensor, *_args, **_kwargs):
                    nonlocal num_gathers
                    num_gathers += 1
                    gathered = tensor * 2
                    # remat retains a detached alias, so track the storage.
                    gathered_refs.append(StorageWeakRef(gathered.untyped_storage()))
                    return gathered

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
                    x_TD = torch.randn(3, 4)
                    expected = _run_forward_backward(baseline, x_TD)
                    num_gathers = 0
                    gathered_refs.clear()

                    x_remat_TD = x_TD.clone().requires_grad_()
                    loss = remat_model(x_remat_TD)
                    gc.collect()
                    retained = not gathered_refs[0].expired()
                    loss.backward()

                self.assertEqual(num_gathers, expected_gathers)
                self.assertEqual(retained, expect_retained)
                block = remat_model.layers["0"]
                assert isinstance(block, _FeedForwardBlock)
                self.assertEqual(
                    block.feed_forward.w13.num_forwards, expected_projections
                )
                torch.testing.assert_close(x_remat_TD.grad, expected[1], rtol=0, atol=0)
                for actual, reference in zip(remat_model.parameters(), expected[2]):
                    torch.testing.assert_close(actual.grad, reference, rtol=0, atol=0)

    def test_shared_tp_gather_region_controls_regather(self):
        # maybe_gather_tp_input declares <module fqn>.tp_gather once for all
        # projections consuming the gathered input.
        for save_regions, expected_gathers in (
            ([], 2),
            (["attention.wa.linear", "attention.wb.linear"], 2),
            (["attention.tp_gather"], 1),
        ):
            with self.subTest(save_regions=save_regions):
                torch.manual_seed(42)
                baseline = _RematModel(_SharedInputBlock())
                remat_model = deepcopy(baseline)
                RegionAC.Config(save_regions=save_regions).build().apply(remat_model)
                num_gathers = 0

                def counted_redistribute(tensor, *_args, **_kwargs):
                    nonlocal num_gathers
                    num_gathers += 1
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
                    x_TD = torch.randn(3, 4)
                    expected = _run_forward_backward(baseline, x_TD)
                    num_gathers = 0
                    names = _trace_region_names(
                        lambda: _run_forward_backward(remat_model, x_TD)
                    )
                    self.assertEqual(num_gathers, expected_gathers)
                    actual = _run_forward_backward(remat_model, x_TD)

                self.assertEqual(names[0], "attention.tp_gather")
                torch.testing.assert_close(actual[1], expected[1], rtol=0, atol=0)
                for actual_grad, expected_grad in zip(actual[2], expected[2]):
                    torch.testing.assert_close(
                        actual_grad, expected_grad, rtol=0, atol=0
                    )

    def test_row_parallel_tp_reduce_has_its_own_policy(self):
        # A saved projection followed by a recomputed reduction keeps the
        # TP-times larger partial output for replay; any other combination
        # frees it after the forward.
        linear, reduce = "feed_forward.w2.linear", "feed_forward.w2.tp_reduce"
        for (
            save_regions,
            expected_reductions,
            expected_projections,
            partial_kept,
        ) in (
            ([], 2, 2, False),
            ([linear], 2, 1, True),
            ([reduce], 1, 2, False),
            ([linear, reduce], 1, 1, False),
        ):
            with self.subTest(save_regions=save_regions):
                torch.manual_seed(42)
                feed_forward = FeedForward.Config(
                    w13=Linear.Config(in_features=4, out_features=8, num_linears=2),
                    w2=RowParallelLinear.Config(in_features=8, out_features=4),
                ).build()
                feed_forward.w2 = _CountingRowParallelLinear(
                    RowParallelLinear.Config(in_features=8, out_features=4)
                )
                baseline = _RematModel(_FeedForwardBlock(feed_forward))
                remat_model = deepcopy(baseline)
                RegionAC.Config(save_regions=save_regions).build().apply(remat_model)
                num_reductions = 0
                partial_refs = []

                def counted_redistribute(tensor, *_args, **_kwargs):
                    nonlocal num_reductions
                    num_reductions += 1
                    partial_refs.append(StorageWeakRef(tensor.untyped_storage()))
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
                    x_TD = torch.randn(3, 4)
                    expected = _run_forward_backward(baseline, x_TD)
                    num_reductions = 0
                    partial_refs.clear()

                    x_remat_TD = x_TD.clone().requires_grad_()
                    with remat.collect_trace() as trace:
                        loss = remat_model(x_remat_TD)
                    gc.collect()
                    self.assertEqual(not partial_refs[0].expired(), partial_kept)
                    loss.backward()

                self.assertEqual(
                    [entry.name for entry in trace.entries],
                    [
                        "feed_forward.w13.linear",
                        "feed_forward.w2.linear",
                        "feed_forward.w2.tp_reduce",
                    ],
                )
                self.assertEqual(num_reductions, expected_reductions)
                block = remat_model.layers["0"]
                assert isinstance(block, _FeedForwardBlock)
                self.assertEqual(
                    block.feed_forward.w2.num_forwards, expected_projections
                )
                torch.testing.assert_close(x_remat_TD.grad, expected[1], rtol=0, atol=0)
                for actual, reference in zip(remat_model.parameters(), expected[2]):
                    torch.testing.assert_close(actual.grad, reference, rtol=0, atol=0)

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
            gate_up_R2F: torch.Tensor, offsets_E: torch.Tensor
        ) -> torch.Tensor:
            del offsets_E
            return torch.nn.functional.silu(gate_up_R2F[:, 0]) * gate_up_R2F[:, 1]

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
                        "torchtitan_recipes.overrides.fused_swiglu.silu_and_mul_op",
                        side_effect=silu_and_mul,
                    ),
                ):
                    x_TD = torch.randn(3, 4, requires_grad=True)
                    with remat.collect_trace() as trace:
                        output = model(x_TD)
                    output.backward()

                self.assertEqual(
                    [entry.name for entry in trace.entries],
                    [
                        "routed_experts.w13.grouped_mm",
                        "routed_experts.w2.grouped_mm",
                    ],
                )
                self.assertIsNotNone(x_TD.grad)

    def test_vision_save_regions_control_recomputation(self):
        for save_regions, expected_counts in (
            ([], (2, 2, 2, 2, 2, 2, 2)),
            (["attn.w[qkv].linear"], (1, 1, 1, 2, 2, 2, 2)),
            (["attn.inner_attention"], (2, 2, 2, 1, 2, 2, 2)),
            (["attn.proj.linear"], (2, 2, 2, 2, 1, 2, 2)),
            (["mlp.linear_fc1.linear"], (2, 2, 2, 2, 2, 1, 2)),
            (["mlp.linear_fc2.linear"], (2, 2, 2, 2, 2, 2, 1)),
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
            gate=HiMidLoLinear.Config(in_features=4, out_features=4),
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
            gate=HiMidLoLinear.Config(in_features=4, out_features=4),
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
        router = RoundRobinTokenChoiceTopKRouter.Config(
            num_experts=4,
            gate=HiMidLoLinear.Config(in_features=4, out_features=4),
            score_func=Sigmoid.Config(),
            top_k=1,
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
