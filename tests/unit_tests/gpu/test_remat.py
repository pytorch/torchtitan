# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest

from collections.abc import Callable
from copy import deepcopy
from dataclasses import fields, MISSING
from typing import Any
from unittest.mock import patch

import torch
import torch_remat as remat

from torchtitan.distributed.activation_checkpoint import RegionAC
from torchtitan.models.common.attention import GQAttention
from torchtitan.models.common.dist_gemm import DistGEMMFeedForward
from torchtitan.models.common.feed_forward import FeedForward, SigmoidGatedFeedForward
from torchtitan.models.common.linear import Linear, RouterGateLinear
from torchtitan.models.common.moe import GroupedExperts, TokenChoiceTopKRouter
from torchtitan.models.common.vision_encoder import (
    VisionAttention,
    VisionMLP,
    VisionTransformerBlock,
)
from torchtitan.models.gpt_oss.moe import GptOssGroupedExperts
from torchtitan.overrides.fused_swiglu import (
    dist_gemm_fused_swiglu,
    fused_swiglu,
    FusedSwiGLUGroupedExperts,
)
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


class _AttentionBlock(Module):
    def __init__(self):
        super().__init__()
        self.attention = _CountingGQAttention()

    def forward(self, x_TD: torch.Tensor) -> torch.Tensor:
        return self.attention(x_TD, attention_masks=None).sum()


class _FeedForwardBlock(Module):
    def __init__(self, feed_forward: Module):
        super().__init__()
        self.feed_forward = feed_forward

    def forward(self, x_TD: torch.Tensor) -> torch.Tensor:
        return self.feed_forward(x_TD).sum()


class _GroupedExpertsBlock(Module):
    def __init__(self, grouped_experts: GroupedExperts):
        super().__init__()
        self.grouped_experts = grouped_experts

    def forward(self, x_RD: torch.Tensor) -> torch.Tensor:
        num_tokens_per_expert_E = torch.tensor([x_RD.shape[0]], device=x_RD.device)
        return self.grouped_experts(x_RD, num_tokens_per_expert_E).sum()


class _CountingGroupedExperts(GroupedExperts):
    """GroupedExperts using a CPU reference GEMM and counted region bodies."""

    def __init__(self):
        super().__init__(GroupedExperts.Config(dim=4, hidden_dim=8, num_experts=1))
        self.num_w13_forwards = 0
        self.num_w2_forwards = 0
        for parameter in self.parameters():
            torch.nn.init.normal_(parameter)

    def _grouped_mm(
        self, *, A: torch.Tensor, weight_EOI: torch.Tensor, offs: torch.Tensor
    ) -> torch.Tensor:
        if weight_EOI is self.w2_EDF:
            self.num_w2_forwards += 1
        else:
            self.num_w13_forwards += 1
        return A.float() @ weight_EOI[0].float().transpose(0, 1)


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
        w1=_linear_config(4, 8),
        w2=_linear_config(8, 4),
        w3=_linear_config(4, 8),
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
            model = model_registry("debugmodel").model.build()
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
        sigmoid_config = SigmoidGatedFeedForward.Config(
            w1=feed_forward_config.w1,
            w2=feed_forward_config.w2,
            w3=feed_forward_config.w3,
            gate=_linear_config(4, 4),
        )

        def silu_and_mul(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
            return torch.nn.functional.silu(gate) * up

        with patch(
            "torchtitan.overrides.fused_swiglu._fused_silu_and_mul",
            side_effect=silu_and_mul,
        ):
            dist_gemm_config = DistGEMMFeedForward.Config(
                w1=feed_forward_config.w1,
                w2=feed_forward_config.w2,
                w3=feed_forward_config.w3,
            )
            variants = (
                (sigmoid_config.build(), ["w13", "w2", "gate"]),
                (dist_gemm_config.build(), ["w13", "w2"]),
                (fused_swiglu(feed_forward_config).build(), ["w13", "w2"]),
                (
                    dist_gemm_fused_swiglu(dist_gemm_config).build(),
                    ["w13", "w2"],
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

    def test_grouped_expert_save_regions_control_recomputation(self):
        for save_regions, expected_counts in (
            ([], (2, 2)),
            (["grouped_experts.*"], (1, 1)),
            (["grouped_experts.w13"], (1, 2)),
            (["grouped_experts.w2"], (2, 1)),
        ):
            with self.subTest(save_regions=save_regions):
                torch.manual_seed(42)
                baseline = _RematModel(_GroupedExpertsBlock(_CountingGroupedExperts()))
                remat_model = deepcopy(baseline)
                RegionAC.Config(save_regions=save_regions).build().apply(remat_model)

                x_RD = torch.randn(3, 4)
                expected = _run_forward_backward(baseline, x_RD)
                actual = _run_forward_backward(remat_model, x_RD)

                torch.testing.assert_close(actual[0], expected[0], rtol=0, atol=0)
                torch.testing.assert_close(actual[1], expected[1], rtol=0, atol=0)
                for actual_grad, expected_grad in zip(actual[2], expected[2]):
                    torch.testing.assert_close(
                        actual_grad, expected_grad, rtol=0, atol=0
                    )

                block = remat_model.layers["0"]
                assert isinstance(block, _GroupedExpertsBlock)
                grouped_experts = block.grouped_experts
                assert isinstance(grouped_experts, _CountingGroupedExperts)
                self.assertEqual(
                    (
                        grouped_experts.num_w13_forwards,
                        grouped_experts.num_w2_forwards,
                    ),
                    expected_counts,
                )

    def test_grouped_expert_variants_use_expected_region_boundaries(self):
        configs = (
            GroupedExperts.Config(dim=4, hidden_dim=8, num_experts=1),
            GptOssGroupedExperts.Config(dim=4, hidden_dim=8, num_experts=1),
            FusedSwiGLUGroupedExperts.Config(dim=4, hidden_dim=8, num_experts=1),
        )

        def grouped_mm(
            *, A: torch.Tensor, weight_EOI: torch.Tensor, offs: torch.Tensor
        ) -> torch.Tensor:
            return A.float() @ weight_EOI[0].float().transpose(0, 1)

        def silu_and_mul(
            gate_RF: torch.Tensor,
            up_RF: torch.Tensor,
            offsets_E: torch.Tensor,
        ) -> torch.Tensor:
            return torch.nn.functional.silu(gate_RF) * up_RF

        for config in configs:
            grouped_experts = config.build()
            with self.subTest(grouped_experts=type(grouped_experts).__name__):
                for parameter in grouped_experts.parameters():
                    torch.nn.init.normal_(parameter)
                model = _RematModel(_GroupedExpertsBlock(grouped_experts))
                RegionAC.Config(save_regions=["grouped_experts.*"]).build().apply(model)

                with (
                    patch.object(
                        grouped_experts, "_grouped_mm", side_effect=grouped_mm
                    ),
                    patch(
                        "torchtitan.overrides.fused_swiglu.silu_and_mul_op",
                        side_effect=silu_and_mul,
                    ),
                ):
                    x_RD = torch.randn(3, 4, requires_grad=True)
                    with remat.collect_trace() as trace:
                        output = model(x_RD)
                    output.backward()

                self.assertEqual(
                    [entry.name for entry in trace.entries],
                    ["grouped_experts.w13", "grouped_experts.w2"],
                )
                self.assertIsNotNone(x_RD.grad)

    def test_vision_save_regions_control_recomputation(self):
        for save_regions, expected_counts in (
            ([], (2, 2, 2, 2, 2, 2, 2)),
            (["attn.qkv"], (1, 1, 1, 2, 2, 2, 2)),
            (["attn.inner_attention"], (2, 2, 2, 1, 2, 2, 2)),
            (["attn.proj"], (2, 2, 2, 2, 1, 2, 2)),
            (["mlp.fc1"], (2, 2, 2, 2, 2, 1, 2)),
            (["mlp.fc2"], (2, 2, 2, 2, 2, 2, 1)),
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
            num_expert_groups=2,
            num_limited_groups=1,
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


if __name__ == "__main__":
    unittest.main()
