# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest
from copy import deepcopy
from dataclasses import fields, MISSING
from fnmatch import fnmatch
from unittest.mock import patch

import torch
import torch_remat as remat

from torchtitan.distributed.activation_checkpoint import (
    RematAC,
    validate_activation_checkpointing_compile,
)
from torchtitan.models.common.attention import GQAttention
from torchtitan.models.common.dist_gemm import DistGEMMFeedForward
from torchtitan.models.common.feed_forward import SigmoidGatedFeedForward
from torchtitan.models.common.linear import Linear
from torchtitan.models.common.moe import (
    GroupedExperts,
    RoutedExperts,
    TokenChoiceTopKRouter,
)
from torchtitan.models.common.remat import (
    available_remat_save_regions,
    configure_remat_save_regions,
    required_remat_save_regions,
)
from torchtitan.models.common.token_dispatcher import (
    AllToAllTokenDispatcher,
    DeepEPTokenDispatcher,
    LocalTokenDispatcher,
    TorchAOTokenDispatcher,
)
from torchtitan.models.common.vision_encoder import VisionAttention, VisionMLP
from torchtitan.models.deepseek_v3.model import Attention as DeepSeekV3Attention
from torchtitan.protocols.module import Module, ModuleDict


class _RegionOwner(Module):
    AVAILABLE_REMAT_SAVE_REGIONS = ("w1", "w2")


class _InheritedRegionOwner(_RegionOwner):
    pass


class _OverriddenRegionOwner(_RegionOwner):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x


class _RequiredRegionOwner(_RegionOwner):
    REQUIRED_REMAT_SAVE_REGIONS = ("w2",)


class _InvalidRequiredRegionOwner(_RegionOwner):
    REQUIRED_REMAT_SAVE_REGIONS = ("missing",)


class _RegionTree(Module):
    def __init__(self):
        super().__init__()
        self.left = _RegionOwner()
        self.right = _RegionOwner()


class _DeeplyNestedRegionBlock(Module):
    def __init__(self):
        super().__init__()
        self.attention = ModuleDict(
            {"projections": ModuleDict({"query": _RegionOwner()})}
        )
        self.feed_forward = ModuleDict(
            {"experts": ModuleDict({"shared": _RegionOwner()})}
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.sum()


class _CountingLinear(Linear):
    def __init__(self, in_features: int = 4, out_features: int = 4):
        super().__init__(
            Linear.Config(in_features=in_features, out_features=out_features)
        )
        self.num_forwards = 0

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        self.num_forwards += 1
        return super().forward(input)


class _CountingQKV(Module):
    def __init__(self):
        super().__init__()
        self.num_forwards = 0

    def forward(
        self, x_TD: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        self.num_forwards += 1
        x_TNH = x_TD.unsqueeze(1)
        return x_TNH, x_TNH, x_TNH


class _CountingInnerAttention(Module):
    def __init__(self):
        super().__init__()
        self.num_forwards = 0

    def forward(
        self,
        q_TNH: torch.Tensor,
        k_TNH: torch.Tensor,
        v_TNH: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        self.num_forwards += 1
        return q_TNH + k_TNH + v_TNH


class _CountingMLAInnerAttention(Module):
    def __init__(self):
        super().__init__()
        self.num_forwards = 0

    def forward(
        self,
        q_TNH: torch.Tensor,
        k_TNH: torch.Tensor,
        v_TNH: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        self.num_forwards += 1
        return q_TNH[..., :2] + k_TNH[..., :2] + v_TNH


class _IdentityRoPE(Module):
    def forward(
        self,
        q_TNH: torch.Tensor,
        k_TNH: torch.Tensor,
        positions: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return q_TNH, k_TNH


class _CountingGQAttention(GQAttention):
    def __init__(self):
        Module.__init__(self)
        self.n_heads = 1
        self.n_kv_heads = 1
        self.head_dim = 4
        self.enable_gqa = False
        self.rope = _IdentityRoPE()
        self.qkv_linear = _CountingQKV()
        self.wo = _CountingLinear()
        self.inner_attention = _CountingInnerAttention()
        self.q_norm = None
        self.k_norm = None
        self.scaling = None


class _RematBlock(Module):
    AVAILABLE_REMAT_SAVE_REGIONS = ("saved",)

    def __init__(self):
        super().__init__()
        self.saved = _CountingLinear()
        self.recomputed = _CountingLinear()

    def forward(self, x_BD: torch.Tensor) -> torch.Tensor:
        saved_BD = remat.region(
            self.saved,
            self.remat_region_name("saved"),
            recompute=self.should_recompute_remat_region("saved"),
        )(x_BD)
        remat.recompute_needs_tensor(saved_BD)
        return self.recomputed(torch.sin(saved_BD)).sum()


class _GQARematBlock(Module):
    def __init__(self):
        super().__init__()
        self.attention = _CountingGQAttention()

    def forward(self, x_TD: torch.Tensor) -> torch.Tensor:
        return self.attention(x_TD, attention_masks=None).sum()


class _CountingVisionAttention(VisionAttention):
    def __init__(self):
        Module.__init__(self)
        self.head_dim = 4
        self.wq = _CountingLinear()
        self.wk = _CountingLinear()
        self.wv = _CountingLinear()
        self.proj = _CountingLinear()
        self.flex_attention = _CountingInnerAttention()


class _CountingVisionMLP(VisionMLP):
    def __init__(self):
        Module.__init__(self)
        self.linear_fc1 = _CountingLinear()
        self.linear_fc2 = _CountingLinear()
        self.act_fn = torch.nn.GELU(approximate="tanh")


def _identity_vision_rope(
    q_THDh: torch.Tensor,
    k_THDh: torch.Tensor,
    rope_cache: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    del rope_cache
    return q_THDh, k_THDh


class _VisionRematBlock(Module):
    def __init__(self):
        super().__init__()
        self.attn = _CountingVisionAttention()
        self.mlp = _CountingVisionMLP()

    def forward(
        self,
        x_TD: torch.Tensor,
        *,
        rope_cache: torch.Tensor,
        rope_apply,
        attention_mask,
    ) -> torch.Tensor:
        attn_out_TD = self.attn(
            x_TD,
            rope_cache=rope_cache,
            rope_apply=rope_apply,
            attention_mask=attention_mask,
        )
        return (attn_out_TD + self.mlp(x_TD)).sum()


class _VisionRematModel(Module):
    def __init__(self):
        super().__init__()
        self.layers = ModuleDict({"0": _VisionRematBlock()})

    def forward(self, x_TD: torch.Tensor) -> torch.Tensor:
        return self.layers["0"](
            x_TD,
            rope_cache=x_TD.new_empty(0),
            rope_apply=_identity_vision_rope,
            attention_mask=None,
        )


class _CountingDeepSeekV3Attention(DeepSeekV3Attention):
    def __init__(self, q_lora_rank: int = 0):
        Module.__init__(self)
        self.dim = 4
        self.n_heads = 1
        self.q_lora_rank = q_lora_rank
        self.kv_lora_rank = 2
        self.qk_nope_head_dim = 2
        self.qk_rope_head_dim = 2
        self.qk_head_dim = 4
        self.v_head_dim = 2
        if self.q_lora_rank == 0:
            self.wq = _CountingLinear()
            self.AVAILABLE_REMAT_SAVE_REGIONS = (
                "wq",
                "wkv_a",
                "wkv_b",
                "inner_attention",
                "wo",
            )
        else:
            self.wq_a = _CountingLinear(4, 2)
            self.q_norm = torch.nn.Identity()
            self.wq_b = _CountingLinear(2, 4)
            self.AVAILABLE_REMAT_SAVE_REGIONS = (
                "wq_a",
                "wq_b",
                "wkv_a",
                "wkv_b",
                "inner_attention",
                "wo",
            )
        self.wkv_a = _CountingLinear()
        self.kv_norm = torch.nn.Identity()
        self.wkv_b = _CountingLinear(2, 4)
        self.wo = _CountingLinear(2, 4)
        self.softmax_scale = self.qk_head_dim**-0.5
        self.inner_attention = _CountingMLAInnerAttention()
        self.rope = _IdentityRoPE()


class _DeepSeekV3AttentionRematBlock(Module):
    def __init__(self, q_lora_rank: int = 0):
        super().__init__()
        self.attention = _CountingDeepSeekV3Attention(q_lora_rank)

    def forward(self, x_TD: torch.Tensor) -> torch.Tensor:
        return self.attention(x_TD, attention_masks=None).sum()


class _CountingDistGEMMFeedForward(DistGEMMFeedForward):
    def __init__(self):
        Module.__init__(self)
        self.w1 = _CountingLinear()
        self.w2 = _CountingLinear()
        self.w3 = _CountingLinear()


class _DistGEMMRematBlock(Module):
    def __init__(self):
        super().__init__()
        self.feed_forward = _CountingDistGEMMFeedForward()

    def forward(self, x_TD: torch.Tensor) -> torch.Tensor:
        return self.feed_forward(x_TD).sum()


class _CountingSigmoidGatedFeedForward(SigmoidGatedFeedForward):
    def __init__(self):
        Module.__init__(self)
        self.w1 = _CountingLinear()
        self.w2 = _CountingLinear()
        self.w3 = _CountingLinear()
        self.gate = _CountingLinear()


class _SigmoidGatedFeedForwardRematBlock(Module):
    def __init__(self):
        super().__init__()
        self.feed_forward = _CountingSigmoidGatedFeedForward()

    def forward(self, x_TD: torch.Tensor) -> torch.Tensor:
        return self.feed_forward(x_TD).sum()


class _CountingTopKRouter(TokenChoiceTopKRouter):
    def __init__(self):
        super().__init__(
            TokenChoiceTopKRouter.Config(
                num_experts=4,
                gate=Linear.Config(in_features=4, out_features=4),
                num_expert_groups=2,
                num_limited_groups=1,
                top_k=1,
            )
        )
        self.gate = _CountingLinear()
        self.num_routing_decisions = 0

    def _select_expert_ids(self, scores_for_choice_TE: torch.Tensor) -> torch.Tensor:
        self.num_routing_decisions += 1
        return super()._select_expert_ids(scores_for_choice_TE)


class _RouterRematBlock(Module):
    AVAILABLE_REMAT_SAVE_REGIONS = ("router",)

    def __init__(self):
        super().__init__()
        self.router = _CountingTopKRouter()

    def forward(self, x_TD: torch.Tensor) -> torch.Tensor:
        topk_scores_TK, topk_expert_ids_TK, scores_TE = remat.region(
            self.router,
            self.remat_region_name("router"),
            recompute=self.should_recompute_remat_region("router"),
        )(x_TD)
        remat.recompute_needs_tensor(topk_scores_TK, topk_expert_ids_TK, scores_TE)
        return topk_scores_TK.sum() + scores_TE.sum()


class _CountingLocalTokenDispatcher(LocalTokenDispatcher):
    def __init__(self):
        super().__init__(LocalTokenDispatcher.Config(num_experts=2, top_k=1))
        self.num_dispatches = 0
        self.num_combines = 0

    def dispatch(
        self,
        x_TD: torch.Tensor,
        topk_scores_TK: torch.Tensor,
        topk_expert_ids_TK: torch.Tensor,
        num_local_tokens_per_expert_E: torch.Tensor,
    ):
        self.num_dispatches += 1
        return super().dispatch(
            x_TD,
            topk_scores_TK,
            topk_expert_ids_TK,
            num_local_tokens_per_expert_E,
        )

    def combine(self, routed_output_RD, metadata, x_TD):
        self.num_combines += 1
        return super().combine(routed_output_RD, metadata, x_TD)


class _CountingExperts(Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.randn(()))
        self.num_forwards = 0

    def forward(
        self,
        x_RD: torch.Tensor,
        num_tokens_per_expert_E: torch.Tensor,
    ) -> torch.Tensor:
        del num_tokens_per_expert_E
        self.num_forwards += 1
        return x_RD * self.weight


class _CountingRoutedExperts(RoutedExperts):
    AVAILABLE_REMAT_SAVE_REGIONS = ("dispatch", "combine")

    def __init__(self):
        Module.__init__(self)
        self.inner_experts = _CountingExperts()
        self.token_dispatcher = _CountingLocalTokenDispatcher()


class _RoutedExpertsRematBlock(Module):
    def __init__(self):
        super().__init__()
        self.routed_experts = _CountingRoutedExperts()

    def forward(self, x_TD: torch.Tensor) -> torch.Tensor:
        topk_scores_TK = torch.sigmoid(x_TD[:, :1])
        topk_expert_ids_TK = (
            torch.arange(x_TD.shape[0], device=x_TD.device, dtype=torch.int64)
            .remainder(2)
            .unsqueeze(-1)
        )
        num_local_tokens_per_expert_E = torch.bincount(
            topk_expert_ids_TK.flatten(), minlength=2
        )
        return self.routed_experts(
            x_TD,
            topk_scores_TK,
            topk_expert_ids_TK,
            num_local_tokens_per_expert_E,
        ).sum()


class _SideEffectBlock(_RematBlock):
    AVAILABLE_REMAT_SAVE_REGIONS = _RematBlock.AVAILABLE_REMAT_SAVE_REGIONS
    num_forwards: torch.Tensor

    def __init__(self):
        super().__init__()
        self.register_buffer("num_forwards", torch.zeros((), dtype=torch.int64))

    def forward(self, x_BD: torch.Tensor) -> torch.Tensor:
        if not remat.is_recomputing():
            self.num_forwards.add_(1)
        return super().forward(x_BD)


class _RematModel(Module):
    def __init__(self, block: Module | None = None):
        super().__init__()
        self.layers = ModuleDict({"0": block if block is not None else _RematBlock()})

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


class TestRematRegions(unittest.TestCase):
    def test_region_policy_accessors_default_to_local_recomputation(self):
        owner = _RegionOwner()

        self.assertEqual(owner.remat_region_name("w1"), "w1")
        self.assertTrue(owner.should_recompute_remat_region("w1"))

        configure_remat_save_regions(owner, ["w1"])

        self.assertEqual(owner.remat_region_name("w1"), "w1")
        self.assertFalse(owner.should_recompute_remat_region("w1"))
        self.assertTrue(owner.should_recompute_remat_region("w2"))

    def test_region_discovery_and_glob_selection(self):
        tree = _RegionTree()

        self.assertEqual(
            available_remat_save_regions(tree),
            ["left.w1", "left.w2", "right.w1", "right.w2"],
        )
        selected_save_regions, available_save_regions = configure_remat_save_regions(
            tree, ["*.w2", "left.w1"]
        )

        self.assertEqual(
            selected_save_regions,
            ["left.w1", "left.w2", "right.w2"],
        )
        self.assertEqual(available_save_regions, available_remat_save_regions(tree))
        self.assertEqual(
            tree.left.remat_region_names,
            {"w1": "left.w1", "w2": "left.w2"},
        )
        self.assertEqual(
            tree.left.is_remat_save_region,
            {"w1": True, "w2": True},
        )
        self.assertEqual(tree.left.remat_region_name("w1"), "left.w1")
        self.assertFalse(tree.left.should_recompute_remat_region("w1"))
        self.assertEqual(
            tree.right.remat_region_names,
            {"w1": "right.w1", "w2": "right.w2"},
        )
        self.assertEqual(
            tree.right.is_remat_save_region,
            {"w1": False, "w2": True},
        )

    def test_required_regions_are_selected_without_a_user_pattern(self):
        owner = _RequiredRegionOwner()

        selected_save_regions, available_save_regions = configure_remat_save_regions(
            owner, []
        )

        self.assertEqual(available_save_regions, ["w1", "w2"])
        self.assertEqual(required_remat_save_regions(owner), ["w2"])
        self.assertEqual(selected_save_regions, ["w2"])
        self.assertEqual(owner.remat_region_names, {"w1": "w1", "w2": "w2"})
        self.assertEqual(
            owner.is_remat_save_region,
            {"w1": False, "w2": True},
        )

    def test_required_regions_must_be_available(self):
        with self.assertRaisesRegex(AssertionError, "must be a subset"):
            configure_remat_save_regions(_InvalidRequiredRegionOwner(), [])

    def test_deeply_nested_save_region_names(self):
        model = _RematModel(_DeeplyNestedRegionBlock())
        block = model.layers["0"]
        expected_save_regions = [
            "attention.projections.query.w1",
            "attention.projections.query.w2",
            "feed_forward.experts.shared.w1",
            "feed_forward.experts.shared.w2",
        ]

        self.assertEqual(available_remat_save_regions(block), expected_save_regions)
        RematAC.Config(
            save_regions=[
                "attention.projections.query.w1",
                "feed_forward.experts.shared.*",
            ]
        ).build().apply(model)

    def test_forward_override_must_redeclare_regions(self):
        self.assertEqual(
            available_remat_save_regions(_InheritedRegionOwner()), ["w1", "w2"]
        )
        self.assertEqual(available_remat_save_regions(_OverriddenRegionOwner()), [])

    def test_compile_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "does not support torch.compile"):
            validate_activation_checkpointing_compile(
                RematAC.Config(save_regions=[]), model_compile_enabled=True
            )

        validate_activation_checkpointing_compile(
            RematAC.Config(save_regions=[]), model_compile_enabled=False
        )

    def test_save_regions_are_required(self):
        save_regions_field = next(
            field for field in fields(RematAC.Config) if field.name == "save_regions"
        )

        self.assertIs(save_regions_field.default, MISSING)
        self.assertIs(save_regions_field.default_factory, MISSING)

    def test_unmatched_region_pattern_warns(self):
        model = _RematModel()

        with self.assertLogs(level="WARNING"):
            RematAC.Config(save_regions=["missing"]).build().apply(model)

        _run_forward_backward(model, torch.randn(3, 4))
        block = model.layers["0"]
        assert isinstance(block, _RematBlock)
        self.assertEqual(block.saved.num_forwards, 2)
        self.assertEqual(block.recomputed.num_forwards, 2)

    def test_model_without_available_save_regions_is_rejected(self):
        model = _RematModel(Module())

        with self.assertRaisesRegex(ValueError, "does not provide any"):
            RematAC.Config(save_regions=[]).build().apply(model)

    def test_empty_pipeline_partition_is_ignored(self):
        model = _RematModel()
        model.layers = ModuleDict()

        RematAC.Config(save_regions=["missing"]).build().apply(model)

    def test_forward_backward_matches_without_checkpointing(self):
        torch.manual_seed(42)
        baseline = _RematModel()
        remat_model = deepcopy(baseline)
        baseline_state_keys = list(baseline.state_dict())

        RematAC.Config(save_regions=["saved"]).build().apply(remat_model)

        x_BD = torch.randn(3, 4)
        expected = _run_forward_backward(baseline, x_BD)
        actual = _run_forward_backward(remat_model, x_BD)

        torch.testing.assert_close(actual[0], expected[0], rtol=0, atol=0)
        torch.testing.assert_close(actual[1], expected[1], rtol=0, atol=0)
        self.assertEqual(len(actual[2]), len(expected[2]))
        for actual_grad, expected_grad in zip(actual[2], expected[2]):
            torch.testing.assert_close(actual_grad, expected_grad, rtol=0, atol=0)

        remat_block = remat_model.layers["0"]
        assert isinstance(remat_block, _RematBlock)
        self.assertEqual(remat_block.saved.num_forwards, 1)
        self.assertEqual(remat_block.recomputed.num_forwards, 2)
        self.assertEqual(list(remat_model.state_dict()), baseline_state_keys)

    def test_common_vision_regions_are_available(self):
        self.assertEqual(
            available_remat_save_regions(_VisionRematBlock()),
            [
                "attn.qkv",
                "attn.inner_attention",
                "attn.proj",
                "mlp.fc1",
                "mlp.fc2",
            ],
        )

    def test_each_common_vision_region_skips_its_recomputation(self):
        for save_region, expected_counts in (
            ("attn.qkv", (1, 1, 1, 2, 2, 2, 2)),
            ("attn.inner_attention", (2, 2, 2, 1, 2, 2, 2)),
            ("attn.proj", (2, 2, 2, 2, 1, 2, 2)),
            ("mlp.fc1", (2, 2, 2, 2, 2, 1, 2)),
            ("mlp.fc2", (2, 2, 2, 2, 2, 2, 1)),
        ):
            with self.subTest(save_region=save_region):
                torch.manual_seed(42)
                baseline = _VisionRematModel()
                remat_model = deepcopy(baseline)
                RematAC.Config(save_regions=[save_region]).build().apply(remat_model)

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
                assert isinstance(block, _VisionRematBlock)
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

    def test_forward_side_effect_can_skip_recompute(self):
        model = _RematModel(_SideEffectBlock())
        RematAC.Config(save_regions=["saved"]).build().apply(model)

        _run_forward_backward(model, torch.randn(3, 4))

        block = model.layers["0"]
        assert isinstance(block, _SideEffectBlock)
        self.assertEqual(block.num_forwards.item(), 1)

    def test_router_decision_is_retained_without_a_user_pattern(self):
        torch.manual_seed(42)
        baseline = _RematModel(_RouterRematBlock())
        remat_model = deepcopy(baseline)

        RematAC.Config(save_regions=[]).build().apply(remat_model)

        x_TD = torch.randn(3, 4)
        expected = _run_forward_backward(baseline, x_TD)
        actual = _run_forward_backward(remat_model, x_TD)

        torch.testing.assert_close(actual[0], expected[0], rtol=0, atol=0)
        torch.testing.assert_close(actual[1], expected[1], rtol=0, atol=0)
        for actual_grad, expected_grad in zip(actual[2], expected[2]):
            torch.testing.assert_close(actual_grad, expected_grad, rtol=0, atol=0)

        block = remat_model.layers["0"]
        assert isinstance(block, _RouterRematBlock)
        self.assertEqual(block.router.gate.num_forwards, 2)
        self.assertEqual(block.router.num_routing_decisions, 1)
        self.assertEqual(
            block.router.remat_region_names,
            {"routing_decision": "router.routing_decision"},
        )
        self.assertEqual(
            block.router.is_remat_save_region,
            {"routing_decision": True},
        )

    def test_required_router_decision_can_be_nested_in_saved_router(self):
        model = _RematModel(_RouterRematBlock())

        RematAC.Config(save_regions=["router"]).build().apply(model)
        _run_forward_backward(model, torch.randn(3, 4))

        block = model.layers["0"]
        assert isinstance(block, _RouterRematBlock)
        self.assertEqual(block.router.gate.num_forwards, 1)
        self.assertEqual(block.router.num_routing_decisions, 1)

    def test_routed_expert_dispatch_regions_are_backend_specific(self):
        inner_experts = GroupedExperts.Config(dim=4, hidden_dim=8, num_experts=2)
        local = RoutedExperts.Config(
            inner_experts=inner_experts,
            token_dispatcher=LocalTokenDispatcher.Config(num_experts=2, top_k=1),
        ).build()
        all_to_all = RoutedExperts.Config(
            inner_experts=inner_experts,
            token_dispatcher=AllToAllTokenDispatcher.Config(num_experts=2, top_k=1),
        ).build()
        with patch.object(DeepEPTokenDispatcher, "__init__", return_value=None):
            deep_ep = RoutedExperts.Config(
                inner_experts=inner_experts,
                token_dispatcher=DeepEPTokenDispatcher.Config(
                    num_experts=2,
                    top_k=1,
                    hidden_dim=4,
                    num_max_tokens_per_rank=4,
                ),
            ).build()

        self.assertEqual(
            available_remat_save_regions(local),
            [
                "dispatch",
                "combine",
                "inner_experts.w1",
                "inner_experts.w3",
                "inner_experts.w2",
            ],
        )
        self.assertEqual(
            available_remat_save_regions(all_to_all),
            [
                "dispatch",
                "combine",
                "inner_experts.w1",
                "inner_experts.w3",
                "inner_experts.w2",
            ],
        )
        self.assertEqual(
            available_remat_save_regions(deep_ep),
            ["inner_experts.w1", "inner_experts.w3", "inner_experts.w2"],
        )

        unsupported = RoutedExperts.Config(
            inner_experts=inner_experts,
            token_dispatcher=TorchAOTokenDispatcher.Config(
                num_experts=2, top_k=1, pad_multiple=16
            ),
        ).build()
        self.assertEqual(
            available_remat_save_regions(unsupported),
            ["inner_experts.w1", "inner_experts.w3", "inner_experts.w2"],
        )

    def test_routed_expert_regions_skip_collective_recomputation(self):
        for save_regions, expected_counts in (
            (["routed_experts.dispatch"], (1, 2, 2)),
            (["routed_experts.combine"], (2, 2, 1)),
            (["routed_experts.*"], (1, 2, 1)),
        ):
            with self.subTest(save_regions=save_regions):
                torch.manual_seed(42)
                baseline = _RematModel(_RoutedExpertsRematBlock())
                remat_model = deepcopy(baseline)
                RematAC.Config(save_regions=save_regions).build().apply(remat_model)

                x_TD = torch.randn(4, 4)
                expected = _run_forward_backward(baseline, x_TD)
                actual = _run_forward_backward(remat_model, x_TD)

                torch.testing.assert_close(actual[0], expected[0], rtol=0, atol=0)
                torch.testing.assert_close(actual[1], expected[1], rtol=0, atol=0)
                for actual_grad, expected_grad in zip(actual[2], expected[2]):
                    torch.testing.assert_close(
                        actual_grad, expected_grad, rtol=0, atol=0
                    )

                block = remat_model.layers["0"]
                assert isinstance(block, _RoutedExpertsRematBlock)
                self.assertEqual(
                    (
                        block.routed_experts.token_dispatcher.num_dispatches,
                        block.routed_experts.inner_experts.num_forwards,
                        block.routed_experts.token_dispatcher.num_combines,
                    ),
                    expected_counts,
                )

    def test_deepseek_nvidia_megatron_h100_policy_regions_are_available(self):
        from torchtitan.models.deepseek_v3 import model_registry
        from torchtitan.models.deepseek_v3.remat import (
            deepseek_v3_nvidia_megatron_h100_remat_config,
        )

        model_spec = model_registry("debugmodel")
        with torch.device("meta"):
            model = model_spec.model.build()
        baseline_state_keys = list(model.state_dict())

        remat_config = deepseek_v3_nvidia_megatron_h100_remat_config()
        available_save_regions = list(
            dict.fromkeys(
                region
                for block in model.layers.values()
                for region in available_remat_save_regions(block)
            )
        )
        self.assertTrue(
            all(
                any(fnmatch(region, pattern) for region in available_save_regions)
                for pattern in remat_config.save_regions
            )
        )
        self.assertIn("attention.wkv_b", available_save_regions)
        self.assertNotIn("attention.wkv_b", remat_config.save_regions)
        remat_config.build().apply(model)

        self.assertEqual(list(model.state_dict()), baseline_state_keys)

    def test_each_deepseek_attention_region_skips_its_recomputation(self):
        for save_region, expected_counts in (
            ("attention.wq", (1, 2, 2, 2, 2)),
            ("attention.wkv_a", (2, 1, 2, 2, 2)),
            ("attention.wkv_b", (2, 2, 1, 2, 2)),
            ("attention.inner_attention", (2, 2, 2, 1, 2)),
            ("attention.wo", (2, 2, 2, 2, 1)),
        ):
            with self.subTest(save_region=save_region):
                torch.manual_seed(42)
                baseline = _RematModel(_DeepSeekV3AttentionRematBlock())
                remat_model = deepcopy(baseline)
                RematAC.Config(save_regions=[save_region]).build().apply(remat_model)

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
                assert isinstance(block, _DeepSeekV3AttentionRematBlock)
                actual_counts = (
                    block.attention.wq.num_forwards,
                    block.attention.wkv_a.num_forwards,
                    block.attention.wkv_b.num_forwards,
                    block.attention.inner_attention.num_forwards,
                    block.attention.wo.num_forwards,
                )
                self.assertEqual(actual_counts, expected_counts)

    def test_deepseek_query_regions_follow_q_lora_configuration(self):
        direct_attention = _CountingDeepSeekV3Attention(q_lora_rank=0)
        lora_attention = _CountingDeepSeekV3Attention(q_lora_rank=2)

        self.assertEqual(
            available_remat_save_regions(direct_attention),
            ["wq", "wkv_a", "wkv_b", "inner_attention", "wo"],
        )
        self.assertEqual(
            available_remat_save_regions(lora_attention),
            ["wq_a", "wq_b", "wkv_a", "wkv_b", "inner_attention", "wo"],
        )

    def test_deepseek_model_flavors_expose_their_query_regions(self):
        from torchtitan.models.deepseek_v3 import model_registry

        for flavor, expected_query_regions in (
            ("debugmodel", ["attention.wq"]),
            ("debugmodel_q_lora", ["attention.wq_a", "attention.wq_b"]),
        ):
            with self.subTest(flavor=flavor), torch.device("meta"):
                model = model_registry(flavor).model.build()

            available_save_regions = available_remat_save_regions(model.layers["0"])
            actual_query_regions = [
                region
                for region in available_save_regions
                if region.startswith("attention.wq")
            ]
            self.assertEqual(actual_query_regions, expected_query_regions)

    def test_deepseek_query_lora_regions_skip_recomputation_independently(self):
        for save_region, expected_counts in (
            ("attention.wq_a", (1, 2)),
            ("attention.wq_b", (2, 1)),
        ):
            with self.subTest(save_region=save_region):
                torch.manual_seed(42)
                baseline = _RematModel(_DeepSeekV3AttentionRematBlock(q_lora_rank=2))
                remat_model = deepcopy(baseline)
                RematAC.Config(save_regions=[save_region]).build().apply(remat_model)

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
                assert isinstance(block, _DeepSeekV3AttentionRematBlock)
                self.assertEqual(
                    (
                        block.attention.wq_a.num_forwards,
                        block.attention.wq_b.num_forwards,
                    ),
                    expected_counts,
                )

    def test_gqa_save_regions_cover_fused_and_non_fused_qkv(self):
        from torchtitan.models.llama3 import model_registry as llama3_model_registry
        from torchtitan.models.qwen3 import model_registry as qwen3_model_registry

        expected_save_regions = [
            "attention.qkv",
            "attention.inner_attention",
            "attention.wo",
            "feed_forward.w1",
            "feed_forward.w3",
            "feed_forward.w2",
        ]
        model_specs = (
            llama3_model_registry("debugmodel"),
            qwen3_model_registry("debugmodel"),
            qwen3_model_registry("debugmodel_non_fused_qkv"),
        )

        for model_spec in model_specs:
            with self.subTest(model=model_spec.name, flavor=model_spec.flavor):
                with torch.device("meta"):
                    model = model_spec.model.build()
                self.assertEqual(
                    available_remat_save_regions(model.layers["0"]),
                    expected_save_regions,
                )

    def test_integration_recipe_remat_policies_match_model_variants(self):
        from torchtitan.models.llama3 import model_registry as llama3_model_registry
        from torchtitan.models.muse_glimmer import (
            model_registry as muse_glimmer_model_registry,
        )
        from torchtitan.models.qwen3 import model_registry as qwen3_model_registry
        from torchtitan_recipes.tests import (
            h100 as h100_recipes,
            models as model_recipes,
        )

        cases = (
            (
                llama3_model_registry("debugmodel"),
                model_recipes.llama3_debugmodel_remat_fsdp2_tp2_cp2().activation_checkpoint,
            ),
            (
                llama3_model_registry("debugmodel", tp_gemm_backend="dist_gemm"),
                h100_recipes.llama3_debugmodel_dist_gemm_remat_tp2().activation_checkpoint,
            ),
            (
                qwen3_model_registry("debugmodel_moe"),
                model_recipes.qwen3_debugmodel_moe_param_groups_remat_fsdp2_tp2_cp2_ep8().activation_checkpoint,
            ),
            (
                muse_glimmer_model_registry("debugmodel"),
                model_recipes.muse_glimmer_debugmodel_remat_fsdp8().activation_checkpoint,
            ),
        )

        for model_spec, remat_config in cases:
            with self.subTest(model=model_spec.name, flavor=model_spec.flavor):
                assert isinstance(remat_config, RematAC.Config)
                with torch.device("meta"):
                    model = model_spec.model.build()
                available_save_regions = list(
                    dict.fromkeys(
                        region
                        for block in model.layers.values()
                        for region in available_remat_save_regions(block)
                    )
                )
                selected_save_regions = list(
                    dict.fromkeys(
                        region
                        for block in model.layers.values()
                        for region in configure_remat_save_regions(
                            block, remat_config.save_regions
                        )[0]
                    )
                )
                self.assertCountEqual(selected_save_regions, available_save_regions)

    def test_each_gqa_save_region_skips_its_recomputation(self):
        for save_region, expected_counts in (
            ("attention.qkv", (1, 2, 2)),
            ("attention.inner_attention", (2, 1, 2)),
            ("attention.wo", (2, 2, 1)),
            ("attention.*", (1, 1, 1)),
        ):
            with self.subTest(save_region=save_region):
                torch.manual_seed(42)
                baseline = _RematModel(_GQARematBlock())
                remat_model = deepcopy(baseline)
                RematAC.Config(save_regions=[save_region]).build().apply(remat_model)

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
                assert isinstance(block, _GQARematBlock)
                actual_counts = (
                    block.attention.qkv_linear.num_forwards,
                    block.attention.inner_attention.num_forwards,
                    block.attention.wo.num_forwards,
                )
                self.assertEqual(actual_counts, expected_counts)

    def test_each_muse_attention_region_skips_its_recomputation(self):
        from torchtitan.models.muse_glimmer import model_registry
        from torchtitan.models.muse_glimmer.model import Attention as MuseAttention

        with torch.device("meta"):
            model = model_registry("debugmodel").model.build()
        self.assertEqual(
            available_remat_save_regions(model.layers["0"]),
            [
                "attention.qkv",
                "attention.inner_attention",
                "attention.o_gate",
                "attention.wo",
                "feed_forward.w1",
                "feed_forward.w3",
                "feed_forward.w2",
            ],
        )

        class CountingMuseAttention(MuseAttention):
            def __init__(self):
                Module.__init__(self)
                self.n_heads = 1
                self.n_kv_heads = 1
                self.head_dim = 4
                self.enable_gqa = False
                self.rope = _IdentityRoPE()
                self.qkv_linear = _CountingQKV()
                self.wo = _CountingLinear()
                self.inner_attention = _CountingInnerAttention()
                self.o_gate = _CountingLinear()
                self.q_norm = None
                self.k_norm = None
                self.scaling = None
                self.scale_query_by = 1.0
                self.use_rope = False
                self.window_size = None

        class MuseRematBlock(Module):
            def __init__(self):
                super().__init__()
                self.attention = CountingMuseAttention()

            def forward(self, x_TD: torch.Tensor) -> torch.Tensor:
                return self.attention(x_TD, attention_masks=None).sum()

        for save_region, expected_counts in (
            ("attention.qkv", (1, 2, 2, 2)),
            ("attention.inner_attention", (2, 1, 2, 2)),
            ("attention.o_gate", (2, 2, 1, 2)),
            ("attention.wo", (2, 2, 2, 1)),
            ("attention.*", (1, 1, 1, 1)),
        ):
            with self.subTest(save_region=save_region):
                torch.manual_seed(42)
                baseline = _RematModel(MuseRematBlock())
                remat_model = deepcopy(baseline)
                RematAC.Config(save_regions=[save_region]).build().apply(remat_model)

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
                assert isinstance(block, MuseRematBlock)
                assert block.attention.o_gate is not None
                actual_counts = (
                    block.attention.qkv_linear.num_forwards,
                    block.attention.inner_attention.num_forwards,
                    block.attention.o_gate.num_forwards,
                    block.attention.wo.num_forwards,
                )
                self.assertEqual(actual_counts, expected_counts)

    def test_each_dist_gemm_ffn_region_skips_its_recomputation(self):
        for save_region, expected_counts in (
            ("feed_forward.w13", (1, 1, 2)),
            ("feed_forward.w2", (2, 2, 1)),
            ("feed_forward.*", (1, 1, 1)),
        ):
            with self.subTest(save_region=save_region):
                torch.manual_seed(42)
                baseline = _RematModel(_DistGEMMRematBlock())
                remat_model = deepcopy(baseline)
                RematAC.Config(save_regions=[save_region]).build().apply(remat_model)

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
                assert isinstance(block, _DistGEMMRematBlock)
                actual_counts = (
                    block.feed_forward.w1.num_forwards,
                    block.feed_forward.w3.num_forwards,
                    block.feed_forward.w2.num_forwards,
                )
                self.assertEqual(actual_counts, expected_counts)

    def test_sigmoid_gated_feed_forward_gate_skips_recomputation(self):
        torch.manual_seed(42)
        baseline = _RematModel(_SigmoidGatedFeedForwardRematBlock())
        remat_model = deepcopy(baseline)

        block = remat_model.layers["0"]
        assert isinstance(block, _SigmoidGatedFeedForwardRematBlock)
        self.assertEqual(
            available_remat_save_regions(block),
            [
                "feed_forward.w1",
                "feed_forward.w3",
                "feed_forward.w2",
                "feed_forward.gate",
            ],
        )
        RematAC.Config(save_regions=["feed_forward.gate"]).build().apply(remat_model)

        x_TD = torch.randn(3, 4)
        expected = _run_forward_backward(baseline, x_TD)
        actual = _run_forward_backward(remat_model, x_TD)

        torch.testing.assert_close(actual[0], expected[0], rtol=0, atol=0)
        torch.testing.assert_close(actual[1], expected[1], rtol=0, atol=0)
        for actual_grad, expected_grad in zip(actual[2], expected[2]):
            torch.testing.assert_close(actual_grad, expected_grad, rtol=0, atol=0)

        self.assertEqual(block.feed_forward.w1.num_forwards, 2)
        self.assertEqual(block.feed_forward.w3.num_forwards, 2)
        self.assertEqual(block.feed_forward.w2.num_forwards, 2)
        self.assertEqual(block.feed_forward.gate.num_forwards, 1)

    def test_region_patterns_for_other_pipeline_partitions_are_ignored(self):
        from torchtitan.models.deepseek_v3 import model_registry

        for layer_id, save_regions in (
            ("0", ["moe.router"]),
            ("1", ["feed_forward.w1"]),
        ):
            with self.subTest(layer_id=layer_id), torch.device("meta"):
                model = model_registry("debugmodel").model.build()
                model.layers = ModuleDict({layer_id: model.layers[layer_id]})

            with self.assertLogs(level="WARNING"):
                RematAC.Config(save_regions=save_regions).build().apply(model)


if __name__ == "__main__":
    unittest.main()
