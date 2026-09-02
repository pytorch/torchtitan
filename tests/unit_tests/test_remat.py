# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest

from copy import deepcopy
from dataclasses import fields, MISSING

import torch
import torch_remat as remat

from torchtitan.distributed.activation_checkpoint import (
    RematAC,
    validate_activation_checkpointing_compile,
)
from torchtitan.models.common.attention import GQAttention
from torchtitan.models.common.linear import Linear
from torchtitan.models.common.moe import TokenChoiceTopKRouter
from torchtitan.models.common.remat import (
    available_remat_save_regions,
    configure_remat_save_regions,
    resolve_remat_save_policy,
)
from torchtitan.protocols.module import Module, ModuleDict


class _RegionOwner(Module):
    AVAILABLE_REMAT_SAVE_REGIONS = ("w1", "w2")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x


class _InheritedRegionOwner(_RegionOwner):
    pass


class _OverriddenRegionOwner(_RegionOwner):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x


class _RegionTree(Module):
    def __init__(self):
        super().__init__()
        self.left = _RegionOwner()
        self.right = _RegionOwner()


class _CountingLinear(Linear):
    def __init__(self):
        super().__init__(Linear.Config(in_features=4, out_features=4))
        self.num_forwards = 0

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        self.num_forwards += 1
        return super().forward(input)


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


class _AttentionBlock(Module):
    def __init__(self):
        super().__init__()
        self.attention = _CountingGQAttention()

    def forward(self, x_TD: torch.Tensor) -> torch.Tensor:
        return self.attention(x_TD, attention_masks=None).sum()


class _FirstRegionBlock(Module):
    AVAILABLE_REMAT_SAVE_REGIONS = ("first",)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.sum()


class _SecondRegionBlock(Module):
    AVAILABLE_REMAT_SAVE_REGIONS = ("second",)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.sum()


class _RematModel(Module):
    def __init__(self, *blocks: Module):
        super().__init__()
        configured_blocks: tuple[Module, ...] = blocks or (_RematBlock(),)
        self.layers = ModuleDict(
            {str(layer_id): block for layer_id, block in enumerate(configured_blocks)}
        )

    def forward(self, x_BD: torch.Tensor) -> torch.Tensor:
        output = x_BD
        for block in self.layers.values():
            output = block(output)
        return output


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
        self.num_routing_decisions = 0

    def _select_expert_ids(self, scores_for_choice_TE: torch.Tensor) -> torch.Tensor:
        self.num_routing_decisions += 1
        return super()._select_expert_ids(scores_for_choice_TE)


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
    def test_module_region_configuration(self):
        owner = _RegionOwner()

        self.assertEqual(owner.remat_region_name("w1"), "w1")
        self.assertTrue(owner.should_recompute_remat_region("w1"))

        selected, available = configure_remat_save_regions(owner, ["w1"])

        self.assertEqual(selected, ["w1"])
        self.assertEqual(available, ["w1", "w2"])
        self.assertFalse(owner.should_recompute_remat_region("w1"))
        self.assertTrue(owner.should_recompute_remat_region("w2"))

    def test_region_discovery_and_glob_selection(self):
        tree = _RegionTree()

        self.assertEqual(
            available_remat_save_regions(tree),
            ["left.w1", "left.w2", "right.w1", "right.w2"],
        )
        selected, _ = configure_remat_save_regions(tree, ["*.w1"])
        self.assertEqual(selected, ["left.w1", "right.w1"])

    def test_forward_override_must_redeclare_regions(self):
        self.assertEqual(
            available_remat_save_regions(_InheritedRegionOwner()), ["w1", "w2"]
        )
        self.assertEqual(available_remat_save_regions(_OverriddenRegionOwner()), [])

    def test_save_regions_config_is_required(self):
        save_regions_field = next(
            field for field in fields(RematAC.Config) if field.name == "save_regions"
        )
        self.assertIs(save_regions_field.default, MISSING)
        self.assertIs(save_regions_field.default_factory, MISSING)

    def test_compile_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "does not support torch.compile"):
            validate_activation_checkpointing_compile(
                RematAC.Config(save_regions=[]), model_compile_enabled=True
            )

    def test_unmatched_region_pattern_errors(self):
        with self.assertRaisesRegex(ValueError, "did not match any"):
            RematAC.Config(save_regions=["missing"]).build().apply(_RematModel())

    def test_wildcard_expansion_and_grouped_policy_report(self):
        policy = resolve_remat_save_policy(
            (("layers.0", _AttentionBlock()), ("layers.1", _AttentionBlock())),
            ["attention.*"],
        )

        policy.validate()

        self.assertEqual(
            policy.matches_by_pattern(),
            {
                "attention.*": [
                    "attention.qkv",
                    "attention.inner_attention",
                    "attention.wo",
                ]
            },
        )
        report = policy.format()
        self.assertIn("[layers.0-1] [_AttentionBlock]", report)
        self.assertIn("attention.qkv [_CountingGQAttention] -> SAVE", report)

    def test_model_without_available_save_regions_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "does not provide any"):
            RematAC.Config(save_regions=[]).build().apply(_RematModel(Module()))

    def test_globally_validated_pattern_can_be_absent_from_pipeline_partition(self):
        model = _RematModel(_FirstRegionBlock(), _SecondRegionBlock())
        config = RematAC.Config(save_regions=["second"])
        RematAC.validate_save_regions(config, model)
        model.layers = ModuleDict({"0": model.layers["0"]})

        config.build().apply(model)

    def test_forward_backward_matches_without_checkpointing(self):
        torch.manual_seed(42)
        baseline = _RematModel()
        remat_model = deepcopy(baseline)
        RematAC.Config(save_regions=["saved"]).build().apply(remat_model)

        x_BD = torch.randn(3, 4)
        expected = _run_forward_backward(baseline, x_BD)
        actual = _run_forward_backward(remat_model, x_BD)

        torch.testing.assert_close(actual[0], expected[0], rtol=0, atol=0)
        torch.testing.assert_close(actual[1], expected[1], rtol=0, atol=0)
        for actual_grad, expected_grad in zip(actual[2], expected[2]):
            torch.testing.assert_close(actual_grad, expected_grad, rtol=0, atol=0)

        block = remat_model.layers["0"]
        assert isinstance(block, _RematBlock)
        self.assertEqual(block.saved.num_forwards, 1)
        self.assertEqual(block.recomputed.num_forwards, 2)

    def test_common_attention_regions_are_available(self):
        self.assertEqual(
            available_remat_save_regions(_AttentionBlock()),
            ["attention.qkv", "attention.inner_attention", "attention.wo"],
        )

    def test_llama_attention_policy_applies_without_changing_state_dict(self):
        from torchtitan.models.llama3 import model_registry

        with torch.device("meta"):
            model = model_registry("debugmodel").model.build()
        state_keys = list(model.state_dict())

        RematAC.Config(save_regions=["attention.*"]).build().apply(model)

        self.assertEqual(list(model.state_dict()), state_keys)
        self.assertEqual(
            available_remat_save_regions(model.layers["0"]),
            ["attention.qkv", "attention.inner_attention", "attention.wo"],
        )

    def test_each_common_attention_region_skips_recomputation(self):
        for save_region, expected_counts in (
            ("attention.qkv", (1, 2, 2)),
            ("attention.inner_attention", (2, 1, 2)),
            ("attention.wo", (2, 2, 1)),
        ):
            with self.subTest(save_region=save_region):
                torch.manual_seed(42)
                baseline = _RematModel(_AttentionBlock())
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
                assert isinstance(block, _AttentionBlock)
                self.assertEqual(
                    (
                        block.attention.qkv_linear.num_forwards,
                        block.attention.inner_attention.num_forwards,
                        block.attention.wo.num_forwards,
                    ),
                    expected_counts,
                )

    def test_router_decision_is_always_saved(self):
        router = _CountingTopKRouter()
        self.assertEqual(available_remat_save_regions(router), [])

        def forward(x_TD: torch.Tensor) -> torch.Tensor:
            topk_scores_TK, _, scores_TE = router(x_TD)
            return topk_scores_TK.sum() + scores_TE.sum()

        checkpointed_forward = remat.checkpoint(
            region_name="transformer_block", preserve_rng_state=False
        )(forward)
        output = checkpointed_forward(torch.randn(3, 4, requires_grad=True))
        output.backward()

        self.assertEqual(router.num_routing_decisions, 1)


if __name__ == "__main__":
    unittest.main()
