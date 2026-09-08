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
from torchtitan.models.common.linear import Linear, RouterGateLinear
from torchtitan.models.common.moe import TokenChoiceTopKRouter
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
    """Return the remat region names reached by one forward."""
    with remat.collect_trace() as trace:
        forward()
    return [entry.name for entry in trace.entries]


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

        with self.assertLogs(level="INFO") as logs:
            RegionAC.Config(
                save_regions=["attention.*"], log_available_regions=True
            ).build().apply(model)

        self.assertEqual(list(model.state_dict()), state_keys)
        self.assertIn(
            "RegionAC available save regions for Llama3TransformerBlock layers",
            "\n".join(logs.output),
        )
        self.assertIn(
            "['attention.qkv', 'attention.inner_attention', 'attention.wo']",
            "\n".join(logs.output),
        )

    def test_attention_regions_are_discoverable(self):
        block = _AttentionBlock()
        self.assertEqual(
            block.available_remat_save_regions(),
            ["attention.qkv", "attention.inner_attention", "attention.wo"],
        )

    def test_available_regions_are_grouped_by_block_type(self):
        class _FirstBlock(Module):
            first_region = Module.register_remat_region("first")

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return x

        class _SecondBlock(Module):
            second_region = Module.register_remat_region("second")

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return x

        model = _RematModel(_FirstBlock())
        model.layers["1"] = _SecondBlock()

        with self.assertLogs(level="INFO") as logs:
            RegionAC.Config(save_regions=[], log_available_regions=True).build().apply(
                model
            )

        output = "\n".join(logs.output)
        self.assertIn(
            "RegionAC available save regions for _FirstBlock layers ['0']: "
            "['first']",
            output,
        )
        self.assertIn(
            "RegionAC available save regions for _SecondBlock layers ['1']: "
            "['second']",
            output,
        )

    def test_available_regions_are_not_logged_by_default(self):
        with self.assertLogs(level="INFO") as logs:
            RegionAC.Config(save_regions=[]).build().apply(
                _RematModel(_AttentionBlock())
            )

        self.assertNotIn("available save regions", "\n".join(logs.output))

    def test_unmatched_save_region_patterns_error(self):
        for save_regions in (
            ["attention.qvk"],
            ["attention.qkv", "attention.qvk"],
        ):
            with self.subTest(save_regions=save_regions), self.assertRaisesRegex(
                ValueError,
                "attention.qvk.*Available regions.*attention.qkv",
            ):
                RegionAC.Config(save_regions=save_regions).build().apply(
                    _RematModel(_AttentionBlock())
                )

    def test_attention_region_catalog_matches_forward(self):
        block = _AttentionBlock()
        model = _RematModel(block)
        RegionAC.Config(save_regions=[]).build().apply(model)
        self.assertEqual(
            _trace_region_names(lambda: model(torch.randn(3, 4))),
            block.available_remat_save_regions(),
        )

    def test_overridden_forward_does_not_inherit_regions(self):
        class _OverriddenAttention(_CountingGQAttention):
            def forward(
                self,
                x_TD: torch.Tensor,
                attention_masks,
                positions: torch.Tensor | None = None,
            ) -> torch.Tensor:
                return x_TD

        overridden_attention = _OverriddenAttention()
        self.assertEqual(overridden_attention.available_remat_save_regions(), [])
        with self.assertRaisesRegex(ValueError, "is not registered"):
            overridden_attention.remat_region_name(
                overridden_attention.qkv_remat_handle
            )

    def test_invalid_region_declarations_error(self):
        with self.assertRaisesRegex(ValueError, "must not be empty"):
            Module.register_remat_region("")

        class _DuplicateRegions(Module):
            first_region = Module.register_remat_region("duplicate")
            second_region = Module.register_remat_region("duplicate")

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return x

        with self.assertRaisesRegex(ValueError, "declares duplicate remat regions"):
            _DuplicateRegions().available_remat_save_regions()

        unregistered_region = Module.register_remat_region("unregistered")
        with self.assertRaisesRegex(
            ValueError, "must be assigned as a class attribute"
        ):
            _CountingGQAttention().remat_region_name(unregistered_region)

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

    def test_router_decision_is_always_saved(self):
        router = TokenChoiceTopKRouter.Config(
            num_experts=4,
            gate=RouterGateLinear.Config(in_features=4, out_features=4),
            num_expert_groups=2,
            num_limited_groups=1,
            top_k=1,
        ).build()

        def forward(x_TD: torch.Tensor) -> torch.Tensor:
            topk_scores_TK, _, scores_TE = router(x_TD)
            return topk_scores_TK.sum() + scores_TE.sum()

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
