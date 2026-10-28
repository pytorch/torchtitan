# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""The local combine remat region, shared by the standard dispatchers at EP=1."""

import gc
import unittest
from unittest.mock import patch

import torch
import torch.nn as nn
import torch_remat as remat

from torch.multiprocessing.reductions import StorageWeakRef

import torchtitan.models.common.moe.token_dispatcher as token_dispatcher_module
from torchtitan.distributed.activation_remat import (
    ActivationRematConfig,
    DefaultActivationRematConfig,
    FullActivationRematConfig,
)
from torchtitan.models.base.module import Module, ModuleDict
from torchtitan.models.common.moe.token_dispatcher import (
    AllToAllTokenDispatcher,
    LocalTokenDispatcher,
)

_MODEL_DIM = 8
_NUM_TOKENS = 16
_NUM_EXPERTS = 4
_TOP_K = 2


class _W2Region(Module):
    """Stand-in for the w2 grouped projection: one policy-controlled region."""

    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(_MODEL_DIM, _MODEL_DIM))
        self.num_forwards = 0

    def forward(self, x_RD: torch.Tensor) -> torch.Tensor:
        return remat.region(
            self._project,
            self.remat_region_name("grouped_mm"),
            recompute=self.remat_should_recompute("grouped_mm"),
        )(x_RD)

    def _project(self, x_RD: torch.Tensor) -> torch.Tensor:
        self.num_forwards += 1
        return x_RD @ self.weight


class _DispatcherBlock(Module):
    """Routed-experts block: dispatch, one projection, combine."""

    def __init__(self, dispatcher_config: LocalTokenDispatcher.Config):
        super().__init__()
        self.token_dispatcher = dispatcher_config.build()
        self.w2 = _W2Region()
        self.gate = nn.Linear(_MODEL_DIM, _NUM_EXPERTS, bias=False)
        self.num_experts = _NUM_EXPERTS
        self.top_k = dispatcher_config.top_k

    def forward(self, x_TD: torch.Tensor) -> torch.Tensor:
        num_tokens = x_TD.shape[0]
        token_ids_T1 = torch.arange(num_tokens, device=x_TD.device).unsqueeze(-1)
        offsets_1K = torch.arange(self.top_k, device=x_TD.device).unsqueeze(0)
        expert_ids_TK = (token_ids_T1 + offsets_1K) % self.num_experts
        scores_TK = torch.sigmoid(self.gate(x_TD)).gather(1, expert_ids_TK)
        routing_map_TE = torch.nn.functional.one_hot(
            expert_ids_TK, num_classes=self.num_experts
        ).any(dim=1)
        routed_input_RD, _num_tokens_E, metadata = self.token_dispatcher.dispatch(
            x_TD, scores_TK, expert_ids_TK, routing_map_TE
        )
        routed_scores_R = getattr(metadata, "routed_scores_R", None)
        if routed_scores_R is not None:
            # Absorbed scores scale the projection input, as GroupedLinear does.
            remat.recompute_needs_tensor(routed_scores_R)
            routed_input_RD = (
                routed_input_RD.float() * routed_scores_R.reshape(-1, 1)
            ).to(routed_input_RD.dtype)
        routed_output_RD = self.w2(routed_input_RD)
        out_TD = self.token_dispatcher.combine(routed_output_RD, metadata, x_TD)
        # The loss is a bare consumer of the combine output.
        remat.recompute_needs_tensor(out_TD)
        return out_TD.double().sum()


class _Model(Module):
    def __init__(self, block: Module):
        super().__init__()
        self.layers = ModuleDict({"0": block})

    def forward(self, x_TD: torch.Tensor) -> torch.Tensor:
        return self.layers["0"](x_TD)


def _forward_backward(
    model: Module, x_TD: torch.Tensor
) -> tuple[torch.Tensor, list[torch.Tensor]]:
    """Run one forward/backward; return the loss and input/parameter grads."""
    model.zero_grad(set_to_none=True)
    input_TD = x_TD.detach().clone().requires_grad_(True)
    loss = model(input_TD)
    loss.backward()
    assert input_TD.grad is not None
    grads = [input_TD.grad.detach().clone()]
    grads += [parameter.grad.detach().clone() for parameter in model.parameters()]
    return loss.detach().clone(), grads


class TestLocalCombineRematRegion(unittest.TestCase):
    @staticmethod
    def _dispatcher_cases():
        # Without a sparse mesh, AllToAll uses the local dispatch/combine.
        return (
            ("local", LocalTokenDispatcher.Config),
            ("alltoall_ep1", AllToAllTokenDispatcher.Config),
        )

    @staticmethod
    def _build(config_cls, absorb_router_scores: bool) -> _Model:
        torch.manual_seed(42)
        return _Model(
            _DispatcherBlock(
                config_cls(
                    num_experts=_NUM_EXPERTS,
                    top_k=_TOP_K,
                    absorb_router_scores=absorb_router_scores,
                )
            )
        )

    def test_combine_never_replays_and_matches_eager(self):
        for name, config_cls in self._dispatcher_cases():
            for absorb in (False, True):
                for policy, expected_w2_forwards in (
                    (FullActivationRematConfig(), 2),
                    (DefaultActivationRematConfig(), 1),
                ):
                    with self.subTest(
                        dispatcher=name,
                        absorb=absorb,
                        policy=type(policy).__qualname__,
                    ):
                        baseline = self._build(config_cls, absorb)
                        remat_model = self._build(config_cls, absorb)
                        remat_model.load_state_dict(baseline.state_dict())
                        policy.apply(remat_model)
                        block = remat_model.layers["0"]

                        num_scatter_adds = 0
                        original_scatter_add = (
                            token_dispatcher_module.deterministic_scatter_add
                        )

                        def counted_scatter_add(*args, **kwargs):
                            nonlocal num_scatter_adds
                            num_scatter_adds += 1
                            return original_scatter_add(*args, **kwargs)

                        torch.manual_seed(0)
                        x_TD = torch.randn(_NUM_TOKENS, _MODEL_DIM)
                        expected = _forward_backward(baseline, x_TD)
                        with patch.object(
                            token_dispatcher_module,
                            "deterministic_scatter_add",
                            counted_scatter_add,
                        ):
                            actual = _forward_backward(remat_model, x_TD)

                        # The combine body runs once; the policy still drives w2.
                        self.assertEqual(num_scatter_adds, 1)
                        self.assertEqual(block.w2.num_forwards, expected_w2_forwards)
                        torch.testing.assert_close(
                            actual[0], expected[0], rtol=0, atol=0
                        )
                        for actual_grad, expected_grad in zip(actual[1], expected[1]):
                            torch.testing.assert_close(
                                actual_grad, expected_grad, rtol=0, atol=0
                            )

    def test_absorbed_combine_does_not_pin_w2_output(self):
        # Absorbed scores leave no backward consumer of the w2 output, so saving w2 must not retain it.
        for name, config_cls in self._dispatcher_cases():
            with self.subTest(dispatcher=name):
                baseline = self._build(config_cls, absorb_router_scores=True)
                remat_model = self._build(config_cls, absorb_router_scores=True)
                remat_model.load_state_dict(baseline.state_dict())
                ActivationRematConfig(save_regions=["w2.grouped_mm"]).apply(remat_model)
                w2 = remat_model.layers["0"].w2
                w2_output_storages = []
                original_forward = _W2Region.forward

                def recorded_forward(module, *args, **kwargs):
                    output = original_forward(module, *args, **kwargs)
                    # The forward also runs (region skipped) during replay.
                    if module is w2 and not remat.is_recomputing():
                        # remat retains a detached alias, so track the storage.
                        w2_output_storages.append(
                            StorageWeakRef(output.untyped_storage())
                        )
                    return output

                torch.manual_seed(0)
                x_TD = torch.randn(_NUM_TOKENS, _MODEL_DIM)
                expected = _forward_backward(baseline, x_TD)
                with patch.object(
                    _W2Region, "forward", autospec=True, side_effect=recorded_forward
                ):
                    input_TD = x_TD.detach().clone().requires_grad_(True)
                    loss = remat_model(input_TD)
                    gc.collect()
                    retained = not w2_output_storages[0].expired()
                    loss.backward()

                self.assertFalse(retained)
                self.assertEqual(len(w2_output_storages), 1)
                torch.testing.assert_close(loss.detach(), expected[0], rtol=0, atol=0)
                actual_grads = [input_TD.grad.detach().clone()]
                actual_grads += [
                    parameter.grad.detach().clone()
                    for parameter in remat_model.parameters()
                ]
                for actual_grad, expected_grad in zip(actual_grads, expected[1]):
                    torch.testing.assert_close(
                        actual_grad, expected_grad, rtol=0, atol=0
                    )


if __name__ == "__main__":
    unittest.main()
