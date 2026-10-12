# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import os
import unittest

from unittest.mock import patch

import pytest
import torch
import torch_remat as remat
from torch.distributed.device_mesh import init_device_mesh
from torch.testing._internal.distributed._tensor.common_dtensor import (
    DTensorTestBase,
    with_comms,
)

from torchtitan.distributed.activation_checkpoint import FullAC, RegionAC, SelectiveAC
from torchtitan.distributed.spmd_types import set_current_spmd_mesh, set_spmd_meshes
from torchtitan.models.base.module import Module, ModuleDict
from torchtitan.models.common.activation import SwiGLU
from torchtitan.models.common.linear import GroupedLinear
from torchtitan.models.common.moe import RoutedExperts
from torchtitan.models.common.token_dispatcher import (
    AllToAllTokenDispatcher,
    MoriTokenDispatcher,
)

mori = pytest.importorskip("mori")

pytestmark = pytest.mark.multi_gpu

# MORI allocates its buffers from a symmetric heap sized by this variable.
os.environ.setdefault("MORI_SHMEM_HEAP_SIZE", "4G")

_MODEL_DIM = 256
_NUM_TOKENS = 64
_TOP_K = 4


_DispatcherConfig = MoriTokenDispatcher.Config | AllToAllTokenDispatcher.Config


class _MoEBlock(Module):
    def __init__(
        self,
        num_experts: int,
        dispatcher: _DispatcherConfig,
        *,
        route_to_first_rank: bool = False,
    ):
        super().__init__()
        # RoutedExperts.forward runs after EP has selected this rank's local
        # expert-weight shard. This test bypasses parallelization, so construct
        # that local view directly while the dispatcher retains the global E.
        num_local_experts = num_experts // torch.distributed.get_world_size()
        routed_experts = RoutedExperts.__new__(RoutedExperts)
        Module.__init__(routed_experts)
        routed_experts.w13 = GroupedLinear.Config(
            group_size=num_local_experts,
            in_features=_MODEL_DIM,
            out_features=_MODEL_DIM,
            num_linears=2,
        ).build()
        routed_experts.w2 = GroupedLinear.Config(
            group_size=num_local_experts,
            in_features=_MODEL_DIM,
            out_features=_MODEL_DIM,
        ).build()
        routed_experts.activation_fn = SwiGLU.Config().build()
        routed_experts.output_postprocess = None
        routed_experts.token_dispatcher = dispatcher.build()
        self.gate = torch.nn.Linear(_MODEL_DIM, num_experts, bias=False)
        self.routed_experts = routed_experts
        self.num_experts = num_experts
        self.num_local_experts = num_local_experts
        self.route_to_first_rank = route_to_first_rank

    def forward(self, x_TD: torch.Tensor) -> torch.Tensor:
        # Each token picks _TOP_K distinct experts spread over the EP ranks, so
        # tokens reach several ranks and several local experts per rank.
        num_tokens = x_TD.shape[0]
        token_ids_T1 = torch.arange(num_tokens, device=x_TD.device).unsqueeze(-1)
        offsets_1K = 3 * torch.arange(_TOP_K, device=x_TD.device).unsqueeze(0)
        expert_ids_TK = (token_ids_T1 + offsets_1K) % self.num_experts
        if self.route_to_first_rank:
            # Every token picks experts on EP rank 0, so the other ranks
            # receive no tokens.
            expert_ids_TK = expert_ids_TK % self.num_local_experts
        # Scores depend on the gate, so the router gets a gradient too.
        scores_TK = torch.sigmoid(self.gate(x_TD)).gather(1, expert_ids_TK)
        num_tokens_per_expert_E = torch.bincount(
            expert_ids_TK.flatten(), minlength=self.num_experts
        )
        out_TD = self.routed_experts(
            x_TD,
            scores_TK,
            expert_ids_TK,
            num_tokens_per_expert_E,
        )
        # The loss is a bare consumer of the routed-expert output.
        remat.recompute_needs_tensor(out_TD)
        return out_TD.float().square().sum()


class _Model(Module):
    def __init__(self, block: Module):
        super().__init__()
        self.layers = ModuleDict({"0": block})

    def forward(self, x_TD: torch.Tensor) -> torch.Tensor:
        return self.layers["0"](x_TD)


def _run_forward_backward(
    model: Module, x_TD: torch.Tensor
) -> tuple[torch.Tensor, list[torch.Tensor]]:
    model.zero_grad(set_to_none=True)
    input_TD = x_TD.detach().clone().requires_grad_(True)
    loss = model(input_TD)
    loss.backward()
    assert input_TD.grad is not None
    grads = [input_TD.grad.detach().clone()]
    grads += [parameter.grad.detach().clone() for parameter in model.parameters()]
    return loss.detach(), grads


def _assert_close_bf16(actual: torch.Tensor, expected: torch.Tensor) -> None:
    """Compare bf16 results that differ only in summation order.

    MORI sums each token's per-rank partial outputs, so its rounding differs
    from the all-to-all's. Elementwise tolerances fail where values cancel to
    near zero; the relative norm error stays at bf16 resolution (~1e-3), while
    a routing error (a wrong or missing token) makes it O(1).
    """
    expected_norm = expected.float().norm()
    if expected_norm == 0:
        # E.g. the expert weights of a rank that received no tokens.
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        return
    error = (actual.float() - expected.float()).norm() / expected_norm
    assert error < 1e-2, f"relative error {error.item():.3e}"


@unittest.skipUnless(torch.cuda.device_count() >= 4, "requires four GPUs")
class TestMoriTokenDispatcher(DTensorTestBase):
    @property
    def world_size(self) -> int:
        return 4

    def _setup_mesh(self):
        mesh = init_device_mesh(
            self.device_type, (self.world_size,), mesh_dim_names=("ep",)
        )
        set_spmd_meshes(dense_mesh=mesh, sparse_mesh=mesh, dense_sp_enabled=False)
        return mesh

    def _build(
        self, dispatcher, num_experts: int, *, route_to_first_rank: bool = False
    ) -> Module:
        torch.manual_seed(42)
        block = _MoEBlock(
            num_experts, dispatcher, route_to_first_rank=route_to_first_rank
        )
        model = _Model(block).to(self.device_type, torch.bfloat16)
        with torch.no_grad():
            for parameter in model.parameters():
                parameter.normal_(std=0.05)
        return model

    def _input(self) -> torch.Tensor:
        torch.manual_seed(self.rank)
        return torch.randn(
            _NUM_TOKENS, _MODEL_DIM, device=self.device_type, dtype=torch.bfloat16
        )

    @with_comms
    def test_matches_all_to_all(self):
        self._check_matches_all_to_all(route_to_first_rank=False)

    @with_comms
    def test_ranks_without_received_tokens(self):
        # Ranks that receive no tokens still take part in dispatch and combine,
        # and their source tokens still get routing-score gradients.
        self._check_matches_all_to_all(route_to_first_rank=True)

    def _check_matches_all_to_all(self, *, route_to_first_rank: bool) -> None:
        # MORI sends each token once per destination rank and sums the per-rank
        # partials on the source; the all-to-all sends it once per expert. Both
        # compute the same MoE output and gradients, up to bf16 summation order.
        mesh = self._setup_mesh()
        num_experts = 4 * self.world_size
        for pad_multiple in (None, 32):
            with (
                self.subTest(pad_multiple=pad_multiple),
                torch.autograd.set_multithreading_enabled(False),
                set_current_spmd_mesh(mesh),
            ):
                baseline = self._build(
                    AllToAllTokenDispatcher.Config(
                        num_experts=num_experts, top_k=_TOP_K
                    ),
                    num_experts,
                    route_to_first_rank=route_to_first_rank,
                )
                mori_model = self._build(
                    MoriTokenDispatcher.Config(
                        num_experts=num_experts,
                        top_k=_TOP_K,
                        hidden_dim=_MODEL_DIM,
                        num_max_tokens_per_rank=_NUM_TOKENS,
                        pad_multiple=pad_multiple,
                    ),
                    num_experts,
                    route_to_first_rank=route_to_first_rank,
                )
                mori_model.layers["0"].routed_experts.token_dispatcher.init_buffer()

                x_TD = self._input()
                expected_loss, expected = _run_forward_backward(baseline, x_TD)
                actual_loss, actual = _run_forward_backward(mori_model, x_TD)

                _assert_close_bf16(actual_loss, expected_loss)
                for actual_grad, expected_grad in zip(actual, expected):
                    _assert_close_bf16(actual_grad, expected_grad)

    @with_comms
    def test_activation_checkpointing_never_replays_communication(self):
        # MORI assigns receive slots with atomics, so a replayed dispatch could
        # receive rows in another order than the forward's routing handle. The
        # dispatcher retains its communication under every policy, FullAC too,
        # while the expert computation around it is still recomputed.
        mesh = self._setup_mesh()
        num_experts = 4 * self.world_size
        dispatcher = MoriTokenDispatcher.Config(
            num_experts=num_experts,
            top_k=_TOP_K,
            hidden_dim=_MODEL_DIM,
            num_max_tokens_per_rank=_NUM_TOKENS,
        )
        for ac_config in (
            FullAC.Config(),
            RegionAC.Config(save_regions=[]),
            RegionAC.Config(save_regions=["*ep_communication"]),
            SelectiveAC.Config(),
        ):
            with (
                self.subTest(ac=type(ac_config).__qualname__),
                torch.autograd.set_multithreading_enabled(False),
                set_current_spmd_mesh(mesh),
            ):
                baseline = self._build(dispatcher, num_experts)
                ac_model = self._build(dispatcher, num_experts)
                ac_config.build().apply(ac_model)
                baseline.layers["0"].routed_experts.token_dispatcher.init_buffer()
                ac_model.layers["0"].routed_experts.token_dispatcher.init_buffer()

                num_calls = {"dispatch": 0, "combine": 0}
                original = {
                    name: getattr(mori.ops.EpDispatchCombineOp, name)
                    for name in num_calls
                }

                def counted(name):
                    def call(*args, **kwargs):
                        num_calls[name] += 1
                        return original[name](*args, **kwargs)

                    return call

                x_TD = self._input()
                with (
                    patch.object(
                        mori.ops.EpDispatchCombineOp,
                        "dispatch",
                        autospec=True,
                        side_effect=counted("dispatch"),
                    ),
                    patch.object(
                        mori.ops.EpDispatchCombineOp,
                        "combine",
                        autospec=True,
                        side_effect=counted("combine"),
                    ),
                ):
                    expected_loss, expected = _run_forward_backward(baseline, x_TD)
                    baseline_calls = dict(num_calls)
                    num_calls.update(dispatch=0, combine=0)
                    actual_loss, actual = _run_forward_backward(ac_model, x_TD)

                # Forward dispatch and combine, plus their backward passes.
                self.assertEqual(baseline_calls, {"dispatch": 2, "combine": 2})
                self.assertEqual(
                    num_calls,
                    {"dispatch": 2, "combine": 2},
                    msg=f"unexpected MORI replays: {num_calls}",
                )
                # Not bitwise: the receive order can differ between the two
                # runs, which changes the summation order.
                _assert_close_bf16(actual_loss, expected_loss)
                for actual_grad, expected_grad in zip(actual, expected):
                    _assert_close_bf16(actual_grad, expected_grad)
