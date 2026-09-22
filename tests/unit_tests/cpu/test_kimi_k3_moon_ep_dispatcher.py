# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""CPU checks for the MoonEP wiring: what the spec selects, the import guard, and the mesh precondition.

Nothing here touches MoonEP itself, which needs its package and NVLink
multicast. The comparison against a dense reference runs against the real
package in ``tests/unit_tests/gpu/test_kimi_k3_moon_ep.py``.
"""

from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
from torch.testing._internal.distributed._tensor.common_dtensor import (
    DTensorTestBase,
    with_comms,
)
from torchtitan.distributed.moonep.moonep import _import_moonep

from torchtitan.distributed.parallel_dims import ParallelDims
from torchtitan.models.common.moe import check_moonep_mesh, MoonEPGroupedExperts
from torchtitan.models.common.token_dispatcher import MoonEPTokenDispatcher
from torchtitan.models.kimi_k3 import model_registry


def _find_dispatcher(model):
    for _, mod in model.named_modules():
        if hasattr(mod, "token_dispatcher"):
            return mod.token_dispatcher
    raise AssertionError("no routed experts found")


def test_moonep_spec_selects_dispatcher_and_experts_sized_by_latent():
    spec = model_registry("debugmodel", moe_comm_backend="moonep")
    model = spec.model.build()
    dispatcher = _find_dispatcher(model)
    assert isinstance(dispatcher, MoonEPTokenDispatcher)
    layer = next(m for _, m in model.named_modules() if hasattr(m, "routed_down"))
    # The routed experts consume routed_down's output, so the buffer width is
    # the latent width, not the model width.
    assert dispatcher.hidden_dim == layer.routed_down.weight.shape[0]
    assert isinstance(layer.routed_experts.inner_experts, MoonEPGroupedExperts)


def test_moonep_ep1_falls_back_to_local_dispatch():
    """With no EP mesh the local fallback runs and moonep is never imported."""
    spec = model_registry("debugmodel", moe_comm_backend="moonep")
    model = spec.model.build()
    dispatcher = _find_dispatcher(model)
    dispatcher.wire_meshes(ep_mesh=None)
    num_tokens, num_experts, top_k = 8, dispatcher.num_experts, dispatcher.top_k
    x_TD = torch.randn(num_tokens, dispatcher.hidden_dim)
    scores_TK, ids_TK = torch.rand(num_tokens, num_experts).topk(top_k, dim=-1)
    counts_E = torch.zeros(num_experts, dtype=torch.long).scatter_add_(
        0, ids_TK.reshape(-1), torch.ones(num_tokens * top_k, dtype=torch.long)
    )
    routed_RD, counts_e, metadata = dispatcher.dispatch(
        x_TD, scores_TK, ids_TK, counts_E
    )
    assert counts_e.sum().item() == num_tokens * top_k
    assert dispatcher.combine(routed_RD, metadata, x_TD).shape == x_TD.shape


def test_moonep_import_guard_names_the_package():
    try:
        import moonep  # noqa: F401
    except ImportError:
        pass
    else:
        pytest.skip("moonep installed; the guard has nothing to raise")
    with pytest.raises(ImportError, match="MoonshotAI/MoonEP"):
        _import_moonep()


class TestMoonEPMeshPrecondition(DTensorTestBase):
    """The mesh check against real ParallelDims meshes."""

    @property
    def world_size(self) -> int:
        return 4

    def _dims(self, **kwargs) -> ParallelDims:
        dims = ParallelDims(pp=1, world_size=self.world_size, **kwargs)
        dims.build_mesh()
        return dims

    @with_comms
    def test_moonep_mesh_requires_efsdp_of_one(self):
        with patch(
            "torchtitan.distributed.parallel_dims.device_type", self.device_type
        ):
            check_moonep_mesh(self._dims(dp_replicate=1, dp_shard=4, cp=1, tp=1, ep=4))
            with self.assertRaisesRegex(NotImplementedError, "efsdp == 1"):
                check_moonep_mesh(
                    self._dims(dp_replicate=1, dp_shard=4, cp=1, tp=1, ep=2)
                )
            with self.assertRaisesRegex(NotImplementedError, "dp_replicate"):
                check_moonep_mesh(
                    self._dims(dp_replicate=2, dp_shard=2, cp=1, tp=1, ep=2)
                )


class _Plan:
    def __init__(self, experts_to_copy: torch.Tensor) -> None:
        self.experts_to_copy = experts_to_copy


class _StubDispatcher:
    """What the experts read off the dispatcher: the slot count and the plan in flight."""

    def __init__(self, num_prefetch_slots: int, cu_seqlens: torch.Tensor) -> None:
        self.num_prefetch_slots = num_prefetch_slots
        self.cu_seqlens = cu_seqlens
        self.plan = None

    def current_plan(self):
        return self.plan, self.cu_seqlens


class _StubBackend:
    """MoonEP's table backend in plain memory: prefetch copies the planned rows
    into the slots, which is the behaviour the interleaving test turns on."""

    def __init__(self, rows_by_name: dict[str, torch.Tensor]) -> None:
        self.rows_by_name = rows_by_name
        self.num_experts = 0
        self.num_slots = 0
        self.own_rows = 0

    def configure(self, *, num_experts: int, num_slots: int) -> None:
        self.num_experts, self.num_slots = num_experts, num_slots
        self.own_rows = num_experts // 2

    def alloc_expert_rows(self, name, in_dim, out_dim):
        return torch.zeros(self.own_rows + self.num_slots, in_dim, out_dim)

    def alloc_grad_rows(self, name, in_dim, out_dim):
        return (
            torch.zeros(self.own_rows, in_dim, out_dim),
            torch.zeros(self.num_slots, in_dim, out_dim),
        )

    def prefetch(self, plan, tables) -> None:
        ids = plan.experts_to_copy[0]
        for slot, expert in enumerate(ids.tolist()):
            if expert < 0:
                continue
            for name, table in tables.items():
                table[self.own_rows + slot] = self.rows_by_name[name][expert]

    def reduce_grad(self, plan, grads) -> None:
        pass


def _interleaving_case():
    """Two plans that put different experts in the slot, over one module."""
    torch.manual_seed(0)
    # the grouped GEMM wants rows on a 16-byte stride
    num_experts, dim, hidden = 4, 64, 32
    rows = {
        "gate": torch.randn(num_experts, dim, hidden),
        "up": torch.randn(num_experts, dim, hidden),
        "down": torch.randn(num_experts, hidden, dim),
    }
    experts = MoonEPGroupedExperts(
        MoonEPGroupedExperts.Config(dim=dim, hidden_dim=hidden, num_experts=num_experts)
    )
    experts.w1_EFD = torch.nn.Parameter(rows["gate"][:2].transpose(-2, -1).contiguous())
    experts.w3_EFD = torch.nn.Parameter(rows["up"][:2].transpose(-2, -1).contiguous())
    experts.w2_EDF = torch.nn.Parameter(rows["down"][:2].transpose(-2, -1).contiguous())
    # rows 0-2 to expert 0, 3-4 to expert 1, 5-6 to the slot; the remote experts
    # take none here, which is what the offsets assert.
    cu = torch.tensor([3, 5, 5, 5, 7, 7], dtype=torch.int32)
    dispatcher = _StubDispatcher(2, cu)
    backend = _StubBackend(rows)
    mesh = SimpleNamespace(get_local_rank=lambda: 0, size=lambda: 2)
    experts.attach(dispatcher, backend, mesh)
    plan_a = _Plan(torch.tensor([[2, -1]], dtype=torch.int32))
    plan_b = _Plan(torch.tensor([[3, -1]], dtype=torch.int32))
    torch.manual_seed(1)
    x_a = torch.randn(7, dim, requires_grad=True)
    x_b = torch.randn(7, dim, requires_grad=True)
    grad_a = torch.randn(7, dim)
    return experts, dispatcher, x_a, x_b, grad_a, plan_a, plan_b


def _run(experts, dispatcher, x, plan):
    dispatcher.plan = plan
    return experts(x, torch.empty(0))


def test_a_later_microbatch_forward_does_not_move_an_earlier_backward():
    """A pipeline schedule runs the next micro-batch's forward before this
    backward, which overwrites the slots the backward recomputes from."""
    experts, dispatcher, x_a, x_b, grad_a, plan_a, plan_b = _interleaving_case()
    _run(experts, dispatcher, x_a, plan_a).backward(grad_a)
    sequential = (
        x_a.grad.clone(),
        experts.w1_EFD.grad.clone(),
        experts.w2_EDF.grad.clone(),
    )

    experts, dispatcher, x_a, x_b, grad_a, plan_a, plan_b = _interleaving_case()
    out_a = _run(experts, dispatcher, x_a, plan_a)
    _run(experts, dispatcher, x_b, plan_b)
    out_a.backward(grad_a)
    interleaved = (
        x_a.grad.clone(),
        experts.w1_EFD.grad.clone(),
        experts.w2_EDF.grad.clone(),
    )

    for name, first, second in zip(("dgrad", "w1", "w2"), sequential, interleaved):
        torch.testing.assert_close(first, second, msg=f"{name} moved")
