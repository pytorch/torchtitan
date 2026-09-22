# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""MoonEP's dispatcher and expert tables on real GPUs against a dense reference."""

import pytest
import torch
import torch.distributed as dist
from torch import nn
from torch.distributed.device_mesh import init_device_mesh
from torch.testing._internal.distributed._tensor.common_dtensor import (
    DTensorTestBase,
    with_comms,
)
from torchtitan.distributed.moonep.moonep import MoonEPTableBackendNVLink

from torchtitan.models.common.activation import SiTUGLU
from torchtitan.models.common.moe import MoonEPGroupedExperts
from torchtitan.models.common.token_dispatcher import MoonEPTokenDispatcher

E, K, S, D, F = 32, 4, 256, 512, 384
BETA, LINEAR_BETA = 4.0, 25.0


def _moonep_runs_here() -> bool:
    try:
        import moonep  # noqa: F401
        from torch._C._distributed_c10d import _SymmetricMemory
    except ImportError:
        return False
    if not torch.cuda.is_available() or torch.cuda.device_count() < 2:
        return False
    return all(
        _SymmetricMemory.has_multicast_support(torch._C._autograd.DeviceType.CUDA, i)
        for i in range(torch.cuda.device_count())
    )


def _reference(x_TD, weights_TK, ids_TK, w1, w2, w3):
    act = SiTUGLU.Config(beta=BETA, linear_beta=LINEAR_BETA).build()
    out = torch.zeros_like(x_TD)
    for k in range(K):
        e = ids_TK[:, k]
        gate = torch.einsum("td,tfd->tf", x_TD, w1[e])
        up = torch.einsum("td,tfd->tf", x_TD, w3[e])
        out = out + weights_TK[:, k : k + 1] * torch.einsum(
            "tf,tdf->td", act(gate, up), w2[e]
        )
    return out


@pytest.mark.skipif(
    not _moonep_runs_here(), reason="needs the moonep package and NVLink multicast"
)
class TestMoonEPOnDevice(DTensorTestBase):
    @property
    def world_size(self) -> int:
        return 2

    def _check(self, routing: str) -> None:
        rank, size = self.rank, self.world_size
        dev = torch.device("cuda", torch.cuda.current_device())
        mesh = init_device_mesh("cuda", (size,), mesh_dim_names=("ep",))
        torch.manual_seed(1)
        params = {
            "w1": torch.randn(E, F, D) * 0.05,
            "w2": torch.randn(E, D, F) * 0.05,
            "w3": torch.randn(E, F, D) * 0.05,
        }
        experts = MoonEPGroupedExperts(
            MoonEPGroupedExperts.Config(
                dim=D,
                hidden_dim=F,
                num_experts=E,
                activation_fn=SiTUGLU.Config(beta=BETA, linear_beta=LINEAR_BETA),
            )
        ).to(dev)
        lo, hi = rank * (E // size), (rank + 1) * (E // size)
        experts.w1_EFD = nn.Parameter(params["w1"][lo:hi].to(dev, torch.bfloat16))
        experts.w2_EDF = nn.Parameter(params["w2"][lo:hi].to(dev, torch.bfloat16))
        experts.w3_EFD = nn.Parameter(params["w3"][lo:hi].to(dev, torch.bfloat16))
        dispatcher = MoonEPTokenDispatcher(
            MoonEPTokenDispatcher.Config(
                num_experts=E,
                top_k=K,
                hidden_dim=D,
                num_max_tokens_per_rank=S,
            )
        )
        dispatcher.wire_meshes(ep_mesh=mesh)
        experts.attach(dispatcher, MoonEPTableBackendNVLink(mesh, dispatcher), mesh)

        torch.manual_seed(100 + rank)
        x = (torch.randn(S, D, device=dev) * 0.5).to(torch.bfloat16)
        if routing == "hot":
            # Every token to experts 0..K-1, all home on rank 0: the other rank's
            # slots must fill.
            ids = torch.arange(K, device=dev, dtype=torch.int64).repeat(S, 1)
            weights = torch.softmax(torch.randn(S, K, device=dev), dim=-1)
        else:
            weights, ids = torch.rand(S, E, device=dev).topk(K, dim=-1)
            weights = weights / weights.sum(-1, keepdim=True)
        counts = torch.zeros(E, dtype=torch.long, device=dev).scatter_add_(
            0, ids.reshape(-1), torch.ones(S * K, dtype=torch.long, device=dev)
        )
        x_in = x.clone().requires_grad_(True)
        routed, rows, metadata = dispatcher.dispatch(x_in, weights, ids, counts)
        plan, _ = dispatcher.current_plan()
        slot_here = int((plan.experts_to_copy[rank] >= 0).any())
        slot_rows = int(rows[E:].sum())
        out = dispatcher.combine(experts(routed, rows), metadata, x_in)
        out.float().sum().backward()
        torch.cuda.synchronize()

        gathered = []
        for t in (x, weights, ids):
            parts = [torch.empty_like(t) for _ in range(size)]
            dist.all_gather(parts, t)
            gathered.append(torch.cat(parts))
        x_all = gathered[0].float().requires_grad_(True)
        p = {
            n: v.to(dev, torch.bfloat16).float().requires_grad_(True)
            for n, v in params.items()
        }
        ref = _reference(
            x_all, gathered[1].float(), gathered[2], p["w1"], p["w2"], p["w3"]
        )
        ref.sum().backward()
        mine = slice(rank * S, (rank + 1) * S)

        populated = torch.tensor([slot_here], device=dev)
        dist.all_reduce(populated, op=dist.ReduceOp.MAX)
        self.assertEqual(int(populated), 1, "no rank filled a prefetch slot")
        if slot_here:
            self.assertGreater(slot_rows, 0, "a filled slot received no tokens")
        # bf16 tables and activations against an fp32 reference.
        torch.testing.assert_close(out.float(), ref[mine], atol=5e-2, rtol=5e-2)
        torch.testing.assert_close(
            x_in.grad.float(), x_all.grad[mine], atol=5e-2, rtol=5e-2
        )
        for name, attr in (("w1", "w1_EFD"), ("w2", "w2_EDF"), ("w3", "w3_EFD")):
            torch.testing.assert_close(
                getattr(experts, attr).grad.float(),
                p[name].grad[lo:hi],
                atol=1e-1,
                rtol=5e-2,
            )
        buffer = dispatcher.buffer
        if buffer is not None and hasattr(buffer, "destroy"):
            buffer.destroy()

    @with_comms
    def test_forced_hot_routing_fills_a_slot_and_matches_the_dense_reference(self):
        self._check("hot")

    @with_comms
    def test_uniform_routing_matches_the_dense_reference(self):
        self._check("uniform")
