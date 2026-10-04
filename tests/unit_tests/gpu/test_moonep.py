# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import pytest
import torch
import torch.distributed as dist
from torch import nn
from torch.distributed.device_mesh import init_device_mesh
from torch.testing._internal.distributed._tensor.common_dtensor import (
    DTensorTestBase,
    with_comms,
)

from torchtitan.config.transform import convert_config_type
from torchtitan.distributed.activation_checkpoint import (
    ActivationCheckpointing,
    FullAC,
    RegionAC,
    SelectiveAC,
)
from torchtitan.distributed.moonep import ops
from torchtitan.distributed.moonep.experts import MoonEPRoutedExperts
from torchtitan.distributed.spmd_types import set_current_spmd_mesh, set_spmd_meshes
from torchtitan.models.common.activation import SiTUGLU
from torchtitan.models.common.config_utils import make_routed_experts_config
from torchtitan.models.common.token_dispatcher import MoonEPTokenDispatcher
from torchtitan.protocols.module import Module, ModuleDict

pytestmark = pytest.mark.multi_gpu

E, K, S, D, F = 32, 4, 256, 512, 384
BETA, LINEAR_BETA = 4.0, 25.0


def _moonep_runs_here(num_gpus: int) -> bool:
    try:
        import moonep  # noqa: F401
        from torch._C._distributed_c10d import _SymmetricMemory
    except ImportError:
        return False
    if not torch.cuda.is_available() or torch.cuda.device_count() < num_gpus:
        return False
    return all(
        _SymmetricMemory.has_multicast_support(torch._C._autograd.DeviceType.CUDA, i)
        for i in range(torch.cuda.device_count())
    )


class _Block(Module):
    def __init__(self, experts: nn.Module):
        super().__init__()
        self.experts = experts

    def forward(self, *args: torch.Tensor) -> torch.Tensor:
        return self.experts(*args)


class _Model(Module):
    def __init__(self, experts: list[nn.Module]):
        super().__init__()
        self.layers = ModuleDict({str(i): _Block(e) for i, e in enumerate(experts)})


def _reference(x_TD, weights_TK, ids_TK, w13_E2FD, w2_EDF):
    activation = SiTUGLU.Config(beta=BETA, linear_beta=LINEAR_BETA).build()
    out_TD = torch.zeros_like(x_TD)
    for e in ids_TK.unique().tolist():
        t, k = (ids_TK == e).nonzero(as_tuple=True)
        x_ND = x_TD[t]
        hidden_NF = activation(x_ND @ w13_E2FD[e, 0].T, x_ND @ w13_E2FD[e, 1].T)
        out_ND = weights_TK[t, k, None] * (hidden_NF @ w2_EDF[e].T)
        out_TD = out_TD.index_add(0, t, out_ND)
    return out_TD


def _routing(
    kind: str, home: int, size: int, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    if kind == "uniform":
        weights_TK, ids_TK = torch.rand(S, E, device=device).topk(K, dim=-1)
        return weights_TK / weights_TK.sum(-1, keepdim=True), ids_TK
    # Every token goes to experts homed on these ranks, so balancing needs the slots.
    homes = [home] if kind == "hot" else [home, (home + 1) % size]
    ids_K = torch.cat(
        [torch.arange(K // len(homes), device=device) + h * (E // size) for h in homes]
    )
    weights_TK = torch.softmax(torch.randn(S, K, device=device), dim=-1)
    return weights_TK, ids_K.repeat(S, 1)


class _MoonEPTestBase(DTensorTestBase):
    def _check(
        self,
        routing: str,
        ac: ActivationCheckpointing.Config | None = None,
        num_layers: int = 1,
        num_microbatches: int = 1,
    ) -> None:
        rank, size = self.rank, self.world_size
        device = torch.device("cuda", torch.cuda.current_device())
        mesh = init_device_mesh("cuda", (size,), mesh_dim_names=("ep",))
        set_spmd_meshes(dense_mesh=mesh, sparse_mesh=mesh, dense_sp_enabled=False)
        lo, hi = rank * (E // size), (rank + 1) * (E // size)

        weights, layers = [], []
        for layer in range(num_layers):
            torch.manual_seed(1 + layer)
            w13_E2FD = torch.randn(E, 2, F, D) * 0.05
            w2_EDF = torch.randn(E, D, F) * 0.05
            config = convert_config_type(
                make_routed_experts_config(
                    dim=D, hidden_dim=F, num_experts=E, top_k=K, param_init={}
                ),
                MoonEPRoutedExperts,
            )
            config.token_dispatcher = MoonEPTokenDispatcher.Config(
                num_experts=E, top_k=K, hidden_dim=D, num_max_tokens_per_rank=S
            )
            config.activation_fn = SiTUGLU.Config(beta=BETA, linear_beta=LINEAR_BETA)
            experts = config.build().to(device)
            experts.w13.weight = nn.Parameter(
                w13_E2FD[lo:hi].to(device, torch.bfloat16)
            )
            experts.w2.weight = nn.Parameter(w2_EDF[lo:hi].to(device, torch.bfloat16))
            weights.append((w13_E2FD, w2_EDF))
            layers.append(experts)
        model = _Model(layers)
        if ac is not None:
            ac.build().apply(model)

        microbatches = []
        for mb in range(num_microbatches):
            torch.manual_seed(100 + rank + size * mb)
            x_TD = (torch.randn(S, D, device=device) * 0.5).to(torch.bfloat16)
            routes = [
                _routing(routing, (mb + layer) % size, size, device)
                for layer in range(num_layers)
            ]
            for weights_TK, _ in routes:
                weights_TK.requires_grad_(True)
            microbatches.append((x_TD, routes))

        dispatched = []
        for experts in layers:
            dispatch = experts.token_dispatcher.dispatch

            def recording_dispatch(*args, dispatch=dispatch):
                result = dispatch(*args)
                dispatched.append(result)
                return result

            experts.token_dispatcher.dispatch = recording_dispatch

        from torchtitan.distributed.moonep import moonep

        x_ins, outs = [], []
        first_plan = ops._next_plan_id
        with set_current_spmd_mesh(mesh):
            for experts in layers:
                experts._init_self_buffers()
            pools = set(moonep._pools)
            for x_TD, routes in microbatches:
                h_TD = x_TD.clone().requires_grad_(True)
                x_ins.append(h_TD)
                for layer, (weights_TK, ids_TK) in enumerate(routes):
                    counts_E = torch.bincount(ids_TK.flatten(), minlength=E)
                    h_TD = model.layers[str(layer)](h_TD, weights_TK, ids_TK, counts_E)
                outs.append(h_TD)
            for out_TD in outs:
                out_TD.float().sum().backward()
        torch.cuda.synchronize()
        self.assertEqual(
            ops._next_plan_id - first_plan,
            num_layers * num_microbatches,
            "a dispatch was replayed",
        )
        self.assertEqual(ops._plans, {}, "a plan outlived its combine")
        self.assertEqual(set(moonep._pools), pools, "a pool was created in the step")

        slot_rows = sum(result[1][E // size :].sum() for result in dispatched)
        slot_rows = slot_rows.reshape(1)
        dist.all_reduce(slot_rows)

        refs = [
            tuple(
                w.to(device, torch.bfloat16).float().requires_grad_(True)
                for w in layer_weights
            )
            for layer_weights in weights
        ]
        mine = slice(rank * S, (rank + 1) * S)
        expected = []
        for x_TD, routes in microbatches:
            gathered = []
            for t in [x_TD, *[part for route in routes for part in route]]:
                parts = [torch.empty_like(t) for _ in range(size)]
                dist.all_gather(parts, t)
                gathered.append(torch.cat(parts))
            x_all_TD = gathered[0].float().requires_grad_(True)
            weights_all = [
                gathered[1 + 2 * layer].float().requires_grad_(True)
                for layer in range(num_layers)
            ]
            h_TD = x_all_TD
            for layer, (w13_ref, w2_ref) in enumerate(refs):
                ids_TK = gathered[2 + 2 * layer]
                h_TD = _reference(h_TD, weights_all[layer], ids_TK, w13_ref, w2_ref)
            h_TD.sum().backward()
            weight_grads = [w.grad[mine] for w in weights_all]
            expected.append((h_TD[mine], x_all_TD.grad[mine], weight_grads))
        layers[0].token_dispatcher.buffer.destroy()

        if routing != "uniform":
            self.assertGreater(int(slot_rows.item()), 0, "no token reached a slot")
        # bf16 weights and activations against an fp32 reference.
        for (_, routes), out_TD, x_in_TD, (ref_TD, ref_grad_TD, weight_grads) in zip(
            microbatches, outs, x_ins, expected, strict=True
        ):
            torch.testing.assert_close(out_TD.float(), ref_TD, atol=5e-2, rtol=5e-2)
            torch.testing.assert_close(
                x_in_TD.grad.float(), ref_grad_TD, atol=5e-2, rtol=5e-2
            )
            for (weights_TK, _), ref_weight_grad in zip(
                routes, weight_grads, strict=True
            ):
                torch.testing.assert_close(
                    weights_TK.grad, ref_weight_grad, atol=1e-1, rtol=5e-2
                )
        for experts, (w13_ref, w2_ref) in zip(layers, refs, strict=True):
            torch.testing.assert_close(
                experts.w13.weight.grad.float(),
                w13_ref.grad[lo:hi],
                atol=1e-1,
                rtol=5e-2,
            )
            torch.testing.assert_close(
                experts.w2.weight.grad.float(), w2_ref.grad[lo:hi], atol=1e-1, rtol=5e-2
            )


@pytest.mark.skipif(
    not _moonep_runs_here(2), reason="needs the moonep package and NVLink multicast"
)
class TestMoonEPRoutedExperts(_MoonEPTestBase):
    @property
    def world_size(self) -> int:
        return 2

    @with_comms
    def test_hot_routing_fills_slots_and_matches_dense_reference(self):
        self._check("hot")

    @with_comms
    def test_uniform_routing_matches_dense_reference(self):
        self._check("uniform")

    @with_comms
    def test_selective_ac_matches_dense_reference(self):
        self._check("hot", SelectiveAC.Config())

    @with_comms
    def test_full_ac_matches_dense_reference(self):
        self._check("hot", FullAC.Config())

    @with_comms
    def test_region_ac_matches_dense_reference(self):
        self._check("hot", RegionAC.Config(save_regions=[]))

    @with_comms
    def test_two_layers_refill_the_shared_pools(self):
        self._check("hot", num_layers=2)

    @with_comms
    def test_interleaved_microbatches_keep_their_plans(self):
        self._check("hot", num_layers=2, num_microbatches=2)


@pytest.mark.skipif(
    not _moonep_runs_here(4),
    reason="needs the moonep package, NVLink multicast and four GPUs",
)
class TestMoonEPRoutedExpertsOnFourRanks(_MoonEPTestBase):
    @property
    def world_size(self) -> int:
        return 4

    @with_comms
    def test_two_hot_home_ranks_match_dense_reference(self):
        self._check("hot2")

    @with_comms
    def test_interleaved_microbatches_keep_their_plans(self):
        self._check("hot2", num_layers=2, num_microbatches=2)


if __name__ == "__main__":
    from torch.testing._internal.common_utils import run_tests

    run_tests()
