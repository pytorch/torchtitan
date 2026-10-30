# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""The forced-routing recipe balances real expert-ownership groups."""

from threading import local

import pytest
import torch
import torch.distributed as dist

from torchtitan.distributed import spmd_types
from torchtitan.distributed.parallelism_context import ParallelismContext
from torchtitan.models.common import HiMidLoLinear, Sigmoid
from torchtitan.models.common.moe import RoundRobinTokenChoiceTopKRouter
from torchtitan_recipes.models import kimi_k3


def test_phased_routing_balances_a_partial_expert_cycle(monkeypatch) -> None:
    """Unphased sources overload ranks when T * K is not divisible by E."""
    router = kimi_k3._PhasedRoundRobinRouter.Config(
        num_experts=896,
        top_k=8,
        gate=HiMidLoLinear.Config(in_features=16, out_features=896, bias=False),
        score_func=Sigmoid.Config(),
    ).build()
    scores_TE = torch.empty(4096, 896)
    phased_counts = torch.zeros(64, dtype=torch.int64)
    unphased_counts = torch.zeros_like(phased_counts)
    # Isolate the registry, but exercise real dense/sparse DeviceMesh groups.
    monkeypatch.setattr(spmd_types, "_MESH_TLS", local())
    for source_rank in range(64):
        dist.init_process_group(
            "fake", store=dist.HashStore(), rank=source_rank, world_size=64
        )
        try:
            context = ParallelismContext(
                dp_replicate=1,
                dp_shard=64,
                cp=1,
                tp=1,
                pp=1,
                ep=64,
                world_size=64,
                enable_sequence_parallel=True,
            )
            with context.activate_spmd():
                dense_mesh = spmd_types.current_spmd_mesh()
                assert spmd_types.spmd_mesh_group("ep") is None
                expert_ids_TK = router._select_experts(scores_TE)
                assert spmd_types.current_spmd_mesh() is dense_mesh
                assert expert_ids_TK[0, 0].item() == source_rank * 14
        finally:
            dist.destroy_process_group()
        assert torch.all(expert_ids_TK.sort(dim=-1).values.diff(dim=-1) > 0)
        phased_counts += torch.bincount(expert_ids_TK.flatten() // 14, minlength=64)
        unphased_ids = RoundRobinTokenChoiceTopKRouter._select_experts(
            router, scores_TE
        )
        unphased_counts += torch.bincount(unphased_ids.flatten() // 14, minlength=64)
    torch.testing.assert_close(
        phased_counts, torch.full_like(phased_counts, 4096 * 8), rtol=0, atol=0
    )
    assert unphased_counts.min() != unphased_counts.max()


def test_phased_routing_without_ep_keeps_the_native_ids(monkeypatch) -> None:
    """The singleton path is the native round-robin sequence, not a new policy."""
    monkeypatch.setattr(spmd_types, "_MESH_TLS", local())
    router = kimi_k3._PhasedRoundRobinRouter.Config(
        num_experts=16,
        top_k=4,
        gate=HiMidLoLinear.Config(in_features=8, out_features=16, bias=False),
        score_func=Sigmoid.Config(),
    ).build()
    scores_TE = torch.empty(13, 16)
    torch.testing.assert_close(
        router._select_experts(scores_TE),
        RoundRobinTokenChoiceTopKRouter._select_experts(router, scores_TE),
        rtol=0,
        atol=0,
    )


def test_full_recipe_selects_dist_moe_before_graph_validation(monkeypatch) -> None:
    """Graph support validates the transformed backend, not the stock dispatcher."""
    pytest.importorskip(
        "dist_moe", reason="Full MXFP8 recipes require the optional Dist-MoE package"
    )
    monkeypatch.setattr(
        "torchtitan.distributed.accelerator_graph.accelerator_graphs_supported",
        lambda: True,
    )
    kimi_k3.kimi_k3_dist_moe_mxfp8_gb200_512()


@pytest.mark.parametrize(
    ("owner", "api"),
    [
        (kimi_k3.PipelineStage, "set_reshard_after_backward"),
        (kimi_k3.FSDPModule, "set_reduce_scatter_on_current_stream"),
    ],
)
def test_missing_native_memory_policy_is_not_silently_ignored(
    monkeypatch, owner, api
) -> None:
    """A native API requirement fails before optional backend import or build."""
    # The pipeline API is inherited. Delete it from the defining class so the
    # test models an unsupported installation and pytest can restore it.
    api_owner = next((cls for cls in owner.__mro__ if api in cls.__dict__), owner)
    monkeypatch.delattr(api_owner, api, raising=False)
    with pytest.raises(ValueError, match="requires native"):
        kimi_k3.kimi_k3_dist_moe_mxfp8_gb300_256()
