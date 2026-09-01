# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import os
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
import torch.nn as nn
from torch._inductor.fx_passes.bucketing import (
    is_all_gather_into_tensor,
    is_reduce_scatter_tensor,
)
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import DTensor
from torch.fx.experimental.proxy_tensor import make_fx

from torchtitan.distributed.utils import set_spmd_backend
from torchtitan.experiments.graph_trainer.common_utils import annotate_module_fqns
from torchtitan.experiments.graph_trainer.fsdp_passes import (
    joint_transformer_block_bucketing_reordering_pass,
)
from torchtitan.experiments.graph_trainer.make_fx_tracer import minimal_fx_tracer
from torchtitan.experiments.graph_trainer.remove_noop_passes import (
    canonicalize_graph_pass,
)
from torchtitan.experiments.graph_trainer.simple_fsdp import (
    _FSDPPaddedParamUnshard,
    data_parallel,
)


@pytest.mark.skipif(
    int(os.environ.get("WORLD_SIZE", "1")) != 2 or not torch.cuda.is_available(),
    reason="run with torchrun --standalone --nproc-per-node=2 on CUDA",
)
def test_uneven_fsdp():
    """Check padding, SPMD graphs, DCP, and bucketing in one FSDP setup."""

    class OddModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = nn.ModuleList(
                [nn.Linear(7, 5, bias=False), nn.Linear(5, 7, bias=False)]
            )

        def forward(self, value):
            return self.layers[1](torch.relu(self.layers[0](value))).sum()

    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl")
    try:
        rank = dist.get_rank()
        device = torch.device("cuda", local_rank)
        mesh = init_device_mesh("cuda", (2,), mesh_dim_names=("fsdp",))
        set_spmd_backend("spmd_types")

        logical_weight = torch.arange(35, device=device, dtype=torch.float32).view(5, 7)
        model = nn.Linear(7, 5, bias=False, device=device)
        model.weight.data.copy_(logical_weight)
        model = data_parallel(model, mesh, mode="fully_shard", shard_dim=0)

        storage_weight = model._parameters["weight"]
        assert isinstance(storage_weight, DTensor)
        assert storage_weight.shape == torch.Size([5, 7])
        assert storage_weight.to_local().shape == torch.Size([3, 7])
        torch.testing.assert_close(model.weight, logical_weight)

        inputs = torch.arange(14, device=device, dtype=torch.float32).view(2, 7) + rank
        model(inputs).sum().backward()
        expected_full_grad = inputs.sum(dim=0).expand(5, -1).clone()
        dist.all_reduce(expected_full_grad)
        expected_padded_grad = torch.cat(
            [expected_full_grad, torch.zeros_like(expected_full_grad[:1])], dim=0
        )
        expected_local_grad = expected_padded_grad.chunk(2, dim=0)[rank]
        assert isinstance(storage_weight.grad, DTensor)
        torch.testing.assert_close(storage_weight.grad.to_local(), expected_local_grad)

        group = mesh.get_group()

        def traced_fwd_bwd(local_shard):
            gathered = _FSDPPaddedParamUnshard.apply(
                local_shard,
                group,
                0,
                5,
                None,
                None,
            )
            (local_grad,) = torch.autograd.grad(gathered.square().sum(), local_shard)
            return gathered, local_grad

        trace_input = torch.arange(
            rank * 3,
            rank * 3 + 3,
            device=device,
            dtype=torch.float32,
            requires_grad=True,
        )
        graph_code = make_fx(traced_fwd_bwd)(trace_input).code
        assert "all_gather_into_tensor" in graph_code
        assert "reduce_scatter_tensor" in graph_code
        assert "slice" in graph_code
        assert "constant_pad_nd" in graph_code
        if graph_dir := os.environ.get("TORCHTITAN_TEST_GRAPH_DIR"):
            output_dir = Path(graph_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            (output_dir / f"rank{rank}.py").write_text(graph_code)
        rank_graphs: list[object] = [None] * dist.get_world_size()
        dist.all_gather_object(rank_graphs, graph_code)
        assert all(graph == rank_graphs[0] for graph in rank_graphs)

        original_local = storage_weight.to_local().clone()
        checkpoint_dir = os.path.join(
            "/tmp",
            f"torchtitan_uneven_dcp_{os.environ.get('TORCHELASTIC_RUN_ID', 'test')}",
        )
        state = {"model": model.state_dict()}
        assert state["model"]["weight"].shape == torch.Size([5, 7])
        dcp.save(state, checkpoint_id=checkpoint_dir)
        with torch.no_grad():
            storage_weight.to_local().zero_()
        loaded_state = {"model": model.state_dict()}
        dcp.load(loaded_state, checkpoint_id=checkpoint_dir)
        model.load_state_dict(loaded_state["model"])
        torch.testing.assert_close(storage_weight.to_local(), original_local)

        bucketed_model = OddModel().cuda()
        annotate_module_fqns(bucketed_model)
        bucketed_model = data_parallel(bucketed_model, mesh, mode="fully_shard")
        example = torch.randn(3, 7, device=device)

        def train_step(value):
            loss = bucketed_model(value)
            grads = torch.autograd.grad(loss, tuple(bucketed_model.parameters()))
            return [loss, *grads]

        traced = minimal_fx_tracer(train_step, module=bucketed_model)(example)
        canonicalize_graph_pass(traced.gm)
        nodes = list(traced.gm.graph.nodes)
        assert sum(is_all_gather_into_tensor(node) for node in nodes) == 2
        assert sum(is_reduce_scatter_tensor(node) for node in nodes) == 2

        joint_transformer_block_bucketing_reordering_pass(
            traced.gm,
            module_bucket_plans=[["layers.0", "layers.1"]],
        )
        nodes = list(traced.gm.graph.nodes)
        assert sum(is_all_gather_into_tensor(node) for node in nodes) == 1
        assert sum(is_reduce_scatter_tensor(node) for node in nodes) == 1
        assert sum("constant_pad_nd" in str(node.target) for node in nodes) == 2
        assert sum("bucketing" in str(node.target) for node in nodes) == 2
        code = traced.gm.code
        assert code.index("_pre_bucket_all_gather") < code.index("slice_1")
        assert code.index("constant_pad_nd") < code.index("_pre_bucket_reduce_scatter")
    finally:
        dist.barrier()
        dist.destroy_process_group()
