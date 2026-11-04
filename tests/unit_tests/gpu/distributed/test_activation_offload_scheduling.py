# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import copy
import os
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch._functorch._activation_offloading.offload_ops import pinned_memory_pool
from torch._inductor.fx_passes.overlap_scheduling import is_compute_node
from torch.distributed.elastic.utils.distributed import get_free_port
from torch.distributed.fsdp import MixedPrecisionPolicy

from torchtitan.config import TrainingConfig
from torchtitan.config.parallelism import ParallelismConfig
from torchtitan.distributed import ParallelismContext
from torchtitan.experiments.graph_trainer.common_utils import (
    annotate_module_fqns,
    compute_parameter_gradients,
    get_simple_fsdp_mesh,
)
from torchtitan.experiments.graph_trainer.configs import GraphTrainerCompileConfig
from torchtitan.experiments.graph_trainer.make_fx_tracer import (
    minimal_fx_tracer,
    run_traced,
)
from torchtitan.experiments.graph_trainer.passes import compile_time_passes
from torchtitan.experiments.graph_trainer.simple_fsdp import data_parallel


class _ResidualModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = torch.nn.ModuleList(
            [torch.nn.Linear(128, 128, bias=False) for _ in range(3)]
        )
        for parameter in self.parameters():
            torch.nn.init.normal_(parameter, std=0.01)

    def forward(self, x):
        for layer in self.layers:
            x = x + torch.nn.functional.silu(layer(x))
        return x


def _run_training(rank, port):
    os.environ.update(
        MASTER_ADDR="localhost",
        MASTER_PORT=str(port),
        CUBLAS_WORKSPACE_CONFIG=":4096:8",
    )
    torch.cuda.set_device(rank)
    torch.manual_seed(42)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    dist.init_process_group("cpu:gloo,cuda:nccl", rank=rank, world_size=2)
    try:
        pc = ParallelismContext(
            dp_replicate=1,
            dp_shard=2,
            cp=1,
            tp=1,
            pp=1,
            ep=1,
            world_size=2,
            enable_sequence_parallel=False,
        )
        pc.build_mesh()
        model = _ResidualModel().cuda()
        reference = copy.deepcopy(model)
        for module in (model, reference):
            data_parallel(
                module,
                get_simple_fsdp_mesh(pc),
                "fully_shard",
                mp_policy=MixedPrecisionPolicy(
                    param_dtype=torch.bfloat16, reduce_dtype=torch.float32
                ),
            )
            annotate_module_fqns(module)
        inputs = torch.randn(256, 128, device="cuda", dtype=torch.bfloat16)

        def step(x):
            loss = model(x).float().square().mean()
            return [loss, *compute_parameter_gradients(loss, model.named_parameters())]

        traced = minimal_fx_tracer(step, module=model)(inputs)
        config = SimpleNamespace(
            training=TrainingConfig(disable_cuda_graphs=True),
            compile=GraphTrainerCompileConfig(
                memory_policy="sac_and_offload", cpu_offload_budget_gb=1
            ),
            parallelism=ParallelismConfig(
                data_parallel_shard_degree=2, fsdp_reshard_after_forward="always"
            ),
            model=SimpleNamespace(
                layers=[SimpleNamespace(feed_forward=None) for _ in range(3)]
            ),
        )
        for graph_pass in compile_time_passes(
            traced, config, parallelism_context=pc, include_inductor=False
        ):
            traced.gm = graph_pass(traced.gm, ())
        nodes = list(traced.gm.graph.nodes)
        positions = {node: index for index, node in enumerate(nodes)}
        assert any("bucket_all_gather" in str(n.target) for n in nodes)
        assert any(
            is_compute_node(node)
            for reload in nodes
            if reload.target == torch.ops.ao.reload.default
            for wait in reload.users
            if wait.target == torch.ops.ao.wait_tensor.default
            for node in nodes[positions[reload] + 1 : positions[wait]]
        )
        execute = run_traced(traced, module=model)
        optimizers = [
            torch.optim.SGD(m.parameters(), lr=0.001) for m in (model, reference)
        ]
        with pinned_memory_pool():
            for _ in range(3):
                outputs = execute(inputs)
            del outputs
            torch.cuda.synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                outputs = execute(inputs)
            for _ in range(10):
                for optimizer in optimizers:
                    optimizer.zero_grad(set_to_none=True)
                inputs.copy_(torch.randn_like(inputs))
                expected_loss = reference(inputs).float().square().mean()
                expected_loss.backward()
                graph.replay()
                assert torch.equal(outputs[0], expected_loss.detach())
                for actual, expected, gradient in zip(
                    model.parameters(), reference.parameters(), outputs[1:], strict=True
                ):
                    actual.grad = gradient
                    assert gradient.dtype == expected.grad.dtype == torch.float32
                    assert torch.equal(gradient.to_local(), expected.grad.to_local())
                norms = [
                    torch.nn.utils.clip_grad_norm_(
                        list(m.parameters()), 1.0, foreach=True
                    )
                    for m in (model, reference)
                ]
                assert torch.equal(norms[0].to_local(), norms[1].to_local())
                for optimizer in optimizers:
                    optimizer.step()
            for actual, expected in zip(
                model.parameters(), reference.parameters(), strict=True
            ):
                assert torch.equal(actual.to_local(), expected.to_local())
            graph.reset()
    finally:
        dist.destroy_process_group()


@pytest.mark.multi_gpu
@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="Requires two CUDA GPUs")
def test_activation_offload_with_simple_fsdp_cuda_replay():
    mp.spawn(_run_training, args=(get_free_port(),), nprocs=2, join=True)
