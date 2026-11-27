# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import gc
import importlib.util
import os

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.distributed.device_mesh import init_device_mesh


def _worker(rank, world_size, store_path, top_k):
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    torch.cuda.set_device(rank)
    torch.use_deterministic_algorithms(True)
    dist.init_process_group(
        "nccl", init_method=f"file://{store_path}", rank=rank, world_size=world_size
    )
    from torch._C import _distributed_c10d
    from torchtitan.distributed import nccl_ep
    from torchtitan.distributed.accelerator_graph import (
        accelerator_graph_teardown,
        AcceleratorGraphWrapper,
    )
    from torchtitan.models.common.moe.token_dispatcher import NCCLEPTokenDispatcher

    mesh = init_device_mesh("cuda", (world_size,), mesh_dim_names=("ep",))
    from torchtitan.distributed.nccl_ep_capture import (
        NCCLEPBufferOwner,
        NCCLEPRoutingCapturePlan,
    )

    owner = NCCLEPBufferOwner()
    plan = NCCLEPRoutingCapturePlan()
    dispatcher = None
    graph = None
    from torchtitan.distributed.spmd_types import set_current_spmd_mesh

    with set_current_spmd_mesh(mesh):
        try:
            num_experts = 64 if top_k == 6 else world_size * 2
            num_local_experts = num_experts // world_size
            dispatcher = NCCLEPTokenDispatcher.Config(
                num_experts=num_experts,
                top_k=top_k,
                hidden_dim=512,
                num_max_tokens_per_rank=32,
                # Pad only for top_k=6, whose combine derivative then has
                # padding rows inside each group.
                pad_multiple=1 if top_k == 2 else 32,
                ht_flow="pull_push",
            ).build()
            with owner.initialize():
                dispatcher.init_buffer()
            x = (
                torch.tensor(
                    [-0.0, 1.0078125, -1.0078125, 0.25],
                    device="cuda",
                    dtype=torch.bfloat16,
                )
                .repeat(32, 128)
                .mul((rank + 1) / 4)
                .requires_grad_()
            )
            scores = (
                torch.arange(32 * top_k, device="cuda", dtype=torch.float32).reshape(
                    32, top_k
                )
                / 1000
                + 0.5
            ).requires_grad_()
            ids = (
                rank * num_local_experts + torch.arange(top_k, device="cuda")
            ).repeat(32, 1)
            routing_map = torch.empty(0, device="cuda", dtype=torch.bool)

            def forward(x, scores, ids):
                assert dispatcher is not None
                routed, _, metadata = dispatcher.dispatch(x, scores, ids, routing_map)
                # Like the MoE experts, scores scale the rows before combine.
                scored = (routed.square() * metadata.scores_R[:, None]).to(routed.dtype)
                return dispatcher.combine(scored, metadata, x)

            compiled = torch.compile(
                forward, fullgraph=True, options={"emulate_precision_casts": True}
            )
            x.grad = torch.zeros_like(x)
            scores.grad = torch.zeros_like(scores)

            def step(x, scores, ids):
                x.grad.zero_()
                scores.grad.zero_()
                output = compiled(x, scores, ids)
                output.float().sum().backward()
                return output.detach().clone(), x.grad.clone(), scores.grad.clone()

            graph = AcceleratorGraphWrapper(
                lambda *args: plan.run(step, *args),
                (x, scores, ids),
                static_input_indices=(0, 1, 2),
                num_warmup_iterations=2,
                prepare_for_capture=plan.prepare_for_capture,
            )
            for _ in range(3):
                graph(x, scores, ids)
            for phase in range(4):
                destination = (
                    phase % world_size if phase % 2 else (rank + phase) % world_size
                )
                ids.copy_(
                    (
                        destination * num_local_experts
                        + torch.arange(top_k, device="cuda")
                    ).repeat(32, 1)
                )
                if phase == 3:
                    ids[::3, -1] = -1
                actual = tuple(value.clone() for value in graph(x, scores, ids))
                reference_x = x.detach().clone().requires_grad_()
                reference_scores = scores.detach().clone().requires_grad_()
                expected = forward(reference_x, reference_scores, ids)
                expected.float().sum().backward()
                for value, reference in zip(
                    actual,
                    (expected, reference_x.grad, reference_scores.grad),
                    strict=True,
                ):
                    assert reference is not None
                    torch.testing.assert_close(value, reference, rtol=0, atol=0)
                    integer_dtype = (
                        torch.int16 if value.dtype == torch.bfloat16 else torch.int32
                    )
                    assert torch.equal(
                        value.contiguous().view(integer_dtype),
                        reference.contiguous().view(integer_dtype),
                    )
                oracle_x = x.detach().clone().requires_grad_()
                oracle_scores = scores.detach().clone().requires_grad_()
                from torchtitan.ops.scatter_add import deterministic_scatter_add

                expert_order_TK = ids.flatten().argsort(stable=True)
                source_R = expert_order_TK // top_k
                # Native combine and its adjoint sum each token's rows in
                # FP32 and round to BF16 once.
                expert_x_RD = oracle_x.float()[source_R].to(torch.bfloat16)
                expert_scores_R = oracle_scores.flatten()[expert_order_TK]
                contributions_RD = (expert_x_RD.square() * expert_scores_R[:, None]).to(
                    torch.bfloat16
                )
                contributions_RD = torch.where(
                    (ids.flatten()[expert_order_TK] >= 0)[:, None],
                    contributions_RD,
                    0,
                )
                oracle_output = deterministic_scatter_add(
                    torch.zeros_like(oracle_x, dtype=torch.float32),
                    source_R[:, None].expand(-1, 512),
                    contributions_RD.float(),
                ).to(torch.bfloat16)
                oracle_output.float().sum().backward()
                for value, reference in zip(
                    actual,
                    (oracle_output, oracle_x.grad, oracle_scores.grad),
                    strict=True,
                ):
                    assert reference is not None
                    torch.testing.assert_close(value, reference, rtol=0, atol=0)
                    integer_dtype = (
                        torch.int16 if value.dtype == torch.bfloat16 else torch.int32
                    )
                    assert torch.equal(
                        value.contiguous().view(integer_dtype),
                        reference.contiguous().view(integer_dtype),
                    )
            graph.teardown()
            plan.release()
            graph = None
            gc.collect()
            assert not any(slot[2] for slot in dispatcher._routing_pool.value.slots)
            assert _distributed_c10d._get_work_registry_size() == 0
        finally:
            accelerator_graph_teardown()
            plan.release()
            owner.close()
            assert not nccl_ep._buffers
            dist.destroy_process_group()


@pytest.mark.multi_gpu
@pytest.mark.skipif(torch.cuda.device_count() < 4, reason="requires four CUDA GPUs")
# TODO: TokenSwitch is not in PyTorch nightly yet; drop this skip once it lands.
@pytest.mark.skipif(
    importlib.util.find_spec("torch.distributed._token_switch") is None
    or torch.version.hip is not None,
    reason="requires TokenSwitch NCCL EP (CUDA only)",
)
@pytest.mark.parametrize("top_k", [2, 6])
def test_dispatcher_fullgraph_and_changing_route_training(tmp_path, top_k):
    mp.spawn(
        _worker,
        args=(4, str(tmp_path / "nccl-ep-store"), top_k),
        nprocs=4,
        join=True,
    )
