# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import os
import tempfile
import unittest
from typing import cast

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.functional as F
from torch.distributed.device_mesh import DeviceMesh
from torch.nn.attention.flex_attention import (
    AuxOutput,
    AuxRequest,
    create_block_mask,
    flex_attention,
)

from torchtitan.experiments.graph_trainer.common_utils import (
    maybe_register_blockmask_pytree_node,
)
from torchtitan.experiments.graph_trainer.make_fx_tracer import (
    minimal_fx_tracer,
    run_traced,
)
from torchtitan.experiments.graph_trainer.simple_cp import (
    simple_cp_flex,
    simple_cp_sdpa,
    SimpleCPTransform,
)
from torchtitan.experiments.graph_trainer.subgraph_regions import (
    apply_subgraph_region_annotations_pass,
)

cp_flex_attention = simple_cp_flex(flex_attention)
cp_sdpa = simple_cp_sdpa(F.scaled_dot_product_attention)


def _head_dependent_causal_mask(_batch, head, query_idx, key_idx):
    return (query_idx >= key_idx) & ((head % 2 == 0) | (key_idx % 2 == 0))


def _run_numerics(rank, world_size, rendezvous):
    torch.cuda.set_device(rank)
    dist.init_process_group(
        "nccl",
        init_method=rendezvous,
        rank=rank,
        world_size=world_size,
    )
    try:
        cp_mesh = DeviceMesh("cuda", list(range(world_size)), mesh_dim_names=("cp",))
        batch, num_query_heads, num_kv_heads, global_seq, head_dim = 1, 4, 2, 256, 64
        local_seq = global_seq // world_size
        seq_start = rank * local_seq
        seq_end = seq_start + local_seq

        torch.manual_seed(1234)
        global_query = torch.randn(
            batch,
            num_query_heads,
            global_seq,
            head_dim,
            device="cuda",
            dtype=torch.float32,
        )
        global_key = torch.randn(
            batch,
            num_kv_heads,
            global_seq,
            head_dim,
            device="cuda",
            dtype=torch.float32,
        )
        global_value = torch.randn_like(global_key)
        block_mask = create_block_mask(
            _head_dependent_causal_mask,
            B=None,
            H=num_query_heads,
            Q_LEN=global_seq,
            KV_LEN=global_seq,
            device="cuda",
        )
        kernel_options = {"BACKEND": "TRITON"}

        query_ref = global_query.detach().requires_grad_()
        key_ref = global_key.detach().requires_grad_()
        value_ref = global_value.detach().requires_grad_()
        output_ref, aux_ref = cast(
            tuple[torch.Tensor, AuxOutput],
            cp_flex_attention(
                query_ref,
                key_ref,
                value_ref,
                block_mask=block_mask,
                kernel_options=kernel_options,
                enable_gqa=True,
                return_aux=AuxRequest(max_scores=True),
            ),
        )
        assert aux_ref.max_scores is not None
        grads_ref = torch.autograd.grad(
            output_ref.float().square().sum(), (query_ref, key_ref, value_ref)
        )

        query = global_query[:, :, seq_start:seq_end].detach().requires_grad_()
        key = global_key[:, :, seq_start:seq_end].detach().requires_grad_()
        value = global_value[:, :, seq_start:seq_end].detach().requires_grad_()

        def simple_cp_step(query, key, value, block_mask):
            output, aux = cast(
                tuple[torch.Tensor, AuxOutput],
                cp_flex_attention(
                    query,
                    key,
                    value,
                    block_mask=block_mask,
                    kernel_options=kernel_options,
                    enable_gqa=True,
                    return_aux=AuxRequest(max_scores=True),
                ),
            )
            assert aux.max_scores is not None
            grads = torch.autograd.grad(
                output.float().square().sum(), (query, key, value)
            )
            return output, aux.max_scores.max(), *grads

        maybe_register_blockmask_pytree_node()
        traced = minimal_fx_tracer(
            simple_cp_step,
            trace_time_transforms=[SimpleCPTransform(cp_mesh)],
        )(query, key, value, block_mask)

        all_to_all_count = sum(
            node.target == torch.ops._c10d_functional.all_to_all_single.default
            for node in traced.gm.graph.nodes
        )
        if all_to_all_count < 5:
            raise AssertionError(
                f"Expected simple_cp all-to-alls in the joint graph, got {all_to_all_count}"
            )
        wait_count = sum(
            node.target == torch.ops._c10d_functional.wait_tensor.default
            for node in traced.gm.graph.nodes
        )

        nodes = list(traced.gm.graph.nodes)
        first_wait = next(
            i
            for i, node in enumerate(nodes)
            if node.target == torch.ops._c10d_functional.wait_tensor.default
        )
        launches_before_wait = sum(
            node.target == torch.ops._c10d_functional.all_to_all_single.default
            for node in nodes[:first_wait]
        )
        if launches_before_wait != 3:
            raise AssertionError(
                "Expected Q, K, and V all-to-alls to start before the first wait, "
                f"got {launches_before_wait} launches"
            )

        apply_subgraph_region_annotations_pass(traced.gm)
        simple_cp_regions = [
            node
            for node in traced.gm.graph.nodes
            if node.target == torch.ops.higher_order.invoke_subgraph
            and node.meta.get("custom", {}).get("subgraph_region_id") == "simple_cp"
        ]
        if len(simple_cp_regions) != 2:
            raise AssertionError(
                "Expected outlined forward and backward simple_cp regions, got "
                f"{len(simple_cp_regions)}"
            )
        region_targets = []
        for region_node in simple_cp_regions:
            region_attr = region_node.args[0]
            region = getattr(traced.gm, region_attr.target)
            for module in region.modules():
                if isinstance(module, torch.fx.GraphModule):
                    region_targets.extend(node.target for node in module.graph.nodes)
        if (
            region_targets.count(torch.ops._c10d_functional.all_to_all_single.default)
            != all_to_all_count
            or region_targets.count(torch.ops._c10d_functional.wait_tensor.default)
            != wait_count
        ):
            raise AssertionError("Expected all simple_cp transfers inside its subgraph")

        output, max_score, *grads = run_traced(traced)(query, key, value, block_mask)

        global_max_score = max_score.clone()
        dist.all_reduce(global_max_score, op=dist.ReduceOp.MAX)

        torch.testing.assert_close(output, output_ref[:, :, seq_start:seq_end])
        torch.testing.assert_close(global_max_score, aux_ref.max_scores.max())
        for grad, grad_ref in zip(grads, grads_ref):
            torch.testing.assert_close(grad, grad_ref[:, :, seq_start:seq_end])

        head = torch.arange(num_query_heads, device="cuda").view(1, -1, 1, 1)
        query_idx = torch.arange(global_seq, device="cuda").view(1, 1, -1, 1)
        key_idx = torch.arange(global_seq, device="cuda").view(1, 1, 1, -1)
        attn_mask = (query_idx >= key_idx) & ((head % 2 == 0) | (key_idx % 2 == 0))

        query_ref = global_query.detach().requires_grad_()
        key_ref = global_key.detach().requires_grad_()
        value_ref = global_value.detach().requires_grad_()
        output_ref = cp_sdpa(
            query_ref,
            key_ref,
            value_ref,
            attn_mask=attn_mask,
            enable_gqa=True,
        )
        grads_ref = torch.autograd.grad(
            output_ref.float().square().sum(), (query_ref, key_ref, value_ref)
        )

        query = global_query[:, :, seq_start:seq_end].detach().requires_grad_()
        key = global_key[:, :, seq_start:seq_end].detach().requires_grad_()
        value = global_value[:, :, seq_start:seq_end].detach().requires_grad_()

        def simple_cp_sdpa_step(query, key, value, attn_mask):
            output = cp_sdpa(
                query,
                key,
                value,
                attn_mask=attn_mask,
                enable_gqa=True,
            )
            grads = torch.autograd.grad(
                output.float().square().sum(), (query, key, value)
            )
            return output, *grads

        traced = minimal_fx_tracer(
            simple_cp_sdpa_step,
            trace_time_transforms=[SimpleCPTransform(cp_mesh)],
        )(query, key, value, attn_mask)
        output, *grads = run_traced(traced)(query, key, value, attn_mask)

        torch.testing.assert_close(output, output_ref[:, :, seq_start:seq_end])
        for grad, grad_ref in zip(grads, grads_ref):
            torch.testing.assert_close(grad, grad_ref[:, :, seq_start:seq_end])
    finally:
        dist.destroy_process_group()


class TestSimpleContextParallel(unittest.TestCase):
    @unittest.skipIf(torch.cuda.device_count() < 2, "simple_cp requires 2 GPUs")
    def test_matches_full_sequence_reference(self):
        os.environ.setdefault("NCCL_SOCKET_IFNAME", "lo")
        with tempfile.TemporaryDirectory() as tmpdir:
            rendezvous = f"file://{tmpdir}/rdzv"
            mp.spawn(_run_numerics, args=(2, rendezvous), nprocs=2, join=True)


if __name__ == "__main__":
    unittest.main()
