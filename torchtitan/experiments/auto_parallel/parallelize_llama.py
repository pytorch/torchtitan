# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import time

import torch

from autoparallel.api import AutoParallel

from torch.distributed.fsdp import MixedPrecisionPolicy
from torch.distributed.tensor.placement_types import Partial, Replicate, Shard

from torchtitan.config import JobConfig, TORCH_DTYPE_MAP
from torchtitan.distributed import ParallelDims

from torchtitan.tools.logging import logger


def group_mm_nodes_with_its_gradients(nodes):
    fwd_nodes = [n for n in nodes if "nn_module_stack" in n.meta]
    bwd_nodes = [n for n in nodes if "fwd_nn_module_stack" in n.meta]
    assert len(fwd_nodes) * 2 == len(bwd_nodes)
    res = {}
    for fwd_node in fwd_nodes:
        o = []
        for bwd_node in bwd_nodes:
            if fwd_node.meta["nn_module_stack"] == bwd_node.meta["fwd_nn_module_stack"]:
                o.append(bwd_node)
        assert len(o) == 2
        res[fwd_node] = o
    return res


def force_tp_constraints(autop, mm_nodes, feat_dim=1, bwd_constraint=False):
    # out = x @ w   - S(0)R, RS(1) -> S(0)S(1)
    # g_w = g.T @ x - S(1)S(0), S(0)R -> PS(0)
    # g_x = g @ w.T - S(0)S(1), RS(0) -> S(0)P

    add_node_constraint = autop.sharding_optimizer.add_node_constraint
    fwd_bwd_groups = group_mm_nodes_with_its_gradients(mm_nodes)
    fwd_nodes = list(fwd_bwd_groups.keys())
    dim1 = 0 if feat_dim == 1 else 1
    dim2 = 1 if feat_dim == 1 else 0
    # assume there are 7 mm nodes per transformer block
    # skip last mm as it's the final projection layer
    assert (
        len(fwd_nodes) - 1
    ) % 7 == 0, f"expected 7 mm nodes per transformer block, {len(fwd_nodes) - 1}"
    for block in range(0, len(fwd_nodes) - 1, 7):
        fwd_nodes_block = fwd_nodes[block : block + 7]
        # force the first 3 mm nodes to be S(0)S(1)
        the_nodes = fwd_nodes_block[:3] + fwd_nodes_block[4:6]
        for n in the_nodes:
            add_node_constraint(n, (Shard(0), Shard(feat_dim)))
            add_node_constraint(n.all_input_nodes[0], (Shard(0), Replicate()))
            add_node_constraint(n.all_input_nodes[1], (Replicate(), Shard(1)))

            if bwd_constraint:
                bwd_nodes = fwd_bwd_groups[n]
                # first is g_w, second is g_x
                add_node_constraint(bwd_nodes[0], (Partial(), Shard(dim1)))
                add_node_constraint(bwd_nodes[1], (Shard(0), Partial()))

        # add reduction to finish TP, yielding S(0)P
        the_nodes = fwd_nodes_block[3:4] + fwd_nodes_block[6:7]
        for n in the_nodes:
            add_node_constraint(n, (Shard(0), Partial()))
            add_node_constraint(n.all_input_nodes[0], (Shard(0), Shard(feat_dim)))
            add_node_constraint(n.all_input_nodes[1], (Replicate(), Shard(0)))

            if bwd_constraint:
                bwd_nodes = fwd_bwd_groups[n]
                # first is g_w, second is g_x
                add_node_constraint(bwd_nodes[0], (Partial(), Shard(dim2)))
                add_node_constraint(bwd_nodes[1], (Shard(0), Shard(feat_dim)))


def add_tp_constraints(autop):
    mm_nodes = autop.gm.graph.find_nodes(
        op="call_function", target=torch.ops.aten.mm.default
    )
    einsum_nodes = autop.gm.graph.find_nodes(
        op="call_function", target=torch.ops.aten.einsum.default
    )
    assert (len(mm_nodes) > 0) ^ (
        len(einsum_nodes) > 0
    ), f"only one should be non-empty, got {len(mm_nodes)} and {len(einsum_nodes)}"
    feat_dim = 1 if len(mm_nodes) > 0 else 2
    tgt_nodes = mm_nodes + einsum_nodes
    force_tp_constraints(autop, tgt_nodes, feat_dim=feat_dim, bwd_constraint=True)

    if einsum_nodes:
        # add sequence parallelism if we have einsum nodes
        autop.sharding_optimizer.add_node_constraint(
            list(tgt_nodes[3].users)[0], (Shard(0), Partial())
        )
        autop.sharding_optimizer.add_node_constraint(
            list(list(tgt_nodes[3].users)[0].users)[0], (Shard(0), Shard(1))
        )


def parallelize_llama(
    model,
    parallel_dims: ParallelDims,
    job_config: JobConfig,
):
    """
    Apply tensor parallelism, activation checkpointing, torch.compile, and data
    parallelism to the model.

    NOTE: The passed-in model preferably should be on meta device. Otherwise,
    the model must fit on GPU or CPU memory.
    """
    world_mesh = parallel_dims.world_mesh

    def input_fn():
        global_batch_size = job_config.training.global_batch_size
        if global_batch_size < 0:
            # This global batch size results in 1 gradient accumulation
            # step.
            dp_degree = parallel_dims.dp_replicate * parallel_dims.dp_shard
            global_batch_size = job_config.training.local_batch_size * dp_degree
        return (
            torch.randint(
                0,
                # job_config.training.vocab_size,
                model.vocab_size,
                (global_batch_size, job_config.training.seq_len),
                device=torch.device("cuda"),
            ),
        )

    # TODO make autop work correctly with different combinations of DP, DP+TP, TP, and support DDP / HSDP
    assert parallel_dims.dp_replicate_enabled is False, "DDP not supported yet"
    assert parallel_dims.cp_enabled is False, "CP not supported yet"
    assert parallel_dims.pp_enabled is False, "PP not supported yet"

    torch._inductor.config.bucket_all_gathers_fx_bucket_size_determinator = (
        lambda bucket_idx: 500 / parallel_dims.tp
    )
    torch._inductor.config.bucket_reduce_scatters_fx_bucket_size_determinator = (
        lambda bucket_idx: 1000 / parallel_dims.tp
    )

    # XXX MICROPIPELINE
    enable_async_tp = True
    if enable_async_tp:
        mesh = world_mesh
        from torch.distributed._symmetric_memory import enable_symm_mem_for_group

        enable_symm_mem_for_group(mesh["tp"].get_group().group_name)
        torch._inductor.config._micro_pipeline_tp = True
        torch._inductor.config.reorder_for_compute_comm_overlap = False
    # XXX--- MICROPIPELINE

    # bail out
    # model = model_fn()
    # return model
    if job_config.experimental.autop_force_bf16:
        logger.info("Forcing bf16 on model")
        model = model.bfloat16()

    param_dtype = TORCH_DTYPE_MAP[job_config.training.mixed_precision_param]
    reduce_dtype = TORCH_DTYPE_MAP[job_config.training.mixed_precision_reduce]
    mp_policy = MixedPrecisionPolicy(param_dtype=param_dtype, reduce_dtype=reduce_dtype)
    with AutoParallel(
        model,
        input_fn,
        world_mesh,
        mp_policy=mp_policy,
        compile=job_config.compile,
        repeated_subgraphs=True,
    ) as autop:
        autop.add_parameter_memory_constraint(low=None, high=None)

        possible_input_shardings = {
            # maps relative to mesh dim names used in torchtitan
            "dp_replicate": Shard(0),
            "dp_shard": Shard(0),
            "tp": Replicate(),
        }
        # only used if loss parallel is enabled
        possible_output_shardings = {
            # maps relative to mesh dim names used in torchtitan
            "dp_shard": Shard(0),
            "tp": Shard(2),
        }
        assert all(
            name in possible_input_shardings for name in world_mesh.mesh_dim_names
        ), f"Unsupported mesh dim in world mesh, only {possible_input_shardings.keys()} are supported by AutoParallel"
        x_sharding = tuple(
            possible_input_shardings[name] for name in world_mesh.mesh_dim_names
        )
        out_sharding = x_sharding
        loss_parallel_enabled = (
            parallel_dims.tp_enabled
            and not job_config.parallelism.disable_loss_parallel
        )
        if loss_parallel_enabled:
            out_sharding = tuple(
                possible_output_shardings[name]
                for name in world_mesh.mesh_dim_names
                if name != "dp_replicate"
            )
        autop.add_input_constraints([x_sharding])
        autop.add_output_constraints([out_sharding])
        enable_manual_constraint = True
        if enable_manual_constraint:
            add_tp_constraints(autop)
        t0 = time.time()
        sharding_placement = autop.optimize_placement()
        t1 = time.time()
        logger.info(f"AutoParallel took {t1 - t0} seconds")
        parallel_mod = autop.apply_placement(sharding_placement)

    if loss_parallel_enabled:

        # current PyTorch's implementation of loss parallel assumes
        # that the DTensor has a 1d device mesh. This is not true
        # in our case, but we can work around it by adding
        # casting the output to a DTensor on a 1d device mesh.
        # We should just use AutoParallel to do this for us, but
        # it would require putting the loss inside the model as well
        def _return_as_dtensor_for_loss_parallel(module, args, output):
            return torch.distributed.tensor.DTensor.from_local(
                output, world_mesh["tp"], (Shard(2),)
            )

        # not keeping a reference to the hook, don't plan on
        # removing it at any point
        parallel_mod.register_forward_hook(_return_as_dtensor_for_loss_parallel)

    return parallel_mod
