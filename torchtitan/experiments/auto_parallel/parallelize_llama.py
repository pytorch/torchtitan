# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import time

import torch

from autoparallel.api import AutoParallel

from torch.distributed import DeviceMesh
from torch.distributed.fsdp import MixedPrecisionPolicy
from torch.distributed.tensor.placement_types import Replicate, Shard

from torchtitan.config_manager import JobConfig, TORCH_DTYPE_MAP
from torchtitan.distributed import ParallelDims

from torchtitan.tools.logging import logger


def parallelize_llama(
    model,
    world_mesh: DeviceMesh,
    parallel_dims: ParallelDims,
    job_config: JobConfig,
):
    """
    Apply tensor parallelism, activation checkpointing, torch.compile, and data
    parallelism to the model.

    NOTE: The passed-in model preferably should be on meta device. Otherwise,
    the model must fit on GPU or CPU memory.
    """

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

    import torch
    from autoparallel.activation_checkpointing import (
        force_recompute_fsdp_all_gather,
        mark_nodes_as_must_save_to_stage_recomputation,
    )

    def custom_joint_pass(graph: torch.fx.Graph):
        # ac_stage_size_in_GiB = 2.0
        ac_stage_size_in_GiB = 6.0  # debugmodel..
        force_recompute_fsdp_all_gather(graph)
        mark_nodes_as_must_save_to_stage_recomputation(
            graph, stage_size_in_GiB=ac_stage_size_in_GiB
        )

    torch._inductor.config.comprehensive_padding = False

    torch._inductor.config.joint_custom_post_pass = custom_joint_pass

    from torch._inductor.fx_passes.bucketing import (
        Any,
        Callable,
        defaultdict,
        is_reduce_scatter_tensor,
        is_wait_tensor,
        Optional,
        OrderedSet,
    )

    def greedy_bucket_collective_by_mb(
        gm: torch.fx.GraphModule,
        bucket_cap_mb_by_bucket_idx: Callable[[int], float],
        filter_node: Callable[[torch.fx.Node], bool],
        node_group_key: Callable[[torch.fx.Node], Any],
        filter_wait_node: Optional[Callable[[torch.fx.Node], bool]] = None,
    ) -> list[list[torch.fx.Node]]:
        g = gm.graph
        found_candidates = False
        for node in g.nodes:
            if filter_node(node):
                found_candidates = True
                break
        if not found_candidates:
            return []

        nodes_groups: dict[Any, list[torch.fx.Node]] = defaultdict(list)
        nodes_successors: dict[torch.fx.Node, OrderedSet[torch.fx.Node]] = defaultdict(
            OrderedSet
        )

        for node in g.nodes:
            for n, successors in nodes_successors.items():
                if any(arg in successors for arg in node.args):
                    successors.add(n)
            if is_wait_tensor(node) and filter_node(node.args[0]):
                if (filter_wait_node is None) or filter_wait_node(node):
                    coll_node = node.args[0]
                    group_key = node_group_key(coll_node)
                    nodes_groups[group_key].append(coll_node)

        buckets: list[list[torch.fx.Node]] = []
        for nodes in nodes_groups.values():
            cur_bucket: list[torch.fx.Node] = []
            cur_bucket_successors: OrderedSet[torch.fx.Node] = OrderedSet()
            cur_bucket_size_bytes: int = 0
            cur_bucket_id: int = 0
            bucket_size_bytes = int(
                bucket_cap_mb_by_bucket_idx(cur_bucket_id) * 1024 * 1024
            )
            for node in nodes:
                if node in cur_bucket_successors:
                    # We can not bucket successors with the node
                    continue
                assert "val" in node.meta
                n_val = node.meta["val"]
                n_arg_val = node.args[0].meta["val"]
                out_size_bytes = n_val.numel() * n_val.element_size()
                in_size_bytes = n_arg_val.numel() * n_arg_val.element_size()
                size_bytes = max(out_size_bytes, in_size_bytes)
                if (
                    cur_bucket_size_bytes + size_bytes > bucket_size_bytes
                    and cur_bucket
                ):
                    # Current bucket is full, create new bucket
                    if len(cur_bucket) > 1:
                        buckets.append(cur_bucket)
                    cur_bucket = []
                    cur_bucket_size_bytes = 0
                    cur_bucket_id += 1
                    cur_bucket_successors = OrderedSet()
                cur_bucket_size_bytes += size_bytes
                cur_bucket.append(node)
                cur_bucket_successors |= nodes_successors[node]
            if len(cur_bucket) > 1:
                buckets.append(cur_bucket)
        return buckets

    def bucket_reduce_scatter_by_mb(
        gm: torch.fx.GraphModule,
        bucket_cap_mb_by_bucket_idx: Callable[[int], float],
        filter_wait_node: Optional[Callable[[torch.fx.Node], bool]] = None,
    ) -> list[list[torch.fx.Node]]:
        """
        Identifies all reduce_scatter nodes and groups them into buckets,
            based on size limit `bucket_cap_mb_by_bucket_idx`.

        Args:
            gm (torch.fx.GraphModule): GraphModule where to bucket reduce_scatters.
            bucket_cap_mb_bucket_idx (Callable[[int], float]): Callable to specify cap of the bucket
                in megabytes by bucket idx.  The idea of `bucket_cap_mb_by_bucket_idx` is to allow
                to specify different sizes of the buckets.
            filter_wait_node (Optional[Callable[[torch.fx.Node], bool]]): If specified,
                only reduce_scatter nodes with wait_node that satisfy `filter_wait_node` will be bucketed.

        Returns:
            list[list[torch.fx.Node]]: List of buckets, where each bucket is a list of all_gather nodes.
        """

        def _rs_group_key(node: torch.fx.Node) -> tuple[str, str, torch.dtype]:
            _, reduce_op, group_size, group_name = node.args
            dtype = node.meta["val"].dtype
            assert isinstance(group_name, str)
            assert isinstance(reduce_op, str)
            return (group_name, reduce_op, dtype)

        return greedy_bucket_collective_by_mb(
            gm,
            bucket_cap_mb_by_bucket_idx,
            is_reduce_scatter_tensor,
            _rs_group_key,
            filter_wait_node,
        )

    def reduce_scatter_merge_fn_to_trace(
        rs_ins: list[torch.Tensor],
        group_size: int,
        group_name: str,
        reduce_op: str,
        reduce_dtype: torch.dtype,  # type: ignore[name-defined]
        device: torch.device,  # type: ignore[name-defined]
    ) -> list[torch.Tensor]:  # type: ignore[no-untyped-def]
        print("This one")
        out = []
        shapes = [(x.shape[0] // group_size,) + x.shape[1:] for x in rs_ins]
        numels = [x.numel() // group_size for x in rs_ins]

        r = [x.view(group_size, -1) for x in rs_ins]
        out = torch.cat(r, dim=1).flatten()

        out = torch.ops._c10d_functional.reduce_scatter_tensor(
            out, reduce_op, group_size, group_name
        )
        out = torch.ops.c10d_functional.wait_tensor(out)
        out = out.split(numels, 0)
        out = [x.view(s) for x, s in zip(out, shapes)]
        return out

    import torch._inductor.fx_passes.bucketing

    torch._inductor.fx_passes.bucketing.reduce_scatter_merge_fn_to_trace = (
        reduce_scatter_merge_fn_to_trace
    )

    def bucket_fsdp_reduce_scatter(
        gm: torch.fx.GraphModule,
        bucket_cap_mb_by_bucket_idx=None,
    ) -> None:
        """
        Bucketing pass for SimpleFSDP reduce_scatter ops.

        Attributes:
            gm (torch.fx.GraphModule): Graph module of the graph.
            bucket_cap_mb_by_bucket_idx (Optional[Callable[[int], float]]): callback function that
                takes in bucket idx and returns size of a bucket in megabytes. By default
                torch._inductor.fx_passes.bucketing.bucket_cap_mb_by_bucket_idx_default is used.

        """
        from torch._inductor.fx_passes.fsdp import (
            is_fsdp_reduce_scatter_wait,
            merge_reduce_scatter,
        )

        if bucket_cap_mb_by_bucket_idx is None:
            from torch._inductor.fx_passes.bucketing import (
                bucket_cap_mb_by_bucket_idx_default,
            )

            bucket_cap_mb_by_bucket_idx = bucket_cap_mb_by_bucket_idx_default
        rs_buckets = bucket_reduce_scatter_by_mb(
            gm,
            bucket_cap_mb_by_bucket_idx,
            filter_wait_node=is_fsdp_reduce_scatter_wait,
        )
        if len(rs_buckets) == 0:
            return
        merge_reduce_scatter(gm, rs_buckets)

    def post_grad_custom_post_pass(graph: torch.fx.Graph):
        s = torch._inductor.config.bucket_reduce_scatters_fx_bucket_size_determinator
        bucket_fsdp_reduce_scatter(graph.owning_module, bucket_cap_mb_by_bucket_idx=s)

    torch._inductor.config.bucket_all_gathers_fx = "fsdp"
    torch._inductor.config.bucket_all_gathers_fx_bucket_size_determinator = (
        lambda bucket_idx: 500
    )
    torch._inductor.config.bucket_reduce_scatters_fx_bucket_size_determinator = (
        lambda bucket_idx: 1000
    )

    torch._inductor.config.post_grad_custom_post_pass = post_grad_custom_post_pass

    # bail out
    # model = model_fn()
    # return model
    if job_config.experimental.autop_force_bf16:
        logger.info("Forcing bf16 on model")
        model = model.bfloat16()

    param_dtype = TORCH_DTYPE_MAP[job_config.training.mixed_precision_param]
    reduce_dtype = TORCH_DTYPE_MAP[job_config.training.mixed_precision_reduce]
    mp_policy = MixedPrecisionPolicy(param_dtype=param_dtype, reduce_dtype=reduce_dtype)
    with AutoParallel(model, input_fn, world_mesh, mp_policy=mp_policy) as autop:
        autop.add_parameter_memory_constraint(low=None, high=None)

        possible_input_shardings = {
            # maps relative to mesh dim names used in torchtitan
            "dp_replicate": Shard(0),
            "dp_shard": Shard(0),
            "tp": Replicate(),
        }
        assert all(
            name in possible_input_shardings for name in world_mesh.mesh_dim_names
        ), f"Unsupported mesh dim in world mesh, only {possible_input_shardings.keys()} are supported by AutoParallel"
        x_sharding = tuple(
            possible_input_shardings[name] for name in world_mesh.mesh_dim_names
        )
        autop.add_input_constraints([x_sharding])
        autop.add_output_constraints([x_sharding])
        t0 = time.time()
        sharding_placement = autop.optimize_placement()
        t1 = time.time()
        logger.info(f"AutoParallel took {t1 - t0} seconds")
        parallel_mod = autop.apply_placement(sharding_placement)

    # if job_config.training.compile:
    #    torch._inductor.config.reorder_for_peak_memory = False
    #    parallel_mod.compile(fullgraph=True)

    return parallel_mod
