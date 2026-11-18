# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Apply FlexShard's bucket runtime before GraphTrainer tracing."""

import logging
from collections import defaultdict
from functools import partial
from itertools import pairwise

import spmd_types as spmd
import torch
import torch.distributed as dist
import torch.fx as fx
from flex_shard import (
    BucketSpec,
    flex_shard,
    FLEX_SHARD_META_ACTION,
    FLEX_SHARD_META_MESH_AXIS_NAMES,
    FLEX_SHARD_META_PARAM_FQNS,
    FLEX_SHARD_META_PHASE,
    get_mesh,
    get_placements,
    MixedPrecisionPolicy,
)
from flex_shard.custom_placements.shard import Shard
from flex_shard.layout_adapters.spmd_types import spmd_types_to_global_layout
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor import DTensor, Shard as DTensorShard
from torch.distributed.tensor.placement_types import (
    _StridedShard,
    Placement as DTensorPlacement,
)

from torchtitan.components.optim import GradientClipper
from torchtitan.config import PrecisionConfig, TORCH_DTYPE_MAP
from torchtitan.distributed import ParallelismContext
from torchtitan.distributed.fsdp import linear_param_shard_placements
from torchtitan.distributed.spmd_types import spmd_axes
from torchtitan.experiments.graph_trainer.graph_pp.split_fsdp_collectives import (
    FSDPGraphRegions,
)
from torchtitan.experiments.graph_trainer.graph_pp.utils import graph_outputs
from torchtitan.models.base.module import Module
from torchtitan.models.common.decoder import Decoder
from torchtitan.models.common.moe import MoE

from .common_utils import get_default_transformer_block_buckets


logger = logging.getLogger(__name__)


def apply_flex_shard(
    model: Decoder,
    *,
    parallelism_context: ParallelismContext,
    precision: PrecisionConfig,
) -> Decoder:
    """Use the manual scheduling module groups as FlexShard buckets."""
    fsdp_mesh = parallelism_context.get_optional_mesh(
        ["dp_shard", "cp"], include_singleton_axes=True
    )
    assert fsdp_mesh is not None
    fsdp_mesh = fsdp_mesh._flatten("fsdp")
    expert_mesh = parallelism_context.get_optional_mesh(
        "edp_shard", include_singleton_axes=True
    )
    mp_policy = MixedPrecisionPolicy(
        param_dtype=TORCH_DTYPE_MAP[precision.param_dtype],
        reduce_dtype=TORCH_DTYPE_MAP[precision.reduce_dtype],
    )
    shard_placements = linear_param_shard_placements(model)
    if parallelism_context.ep_enabled:
        assert expert_mesh is not None
        for module in model.modules():
            if not isinstance(module, MoE):
                continue
            experts = module.routed_experts
            if expert_mesh.size() * parallelism_context.ep <= experts.w13.group_size:
                for param in experts.parameters():
                    shard_placements.pop(param, None)
            else:
                shard_placements.update(
                    linear_param_shard_placements(
                        experts, include_unstacked_grouped=True
                    )
                )

    def placement_fn(named_params, mesh):
        del mesh
        return {
            fqn: (
                Shard(shard_placements[param].dim if param in shard_placements else 0),
            )
            for fqn, param in named_params
        }

    for module in model.modules():
        if not isinstance(module, Module) or module._sharding_config is None:
            continue
        for name, param in module.named_parameters(recurse=False):
            layout = module._sharding_config.state_shardings[name]
            mesh = parallelism_context.get_optional_mesh(
                [axis.value for axis in spmd_axes(layout)],
                include_singleton_axes=True,
            )
            assert mesh is not None
            spmd_types_to_global_layout(module, {name: layout}, mesh)
    module_buckets = get_default_transformer_block_buckets(
        len(model.config.layers),
        num_mtp_layers=len(getattr(model.config, "mtp_layers", ()) or ()),
        moe_layer_ids=frozenset(
            int(layer_id)
            for layer_id, layer in model.layers.items()
            if isinstance(layer.feed_forward, MoE)
        ),
        split_moe_expert_buckets=parallelism_context.ep_enabled,
    )
    module_buckets[-1] = "norm"
    module_buckets.append("lm_head")
    buckets = []
    for module_bucket in module_buckets:
        module_fqns = (
            [module_bucket] if isinstance(module_bucket, str) else module_bucket
        )
        mesh = fsdp_mesh
        if any(fqn.endswith(".routed_experts") for fqn in module_fqns):
            assert expert_mesh is not None
            mesh = expert_mesh
        buckets.append(
            BucketSpec(
                [f"{fqn}.*" for fqn in module_fqns],
                mesh=mesh,
                placement_fn=placement_fn,
                mp_policy=mp_policy,
                gradient_reduce_op=dist.ReduceOp.SUM,
                reshard_after_forward=False,
            )
        )
    flex_shard(model, buckets=buckets)
    for module in model.modules():
        if not isinstance(module, Module) or module._param_init is None:
            continue
        param_init = module._param_init.copy()
        for name, param in module.named_parameters(recurse=False):
            param_init[name] = partial(
                _init_dtensor_param,
                layout=module._sharding_config.state_shardings[name],
                init_fn=param_init[name],
                parallelism_context=parallelism_context,
            )
        module._param_init = param_init
    logger.info("Applied FlexShard with %s module buckets", len(buckets))
    return model


def _init_dtensor_param(param, *, layout, init_fn, parallelism_context):
    outer_mesh = parallelism_context.get_optional_mesh(
        "ep" if "ep" in layout.local_type else "tp"
    )
    mesh = get_mesh(param)
    assert mesh is not None
    placements: list[DTensorPlacement] = [
        DTensorShard(p.dim) for p in get_placements(param)
    ]
    if outer_mesh is not None:
        outer_placements = []
        for axis in outer_mesh.mesh_dim_names:
            axis_type = layout.local_type[axis]
            if layout.partition_spec is not None:
                for dim, entry in enumerate(layout.partition_spec):
                    axes = entry if isinstance(entry, tuple) else (entry,)
                    if axis in axes:
                        axis_type = spmd.S(dim)
            outer_placements.append(spmd.spmd_type_to_dtensor_placement(axis_type))
        for index, placement in enumerate(placements):
            split_factor = 1
            for axis_index, outer_placement in enumerate(outer_placements):
                if outer_placement.is_shard(placement.dim):
                    split_factor *= outer_mesh.size(axis_index)
            if split_factor > 1:
                placements[index] = _StridedShard(
                    placement.dim, split_factor=split_factor
                )
        placements.extend(outer_placements)
        mesh = DeviceMesh._concatenate([mesh, outer_mesh])
    param._graph_trainer_dtensor_param = torch.nn.Parameter(
        DTensor.from_local(param.detach(), mesh, tuple(placements), run_check=False),
        requires_grad=False,
    )
    init_fn(param._graph_trainer_dtensor_param)


class FlexShardGradientClipper(GradientClipper):
    @torch.no_grad()
    def clip_grad_norm_(self, parameters, max_norm, *, parallelism_context):
        norm_parameters = []
        for param in parameters:
            if param.grad is None:
                continue
            norm_param = param._graph_trainer_dtensor_param
            norm_param.grad = DTensor.from_local(
                param.grad,
                norm_param.device_mesh,
                norm_param.placements,
                run_check=False,
            )
            norm_parameters.append(norm_param)
        norm = super().clip_grad_norm_(
            norm_parameters, max_norm, parallelism_context=parallelism_context
        )
        for norm_param in norm_parameters:
            norm_param.grad = None
        return norm


def _collect_bucket_regions(graph: fx.Graph):
    buckets = defaultdict(list)
    region_key = None
    phase = None
    for node in graph.nodes:
        if node.op in ("placeholder", "output"):
            continue
        meta = node.meta.get("custom", {})
        action = meta.get(FLEX_SHARD_META_ACTION)
        if action not in ("unshard", "reduce_grad"):
            region_key = None
            continue
        key = (
            action,
            meta[FLEX_SHARD_META_PARAM_FQNS],
            meta[FLEX_SHARD_META_MESH_AXIS_NAMES],
            node.meta.get("autograd_backward", False),
        )
        next_phase = meta[FLEX_SHARD_META_PHASE]
        # A new prepare phase also separates consecutive uses of one bucket.
        if key != region_key or (phase == "execute" and next_phase == "prepare"):
            region = []
            buckets[action, key[-1]].append(region)
        region.append(node)
        region_key, phase = key, next_phase
    return buckets


def get_flex_shard_graph_regions(
    gm: fx.GraphModule, *, include_reduce_grad_inputs: bool = True
) -> FSDPGraphRegions:
    unshard_nodes = set()
    reduce_regions = []
    for (action, backward), regions in _collect_bucket_regions(gm.graph).items():
        for region in regions:
            if action == "unshard" and not backward:
                unshard_nodes.update(region)
            elif action == "reduce_grad":
                execution = {
                    node
                    for node in region
                    if node.meta["custom"][FLEX_SHARD_META_PHASE] == "execute"
                }
                if execution:
                    reduce_regions.append(execution)
    unshard_outputs = tuple(
        node.name
        for node in gm.graph.nodes
        if node in unshard_nodes
        and isinstance(node.meta.get("val"), torch.Tensor)
        and any(user not in unshard_nodes for user in node.users)
    )
    if not include_reduce_grad_inputs:
        return FSDPGraphRegions(
            unshard_node_names=frozenset(node.name for node in unshard_nodes),
            unshard_output_names=unshard_outputs,
            reduce_grad_node_names=frozenset(
                node.name for region in reduce_regions for node in region
            ),
            reduce_grad_inputs=(),
        )
    packed_inputs = {}
    region_by_node = {}
    for key, region in enumerate(reduce_regions):
        if not any(
            node.target
            in (
                torch.ops._c10d_functional.reduce_scatter_tensor,
                torch.ops._c10d_functional.reduce_scatter_tensor.default,
                torch.ops._c10d_functional.all_reduce,
                torch.ops._c10d_functional.all_reduce.default,
            )
            for node in region
        ):
            continue
        inputs = {
            dependency
            for node in region
            for dependency in node.all_input_nodes
            if dependency not in region
            and isinstance(dependency.meta.get("val"), torch.Tensor)
        }
        if len(inputs) != 1:
            raise ValueError(
                f"FlexShard gradient bucket {key} requires one packed input, "
                f"got {tuple(node.name for node in inputs)}"
            )
        packed_inputs[key] = next(iter(inputs))
        region_by_node.update((node, key) for node in region)
    output_owners = []
    blocked_owners = set()
    for output in graph_outputs(gm.graph):
        if not isinstance(output, fx.Node):
            continue
        pending = [output]
        visited = set()
        owners = set()
        while pending:
            node = pending.pop()
            if node in visited:
                continue
            visited.add(node)
            if node in region_by_node:
                owners.add(region_by_node[node])
            else:
                pending.extend(node.all_input_nodes)
        output_owners.append((output, owners))
        if len(owners) > 1:
            blocked_owners.update(owners)
    grad_inputs = []
    extracted_owners = set()
    for output, owners in output_owners:
        if len(owners) == 1 and not owners & blocked_owners:
            owner = next(iter(owners))
            grad_inputs.append((output.name, packed_inputs[owner].name))
            extracted_owners.add(owner)
    return FSDPGraphRegions(
        unshard_node_names=frozenset(node.name for node in unshard_nodes),
        unshard_output_names=unshard_outputs,
        reduce_grad_node_names=frozenset(
            node.name
            for node, owner in region_by_node.items()
            if owner in extracted_owners
        ),
        reduce_grad_inputs=tuple(grad_inputs),
    )


def flex_shard_fsdp_scheduling(
    gm: fx.GraphModule,
    example_inputs: tuple | None = None,
    *,
    schedule_all_gathers: bool = True,
    schedule_reduce_scatters: bool = True,
) -> fx.GraphModule:
    """Prefetch one bucket's all-gather and delay one bucket's reduce-scatter wait.

    ``schedule_all_gathers`` enables parameter prefetch;
    ``schedule_reduce_scatters`` enables delaying gradient-reduction waits.
    Disable a type when its collectives will be extracted and scheduled separately.

    All-gather prefetch. Each move must preserve data dependencies.

    INPUT:
        prepare0        # Allocate and fill bucket 0's communication buffers.
        all_gather0     # Launch bucket 0's all-gather.
        wait0           # Wait for bucket 0's parameters.
        unpack0         # Assemble bucket 0's full parameters.
        compute0        # Run the layer using bucket 0.
        prepare1        # Prepare bucket 1's buffers.
        all_gather1     # Launch bucket 1's all-gather.
        wait1           # Wait for bucket 1's parameters.
        unpack1         # Assemble bucket 1's full parameters.

    OUTPUT:
        prepare0
        all_gather0
        prepare1        # Move preparation together with its launch.
        all_gather1     # Prefetch before bucket 0's wait.
        wait0
        unpack0
        compute0        # Bucket 1's all-gather can overlap this compute.
        wait1
        unpack1

    Reduce-scatter wait delay:

    INPUT:
        pack0           # Pack bucket 0's gradients.
        reduce_scatter0 # Launch bucket 0's gradient reduction.
        wait0           # Wait for bucket 0's reduced gradients.
        unpack0         # Produce bucket 0's local gradient shards.
        backward1       # Compute bucket 1's gradients.
        pack1           # Pack bucket 1's gradients.
        reduce_scatter1 # Launch bucket 1's gradient reduction.
        wait1           # Wait for bucket 1's reduced gradients.
        unpack1         # Produce bucket 1's local gradient shards.

    OUTPUT:
        pack0
        reduce_scatter0
        backward1       # Bucket 0's reduce-scatter can overlap this compute.
        pack1
        reduce_scatter1
        wait0           # Delay until after bucket 1's launch.
        unpack0         # Move unpacking together with its wait.
        wait1
        unpack1
    """

    def _is_coll(node: fx.Node, collective) -> bool:
        return node.target in (collective, collective.default)

    collectives = {}
    if schedule_all_gathers:
        collectives["unshard"] = torch.ops._c10d_functional.all_gather_into_tensor
    if schedule_reduce_scatters:
        collectives["reduce_grad"] = torch.ops._c10d_functional.reduce_scatter_tensor
    wait_tensor = torch.ops._c10d_functional.wait_tensor

    def _is_wait_for(node: fx.Node, launch: fx.Node) -> bool:
        return (
            node.target in (wait_tensor, wait_tensor.default) and node.args[0] is launch
        )

    for module in gm.modules():
        if not isinstance(module, fx.GraphModule):
            continue
        buckets = _collect_bucket_regions(module.graph)
        for (action, _), regions in buckets.items():
            if action not in collectives:
                continue
            launch_wait_pairs = []
            for region in regions:
                launches = [
                    node for node in region if _is_coll(node, collectives[action])
                ]
                if not launches:
                    continue  # Singleton meshes do not communicate.
                assert (
                    len(launches) == 1
                ), "Expected one collective per FlexShard region"
                launch = launches[0]
                waits = [node for node in region if _is_wait_for(node, launch)]
                assert len(waits) == 1, "Expected one wait per FlexShard collective"
                wait = waits[0]
                launch_wait_pairs.append(
                    (
                        region[: region.index(launch) + 1],
                        wait,
                        region[region.index(wait) :],
                    )
                )

            for (_, previous_wait, previous_finish), (prepare, _, _) in pairwise(
                launch_wait_pairs
            ):
                if action == "unshard":
                    moving, anchor = prepare, previous_wait
                else:
                    moving, anchor = previous_finish, prepare[-1]
                positions = {
                    node: index for index, node in enumerate(module.graph.nodes)
                }
                moving_set = set(moving)
                if action == "unshard":
                    if any(
                        positions[dependency] >= positions[anchor]
                        for node in moving
                        for dependency in node.all_input_nodes
                        if dependency not in moving_set
                    ):
                        continue
                    for node in moving:
                        anchor.prepend(node)
                else:
                    if any(
                        positions[user] <= positions[anchor]
                        for node in moving
                        for user in node.users
                        if user not in moving_set
                    ):
                        continue
                    for node in reversed(moving):
                        anchor.append(node)
        module.graph.lint()
        module.recompile()
    return gm
