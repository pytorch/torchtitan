# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Apply FlexShard's bucket runtime before GraphTrainer tracing."""

import copy
import logging
import operator
from collections import defaultdict
from collections.abc import Callable, Hashable
from functools import partial
from itertools import pairwise
from typing import Literal

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
    get_flex_shard_global_layouts,
    get_mesh,
    get_placements,
    init_from_full,
    MixedPrecisionPolicy,
    PlacementFn,
)
from flex_shard.custom_placements.block_shard import BlockShard, BucketedBlockShard
from flex_shard.custom_placements.shard import Shard
from flex_shard.layout_adapters.spmd_types import spmd_types_to_global_layout
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor import DTensor, Shard as DTensorShard
from torch.distributed.tensor.placement_types import (
    _StridedShard,
    Placement as DTensorPlacement,
)
from torch.utils._pytree import tree_leaves
from torch.utils.checkpoint import _is_cacheable_effect, CheckpointPolicy

from torchtitan.components.optim import GradientClipper
from torchtitan.config import PrecisionConfig, TORCH_DTYPE_MAP
from torchtitan.config.parallelism import ParallelismConfig
from torchtitan.distributed import ParallelismContext
from torchtitan.distributed.fsdp import (
    get_fsdp_reshard_after_forward_policy,
    linear_param_shard_placements,
)
from torchtitan.distributed.spmd_types import spmd_axes
from torchtitan.experiments.graph_trainer.fsdp_patterns import (
    fsdp_param_fqns,
    is_all_gather_into_tensor,
    is_reduce_grad_collective,
    is_reduce_scatter_tensor,
    is_wait_tensor,
)
from torchtitan.experiments.graph_trainer.graph_pp.split_fsdp_collectives import (
    FSDPGraphRegions,
)
from torchtitan.experiments.graph_trainer.graph_pp.utils import graph_outputs
from torchtitan.experiments.graph_trainer.mutation_utils import (
    base_tensor_for_mutation_target,
    mutation_target_nodes,
)
from torchtitan.models.base.module import Module
from torchtitan.models.common.decoder import Decoder
from torchtitan.models.common.moe import MoE
from torchtitan.models.common.param_init import skip_param_init

from .common_utils import get_default_transformer_block_buckets


logger = logging.getLogger(__name__)


def apply_flex_shard(
    model: Decoder,
    *,
    parallelism_context: ParallelismContext,
    precision: PrecisionConfig,
    placement_fn: PlacementFn | None,
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
    # The generic linear policy shards stacked W13 rows. FlexShard applies our
    # placement_fn, so use SimpleFSDP's expert-axis policy when enough experts
    # are available to preserve seeded initialization.
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

    def _default_placement_fn(named_params, mesh) -> dict[str, tuple[Shard]]:
        del mesh
        return {
            fqn: (
                Shard(shard_placements[param].dim if param in shard_placements else 0),
            )
            for fqn, param in named_params
        }

    for module_fqn, module in model.named_modules():
        if not isinstance(module, Module) or module._sharding_config is None:
            continue
        for name, param in module.named_parameters(recurse=False):
            layout = module._sharding_config.state_shardings[name]
            mesh = parallelism_context.get_optional_mesh(
                [axis.value for axis in spmd_axes(layout)],
                include_singleton_axes=True,
            )
            assert mesh is not None
            for axis_index, axis in enumerate(mesh.mesh_dim_names):
                if (
                    param.requires_grad
                    and axis in ("tp", "ep")
                    and layout.local_type[axis] is spmd.R
                    and mesh.size(axis_index) > 1
                    and not (
                        layout.partition_spec is not None
                        and any(
                            axis in (entry if isinstance(entry, tuple) else (entry,))
                            for entry in layout.partition_spec
                        )
                    )
                ):
                    fqn = f"{module_fqn}.{name}".lstrip(".")
                    raise ValueError(
                        f"FlexShard tracing does not reduce partial gradients for "
                        f"{fqn} replicated across {axis}."
                    )
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
                placement_fn=placement_fn or _default_placement_fn,
                mp_policy=mp_policy,
                gradient_reduce_op=dist.ReduceOp.SUM,
                reshard_after_forward=False,
            )
        )
    flex_shard(model, buckets=buckets)
    global_layouts = get_flex_shard_global_layouts(model)
    for module_fqn, module in model.named_modules():
        if not isinstance(module, Module) or module._param_init is None:
            continue
        param_init = module._param_init.copy()
        for name, param in module.named_parameters(recurse=False):
            param_init[name] = partial(
                _init_flex_shard_param,
                use_global_init=placement_fn is not None,
                layout=module._sharding_config.state_shardings[name],
                global_layout=global_layouts[f"{module_fqn}.{name}".lstrip(".")],
                init_fn=param_init[name],
                parallelism_context=parallelism_context,
            )
        module._param_init = param_init
    logger.info("Applied FlexShard with %s module buckets", len(buckets))
    return model


def _init_flex_shard_param(
    param, *, use_global_init, layout, global_layout, init_fn, parallelism_context
) -> None:
    outer_mesh = parallelism_context.get_optional_mesh(
        "ep" if "ep" in layout.local_type else "tp"
    )
    mesh = get_mesh(param)
    assert mesh is not None
    if use_global_init:
        if not all(
            isinstance(p, (Shard, BlockShard, BucketedBlockShard))
            for p in get_placements(param)
        ):
            raise ValueError("GraphTrainer supports Shard and block-sharded storage.")
        norm_groups = [
            mesh.get_group(axis)
            for axis_index, axis in enumerate(mesh.mesh_dim_names)
            if mesh.size(axis_index) > 1
        ]
        if outer_mesh is not None:
            for axis_index, axis in enumerate(outer_mesh.mesh_dim_names):
                sharded = isinstance(layout.local_type[axis], spmd.Shard) or (
                    layout.partition_spec is not None
                    and any(
                        axis in (entry if isinstance(entry, tuple) else (entry,))
                        for entry in layout.partition_spec
                    )
                )
                if sharded and outer_mesh.size(axis_index) > 1:
                    norm_groups.append(outer_mesh.get_group(axis))
        param._graph_trainer_norm_groups = tuple(norm_groups)
        if init_fn is not skip_param_init:
            init_from_full(param, global_layout, init_fn)
        return
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
        # Storage is EP/TP-local before FSDP shards it. Describe that order on
        # the combined [FSDP, EP/TP] mesh used for initialization.
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
    r"""Adapt ``GradientClipper`` to FlexShard's local gradient tensors.

    ``GradientClipper`` uses DTensor metadata to combine norms across shards.
    FlexShard gradients are plain local tensors.
    This class wraps those plain tensors into DTensors and calls ``GradientClipper``.
    """

    @torch.no_grad()
    def clip_grad_norm_(
        self,
        parameters: list[torch.nn.Parameter],
        max_norm: float | int,
        *,
        parallelism_context: ParallelismContext,
    ) -> torch.Tensor:
        parameters = [param for param in parameters if param.grad is not None]
        if any(hasattr(param, "_graph_trainer_norm_groups") for param in parameters):
            grads_by_groups = defaultdict(list)
            for param in parameters:
                grads_by_groups[param._graph_trainer_norm_groups].append(param.grad)
            squared_norms = []
            for groups, grads in grads_by_groups.items():
                norm = torch.nn.utils.get_total_norm(grads, foreach=True)
                squared_norm = norm.square()
                for group in groups:
                    dist.all_reduce(squared_norm, group=group)
                squared_norms.append(squared_norm)
            squared_norm = sum(squared_norms)
            pp_mesh = parallelism_context.get_optional_mesh("pp")
            if pp_mesh is not None:
                dist.all_reduce(squared_norm, group=pp_mesh.get_group())
            norm = squared_norm.sqrt()
            torch.nn.utils.clip_grads_with_norm_(
                parameters, max_norm, norm, foreach=True
            )
            return norm
        norm_parameters = []
        for param in parameters:
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


def _collect_bucket_regions(
    graph: fx.Graph,
) -> defaultdict[tuple[Literal["unshard", "reduce_grad"], bool], list[list[fx.Node]]]:
    """
    Collect bucket regions based on fx nodes custom meta, propagated from flex_shard.
    """
    buckets = defaultdict(list)
    region_key = None
    phase = None
    for node in graph.nodes:
        if node.op in ("placeholder", "output"):
            continue
        meta = node.meta.get("custom", {})
        if FLEX_SHARD_META_ACTION not in meta:
            base_meta = base_tensor_for_mutation_target(node).meta.get("custom", {})
            if (
                base_meta.get(FLEX_SHARD_META_ACTION) == "reduce_grad"
                and base_meta.get(FLEX_SHARD_META_PHASE) == "finish"
            ):
                node.meta["custom"] = meta = {**copy.deepcopy(base_meta), **meta}
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
        if key != region_key or (
            phase in ("execute", "finish") and next_phase == "prepare"
        ):
            region = []
            buckets[action, key[-1]].append(region)
        region.append(node)
        region_key, phase = key, next_phase
    return buckets


def mark_flex_shard_unshard_recompute_regions_pass(
    gm: fx.GraphModule, *, policy_fn: Callable[[fx.Node], CheckpointPolicy]
) -> fx.GraphModule:
    """Mark fresh-storage unshards so SAC replays their initialization together.

    INPUT:
        gather          # Fetch parameter shards.
        buffer          # Allocate fresh full-parameter storage.
        view(buffer)    # Create a parameter view before its storage is filled.
        copy_(buffer)   # Initialize storage; FX value edges can omit this write.
        forward(view)
        backward(view)

    OUTPUT after SAC:
        gather
        buffer
        view(buffer)
        copy_(buffer)
        forward(view)
        gather_bwd      # Replay the complete marked group.
        buffer_bwd      # Allocate separate backward storage.
        view(buffer_bwd)
        copy_(buffer_bwd)
        backward(view_bwd)
    """
    for module in list(gm.modules()):
        if not isinstance(module, fx.GraphModule):
            continue
        nodes = list(module.graph.nodes)
        positions = {node: index for index, node in enumerate(nodes)}
        storages = {
            node: {
                value.untyped_storage()
                for value in tree_leaves(node.meta.get("val"))
                if isinstance(value, torch.Tensor)
            }
            for node in nodes
        }
        writes = {
            node: set().union(
                *(storages[target] for target in mutation_target_nodes(node))
            )
            for node in nodes
        }
        regions = _collect_bucket_regions(module.graph).get(("unshard", False), ())
        for index, region in enumerate(regions):
            names = region[0].meta["custom"][FLEX_SHARD_META_PARAM_FQNS]
            owners = {tuple(name.split(".")[:2]) for name in names}
            if len(owners) != 1:
                continue
            owner = next(iter(owners))
            if len(owner) != 2 or owner[0] != "layers" or not owner[1].isdigit():
                continue
            policy_nodes = []
            for node in region:
                policy_node = copy.copy(node)
                policy_node.meta = dict(
                    node.meta,
                    custom=dict(node.meta["custom"], module_fqn=".".join(owner)),
                )
                policy_nodes.append(policy_node)
            if any(
                policy_fn(node) is CheckpointPolicy.MUST_SAVE for node in policy_nodes
            ):
                continue
            if any(
                policy_fn(node) is CheckpointPolicy.MUST_SAVE
                or _is_cacheable_effect(node.target)
                or (
                    getattr(node.target, "_schema", None) is None
                    and node.target is not operator.getitem
                )
                for node in region
            ):
                continue
            owned = set(region)
            inputs = {
                dependency
                for node in region
                for dependency in node.all_input_nodes
                if dependency not in owned
            }
            outputs = [
                node for node in region if any(user not in owned for user in node.users)
            ]
            input_storages = set().union(*(storages[node] for node in inputs))
            output_storages = set().union(*(storages[node] for node in outputs))
            if not outputs or output_storages & input_storages:
                continue
            if any(writes[node] & input_storages for node in region):
                continue
            if any(
                writes[node] & output_storages
                or (
                    positions[node] > positions[region[-1]]
                    and writes[node] & input_storages
                )
                for node in nodes
                if node not in owned
            ):
                continue
            if any(
                not storages[node]
                or any(
                    not isinstance(value, torch.Tensor)
                    for value in tree_leaves(node.meta.get("val"))
                )
                for node in outputs
            ):
                continue
            # Default SAC follows value edges, but mutations like copy_ does not have the edge.
            # We must preserve all mutations. For that we add recompute_region meta,
            # to identify the nodes that have to be remat together with activation node.
            #
            # Example:
            # gathered = all_gather(shard)        # Fetch this bucket's shards.
            # ready = wait_tensor(gathered)       # Complete communication.
            # buffer = empty_like(ready)          # Allocate fresh storage.
            # weight = view(buffer)               # Expose a parameter view.
            # copy_(buffer, ready)                # Fill storage; no value user.
            #
            # All nodes get recompute_region=index.
            for node in region:
                node.meta["recompute_region"] = index
                node.meta["custom"] = dict(
                    node.meta["custom"], module_fqn=".".join(owner)
                )
        module.graph.lint()
        module.recompile()
    return gm


def prepare_flex_shard_full_recompute(
    gm: fx.GraphModule,
    *,
    policy_fn: Callable[[fx.Node], CheckpointPolicy],
    parallelism: ParallelismConfig,
) -> set[fx.Node] | None:
    """Mark fresh unshards and return those reused across microbatches to save."""
    mark_flex_shard_unshard_recompute_regions_pass(gm, policy_fn=policy_fn)
    if get_fsdp_reshard_after_forward_policy(
        parallelism.fsdp_reshard_after_forward,
        pp_enabled=parallelism.pipeline_parallel_degree > 1,
    ):
        return None
    return {
        node
        for node in gm.graph.nodes
        if node.meta.get("custom", {}).get(FLEX_SHARD_META_ACTION) == "unshard"
    }


def deduplicate_flex_shard_unshard_pass(
    gm: fx.GraphModule, example_inputs: tuple | None = None
) -> fx.GraphModule:
    """FlexShard counterpart of ``deduplicate_fsdp_unshard_chains_pass``.

    Reuse equivalent annotated bucket programs while preserving storage mutations.
    """
    del example_inputs
    chunk_keys = (
        "chunk_id",
        "chunked_region_fqn",
        "chunked_region_is_backward",
        "chunked_region_producer",
        "chunked_region_role",
    )
    for module in gm.modules():
        if not isinstance(module, fx.GraphModule):
            continue
        graph = module.graph
        storages = {
            node: frozenset(
                value.untyped_storage()
                for value in tree_leaves(node.meta.get("val"))
                if isinstance(value, torch.Tensor)
            )
            for node in graph.nodes
        }
        writes = {
            node: set().union(
                *(storages[target] for target in mutation_target_nodes(node))
            )
            for node in graph.nodes
        }
        regions = _collect_bucket_regions(graph).get(("unshard", False), [])
        starts = {region[0]: region for region in regions}
        epochs = defaultdict(int)
        canonical = {}
        removable = set()
        for node in tuple(graph.nodes):
            if node in starts:
                region = starts[node]
                indices = {value: index for index, value in enumerate(region)}
                inputs = {
                    dependency
                    for value in region
                    for dependency in value.all_input_nodes
                    if dependency not in indices
                }
                input_storages = set().union(*(storages[value] for value in inputs))
                region_storages = set().union(*(storages[value] for value in region))
                fresh_storages = region_storages - input_storages
                safe = all(
                    (
                        getattr(value.target, "_schema", None) is not None
                        or value.target is operator.getitem
                    )
                    and not (writes[value] & input_storages)
                    for value in region
                ) and not any(
                    written & fresh_storages
                    for value, written in writes.items()
                    if value not in indices
                )

                def _key(value) -> Hashable:
                    if isinstance(value, fx.Node):
                        if value in indices:
                            return ("owned", indices[value])
                        return (
                            "input",
                            value,
                            tuple(
                                sorted(epochs[storage] for storage in storages[value])
                            ),
                        )
                    if isinstance(value, (tuple, list)):
                        return (type(value), tuple(_key(item) for item in value))
                    if isinstance(value, dict):
                        return tuple((name, _key(item)) for name, item in value.items())
                    return value

                if safe:
                    signature = tuple(
                        (
                            value.op,
                            value.target,
                            _key(value.args),
                            _key(value.kwargs),
                            tuple(
                                fsdp_param_fqns(
                                    value, meta_key=FLEX_SHARD_META_PARAM_FQNS
                                )
                            ),
                            tuple(
                                value.meta["custom"][FLEX_SHARD_META_MESH_AXIS_NAMES]
                            ),
                            tuple(
                                value.meta.get(name, value.meta["custom"].get(name))
                                for name in chunk_keys
                            ),
                        )
                        for value in region
                    )
                    previous = canonical.setdefault(signature, region)
                    if previous is not region:
                        for duplicate, original in zip(region, previous, strict=True):
                            for user in tuple(duplicate.users):
                                if user not in indices:
                                    user.replace_input_with(duplicate, original)
                        removable.update(region)
            for storage in writes[node]:
                epochs[storage] += 1
        if removable:
            graph.eliminate_dead_code(
                is_impure_node=lambda value: value not in removable
                and value.is_impure()
            )
            graph.lint()
            module.recompile()
            logger.info("Removed %s duplicate FlexShard unshard nodes", len(removable))
    return gm


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
                    if node.meta["custom"][FLEX_SHARD_META_PHASE] != "prepare"
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
        if not any(is_reduce_grad_collective(node) for node in region):
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
        unpack0         # Move unpacking and final gradient casts with its wait.
        wait1
        unpack1
    """

    collective_predicates = {}
    if schedule_all_gathers:
        collective_predicates["unshard"] = is_all_gather_into_tensor
    if schedule_reduce_scatters:
        collective_predicates["reduce_grad"] = is_reduce_scatter_tensor

    for module in gm.modules():
        if not isinstance(module, fx.GraphModule):
            continue
        aliases = {}
        for node in module.graph.nodes:
            base = base_tensor_for_mutation_target(node)
            aliases[node] = {base} | {
                value.untyped_storage()
                for value in tree_leaves(node.meta.get("val"))
                if isinstance(value, torch.Tensor)
            }
            if base is not node:
                aliases[node] |= aliases[base]
        reads = {
            node: set().union(*(aliases[value] for value in node.all_input_nodes))
            for node in module.graph.nodes
        }
        writes = {
            node: set().union(
                *(aliases[target] for target in mutation_target_nodes(node))
            )
            for node in module.graph.nodes
        }
        buckets = _collect_bucket_regions(module.graph)
        for (action, _), regions in buckets.items():
            if action not in collective_predicates:
                continue
            launch_wait_pairs = []
            for region in regions:
                launches = [
                    node for node in region if collective_predicates[action](node)
                ]
                if not launches:
                    continue  # Singleton meshes do not communicate.
                assert (
                    len(launches) == 1
                ), "Expected one collective per FlexShard region"
                launch = launches[0]
                waits = [
                    node
                    for node in region
                    if is_wait_tensor(node) and node.args[0] is launch
                ]
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
                moving_reads = set().union(*(reads[node] for node in moving))
                moving_writes = set().union(*(writes[node] for node in moving))
                first = min(positions[anchor], *(positions[node] for node in moving))
                last = max(positions[anchor], *(positions[node] for node in moving))
                # FX value edges miss dependencies through aliases. Keep
                # all-gather prefetch after input writes and reduce-scatter
                # wait/copy before output reads.
                if any(
                    (writes[node] & (moving_reads | moving_writes))
                    or (reads[node] & moving_writes)
                    for node in tuple(module.graph.nodes)[first : last + 1]
                    if node not in moving_set
                ):
                    continue
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
