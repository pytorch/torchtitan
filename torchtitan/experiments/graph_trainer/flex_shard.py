# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Apply FlexShard's bucket runtime before GraphTrainer tracing."""

import logging
import operator
from collections import defaultdict
from collections.abc import Callable
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
from torch._functorch.compile_utils import fx_graph_cse
from torch._guards import detect_fake_mode
from torch._higher_order_ops.passes.inline_invoke_subgraph import (
    inline_invoke_subgraph_nodes,
)
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor import DTensor, Shard as DTensorShard
from torch.distributed.tensor.placement_types import (
    _StridedShard,
    Placement as DTensorPlacement,
)
from torch.fx.passes.fake_tensor_prop import FakeTensorProp
from torch.utils._pytree import tree_leaves
from torch.utils.checkpoint import CheckpointPolicy

from torchtitan.components.optim import GradientClipper
from torchtitan.config import PrecisionConfig, TORCH_DTYPE_MAP
from torchtitan.distributed import ParallelismContext
from torchtitan.distributed.fsdp import linear_param_shard_placements
from torchtitan.distributed.spmd_types import spmd_axes
from torchtitan.experiments.graph_trainer.graph_pp.split_fsdp_collectives import (
    FSDPGraphRegions,
)
from torchtitan.experiments.graph_trainer.graph_pp.utils import graph_outputs
from torchtitan.experiments.graph_trainer.mutation_utils import (
    base_tensor_for_mutation_target,
    mutation_target_nodes,
)
from torchtitan.experiments.graph_trainer.subgraph_regions import mark_invoke_subgraph
from torchtitan.models.base.module import Module
from torchtitan.models.common.decoder import Decoder
from torchtitan.models.common.moe import MoE

from .common_utils import get_default_transformer_block_buckets


logger = logging.getLogger(__name__)

_CHUNK_OWNERSHIP_KEYS = (
    "chunk_id",
    "chunked_region_fqn",
    "chunked_region_is_backward",
    "chunked_region_producer",
    "chunked_region_role",
)

_UNSHARD_REMAT_REGION = "flex_shard_unshard_remat_region"


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
    def __init__(self, config):
        super().__init__(config)
        self._gradient_views = {}
        self._norm_layouts = {}

    @torch.no_grad()
    def clip_grad_norm_(self, parameters, max_norm, *, parallelism_context):
        parameters = list(parameters)
        params_by_mesh = defaultdict(list)
        for param in parameters:
            if param.grad is not None:
                norm_param = param._graph_trainer_dtensor_param
                mesh = norm_param.device_mesh
                key = (mesh, norm_param.placements)
                if key not in self._norm_layouts:
                    self._norm_layouts[key] = torch.linalg.vector_norm(
                        torch.empty_like(norm_param, device="meta"), 2
                    ).placements
                params_by_mesh[mesh].append((param, self._norm_layouts[key]))
        for items in params_by_mesh.values():
            first, layout = items[0]
            if any(
                candidate != layout
                or param.grad.dtype != first.grad.dtype
                or param.grad.device != first.grad.device
                for param, candidate in items
            ):
                return self._clip_grad_norm_with_views_(
                    parameters, max_norm, parallelism_context=parallelism_context
                )
        norms = []
        for mesh, items in params_by_mesh.items():
            local_norms = torch._foreach_norm([param.grad for param, _ in items], 2)
            norm_inputs = DTensor.from_local(
                torch.stack(local_norms), mesh, items[0][1], run_check=False
            )
            norms.append(torch.linalg.vector_norm(norm_inputs, 2).full_tensor())
        if len(norms) == 1:
            total_norm = norms[0]
        elif norms:
            total_norm = sum(norm**2 for norm in norms) ** 0.5
        else:
            total_norm = torch.tensor(0.0)
        if (pp_mesh := parallelism_context.get_optional_mesh("pp")) is not None:
            total_norm **= 2
            dist.all_reduce(total_norm, op=dist.ReduceOp.SUM, group=pp_mesh.get_group())
            total_norm **= 0.5
        for items in params_by_mesh.values():
            torch.nn.utils.clip_grads_with_norm_(
                [param for param, _ in items], max_norm, total_norm, foreach=True
            )
        return total_norm

    def _clip_grad_norm_with_views_(self, parameters, max_norm, *, parallelism_context):
        norm_parameters = []
        bound_views = []
        try:
            for param in parameters:
                if param.grad is None:
                    continue
                norm_param = param._graph_trainer_dtensor_param
                grad = param.grad
                key = (
                    id(norm_param),
                    grad.dtype,
                    grad.device,
                    norm_param.device_mesh,
                    norm_param.placements,
                    norm_param.shape,
                    norm_param.stride(),
                )
                view = self._gradient_views.get(key)
                if view is None:
                    view = DTensor.from_local(
                        torch.empty(0, dtype=grad.dtype, device=grad.device),
                        norm_param.device_mesh,
                        norm_param.placements,
                        shape=norm_param.shape,
                        stride=norm_param.stride(),
                        run_check=False,
                    )
                    self._gradient_views[key] = view
                view.to_local().set_(grad)
                bound_views.append((norm_param, view))
                norm_param.grad = view
                norm_parameters.append(norm_param)
            return super().clip_grad_norm_(
                norm_parameters, max_norm, parallelism_context=parallelism_context
            )
        finally:
            for norm_param, view in bound_views:
                view.to_local().set_()
                norm_param.grad = None


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


def _graph_tensor_storages_and_writes(graph: fx.Graph):
    storages = {}
    for node in graph.nodes:
        storages[node] = frozenset(
            value.untyped_storage()
            for value in tree_leaves(node.meta.get("val"))
            if isinstance(value, torch.Tensor)
        )
        base = base_tensor_for_mutation_target(node)
        if base is not node:
            storages[node] |= storages[base]
    writes = {
        node: set().union(*(storages[target] for target in mutation_target_nodes(node)))
        for node in graph.nodes
    }
    return storages, writes


def outline_flex_shard_unshards_pass(
    gm: fx.GraphModule,
    example_inputs: tuple | None = None,
    *,
    policy_fn: Callable[[fx.Node], CheckpointPolicy] | None = None,
) -> fx.GraphModule:
    del example_inputs
    count = 0
    for module in list(gm.modules()):
        if not isinstance(module, fx.GraphModule):
            continue
        nodes = list(module.graph.nodes)
        positions = {node: index for index, node in enumerate(nodes)}
        storages, writes = _graph_tensor_storages_and_writes(module.graph)
        regions = _collect_bucket_regions(module.graph).get(("unshard", False), ())
        for region in regions:
            names = region[0].meta["custom"][FLEX_SHARD_META_PARAM_FQNS]
            owners = {tuple(name.split(".")[:2]) for name in names}
            if len(owners) != 1:
                continue
            owner = next(iter(owners))
            if len(owner) != 2 or owner[0] != "layers" or not owner[1].isdigit():
                continue
            if policy_fn is not None and any(
                policy_fn(node) is CheckpointPolicy.MUST_SAVE for node in region
            ):
                continue
            node_set = set(region)
            inputs = {
                dependency
                for node in region
                for dependency in node.all_input_nodes
                if dependency not in node_set
            }
            outputs = [
                node
                for node in region
                if any(user not in node_set for user in node.users)
            ]
            input_storages = set().union(*(storages[node] for node in inputs))
            written_storages = set().union(*(writes[node] for node in region))
            if written_storages & input_storages:
                continue
            targets = {
                base_tensor_for_mutation_target(target)
                for node in region
                for target in mutation_target_nodes(node)
            }
            if any(
                target not in node_set
                or not getattr(
                    getattr(target.target, "_schema", None), "name", ""
                ).startswith("aten::empty")
                for target in targets
            ):
                continue
            if any(
                writes[node] & input_storages
                for node in nodes[positions[region[-1]] + 1 :]
                if node not in node_set
            ):
                continue
            if any(
                tag in getattr(node.target, "tags", ())
                for node in region
                for tag in (
                    torch.Tag.nondeterministic_seeded,
                    torch.Tag.nondeterministic_bitwise,
                    torch.Tag.data_dependent_output,
                    torch.Tag.dynamic_output_shape,
                )
            ) or any(
                node.meta.get("recompute") is CheckpointPolicy.MUST_SAVE
                for node in region
            ):
                continue
            if not outputs or any(
                not tree_leaves(node.meta.get("val"))
                or any(
                    not isinstance(value, torch.Tensor)
                    for value in tree_leaves(node.meta.get("val"))
                )
                or storages[node] & input_storages
                for node in outputs
            ):
                continue
            output_storages = set().union(*(storages[node] for node in outputs))
            if any(
                writes[node] & output_storages for node in nodes if node not in node_set
            ):
                continue
            call = mark_invoke_subgraph(
                module.graph,
                region,
                region_name_prefix=f"flex_shard_unshard_remat_{count}",
            )
            call.meta[_UNSHARD_REMAT_REGION] = True
            call.meta["autograd_backward"] = False
            call.meta["custom"] = {"module_fqn": ".".join(owner)}
            for getter in call.users:
                assert getter.target is operator.getitem
                getter.meta["custom"] = dict(
                    getter.meta.get("custom", {}), module_fqn=".".join(owner)
                )
            count += 1
        module.graph.lint()
        module.recompile()
    logger.info("Outlined %s FlexShard unshard regions for rematerialization", count)
    return gm


def inline_flex_shard_unshards_pass(
    gm: fx.GraphModule, example_inputs: tuple | None = None
) -> fx.GraphModule:
    del example_inputs
    count = 0
    for module in list(gm.modules()):
        if not isinstance(module, fx.GraphModule):
            continue
        calls = [
            node for node in module.graph.nodes if node.meta.get(_UNSHARD_REMAT_REGION)
        ]
        targets = {str(node.args[0].target) for node in calls}
        for call in calls:
            child = getattr(module, str(call.args[0].target))
            inputs = [node.meta["val"] for node in call.args[2:]]
            with torch.no_grad():
                FakeTensorProp(
                    child, mode=detect_fake_mode(inputs)
                ).propagate_dont_convert_inputs(*inputs)
            before = set(module.graph.nodes)
            backward = call.meta.get("autograd_backward", False)
            owner = call.meta["custom"]["module_fqn"]
            inline_invoke_subgraph_nodes(module, [call])
            for node in module.graph.nodes:
                if node not in before:
                    node.meta["autograd_backward"] = backward
                    node.meta["custom"] = dict(
                        node.meta.get("custom", {}), module_fqn=owner
                    )
                    node.meta.pop(_UNSHARD_REMAT_REGION, None)
            count += 1
        for target in targets:
            if not any(
                node.op == "get_attr" and node.target == target
                for node in module.graph.nodes
            ):
                delattr(module, target)
        module.graph.lint()
        module.recompile()
    logger.info("Inlined %s FlexShard unshard regions before scheduling", count)
    return gm


def fold_flex_shard_gradient_casts_pass(
    gm: fx.GraphModule, example_inputs: tuple | None = None
) -> fx.GraphModule:
    del example_inputs
    for module in gm.modules():
        if not isinstance(module, fx.GraphModule):
            continue
        folded = 0
        positions = {node: index for index, node in enumerate(module.graph.nodes)}
        _, written_storages = _graph_tensor_storages_and_writes(module.graph)
        writes = [
            (
                positions[node],
                written_storages[node],
                frozenset(
                    base_tensor_for_mutation_target(target)
                    for target in mutation_target_nodes(node)
                ),
            )
            for node in module.graph.nodes
        ]
        for node in module.graph.nodes:
            if (
                node.target is not torch.ops.fsdp.chunk_cat_mixed_dtype.default
                or node.meta.get("custom", {}).get(FLEX_SHARD_META_ACTION)
                != "reduce_grad"
                or node.meta["custom"].get(FLEX_SHARD_META_PHASE) != "prepare"
            ):
                continue
            output = node.kwargs["out"].meta["val"]
            if output.dtype != torch.float32:
                continue
            tensors = list(node.args[0])
            for index, cast in enumerate(tensors):
                if (
                    cast.target is not torch.ops.aten._to_copy.default
                    or len(cast.users) != 1
                ):
                    continue
                source = cast.args[0]
                before, after = source.meta["val"], cast.meta["val"]
                base = base_tensor_for_mutation_target(source)
                schema = getattr(base.target, "_schema", None)
                if (
                    before.dtype not in (torch.bfloat16, torch.float16)
                    or after.dtype != output.dtype
                    or before.device != after.device
                    or before.layout != after.layout
                    or before.stride() != after.stride()
                    or base.target is operator.getitem
                    or (
                        schema is not None
                        and any(
                            result.alias_info is not None for result in schema.returns
                        )
                    )
                    or any(
                        positions[cast] < position <= positions[node]
                        and (before.untyped_storage() in storages or base in bases)
                        for position, storages, bases in writes
                    )
                ):
                    continue
                tensors[index] = source
                folded += 1
            node.args = (tensors, *node.args[1:])
        if folded:
            module.graph.eliminate_dead_code()
            module.graph.lint()
            module.recompile()
            logger.info("Folded %s gradient upcasts into FlexShard packing", folded)
    return gm


def deduplicate_flex_shard_unshard_pass(
    gm: fx.GraphModule, example_inputs: tuple | None = None
) -> fx.GraphModule:
    del example_inputs
    for module in gm.modules():
        if not isinstance(module, fx.GraphModule):
            continue
        graph = module.graph
        storages, writes = _graph_tensor_storages_and_writes(graph)
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

                def key(value):
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
                        return (type(value), tuple(key(item) for item in value))
                    if isinstance(value, dict):
                        return tuple((name, key(item)) for name, item in value.items())
                    return value

                if safe:
                    signature = tuple(
                        (
                            value.op,
                            value.target,
                            key(value.args),
                            key(value.kwargs),
                            tuple(value.meta["custom"][FLEX_SHARD_META_PARAM_FQNS]),
                            tuple(
                                value.meta["custom"][FLEX_SHARD_META_MESH_AXIS_NAMES]
                            ),
                            tuple(
                                value.meta.get(name, value.meta["custom"].get(name))
                                for name in _CHUNK_OWNERSHIP_KEYS
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


def deduplicate_flex_shard_parameter_transforms_pass(
    gm: fx.GraphModule, example_inputs: tuple | None = None
) -> fx.GraphModule:
    del example_inputs
    for module in gm.modules():
        if not isinstance(module, fx.GraphModule):
            continue
        nodes = list(module.graph.nodes)
        positions = {node: index for index, node in enumerate(nodes)}
        roots = get_flex_shard_graph_regions(
            module, include_reduce_grad_inputs=False
        ).unshard_output_names
        roots = {node for node in nodes if node.name in roots}
        storages, writes = _graph_tensor_storages_and_writes(module.graph)

        def owners(node):
            owned = set()
            pending = [node]
            while pending:
                value = pending.pop()
                if value in owned:
                    continue
                owned.add(value)
                base = base_tensor_for_mutation_target(value)
                if base is not value:
                    pending.append(base)
                if value.target is operator.getitem and isinstance(
                    value.args[0], fx.Node
                ):
                    pending.append(value.args[0])
            return owned

        output_owners = set().union(
            *(owners(node) for node in nodes[-1].all_input_nodes)
        )
        output_storages = set().union(
            *(storages[node] for node in nodes[-1].all_input_nodes)
        )
        written_owners = defaultdict(lambda: -1)
        last_writer = defaultdict(lambda: -1)
        for node in nodes:
            for storage in writes[node]:
                last_writer[storage] = positions[node]
            for target in mutation_target_nodes(node):
                for owner in owners(target):
                    written_owners[owner] = positions[node]
        graph = fx.Graph()
        env = {}
        inputs = {}
        epochs = defaultdict(int)
        originals = []
        for node in nodes:
            schema = getattr(node.target, "_schema", None)
            eligible = (
                node.op == "call_function"
                and not node.meta.get("autograd_backward", False)
                and not node.meta.get("custom", {}).get(FLEX_SHARD_META_ACTION)
                and bool(node.all_input_nodes)
                and all(
                    dependency in roots or dependency in env
                    for dependency in node.all_input_nodes
                )
                and (
                    (
                        node.target is operator.getitem
                        and isinstance(node.args[0], fx.Node)
                        and isinstance(node.args[0].meta.get("val"), (tuple, list))
                    )
                    or (schema is not None and not schema.is_mutable)
                )
                and bool(tree_leaves(node.meta.get("val")))
                and all(
                    isinstance(value, torch.Tensor)
                    for value in tree_leaves(node.meta.get("val"))
                )
                and not node.is_impure()
                and not (
                    schema is not None
                    and any(result.alias_info is not None for result in schema.returns)
                    and base_tensor_for_mutation_target(node) is node
                )
                and not any(
                    tag in getattr(node.target, "tags", ())
                    for tag in (
                        torch.Tag.nondeterministic_seeded,
                        torch.Tag.nondeterministic_bitwise,
                        torch.Tag.data_dependent_output,
                        torch.Tag.dynamic_output_shape,
                    )
                )
                and (
                    schema is None
                    or not schema.name.startswith(("_c10d", "c10d::", "aten::empty"))
                )
                and all(
                    last_writer[storage] <= positions[node]
                    for storage in storages[node]
                )
                and written_owners[node] <= positions[node]
                and node not in output_owners
                and not (storages[node] & output_storages)
            )
            if eligible:

                def argument(dependency):
                    if dependency in roots:
                        key = (
                            dependency,
                            tuple(
                                sorted(
                                    epochs[storage] for storage in storages[dependency]
                                )
                            ),
                        )
                        if key not in inputs:
                            inputs[key] = graph.placeholder(f"parameter_{len(inputs)}")
                        return inputs[key]
                    return env[dependency]

                copy = graph.node_copy(node, argument)
                copy.meta = {
                    "custom": dict(node.meta.get("custom", {})),
                    "recompute": node.meta.get("recompute"),
                    "source": node,
                    "chunk_ownership": tuple(
                        node.meta.get(field, node.meta.get("custom", {}).get(field))
                        for field in _CHUNK_OWNERSHIP_KEYS
                    ),
                }
                env[node] = copy
                originals.append(node)
            for storage in writes[node]:
                epochs[storage] += 1
        if not originals:
            continue
        graph.output([env[node] for node in originals])
        graph = fx_graph_cse(
            graph,
            extra_node_key=lambda node: (
                repr(node.meta.get("custom", {})),
                node.meta.get("recompute"),
                node.meta.get("chunk_ownership"),
            ),
        )
        outputs = list(graph.nodes)[-1].args[0]
        removed = 0
        for original, replacement in zip(originals, outputs, strict=True):
            canonical = replacement.meta["source"]
            if original is not canonical:
                original.replace_all_uses_with(canonical)
                removed += 1
        if removed:
            module.graph.eliminate_dead_code()
            module.graph.lint()
            module.recompile()
            logger.info("Removed %s duplicate FlexShard parameter transforms", removed)
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
        return any(node.target in (op, op.default) for op in collective)

    collectives = {}
    if schedule_all_gathers:
        collectives["unshard"] = (
            torch.ops._c10d_functional.all_gather_into_tensor,
            torch.ops._c10d_functional.all_gather_into_tensor_out,
        )
    if schedule_reduce_scatters:
        collectives["reduce_grad"] = (torch.ops._c10d_functional.reduce_scatter_tensor,)
    wait_tensor = torch.ops._c10d_functional.wait_tensor

    def _is_wait_for(node: fx.Node, launch: fx.Node) -> bool:
        return (
            node.target in (wait_tensor, wait_tensor.default) and node.args[0] is launch
        )

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
                moving_reads = set().union(*(reads[node] for node in moving))
                moving_writes = set().union(*(writes[node] for node in moving))
                first = min(positions[anchor], *(positions[node] for node in moving))
                last = max(positions[anchor], *(positions[node] for node in moving))
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
