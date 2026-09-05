# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from contextlib import nullcontext
from operator import attrgetter, getitem
from typing import Any

import torch
import torch.fx as fx
from torch.fx import Node
from torch.fx._lazy_graph_module import _LazyGraphModule
from torch.fx.traceback import annotate
from torch.utils._ordered_set import OrderedSet

SUBGRAPH_REGION = "graph_trainer_subgraph_region"
SUBGRAPH_REGION_ROLE = "graph_trainer_subgraph_region_role"
SUBGRAPH_REGION_PRESERVE_ORDER = "graph_trainer_subgraph_region_preserve_order"
_SUBGRAPH_REGION_CUSTOM_KEYS = (
    SUBGRAPH_REGION,
    SUBGRAPH_REGION_ROLE,
    SUBGRAPH_REGION_PRESERVE_ORDER,
)


def subgraph(
    name: str | None,
    role: str | None = None,
    *,
    preserve_order: bool = False,
):
    """Mark operations for outlining as explicit ``invoke_subgraph`` regions.

    Use this context inside code traced by ``minimal_fx_tracer``::

        with subgraph(f"loss_chunk_{chunk_idx}", role="loss_chunk"):
            loss = compute_chunk_loss(x)

    ``role`` gives regions with distinct ``name`` values a stable category for
    downstream passes. For example, every iteration of a chunked loss can have
    a unique name and share the ``"loss_chunk"`` role, allowing a later pass to
    select only those outlined subgraphs for memory-policy tagging or
    rematerialization.

    GraphTrainer copies the annotation from differentiable forward operations
    to their generated backward operations.
    ``apply_subgraph_region_annotations_pass`` outlines each contiguous
    annotated segment. Forward and backward segments become separate subgraphs
    with the same ``name`` and optional ``role``. Each outlined subgraph is an
    Inductor fusion and buffer-reuse boundary.
    """
    if name is None:
        return nullcontext()
    if not isinstance(name, str):
        raise AssertionError(
            f"expected subgraph region name to be str, got {type(name)}"
        )
    custom: dict[str, Any] = {SUBGRAPH_REGION: name}
    if role is not None:
        if not isinstance(role, str):
            raise AssertionError(
                f"expected subgraph region role to be str, got {type(role)}"
            )
        custom[SUBGRAPH_REGION_ROLE] = role
    if preserve_order:
        custom[SUBGRAPH_REGION_PRESERVE_ORDER] = True
    return annotate(custom)


def _getattr_or_none(module: torch.fx.GraphModule, target: str) -> Any:
    value: Any = module
    missing = object()
    for atom in target.split("."):
        value = getattr(value, atom, missing)
        if value is missing:
            return None
    return value


def _has_graph_module_arg(node: Node) -> bool:
    gm = node.graph.owning_module
    if gm is None:
        return False
    return any(
        inp.op == "get_attr"
        and isinstance(inp.target, str)
        and isinstance(_getattr_or_none(gm, inp.target), torch.fx.GraphModule)
        for inp in node.all_input_nodes
    )


def subgraph_region_key(node: Node) -> tuple[str, str, str, bool] | None:
    if node.op in ("placeholder", "output", "get_attr"):
        return None

    custom = node.meta.get("custom")
    if not isinstance(custom, dict):
        return None
    region = custom.get(SUBGRAPH_REGION)
    if region is None:
        return None
    if not isinstance(region, str):
        raise AssertionError(
            f"expected custom {SUBGRAPH_REGION} to be a str, got {type(region)}"
        )
    role = custom.get(SUBGRAPH_REGION_ROLE)
    if role is None:
        role = "bw" if node.meta.get("autograd_backward") is True else "fw"
    elif not isinstance(role, str):
        raise AssertionError(
            f"expected custom {SUBGRAPH_REGION_ROLE} to be a str, got {type(role)}"
        )
    preserve_order = custom.get(SUBGRAPH_REGION_PRESERVE_ORDER, False)
    if not isinstance(preserve_order, bool):
        raise TypeError(
            f"expected custom {SUBGRAPH_REGION_PRESERVE_ORDER} to be a bool, "
            f"got {type(preserve_order)}"
        )
    if (
        node.op == "call_function"
        and isinstance(node.target, torch._ops.HigherOrderOperator)
        and (
            node.target is torch.ops.higher_order.invoke_subgraph or not preserve_order
        )
    ):
        return None
    if _has_graph_module_arg(node) and not preserve_order:
        return None
    return f"{region}_{role}", region, role, preserve_order


def collect_subgraph_region_groups(
    graph: torch.fx.Graph,
) -> list[tuple[str, str, str, bool, list[Node]]]:
    groups: list[tuple[str, str, str, bool, list[Node]]] = []
    current_key: tuple[str, str, str, bool] | None = None
    current_nodes: list[Node] = []

    def flush() -> None:
        nonlocal current_key, current_nodes
        if current_key is not None and len(current_nodes) > 1:
            groups.append((*current_key, current_nodes))
        current_key = None
        current_nodes = []

    for node in list(graph.nodes):
        if node.op == "get_attr" and current_key is not None and current_key[-1]:
            continue
        key = subgraph_region_key(node)
        if key is None:
            flush()
            continue
        if key != current_key:
            flush()
            current_key = key
        current_nodes.append(node)
    flush()
    return groups


def _copy_placeholder_meta(
    placeholder: fx.Node, input_node: fx.Node, owning_module: fx.GraphModule
) -> None:
    if "val" in input_node.meta:
        placeholder.meta.update(input_node.meta)
    elif input_node.op == "get_attr" and isinstance(input_node.target, str):
        placeholder.meta["val"] = attrgetter(input_node.target)(owning_module)


def _strip_custom_keys_from_meta(meta: dict[str, Any], keys: tuple[str, ...]) -> None:
    if not keys:
        return
    custom = meta.get("custom")
    if not isinstance(custom, dict):
        return

    custom = custom.copy()
    for key in keys:
        custom.pop(key, None)
    compile_with_inductor = custom.get("compile_with_inductor")
    if isinstance(compile_with_inductor, dict):
        compile_with_inductor = compile_with_inductor.copy()
        for key in keys:
            compile_with_inductor.pop(key, None)
        custom["compile_with_inductor"] = compile_with_inductor

    if custom:
        meta["custom"] = custom
    else:
        meta.pop("custom", None)


def _strip_subgraph_arg_annotations(
    module: torch.fx.GraphModule,
    nodes: list[Node],
    keys: tuple[str, ...],
) -> None:
    for node in nodes:
        for input_node in node.all_input_nodes:
            if input_node.op != "get_attr" or not isinstance(input_node.target, str):
                continue
            submodule = _getattr_or_none(module, input_node.target)
            if not isinstance(submodule, torch.fx.GraphModule):
                continue
            for nested_module in submodule.modules():
                if not isinstance(nested_module, torch.fx.GraphModule):
                    continue
                for subnode in nested_module.graph.nodes:
                    _strip_custom_keys_from_meta(subnode.meta, keys)


def _target_key(target):
    if hasattr(target, "name") and hasattr(target, "overloadpacket"):
        return str(target)
    return repr(target)


def _meta_value_key(value):
    if isinstance(value, torch.Tensor):
        return (
            "tensor",
            str(value.dtype),
            tuple(_meta_value_key(dim) for dim in value.shape),
            tuple(_meta_value_key(stride) for stride in value.stride()),
            str(value.device),
            value.requires_grad,
        )
    if isinstance(value, (torch.SymInt, torch.SymFloat, torch.SymBool)):
        return type(value).__name__, str(value)
    if isinstance(value, (tuple, list)):
        return type(value).__name__, tuple(_meta_value_key(item) for item in value)
    if isinstance(value, dict):
        return "dict", tuple(
            sorted((repr(key), _meta_value_key(item)) for key, item in value.items())
        )
    if isinstance(value, (str, int, float, bool, type(None))):
        return value
    return repr(value)


def _arg_key(value, node_indices):
    if isinstance(value, fx.Node):
        return "node", node_indices[value]
    if isinstance(value, (tuple, list)):
        return type(value).__name__, tuple(
            _arg_key(item, node_indices) for item in value
        )
    if isinstance(value, dict):
        return "dict", tuple(
            sorted(
                (repr(key), _arg_key(item, node_indices)) for key, item in value.items()
            )
        )
    if isinstance(value, slice):
        return (
            "slice",
            _meta_value_key(value.start),
            _meta_value_key(value.stop),
            _meta_value_key(value.step),
        )
    return _meta_value_key(value)


def _custom_key(custom):
    if not isinstance(custom, dict):
        return None
    return tuple(
        sorted(
            (key, _meta_value_key(value))
            for key, value in custom.items()
            if key != SUBGRAPH_REGION
        )
    )


def _node_meta_key(node):
    return (
        _meta_value_key(node.meta.get("val")) if "val" in node.meta else None,
        (
            _meta_value_key(node.meta.get("recompute"))
            if "recompute" in node.meta
            else None
        ),
        node.meta.get("autograd_backward", False),
        _custom_key(node.meta.get("custom")),
    )


def _subgraph_structural_key(gm):
    node_indices = {node: idx for idx, node in enumerate(gm.graph.nodes)}
    return tuple(
        (
            node.op,
            _target_key(node.target),
            _arg_key(node.args, node_indices),
            _arg_key(node.kwargs, node_indices),
            _node_meta_key(node),
        )
        for node in gm.graph.nodes
    )


def _reuse_subgraph_module(module, region_node, canonical_target) -> None:
    attr_node = region_node.args[0]
    if not (
        isinstance(attr_node, Node)
        and attr_node.op == "get_attr"
        and isinstance(attr_node.target, str)
    ):
        return

    duplicate_target = attr_node.target
    if duplicate_target == canonical_target:
        return

    attr_node.target = canonical_target
    region_node.args = (attr_node, canonical_target, *region_node.args[2:])
    if hasattr(module, duplicate_target):
        delattr(module, duplicate_target)


def mark_invoke_subgraph(
    graph: fx.Graph,
    nodes: list[fx.Node],
    *,
    region_name_prefix: str,
) -> fx.Node:
    """Outline FX nodes into an invoke_subgraph HOP and return the HOP node."""
    owning_module = graph.owning_module
    if owning_module is None:
        raise AssertionError("expected graph to have an owning_module")
    if not nodes:
        raise AssertionError("expected non-empty nodes")

    node_set = OrderedSet(nodes)
    ordered_nodes = [node for node in graph.nodes if node in node_set]
    if len(ordered_nodes) != len(node_set):
        raise AssertionError("expected all nodes to belong to graph")
    if any(node.op in ("placeholder", "output") for node in ordered_nodes):
        raise AssertionError(
            "expected invoke_subgraph nodes to exclude graph boundaries"
        )

    region_outputs = [
        node
        for node in ordered_nodes
        if any(user not in node_set for user in node.users)
    ]

    subgraph = fx.Graph(owning_module)
    env: dict[fx.Node, fx.Node] = {}
    input_replacements: dict[fx.Node, Any] = {}
    boundary_args: list[tuple[fx.Node, tuple[int, ...], Any]] = []

    external_inputs: OrderedSet[fx.Node] = OrderedSet()
    preserved_getattrs: OrderedSet[fx.Node] = OrderedSet()

    def collect_external_input(node: fx.Node) -> fx.Node:
        if node not in node_set:
            if (
                node.op == "get_attr"
                and isinstance(node.target, str)
                and isinstance(attrgetter(node.target)(owning_module), fx.GraphModule)
            ):
                preserved_getattrs.add(node)
            else:
                external_inputs.add(node)
        return node

    for node in ordered_nodes:
        fx.map_arg((node.args, node.kwargs), collect_external_input)

    node_order = {node: idx for idx, node in enumerate(graph.nodes)}
    latest_input = max(
        external_inputs,
        key=lambda node: node_order[node],
        default=None,
    )
    first_external_user = min(
        (
            user
            for output_node in region_outputs
            for user in output_node.users
            if user not in node_set
        ),
        key=lambda node: node_order[node],
        default=None,
    )
    if (
        first_external_user is not None
        and latest_input is not None
        and node_order[latest_input] >= node_order[first_external_user]
    ):
        raise AssertionError("expected invoke_subgraph boundary to be acyclic")

    def add_boundary_arg(
        input_node: fx.Node, path: tuple[int, ...], meta_val: Any
    ) -> fx.Node:
        placeholder = subgraph.placeholder(f"arg_{len(boundary_args)}")
        _copy_placeholder_meta(placeholder, input_node, owning_module)
        if path:
            placeholder.meta["val"] = meta_val
        boundary_args.append((input_node, path, meta_val))
        return placeholder

    def make_input_replacement(
        input_node: fx.Node, value: Any, path: tuple[int, ...] = ()
    ) -> Any:
        if isinstance(value, (tuple, list)):
            return type(value)(
                make_input_replacement(input_node, item, (*path, idx))
                for idx, item in enumerate(value)
            )
        return add_boundary_arg(input_node, path, value)

    for input_node in external_inputs:
        value = input_node.meta.get("val")
        if isinstance(value, (tuple, list)):
            input_replacements[input_node] = make_input_replacement(input_node, value)
        else:
            input_replacements[input_node] = add_boundary_arg(input_node, (), value)

    def load_arg(node: fx.Node) -> Any:
        if node in env:
            return env[node]
        if node in node_set:
            raise AssertionError("expected invoke_subgraph nodes to be topological")
        if node in preserved_getattrs:
            if not isinstance(node.target, str):
                raise AssertionError("expected get_attr target to be a string")
            get_attr_node = subgraph.get_attr(node.target)
            get_attr_node.meta.update(node.meta)
            env[node] = get_attr_node
            return get_attr_node
        return input_replacements[node]

    for node in ordered_nodes:
        env[node] = subgraph.node_copy(node, load_arg)

    subgraph_outputs = tuple(env[node] for node in region_outputs)
    out = subgraph.output(subgraph_outputs)
    out.meta["val"] = tuple(node.meta.get("val") for node in region_outputs)
    subgraph.lint()

    subgraph_module = _LazyGraphModule(owning_module, subgraph)
    first_name = ordered_nodes[0].name
    last_name = ordered_nodes[-1].name
    region_name = f"{region_name_prefix}_{first_name}_{last_name}"
    subgraph_attr_name = f"{region_name}_0"
    setattr(owning_module, subgraph_attr_name, subgraph_module)

    if latest_input is None or node_order[latest_input] < node_order[ordered_nodes[0]]:
        with graph.inserting_before(ordered_nodes[0]):
            get_subgraph = graph.get_attr(subgraph_attr_name)
    else:
        with graph.inserting_after(latest_input):
            get_subgraph = graph.get_attr(subgraph_attr_name)

    outer_args: list[fx.Node] = []
    insert_after = get_subgraph

    def make_outer_arg(
        input_node: fx.Node, path: tuple[int, ...], meta_val: Any
    ) -> fx.Node:
        nonlocal insert_after
        source = input_node
        for idx in path:
            with graph.inserting_after(insert_after):
                source = graph.call_function(
                    getitem,
                    args=(source, idx),
                    name=f"{input_node.name}_{region_name_prefix}_arg_{len(outer_args)}",
                )
            insert_after = source
        if path:
            source.meta["val"] = meta_val
        return source

    for input_node, path, meta_val in boundary_args:
        outer_args.append(make_outer_arg(input_node, path, meta_val))

    with graph.inserting_after(insert_after):
        region_node = graph.call_function(
            torch.ops.higher_order.invoke_subgraph,
            args=(get_subgraph, subgraph_attr_name, *outer_args),
            name=region_name,
        )

    replacements: list[fx.Node] = []
    if len(region_outputs) == 0:
        region_node.meta["val"] = ()
    else:
        region_node.meta["val"] = tuple(node.meta.get("val") for node in region_outputs)
        insert_after = region_node
        for idx, output_node in enumerate(region_outputs):
            with graph.inserting_after(insert_after):
                replacement = graph.call_function(
                    getitem,
                    args=(region_node, idx),
                    name=f"{output_node.name}_{region_name_prefix}",
                )
            replacement.meta = output_node.meta.copy()
            replacement.meta.pop("eager_input_vals", None)
            _strip_custom_keys_from_meta(replacement.meta, _SUBGRAPH_REGION_CUSTOM_KEYS)
            replacements.append(replacement)
            insert_after = replacement

    for output_node, replacement in zip(region_outputs, replacements, strict=True):
        for user in list(output_node.users):
            if user not in node_set:
                user.replace_input_with(output_node, replacement)

    for node in reversed(ordered_nodes):
        graph.erase_node(node)
    graph.lint()

    return region_node


def _record_subgraph_region(
    module: torch.fx.GraphModule,
    region_node: Node,
    region: str,
    region_id: str,
    region_role: str,
    preserve_order: bool,
) -> None:
    region_node.meta[SUBGRAPH_REGION] = region
    custom = region_node.meta.setdefault("custom", {})
    custom["subgraph_region_id"] = region_id
    custom["subgraph_region_role"] = region_role
    nested_config = None
    if preserve_order:
        from torch._higher_order_ops.invoke_subgraph import NestedCompileRegionOptions

        nested_config = NestedCompileRegionOptions(
            inductor_config_patches={
                "reorder_for_locality": False,
                "reorder_for_peak_memory": False,
                "reorder_for_compute_comm_overlap": False,
                # Without a later scheduling pass, pre-fusion lifetime metadata
                # would be stale after fusion.
                "fusion_memory_timeline_peak_allowed_increase_mb": None,
                "aten_distributed_optimizations.enable_simple_overlap": False,
                "aten_distributed_optimizations.enable_overlap_scheduling": False,
            }
        )
        custom["nested_region_config"] = nested_config
    get_subgraph = region_node.args[0]
    if not (
        isinstance(get_subgraph, Node)
        and get_subgraph.op == "get_attr"
        and isinstance(get_subgraph.target, str)
    ):
        return
    submod = getattr(module, get_subgraph.target, None)
    if isinstance(submod, torch.fx.GraphModule):
        submod.meta[SUBGRAPH_REGION] = region
        submod_custom = submod.meta.setdefault("custom", {})
        if not isinstance(submod_custom, dict):
            submod_custom = {}
            submod.meta["custom"] = submod_custom
        submod_custom["subgraph_region_id"] = region_id
        submod_custom["subgraph_region_role"] = region_role
        if nested_config is not None:
            submod.meta["nested_region_config"] = nested_config


def apply_subgraph_region_annotations_pass(
    gm: torch.fx.GraphModule,
    example_inputs: tuple | None = None,
) -> torch.fx.GraphModule:
    del example_inputs

    outlined_regions = 0
    for module in list(gm.modules()):
        if not isinstance(module, torch.fx.GraphModule):
            continue
        outlined_subgraphs = {}
        groups = collect_subgraph_region_groups(module.graph)
        if not groups:
            continue
        for region, region_id, region_role, preserve_order, nodes in groups:
            if preserve_order:
                _strip_subgraph_arg_annotations(
                    module, nodes, _SUBGRAPH_REGION_CUSTOM_KEYS
                )
            region_node = mark_invoke_subgraph(
                module.graph,
                nodes,
                region_name_prefix=f"subgraph_region_{outlined_regions}",
            )
            _record_subgraph_region(
                module,
                region_node,
                region,
                region_id,
                region_role,
                preserve_order,
            )
            attr_node = region_node.args[0]
            if (
                isinstance(attr_node, Node)
                and attr_node.op == "get_attr"
                and isinstance(attr_node.target, str)
            ):
                submod = getattr(module, attr_node.target, None)
                if isinstance(submod, torch.fx.GraphModule):
                    subgraph_key = _subgraph_structural_key(submod)
                    canonical_target = outlined_subgraphs.setdefault(
                        subgraph_key, attr_node.target
                    )
                    _reuse_subgraph_module(module, region_node, canonical_target)
            outlined_regions += 1
        module.graph.lint()
        module.recompile()

    return gm
