# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable
from contextlib import nullcontext
from dataclasses import dataclass
from operator import attrgetter, getitem
from typing import Any

import torch
import torch.fx as fx
from torch._functorch.partitioners import get_default_op_list
from torch.fx import Node
from torch.fx._lazy_graph_module import _LazyGraphModule
from torch.fx.traceback import annotate
from torch.utils._ordered_set import OrderedSet
from torch.utils._pytree import tree_flatten
from torch.utils.checkpoint import CheckpointPolicy

from torchtitan.experiments.graph_trainer.common_utils import (
    is_fsdp_unshard_all_gather,
)
from torchtitan.experiments.graph_trainer.min_cut_rematerialization import (
    min_cut_rematerialization_pass,
)
from torchtitan.tools.logging import logger


SUBGRAPH_REGION = "graph_trainer_subgraph_region"
SUBGRAPH_REGION_ROLE = "graph_trainer_subgraph_region_role"
_FSDP_UNSHARD_CHAIN_TARGETS = frozenset(
    {
        torch.ops._c10d_functional.all_gather_into_tensor.default,
        getitem,
        torch.ops._c10d_functional.wait_tensor.default,
        torch.ops.aten.constant_pad_nd.default,
        torch.ops.aten._to_copy.default,
        torch.ops.prims.convert_element_type.default,
    }
)


@dataclass(frozen=True)
class _OutlinedInvokeSubgraph:
    region_node: fx.Node
    replacements: tuple[fx.Node, ...]


def subgraph(
    name: str | None,
    role: str | None = None,
):
    if name is None:
        return nullcontext()
    if not isinstance(name, str):
        raise AssertionError(
            f"expected subgraph region name to be str, got {type(name)}"
        )
    custom = {SUBGRAPH_REGION: name}
    if role is not None:
        if not isinstance(role, str):
            raise AssertionError(
                f"expected subgraph region role to be str, got {type(role)}"
            )
        custom[SUBGRAPH_REGION_ROLE] = role
    return annotate(custom)


def _getattr_or_none(module: torch.fx.GraphModule, target: str) -> Any:
    try:
        return attrgetter(target)(module)
    except AttributeError:
        return None


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


def subgraph_region_key(node: Node) -> tuple[str, str, str] | None:
    if node.op in ("placeholder", "output", "get_attr"):
        return None
    if node.op == "call_function" and isinstance(
        node.target, torch._ops.HigherOrderOperator
    ):
        return None
    if _has_graph_module_arg(node):
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
    return f"{region}_{role}", region, role


def collect_subgraph_region_groups(
    graph: torch.fx.Graph,
) -> list[tuple[str, str, str, list[Node]]]:
    groups: list[tuple[str, str, str, list[Node]]] = []
    current_key: tuple[str, str, str] | None = None
    current_nodes: list[Node] = []

    def flush() -> None:
        nonlocal current_key, current_nodes
        if current_key is not None and len(current_nodes) > 1:
            groups.append((*current_key, current_nodes))
        current_key = None
        current_nodes = []

    for node in list(graph.nodes):
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


def _get_subgraph_name(gm: fx.GraphModule, name: str) -> str:
    i = 0
    while hasattr(gm, f"{name}_{i}"):
        i += 1
    return f"{name}_{i}"


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


def _target_key(target):
    if hasattr(target, "name") and hasattr(target, "overloadpacket"):
        return str(target)
    return repr(target)


def _meta_value_key(value):
    if isinstance(value, torch.Tensor):
        return (
            "tensor",
            str(value.dtype),
            tuple(value.shape),
            tuple(value.stride()),
            str(value.device),
            value.requires_grad,
        )
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
        return type(value).__name__, tuple(_arg_key(item, node_indices) for item in value)
    if isinstance(value, dict):
        return "dict", tuple(
            sorted((repr(key), _arg_key(item, node_indices)) for key, item in value.items())
        )
    if isinstance(value, slice):
        return "slice", value.start, value.stop, value.step
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
        _meta_value_key(node.meta.get("recompute")) if "recompute" in node.meta else None,
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


def _reuse_subgraph_module(module, region_node, canonical_target):
    attr_node = region_node.args[0]
    if not (
        isinstance(attr_node, Node)
        and attr_node.op == "get_attr"
        and isinstance(attr_node.target, str)
    ):
        return False

    duplicate_target = attr_node.target
    if duplicate_target == canonical_target:
        return False

    attr_node.target = canonical_target
    region_node.args = (attr_node, canonical_target, *region_node.args[2:])
    if hasattr(module, duplicate_target):
        delattr(module, duplicate_target)
    return True


def _outline_invoke_subgraph(
    graph: fx.Graph,
    nodes: list[fx.Node],
    *,
    region_name_prefix: str,
    output_name_suffix: str,
    always_return_tuple: bool = False,
    strip_custom_keys: tuple[str, ...] = (),
    sort_region_outputs: Callable[[fx.Node], Any] | None = None,
) -> _OutlinedInvokeSubgraph:
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
    if sort_region_outputs is not None:
        region_outputs.sort(key=sort_region_outputs)

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
    if len(subgraph_outputs) == 0:
        out = subgraph.output(())
        out.meta["val"] = ()
    elif len(subgraph_outputs) == 1 and not always_return_tuple:
        out = subgraph.output(subgraph_outputs[0])
        if "val" in region_outputs[0].meta:
            out.meta["val"] = region_outputs[0].meta["val"]
    else:
        out = subgraph.output(subgraph_outputs)
        out.meta["val"] = tuple(node.meta.get("val") for node in region_outputs)
    subgraph.lint()

    subgraph_module = _LazyGraphModule(owning_module, subgraph)
    first_name = ordered_nodes[0].name
    last_name = ordered_nodes[-1].name
    region_name = f"{region_name_prefix}_{first_name}_{last_name}"
    subgraph_attr_name = _get_subgraph_name(owning_module, region_name)
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
                    name=(
                        f"{input_node.name}_{output_name_suffix}_arg_{len(outer_args)}"
                    ),
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
    elif len(region_outputs) == 1 and not always_return_tuple:
        replacement = region_node
        replacement.meta = region_outputs[0].meta.copy()
        replacement.meta.pop("eager_input_vals", None)
        _strip_custom_keys_from_meta(replacement.meta, strip_custom_keys)
        replacements.append(replacement)
    else:
        region_node.meta["val"] = tuple(node.meta.get("val") for node in region_outputs)
        insert_after = region_node
        for idx, output_node in enumerate(region_outputs):
            with graph.inserting_after(insert_after):
                replacement = graph.call_function(
                    getitem,
                    args=(region_node, idx),
                    name=f"{output_node.name}_{output_name_suffix}",
                )
            replacement.meta = output_node.meta.copy()
            replacement.meta.pop("eager_input_vals", None)
            _strip_custom_keys_from_meta(replacement.meta, strip_custom_keys)
            replacements.append(replacement)
            insert_after = replacement

    for output_node, replacement in zip(region_outputs, replacements, strict=True):
        for user in list(output_node.users):
            if user not in node_set:
                user.replace_input_with(output_node, replacement)

    for node in reversed(ordered_nodes):
        graph.erase_node(node)
    graph.lint()

    return _OutlinedInvokeSubgraph(region_node, tuple(replacements))


def mark_invoke_subgraph(
    graph: fx.Graph,
    nodes: list[fx.Node],
    *,
    region_name_prefix: str = "invoke_subgraph_region",
    strip_custom_keys: tuple[str, ...] = (),
    sort_region_outputs: Callable[[fx.Node], Any] | None = None,
) -> fx.Node:
    """Outline FX nodes into an invoke_subgraph HOP and return the HOP node."""
    return _outline_invoke_subgraph(
        graph,
        nodes,
        region_name_prefix=region_name_prefix,
        output_name_suffix=region_name_prefix,
        always_return_tuple=True,
        strip_custom_keys=strip_custom_keys,
        sort_region_outputs=sort_region_outputs,
    ).region_node


def _record_subgraph_region(
    module: torch.fx.GraphModule,
    region_node: Node,
    region: str,
    region_id: str,
    region_role: str,
    nested_config,
) -> None:
    region_node.meta[SUBGRAPH_REGION] = region
    custom = region_node.meta.setdefault("custom", {})
    custom["subgraph_region_id"] = region_id
    custom["subgraph_region_role"] = region_role
    if nested_config is not None:
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
    *,
    min_cut_rematerialization: bool = False,
) -> torch.fx.GraphModule:
    from torch._inductor.decomposition import select_decomp_table

    decomposition_table = select_decomp_table()
    nested_config = None

    outlined_regions = 0
    reused_regions = 0
    outlined_subgraphs = {}
    for module in list(gm.modules()):
        if not isinstance(module, torch.fx.GraphModule):
            continue
        groups = collect_subgraph_region_groups(module.graph)
        if not groups:
            continue
        for region, region_id, region_role, nodes in groups:
            region_node = mark_invoke_subgraph(
                module.graph,
                nodes,
                region_name_prefix=f"subgraph_region_{outlined_regions}",
                strip_custom_keys=(SUBGRAPH_REGION, SUBGRAPH_REGION_ROLE),
            )
            _record_subgraph_region(
                module, region_node, region, region_id, region_role, nested_config
            )
            attr_node = region_node.args[0]
            if (
                isinstance(attr_node, Node)
                and attr_node.op == "get_attr"
                and isinstance(attr_node.target, str)
            ):
                submod = getattr(module, attr_node.target, None)
                if isinstance(submod, torch.fx.GraphModule):
                    if min_cut_rematerialization:
                        min_cut_rematerialization_pass(
                            submod,
                            example_inputs=None,
                            decomposition_table=decomposition_table,
                        )
                    subgraph_key = _subgraph_structural_key(submod)
                    canonical_target = outlined_subgraphs.setdefault(
                        subgraph_key, attr_node.target
                    )
                    if _reuse_subgraph_module(module, region_node, canonical_target):
                        reused_regions += 1
            outlined_regions += 1
        module.graph.lint()
        module.recompile()

    if outlined_regions:
        logger.info(
            "Outlined %d annotated subgraph regions, reused %d structurally "
            "identical subgraphs",
            outlined_regions,
            reused_regions,
        )
    return gm


def _custom(node):
    custom = node.meta.get("custom", {})
    return custom if isinstance(custom, dict) else {}


def _region(node):
    return _custom(node).get(SUBGRAPH_REGION)


def _role(node):
    return _custom(node).get(SUBGRAPH_REGION_ROLE)


def _drop_region_meta(node):
    custom = dict(_custom(node))
    custom.pop(SUBGRAPH_REGION, None)
    custom.pop(SUBGRAPH_REGION_ROLE, None)
    if custom:
        node.meta["custom"] = custom
    else:
        node.meta.pop("custom", None)
    node.meta["recompute"] = CheckpointPolicy.MUST_SAVE


def _is_cse_target(node, repeated_roles, op_types, chain_targets):
    if (
        node.op != "call_function"
        or node.meta.get("autograd_backward") is True
        or _role(node) not in repeated_roles
    ):
        return False
    return node.target in chain_targets or op_types.is_view(node)


def _repeated_roles(graph):
    regions_by_role = defaultdict(set)
    for node in graph.nodes:
        if _role(node) is not None and _region(node) is not None:
            regions_by_role[_role(node)].add(_region(node))
    return {role for role, regions in regions_by_role.items() if len(regions) > 1}


def _candidate_nodes(
    graph,
    *,
    chain_targets=_FSDP_UNSHARD_CHAIN_TARGETS,
    is_unshard_anchor=is_fsdp_unshard_all_gather,
):
    repeated_roles = _repeated_roles(graph)
    if not repeated_roles:
        return set()

    op_types = get_default_op_list()
    base = {
        node
        for node in graph.nodes
        if _is_cse_target(node, repeated_roles, op_types, chain_targets)
    }

    selected = set()

    def local_inputs(node):
        return [
            inp
            for inp in node.all_input_nodes
            if _region(inp) == _region(node) and _role(inp) == _role(node)
        ]

    def collect_ancestors(node, result):
        if node in result:
            return True
        if node not in base:
            return False
        for inp in local_inputs(node):
            if not collect_ancestors(inp, result):
                return False
        result.add(node)
        return True

    for node in graph.nodes:
        if not is_unshard_anchor(node):
            continue
        ancestors = set()
        if collect_ancestors(node, ancestors):
            selected.update(ancestors)

    changed = True
    while changed:
        changed = False
        for node in graph.nodes:
            if node not in base or node in selected:
                continue
            inputs = local_inputs(node)
            if (
                inputs
                and any(inp in selected for inp in inputs)
                and all(inp in selected for inp in inputs)
            ):
                selected.add(node)
                changed = True

    return selected


def _shared_candidates(graph, candidates):
    shared = set()
    for node in candidates:
        user_regions = {
            _region(user)
            for user in node.users
            if _region(user) is not None and _role(user) == _role(node)
        }
        if len(user_regions) > 1:
            shared.add(node)

    stack = list(shared)
    while stack:
        node = stack.pop()
        for inp in node.all_input_nodes:
            if inp in candidates and _role(inp) == _role(node) and inp not in shared:
                shared.add(inp)
                stack.append(inp)

    return shared


def _hoist_shared_candidates(graph, shared):
    by_role = defaultdict(list)
    for node in graph.nodes:
        if node in shared:
            by_role[_role(node)].append(node)

    for role, nodes in by_role.items():
        anchor = next(
            (
                node
                for node in graph.nodes
                if _role(node) == role and node not in shared
            ),
            None,
        )
        if anchor is not None:
            for node in nodes:
                anchor.prepend(node)
        for node in nodes:
            _drop_region_meta(node)


def _candidate_target_count(graph, chain_targets):
    return sum(
        node.op == "call_function" and node.target in chain_targets
        for node in graph.nodes
    )


def _cse_token(node, env):
    def substitute(values):
        flat, spec = tree_flatten(values)
        result = []
        for value in flat:
            if isinstance(value, torch.fx.Node):
                value = env.get(value, value)
            elif isinstance(value, (torch.SymBool, torch.SymInt, torch.SymFloat)):
                value = value.node
            result.append(value)
        return tuple(result), spec

    args, args_spec = substitute(node.args)
    kwargs, kwargs_spec = substitute(node.kwargs)
    return {
        "target": node.target,
        "role": _role(node),
        "args": args,
        "args_spec": args_spec,
        "kwargs": kwargs,
        "kwargs_spec": kwargs_spec,
    }


def _cse_candidate_nodes(graph, candidates):
    env = {}
    seen = {}
    to_erase = []
    for node in graph.nodes:
        if node not in candidates:
            continue
        token = _cse_token(node, env)
        hash_val = (
            node.target,
            _role(node),
            hash(
                (
                    tuple((arg, type(arg)) for arg in token["args"]),
                    tuple((arg, type(arg)) for arg in token["kwargs"]),
                )
            ),
        )
        prev = seen.get(hash_val)
        if prev is not None and prev[1] == token:
            node.replace_all_uses_with(prev[0])
            env[node] = prev[0]
            to_erase.append(node)
        else:
            env[node] = node
            seen[hash_val] = (node, token)

    for node in reversed(to_erase):
        graph.erase_node(node)


def _extract_module_common_fsdp_unshards(
    module,
    *,
    chain_targets=_FSDP_UNSHARD_CHAIN_TARGETS,
    is_unshard_anchor=is_fsdp_unshard_all_gather,
):
    candidates = _candidate_nodes(
        module.graph,
        chain_targets=chain_targets,
        is_unshard_anchor=is_unshard_anchor,
    )
    if not candidates:
        return 0, 0

    old_count = _candidate_target_count(module.graph, chain_targets)
    _cse_candidate_nodes(module.graph, candidates)
    candidates = _candidate_nodes(
        module.graph,
        chain_targets=chain_targets,
        is_unshard_anchor=is_unshard_anchor,
    )
    shared = _shared_candidates(module.graph, candidates)
    _hoist_shared_candidates(module.graph, shared)
    module.graph.lint()
    module.recompile()
    return len(shared), old_count - _candidate_target_count(
        module.graph, chain_targets
    )


def extract_common_fsdp_unshards_pass(
    gm,
    example_inputs=None,
    *,
    chain_targets=_FSDP_UNSHARD_CHAIN_TARGETS,
    is_unshard_anchor=is_fsdp_unshard_all_gather,
):
    """Share identical forward FSDP unshard chains across same-role regions."""
    total_shared = 0
    total_erased = 0

    for module in list(gm.modules()):
        if not isinstance(module, torch.fx.GraphModule):
            continue
        shared, erased = _extract_module_common_fsdp_unshards(
            module,
            chain_targets=chain_targets,
            is_unshard_anchor=is_unshard_anchor,
        )
        total_shared += shared
        total_erased += erased

    if total_shared:
        logger.info(
            "extract_common_fsdp_unshards_pass: extracted %d shared nodes; "
            "CSE erased %d duplicate nodes",
            total_shared,
            total_erased,
        )
    return gm
