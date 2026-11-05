# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Find fusion regions by topology and lift them into ``call_module`` nodes.

curate_regions.py generates one pass per region (``fusion_regions/g003_rope/lift_pass.py``).
It holds plain Python matchers: starting from an anchor node, a matcher walks the
region's producers (``args``) and, where needed, users, and checks every op, its
literal arguments, and that paths which meet in the region meet at the same node. It
returns the instance's nodes and its external inputs. Nothing refers to node names, so
a matcher finds the region wherever the same subgraph appears.

``lift_matches(gm, region_id, matchers)`` runs the matchers over the graph, drops
matches that are not safe to lift (see ``lift_hazard``), and outlines each remaining
instance: its nodes move into a GraphModule submodule called by one ``call_module``.
Instances with the same structure share one submodule named ``fusion_<id>`` (e.g.
``fusion_G003``); structural variants get ``_v<k>`` suffixes. The body is the region's
original aten ops, so numerics are unchanged; assigning another ``nn.Module`` with the
same signature to that attribute swaps in a kernel for every instance.

The same matchers and the same ``lift_hazard`` run on ``DumpNode`` graphs loaded from
the capture's graph dump (``load_dump_graph``), so curate_regions.py can check offline
that they find every instance the finder found and which matches are safe to lift.
"""

from __future__ import annotations

import gzip
import json
import logging
import operator
from dataclasses import dataclass, field
from typing import Any

import torch
import torch.fx as fx
from torch.utils import _pytree as pytree

logger = logging.getLogger(__name__)


class _NodeSlot:
    def __repr__(self) -> str:
        return "N"


# Marks a node-valued slot in an is_op argument template.
N = _NodeSlot()


@dataclass(eq=False)
class DumpNode:
    """A node of a dumped graph (graph_dump/*_nodes.jsonl.gz), for running matchers offline.

    It has the ``fx.Node`` attributes the matchers and ``lift_hazard`` use; ``target`` is
    the op name as dumped and nodes compare by graph position.
    """

    name: str
    op: str
    target: str
    idx: int
    args: Any = field(default=(), repr=False)
    kwargs: dict[str, Any] = field(default_factory=dict, repr=False)
    users: list[DumpNode] = field(default_factory=list, repr=False)
    next: DumpNode | None = field(default=None, repr=False)
    schema: str | None = field(default=None, repr=False)

    @property
    def all_input_nodes(self) -> list[DumpNode]:
        leaves = pytree.tree_leaves((self.args, self.kwargs))
        return list(dict.fromkeys(x for x in leaves if isinstance(x, DumpNode)))

    def __lt__(self, other: DumpNode) -> bool:
        return self.idx < other.idx

    def __gt__(self, other: DumpNode) -> bool:
        return self.idx > other.idx


def load_dump_graph(path: str) -> list[DumpNode]:
    """The nodes of a graph_dump/*_nodes.jsonl.gz file, in graph order."""
    with gzip.open(path, "rt") as f:
        recs = [json.loads(line) for line in f]
    nodes = {
        r["name"]: DumpNode(
            r["name"], r["op"], r["target"], r["idx"], schema=r.get("schema")
        )
        for r in recs
    }

    def decode(a: Any) -> Any:
        if isinstance(a, dict):
            if set(a) == {"node"}:
                return nodes[a["node"]]
            # Older dumps wrote torch.layout / torch.memory_format as reprs, which equal str().
            if set(a) == {"repr"} and a["repr"].startswith("torch."):
                return a["repr"]
            return {k: decode(v) for k, v in a.items()}
        if isinstance(a, list):
            return [decode(x) for x in a]
        return a

    ordered = [nodes[r["name"]] for r in recs]
    for r, n, nxt in zip(recs, ordered, ordered[1:] + [None]):
        n.args = decode(r["args"])
        n.kwargs = decode(r["kwargs"]) or {}
        n.users = [nodes[u] for u in r["users"]]
        n.next = nxt
    return ordered


_TARGET_STRS: dict[Any, str] = {}


def target_str(node: fx.Node | DumpNode) -> str:
    """Op name as the graph dump writes it, e.g. ``aten.mul.Tensor`` or ``_operator.getitem``."""
    t = node.target
    if isinstance(node, DumpNode):
        return t
    if node.op != "call_function":
        return str(t)
    s = _TARGET_STRS.get(t)
    if s is None:
        if isinstance(t, torch._ops.OpOverload):
            s = str(t)
        elif isinstance(t, torch._ops.HigherOrderOperator):
            s = f"hop.{t.name()}"
        else:
            qual = getattr(t, "__qualname__", getattr(t, "__name__", str(t)))
            s = f"{getattr(t, '__module__', '')}.{qual}"
        _TARGET_STRS[t] = s
    return s


def _is_node(x: Any) -> bool:
    return isinstance(x, (fx.Node, DumpNode))


def _encode(x: Any) -> Any:
    """Literal in the graph dump's encoding (lists for tuples, str for dtypes/devices)."""
    if x is None or isinstance(x, (bool, int, float, str)):
        return x
    if isinstance(x, (list, tuple)):
        return [_encode(v) for v in x]
    if isinstance(x, dict):
        return {str(k): _encode(v) for k, v in x.items()}
    if isinstance(
        x, (torch.dtype, torch.device, torch.layout, torch.memory_format, torch.SymInt)
    ):
        return str(x)
    return {"repr": repr(x)}  # never equal to a template literal


def _fits(value: Any, template: Any) -> bool:
    if template is N:
        return _is_node(value)
    if _is_node(value):
        return False
    if isinstance(template, list):
        return (
            isinstance(value, (list, tuple))
            and len(value) == len(template)
            and all(_fits(v, t) for v, t in zip(value, template))
        )
    if isinstance(template, dict):
        return (
            isinstance(value, dict)
            and value.keys() == template.keys()
            and all(_fits(value[k], t) for k, t in template.items())
        )
    value = _encode(value)
    return type(value) is type(template) and value == template


def is_op(
    node: Any, target: str, args: list[Any], kwargs: dict[str, Any] | None = None
) -> bool:
    """``node`` calls ``target`` with exactly this argument structure.

    In ``args``/``kwargs``, ``N`` marks a node; every other entry is a literal that must
    be equal (dtypes and devices as strings, e.g. ``"torch.float32"``, ``"cuda:0"``).
    """
    return (
        _is_node(node)
        and node.op == "call_function"
        and target_str(node) == target
        and _fits(list(node.args), args)
        and _fits(dict(node.kwargs), kwargs or {})
    )


def user_of(
    node: Any,
    path: tuple[int | str, ...],
    target: str,
    args: list[Any],
    kwargs: dict[str, Any] | None = None,
    skip: tuple[Any, ...] = (),
) -> Any:
    """First user of ``node`` not in ``skip`` that has ``node`` at argument ``path`` and passes ``is_op``.

    ``path`` indexes ``args`` (ints), or ``kwargs`` when its first entry is a name.
    """
    for user in node.users:
        if user in skip or not is_op(user, target, args, kwargs):
            continue
        value: Any = user.kwargs if isinstance(path[0], str) else user.args
        for p in path:
            value = value[p]
        if value is node:
            return user
    return None


_SCHEMAS: dict[str, Any] = {}


def _schema(node: Any) -> torch._C.FunctionSchema | None:
    """Schema of the op ``node`` calls, or None for Python callables and HOPs.

    Graph dumps record it; for older dumps the op is looked up by name, which needs
    the module that registers it to be imported.
    """
    if node.op != "call_function":
        return None
    t = node.target
    if not isinstance(node, DumpNode):
        return t._schema if isinstance(t, torch._ops.OpOverload) else None
    key = node.schema or t
    if key not in _SCHEMAS:
        if node.schema:
            _SCHEMAS[key] = torch._C.parse_schema(node.schema)
        elif t.startswith(("_operator.", "hop.")):
            _SCHEMAS[key] = None
        else:
            ns, op, overload = t.split(".")
            try:
                _SCHEMAS[key] = getattr(
                    getattr(getattr(torch.ops, ns), op), overload
                )._schema
            except (AttributeError, RuntimeError) as e:
                raise ValueError(
                    f"cannot resolve {t}; import the module that registers it"
                ) from e
    return _SCHEMAS[key]


def _is_view(schema: torch._C.FunctionSchema) -> bool:
    """``OpOverload.is_view`` for a schema: aliased arguments, none written."""
    writes = [
        a.alias_info.is_write for a in schema.arguments if a.alias_info is not None
    ]
    return bool(writes) and not any(writes)


def _writes(node: Any) -> list[Any]:
    """Nodes whose storage ``node`` writes (mutable schema arguments, e.g. ``self`` or ``out``)."""
    schema = _schema(node)
    if schema is None or not schema.is_mutable:
        return []
    written = []
    for i, arg in enumerate(schema.arguments):
        if arg.alias_info is None or not arg.alias_info.is_write:
            continue
        if arg.name in node.kwargs:
            value = node.kwargs[arg.name]
        else:
            value = node.args[i] if i < len(node.args) else None
        written += [x for x in pytree.tree_leaves(value) if _is_node(x)]
    return written


def _alias_source(node: Any) -> Any:
    """The node whose storage ``node``'s output shares, or None if the output is new storage."""
    if node.op != "call_function":
        return None
    if target_str(node) == "_operator.getitem":
        src = node.args[0]
        return _alias_source(src) if _is_node(src) else None
    schema = _schema(node)
    if schema is None:
        return None
    # wait_tensor's schema has no alias annotation, but it returns its input tensor.
    if _is_view(schema) or schema.name in (
        "aten::_unsafe_view",
        "_c10d_functional::wait_tensor",
    ):
        src = node.args[0]
        return src if _is_node(src) else None
    returns = schema.returns
    if not returns or returns[0].alias_info is None:
        return None
    # In-place and out= ops return the argument they write.
    alias_set = returns[0].alias_info.before_set
    for i, arg in enumerate(schema.arguments):
        if arg.alias_info is not None and arg.alias_info.before_set & alias_set:
            value = node.kwargs.get(
                arg.name, node.args[i] if i < len(node.args) else None
            )
            return value if _is_node(value) else None
    return None


def _storage_root(node: Any) -> Any:
    while (src := _alias_source(node)) is not None:
        node = src
    return node


def _alias_readers(root: Any) -> list[Any]:
    """Nodes that read ``root``'s storage, directly or through views."""
    readers, stack, seen = [], [root], {root}
    while stack:
        x = stack.pop()
        for u in x.users:
            if u in seen:
                continue
            seen.add(u)
            readers.append(u)
            if _alias_source(u) is x:
                stack.append(u)
    return readers


def writes_outside(node: Any, members: set[Any]) -> bool:
    """``node`` writes storage that no node in ``members`` owns (e.g. a buffer accumulate)."""
    return any(_storage_root(w) not in members for w in _writes(node))


def lift_hazard(members: list[Any]) -> str:
    """Why outlining ``members`` into one call could change results; "" if it cannot.

    The call runs inside the instance's own span, so no other node moves. That is unsafe
    when an output is needed before the last input exists, when a member writes storage
    the instance does not own (or that an outside op reads before the write), or when
    another op inside the span writes storage that a member reads. The joint graph has
    in-place ops and out= writers whose order is not expressed by edges. Works on
    ``fx.Node`` and ``DumpNode`` graphs alike.
    """
    mset = set(members)
    ordered = sorted(members)
    inputs = [i for n in ordered for i in n.all_input_nodes if i not in mset]
    users = [u for n in ordered for u in n.users if u not in mset]
    if inputs and users and not max(inputs) < min(users):
        return "an output is used before the last input is produced"
    for m in ordered:
        for w in _writes(m):
            root = _storage_root(w)
            if root not in mset:
                return f"{target_str(m)} writes storage the instance does not own"
            early = [r for r in _alias_readers(root) if r not in mset and r < m]
            if early:
                return f"{target_str(m)} writes a buffer that {target_str(early[0])} reads earlier"
    read_roots = {_storage_root(a) for m in ordered for a in m.all_input_nodes}
    n = ordered[0]
    while n is not ordered[-1]:
        n = n.next
        if n not in mset and any(_storage_root(w) in read_roots for w in _writes(n)):
            return f"{target_str(n)} writes storage a member reads, inside the instance's span"
    return ""


def _plain_outputs(members: list[Any]) -> list[Any]:
    """Adjust one matched instance so its submodule outputs only plain tensors it computes.

    Matchers come from one example instance, but which values are used outside differs
    per site: a tuple-valued member takes its ``getitem`` users along, and a view of
    external inputs alone that is also used outside stays outside (the region reads it).
    """
    mset = set(members)
    members = members + [
        u
        for n in members
        for u in n.users
        if u not in mset and target_str(u) == "_operator.getitem"
    ]
    mset = set(members)
    changed = True
    while changed:
        changed = False
        for n in members:
            schema = _schema(n)
            if (
                schema is not None
                and _is_view(schema)
                and n.all_input_nodes
                and not any(i in mset for i in n.all_input_nodes)
                and any(u not in mset for u in n.users)
            ):
                members.remove(n)
                mset.discard(n)
                changed = True
                break
    return members


def find_matches(
    nodes: list[Any],
    matchers: list[tuple[str, Any]],
    claimed: set[Any] | None = None,
) -> list[list[Any]]:
    """Non-overlapping instances found by ``matchers`` (anchor op, match function), in graph order.

    ``claimed`` (updated in place) holds nodes already taken, e.g. by regions lifted
    earlier when curate_regions.py replays a sequence of passes on a dumped graph.
    """
    claimed = set() if claimed is None else claimed
    sites = []
    for anchor_target, match in matchers:
        for n in nodes:
            if (
                n.op != "call_function"
                or n in claimed
                or target_str(n) != anchor_target
            ):
                continue
            found = match(n)
            if found is None:
                continue
            members, inputs = found
            members = _plain_outputs(members)
            mset = set(members)
            if mset & claimed or mset & set(inputs):
                continue
            claimed |= mset
            sites.append(members)
    return sites


def _structural_key(gm: fx.GraphModule) -> tuple:
    """Ops and dataflow of a submodule, independent of node names."""
    index = {n: i for i, n in enumerate(gm.graph.nodes)}

    def arg_key(a: Any) -> Any:
        if isinstance(a, fx.Node):
            return ("node", index[a])
        if isinstance(a, (list, tuple)):
            return tuple(arg_key(x) for x in a)
        if isinstance(a, dict):
            return tuple((k, arg_key(v)) for k, v in a.items())
        return repr(a)

    return tuple(
        (
            n.op,
            "" if n.op == "placeholder" else target_str(n),
            arg_key(n.args),
            arg_key(n.kwargs),
        )
        for n in gm.graph.nodes
    )


def _outline(gm: fx.GraphModule, nodes: list[fx.Node], name: str) -> fx.Node:
    """Replace ``nodes`` by one ``call_module name`` placed inside their own span.

    The call goes at the first member, or right after the last external input if that
    comes later, so no other node is reordered (lift_hazard has checked this is safe).
    """
    # TODO: torch.fx.passes.utils.fuser_utils.fuse_as_graphmodule does the same outlining,
    # but its validate_partition walks the whole downstream graph per partition (too slow
    # for thousands of instances) and insert_subgm relies on a later topological re-sort,
    # which reorders mutations; upstream a bounded, order-preserving variant.
    graph = gm.graph
    members = dict.fromkeys(sorted(nodes))
    inputs = list(
        dict.fromkeys(i for n in members for i in n.all_input_nodes if i not in members)
    )
    outputs = [n for n in members if any(u not in members for u in n.users)]

    sub = fx.Graph()
    env: dict[fx.Node, fx.Node] = {}
    for i in inputs:
        p = sub.placeholder(i.name)
        p.meta = dict(i.meta)
        env[i] = p
    for n in members:
        env[n] = sub.node_copy(n, env.__getitem__)
    sub.output(env[outputs[0]] if len(outputs) == 1 else tuple(env[o] for o in outputs))
    gm.add_submodule(name, fx.GraphModule(gm, sub, class_name=name))

    first = next(iter(members))
    last_input = max(inputs) if inputs else None
    anchor = last_input if last_input is not None and last_input > first else first.prev
    with graph.inserting_after(anchor):
        call = graph.call_module(name, tuple(inputs))
    if len(outputs) == 1:
        call.meta = dict(outputs[0].meta)
        replacements = [call]
    else:
        call.meta = {"val": tuple(o.meta.get("val") for o in outputs)}
        replacements = []
        prev = call
        for k, o in enumerate(outputs):
            with graph.inserting_after(prev):
                prev = graph.call_function(operator.getitem, (call, k))
            prev.meta = dict(o.meta)
            replacements.append(prev)
    for o, new in zip(outputs, replacements):
        o.replace_all_uses_with(new, delete_user_cb=lambda u: u not in members)
    for n in reversed(list(members)):
        graph.erase_node(n)
    return call


def lift_matches(
    gm: fx.GraphModule, region_id: str, matchers: list[tuple[str, Any]]
) -> fx.GraphModule:
    """Lift every safe instance that ``matchers`` find into ``call_module fusion_<id>``."""
    sites = find_matches(list(gm.graph.nodes), matchers)
    hazards = [lift_hazard(members) for members in sites]
    safe = [members for members, h in zip(sites, hazards) if not h]
    if not safe:
        unsafe = f", unsafe: {hazards[0]}" if sites else ""
        raise ValueError(
            f"fusion region {region_id}: no instance to lift ({len(sites)} matched{unsafe}); "
            "the graph changed or an overlapping region was lifted first"
        )
    skipped = [h for h in hazards if h]
    if skipped:
        logger.warning(
            "fusion region %s: skipped %d unsafe match(es), e.g. %s",
            region_id,
            len(skipped),
            skipped[0],
        )
    base = f"fusion_{region_id.replace('-', '')}"
    calls = [_outline(gm, members, f"{base}_i{k}") for k, members in enumerate(safe)]

    # Share one submodule across structurally identical instances.
    variants: dict[tuple, list[fx.Node]] = {}
    for call in calls:
        variants.setdefault(_structural_key(getattr(gm, call.target)), []).append(call)
    for v, group in enumerate(variants.values()):
        name = base if len(variants) == 1 else f"{base}_v{v}"
        gm.add_submodule(name, getattr(gm, group[0].target))
        for call in group:
            gm.delete_submodule(call.target)
            call.target = name
            call.meta["fusion_region"] = region_id
    gm.graph.lint()
    gm.recompile()
    logger.info(
        "fusion region %s: lifted %d instance(s) into %s", region_id, len(calls), base
    )
    return gm
