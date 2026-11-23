# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Outline FX regions while preserving tensor and storage dependencies."""

import operator
from typing import Any

import torch
from torch import fx
from torch.utils import _pytree as pytree


def storage_effects(node: fx.Node) -> tuple[list[str], list[str]]:
    """Names of the inputs whose storage ``node`` writes, and of the inputs whose
    storage its output aliases, from the op schema (``Tensor(a!)``, ``Tensor(a)``).

    A ``getitem`` aliases its input.
    """
    if node.target is operator.getitem:
        source = node.args[0]
        return [], [source.name] if isinstance(source, fx.Node) else []
    if not isinstance(node.target, torch._ops.OpOverload):
        return [], []
    schema = node.target._schema
    writes, aliased = [], []
    for i, arg in enumerate(schema.arguments):
        if arg.alias_info is None:
            continue
        if not arg.kwarg_only and i < len(node.args):
            value = node.args[i]
        elif arg.name in node.kwargs:
            value = node.kwargs[arg.name]
        else:
            continue
        names = [x.name for x in pytree.tree_leaves(value) if isinstance(x, fx.Node)]
        aliased += names
        if arg.alias_info.is_write:
            writes += names
    if not any(r.alias_info is not None for r in schema.returns):
        aliased = []
    return writes, aliased


def _submodule_storage(sub: fx.GraphModule) -> tuple[set[int], list[int | str]]:
    """The inputs (placeholder indices) a lifted submodule writes into, and per
    output its storage: an input's index, or the name of the node inside that
    allocated it."""
    by_name = {n.name: n for n in sub.graph.nodes}
    index: dict[fx.Node, int] = {}
    base: dict[fx.Node, fx.Node] = {}
    written: set[fx.Node] = set()
    outputs: tuple = ()
    for n in sub.graph.nodes:
        if n.op == "placeholder":
            index[n] = len(index)
            base[n] = n
        elif n.op == "output":
            outputs = n.args[0]
        else:
            writes, aliases = storage_effects(n)
            base[n] = base[by_name[aliases[0]]] if aliases else n
            written |= {base[by_name[w]] for w in writes}
    return (
        {index[b] for b in written if b in index},
        [index[base[o]] if base[o] in index else base[o].name for o in outputs],
    )


class RegionOutliner:
    """The storage each node reads and writes, as a base node (or a lifted call's
    storage); a view accesses no data. A lifted call reads its inputs and writes
    the inputs its submodule writes into; each of its outputs keeps its storage,
    an input's or one allocated inside the submodule."""

    def __init__(self, gm: fx.GraphModule) -> None:
        self.gm = gm
        by_name = {n.name: n for n in gm.graph.nodes}
        self.base: dict[fx.Node, Any] = {}
        self.reads: dict[fx.Node, set] = {}
        self.writes: dict[fx.Node, set] = {}
        calls: dict[fx.Node, list[int | str]] = {}
        for n in gm.graph.nodes:
            sub = gm.get_submodule(n.target) if n.op == "call_module" else None
            if isinstance(sub, fx.GraphModule):
                written, calls[n] = _submodule_storage(sub)
                self.base[n] = n
                self.reads[n] = {self.base[i] for i in n.all_input_nodes}
                self.writes[n] = {self.base[n.args[i]] for i in written}
                continue
            if n.target is operator.getitem and n.args[0] in calls:
                b = calls[n.args[0]][n.args[1]]
                self.base[n] = (
                    self.base[n.args[0].args[b]]
                    if isinstance(b, int)
                    else (n.args[0], b)
                )
                self.reads[n], self.writes[n] = set(), set()
                continue
            writes, aliases = storage_effects(n)
            self.base[n] = self.base[by_name[aliases[0]]] if aliases else n
            self.writes[n] = {self.base[by_name[w]] for w in writes}
            is_view = aliases and not writes
            self.reads[n] = (
                set() if is_view else {self.base[i] for i in n.all_input_nodes}
            )

    def _add_call(
        self,
        call: fx.Node,
        members: list[fx.Node],
        outputs: list[fx.Node],
        getitems: list[fx.Node],
    ) -> None:
        """Note the call that replaced ``members`` (erased; their storage stays known
        by them) and its ``getitems`` of ``outputs``."""
        self.base[call] = call
        self.reads[call] = set().union(*(self.reads[m] for m in members))
        self.writes[call] = set().union(*(self.writes[m] for m in members))
        for out, getitem in zip(outputs, getitems):
            self.base[getitem] = self.base[out]
            self.reads[getitem], self.writes[getitem] = set(), set()

    def outline(self, members: set[fx.Node], *, name: str) -> fx.Node:
        """Outline a region after moving its interleaved dependents out of the way."""

        moved = []
        moved_set = set()
        reads, writes = set(), set()
        moved_reads, moved_writes = set(), set()
        ordered = sorted(members)
        node = ordered[0]
        while True:
            if node in members:
                if (
                    moved_set.intersection(node.all_input_nodes)
                    or self.reads[node] & moved_writes
                    or self.writes[node] & (moved_reads | moved_writes)
                ):
                    raise ValueError(
                        f"{name}: dependency leaves the region and returns to {node}"
                    )
                reads.update(self.reads[node])
                writes.update(self.writes[node])
            elif (
                any(i in members or i in moved_set for i in node.all_input_nodes)
                or self.reads[node] & writes
                or self.writes[node] & (reads | writes)
            ):
                moved.append(node)
                moved_set.add(node)
                moved_reads.update(self.reads[node])
                moved_writes.update(self.writes[node])
                reads.update(self.reads[node])
                writes.update(self.writes[node])
            if node is ordered[-1]:
                break
            node = node.next
        call, outputs, projections = _outline(self.gm, ordered, name)
        self._add_call(call, ordered, outputs, projections)
        anchor = projections[-1] if projections else call
        for node in moved:
            anchor.append(node)
            anchor = node
        return call


def _outline(
    gm: fx.GraphModule, members: list[fx.Node], name: str
) -> tuple[fx.Node, list[fx.Node], list[fx.Node]]:
    """Move ``members`` (in graph order) into submodule ``name``. Returns the call,
    placed after the last member, the members it outputs, and the ``getitem``
    nodes of its outputs."""
    mset = set(members)
    sub = fx.Graph()
    env: dict[fx.Node, fx.Node] = {}
    inputs: list[fx.Node] = []
    for n in members:
        for i in n.all_input_nodes:
            if i not in mset and i not in env:
                env[i] = sub.placeholder(i.name)
                env[i].meta = dict(i.meta)
                inputs.append(i)
        env[n] = sub.node_copy(n, env.__getitem__)
    outputs = [n for n in members if any(u not in mset for u in n.users)]
    sub.output(tuple(env[n] for n in outputs))
    module = fx.GraphModule(torch.nn.Module(), sub, name)
    # FX treats a GraphModule call as pure: dead code elimination would drop a call
    # whose writes into storage have no FX users.
    module._is_impure = any(storage_effects(n)[0] for n in members)
    gm.add_submodule(name, module)
    with gm.graph.inserting_after(members[-1]):
        call = gm.graph.call_module(name, tuple(inputs))
    call.meta["val"] = tuple(n.meta.get("val") for n in outputs)
    getitems, anchor = [], call
    for k, n in enumerate(outputs):
        with gm.graph.inserting_after(anchor):
            anchor = gm.graph.call_function(operator.getitem, (call, k))
        anchor.meta = dict(n.meta)
        n.replace_all_uses_with(anchor, delete_user_cb=lambda u: u not in mset)
        getitems.append(anchor)
    for n in reversed(members):
        gm.graph.erase_node(n)
    return call, outputs, getitems
