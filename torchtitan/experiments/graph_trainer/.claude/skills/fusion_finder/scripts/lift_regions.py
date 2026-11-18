# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Lift the regions found by ``regions.py`` into ``call_module`` nodes.

A recipe registers the passes and lists them in ``compile.extra_passes``::

    from lift_regions import register_lift_passes

    config.compile.extra_passes = register_lift_passes("<capture dir>")

``register_lift_passes`` registers one graph pass per region above the profit floor
(or per id in ``ids``), named ``lift_fusion_region_<id>``, and returns their names
in the order of ``regions.json``. A pass finds the region's instances by topology,
as ``regions.py`` groups them: the region's pattern (``regions.Pattern``, the first
instance's ``match_key``s) matches the node sets with the same ops, edges among
them and arguments. Node names do not matter, and neither do sizes
(arguments the op schema types ``SymInt``), scalar operands (numbers passed as a
``Tensor`` or ``Scalar``) or device indices: the passes apply to other shapes, layer
counts and ranks of the same model code.

Match ``k`` becomes ``fusion_<id>_i<k> = self.fusion_<id>_i<k>(...)``, a submodule
holding the matched nodes: the computation a kernel replaces, on tensors (a
``getitem`` of a tuple from outside the match stays outside, and the call takes the
tensor). The call takes the place of the match's last node. The nodes between the match's first and last node
that depend on it, through a tensor or through a storage they share
(``storage_effects``), move after the call in their order; nothing else moves. A
pass raises if it finds no match, or if a path leaves a match and comes back (it
cannot become one call).
"""

import json
import logging
import operator
import os
from typing import Any

import torch
import torch.fx as fx

from fusion_finder_capture import match_key, storage_effects, target_name
from regions import key_skeleton, Pattern

from torchtitan.experiments.graph_trainer.registry import register_graph_pass

logger = logging.getLogger(__name__)


def _find(pattern: Pattern, graph: fx.Graph) -> list[list[fx.Node]]:
    """The matches of ``pattern`` in ``graph`` that share no node (in graph order):
    per match, the nodes matched to the members, in the members' order."""
    by_name = {n.name: n for n in graph.nodes}
    skeletons: dict[str, tuple[str, list[str]]] = {}

    def skeleton(name: str) -> tuple[str, list[str]]:
        if name not in skeletons:
            skeletons[name] = key_skeleton(match_key(by_name[name]))
        return skeletons[name]

    def users(name: str) -> list[str]:
        return [u.name for u in by_name[name].users]

    anchors = [
        n.name
        for n in graph.nodes
        if n.op == "call_function" and target_name(n) == pattern.target
    ]
    return [
        [by_name[bound[m]] for m in pattern.members]
        for bound in pattern.find(anchors, skeleton, users)
    ]


def submodule_storage(sub: fx.GraphModule) -> tuple[set[int], list[int | str]]:
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


class _Storage:
    """The storage each node reads and writes, as a base node (or a lifted call's
    storage); a view accesses no data. A lifted call reads its inputs and writes
    the inputs its submodule writes into; each of its outputs keeps its storage,
    an input's or one allocated inside the submodule."""

    def __init__(self, gm: fx.GraphModule) -> None:
        by_name = {n.name: n for n in gm.graph.nodes}
        self.base: dict[fx.Node, Any] = {}
        self.reads: dict[fx.Node, set] = {}
        self.writes: dict[fx.Node, set] = {}
        calls: dict[fx.Node, list[int | str]] = {}
        for n in gm.graph.nodes:
            sub = gm.get_submodule(n.target) if n.op == "call_module" else None
            if isinstance(sub, fx.GraphModule):
                written, calls[n] = submodule_storage(sub)
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

    def add_call(
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


def _outline(
    gm: fx.GraphModule, members: list[fx.Node], order: list[fx.Node], name: str
) -> tuple[fx.Node, list[fx.Node], list[fx.Node]]:
    """Move ``members`` (in graph order) into submodule ``name``, in ``order``.
    Returns the call, placed after the last member, the members it outputs, and the
    ``getitem`` nodes of its outputs."""
    mset = set(members)
    sub = fx.Graph()
    env: dict[fx.Node, fx.Node] = {}
    inputs: list[fx.Node] = []
    for n in order:
        for i in n.all_input_nodes:
            if i not in mset and i not in env:
                env[i] = sub.placeholder(i.name)
                env[i].meta = dict(i.meta)
                inputs.append(i)
        env[n] = sub.node_copy(n, env.__getitem__)
    outputs = [n for n in order if any(u not in mset for u in n.users)]
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


def _lift_region(gm: fx.GraphModule, region: dict, pattern: Pattern) -> fx.GraphModule:
    pass_name = f"lift_fusion_region_{region['id']}"
    matches = _find(pattern, gm.graph)
    if not matches:
        raise ValueError(f"{pass_name}: no instance of the region in this graph")

    # Positions only need to stay increasing along the graph.
    position = {n: float(i) for i, n in enumerate(gm.graph.nodes)}
    storage = _Storage(gm)
    for k, matched in enumerate(matches):
        # A getitem of a tuple from outside the match reads one tensor of it: it
        # stays outside, and the call takes that tensor rather than the tuple.
        members = sorted(
            (
                n
                for n in matched
                if not (n.target is operator.getitem and n.args[0] not in matched)
            ),
            key=position.__getitem__,
        )
        mset = set(members)
        # Walk from the first to the last member. A node that depends on a member,
        # or on a node already moved, moves after the call; a member must not.
        moved: list[fx.Node] = []
        moved_set: set[fx.Node] = set()
        read: set[fx.Node] = set()
        written: set[fx.Node] = set()
        moved_read: set[fx.Node] = set()
        moved_written: set[fx.Node] = set()
        n = members[0]
        while True:
            if n in mset:
                if (
                    any(i in moved_set for i in n.all_input_nodes)
                    or storage.reads[n] & moved_written
                    or storage.writes[n] & (moved_read | moved_written)
                ):
                    raise ValueError(
                        f"{pass_name}: a path leaves instance {k} and comes back to "
                        f"{n.name}"
                    )
                read |= storage.reads[n]
                written |= storage.writes[n]
            elif (
                any(i in mset or i in moved_set for i in n.all_input_nodes)
                or storage.reads[n] & written
                or storage.writes[n] & (read | written)
            ):
                moved.append(n)
                moved_set.add(n)
                for nodes_read, nodes_written in (
                    (read, written),
                    (moved_read, moved_written),
                ):
                    nodes_read |= storage.reads[n]
                    nodes_written |= storage.writes[n]
            if n is members[-1]:
                break
            n = n.next
        # Without writes, the submodule lists the nodes in the pattern's order (a
        # topological order of every match), so every instance has the same code,
        # inputs and outputs, whatever the order of independent nodes in the graph.
        order = (
            members
            if any(storage.writes[n] for n in members)
            else [n for n in matched if n in mset]
        )
        call, outputs, getitems = _outline(
            gm, members, order, f"fusion_{region['id']}_i{k}"
        )
        storage.add_call(call, members, outputs, getitems)
        anchor = getitems[-1] if getitems else call
        for n in moved:
            anchor.append(n)
            anchor = n
        block = [call, *getitems, *moved]
        start = position[members[-1]]
        stop = position.get(anchor.next, start + 1.0)
        for j, n in enumerate(block, 1):
            position[n] = start + (stop - start) * j / (len(block) + 1)
    gm.graph.lint()
    gm.recompile()
    logger.info(
        f"{pass_name}: lifted {len(matches)} instances "
        f"({region['instances']} in the capture)"
    )
    return gm


def register_lift_passes(capture_dir: str, ids: list[str] | None = None) -> list[str]:
    """Register a lift pass per region of ``<capture_dir>/regions.json`` above the
    profit floor, or per region id in ``ids``; return the pass names in the order of
    ``regions.json``, the order they must run in: a node an earlier pass lifted is
    not matched again."""
    with open(os.path.join(capture_dir, "regions.json")) as f:
        found = json.load(f)
    if ids is None:
        regions = [
            r for r in found["regions"] if r["gap_ms"] >= found["profit_floor_ms"]
        ]
    else:
        unknown = set(ids) - {r["id"] for r in found["regions"]}
        if unknown:
            raise ValueError(f"no region {sorted(unknown)} in {capture_dir}")
        regions = [r for r in found["regions"] if r["id"] in ids]
    names = []
    for region in regions:
        if region["pattern"] is None:
            raise ValueError(
                f"region {region['id']} has no pattern: its members are joined only "
                "through a storage"
            )
        pattern = Pattern(**region["pattern"])

        def lift(
            gm: fx.GraphModule,
            example_inputs: Any = None,
            *,
            _region: dict = region,
            _pattern: Pattern = pattern,
        ) -> fx.GraphModule:
            return _lift_region(gm, _region, _pattern)

        lift.__name__ = lift.__qualname__ = f"lift_fusion_region_{region['id']}"
        register_graph_pass(lift)
        names.append(lift.__name__)
    return names
