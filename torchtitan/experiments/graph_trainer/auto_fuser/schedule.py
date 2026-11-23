# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Address the operators of a GraphTrainer joint graph by phase, module and occurrence."""

from __future__ import annotations

import operator
from collections import defaultdict
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from typing import Any

import torch
from torch import fx

from torchtitan.experiments.graph_trainer.auto_fuser.programmable_fusion import op_scope
from torchtitan.experiments.graph_trainer.auto_fuser.region import RegionOutliner


class Scope:
    """Operators in graph order, grouped by their relative module paths."""

    def __init__(self, path: str) -> None:
        self._path = path
        self._modules: dict[str, Scope] = {}
        self._operators: dict[str, list[fx.Node]] = defaultdict(list)

    def __getitem__(self, name: int | str) -> Scope | list[fx.Node]:
        key = str(name)
        if key in self._operators and key in self._modules:
            raise ValueError(f"{self._path}.{key} names both a module and an operator")
        if key in self._operators:
            return self._operators[key]
        return self._modules[key]

    def __getattr__(self, name: str) -> Any:
        # Only the live graph determines if an attribute is a module or an op list.
        try:
            return self[name]
        except KeyError:
            raise AttributeError(
                f"No operator or module at {self._path}.{name}"
            ) from None

    def __iter__(self) -> Iterator[str]:
        return iter(self._modules)

    def __len__(self) -> int:
        return len(self._modules)

    def nodes(self) -> Iterator[fx.Node]:
        for group in self._operators.values():
            yield from group
        for module in self._modules.values():
            yield from module.nodes()


class IndexedGraph:
    def __init__(self, gm: fx.GraphModule) -> None:
        self.graph_module = gm
        self.fwd = Scope("fwd")
        self.bwd = Scope("bwd")
        for node in gm.graph.nodes:
            if node.op != "call_function":
                continue
            phase, module, name = op_scope(node)
            scope = self.bwd if phase == "bwd" else self.fwd
            for part in module.split(".") if module else ():
                if part not in scope._modules:
                    scope._modules[part] = Scope(f"{scope._path}.{part}")
                scope = scope._modules[part]
            scope._operators[name].append(node)


def _is_view(node: fx.Node) -> bool:
    return node.target is operator.getitem or (
        isinstance(node.target, torch._ops.OpOverload) and node.target.is_view
    )


Selection = fx.Node | Scope | Iterable[fx.Node]


def _members(targets: Iterable[Selection]) -> set[fx.Node]:
    selected = set()
    for target in targets:
        if isinstance(target, fx.Node):
            selected.add(target)
        elif isinstance(target, Scope):
            selected.update(target.nodes())
        else:
            selected.update(target)
    # Include view chains connecting two named ops. Views at either boundary
    # determine the kernel signature and can be selected explicitly.
    upstream = set()
    pending = list(selected)
    while pending:
        for node in pending.pop().all_input_nodes:
            if node not in upstream and node not in selected and _is_view(node):
                upstream.add(node)
                pending.append(node)
    for node in sorted(upstream):
        if any(parent in selected for parent in node.all_input_nodes):
            selected.add(node)
    for node in list(selected):
        selected.update(user for user in node.users if user.target is operator.getitem)
    return selected


@dataclass
class SchedulePlan:
    """Collect disjoint selections against the original graph, then outline in order."""

    indexed_graph: IndexedGraph
    _regions: list[tuple[str, set[fx.Node]]] = field(default_factory=list, init=False)
    _claimed: set[fx.Node] = field(default_factory=set, init=False)

    def fuse(self, name: str, targets: Iterable[Selection]) -> None:
        members = _members(targets)
        if not members:
            raise ValueError(f"{name}: empty fusion region")
        if overlap := members & self._claimed:
            raise ValueError(f"{name}: {min(overlap)} is already in another region")
        for node in members:
            if node.graph is not self.indexed_graph.graph_module.graph:
                raise ValueError(f"{name}: {node} belongs to another graph")
            if node.target is operator.getitem:
                continue
            if not isinstance(node.target, torch._ops.OpOverload) or (
                node.target.namespace
                in {"c10d", "_c10d_functional", "_c10d_functional_autograd"}
                or torch.Tag.nondeterministic_seeded in node.target.tags
            ):
                raise ValueError(f"{name}: cannot fuse {node.target}")
        self._regions.append((name, members))
        self._claimed.update(members)

    def apply(self) -> list[fx.Node]:
        gm = self.indexed_graph.graph_module
        outliner = RegionOutliner(gm)
        calls = []
        for i, (name, members) in enumerate(self._regions):
            target = f"fusion_{name}_{i}"
            if hasattr(gm, target):
                raise ValueError(f"Graph attribute already exists: {target}")
            call = outliner.outline(members, name=target)
            call.meta["fusion_region"] = name
            calls.append(call)
        gm.graph.lint()
        gm.recompile()
        return calls
