# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Programmable fusion: recipes name fusion regions with plain loops over layers.

Each call_function node of the traced joint graph (except getitems and
collectives) has an op path::

    [fwd|bwd].<module fqn>.<op>#k      e.g. bwd.layers.3.ffn_norm._fused_rms_norm_backward#0

``k`` counts the ops with the same phase, module and op in graph order. A
prefix selects a subtree: ``fwd.layers.3.attention`` is every forward op of
``layers.3.attention`` and its submodules. A region is its targets minus its
exclusions, plus the getitems of tuple-valued members::

    p = FusionPlan()
    for i in range(num_layers):
        p.fuse(
            "qk_norm_rope_fwd",
            [f"fwd.layers.{i}.attention.q_norm", f"fwd.layers.{i}.attention.k_norm",
             f"fwd.layers.{i}.attention.rope"],
        )
        p.fuse(
            "router_bwd",
            [f"bwd.layers.{i}.feed_forward.router"],
            exclude=[f"bwd.layers.{i}.feed_forward.router.gate.mm#0"],
        )
    config.compile.programmable_fusion = p.build(kernel_cache_dir=...)

Each region is outlined into a ``call_module`` that runs its original ops, then
``substitute_with_kernels_pass`` swaps in a cached kernel where one exists.
Regions with the same name share one submodule per structural variant (the
first instance's in graph order), so all layers find the same kernels.
"""

from __future__ import annotations

import difflib
import operator
from collections import Counter, defaultdict
from collections.abc import Sequence
from dataclasses import dataclass

import torch
import torch.fx as fx

from torchtitan.experiments.graph_trainer.common_utils import (
    _get_module_fqn,
    _is_backward_node,
)
from torchtitan.experiments.graph_trainer.mutation_utils import (
    base_tensor_for_mutation_target,
    mutation_target_nodes,
)
from torchtitan.experiments.graph_trainer.substitute_kernels import (
    normalize_names,
    substitute_with_kernels_pass,
)


@dataclass(frozen=True, kw_only=True, slots=True)
class FusionRegion:
    name: str
    targets: tuple[str, ...]
    exclude: tuple[str, ...] = ()


@dataclass(frozen=True, kw_only=True, slots=True)
class ProgrammableFusionConfig:
    regions: tuple[FusionRegion, ...]
    kernel_cache_dir: str | None = None
    """Kernel store for ``substitute_with_kernels_pass``; None uses its default."""


class FusionPlan:
    """Collects fusion regions from a recipe's loops."""

    def __init__(self) -> None:
        self.regions: list[FusionRegion] = []

    def fuse(
        self, name: str, targets: Sequence[str], *, exclude: Sequence[str] = ()
    ) -> None:
        self.regions.append(
            FusionRegion(name=name, targets=tuple(targets), exclude=tuple(exclude))
        )

    def build(self, *, kernel_cache_dir: str | None = None) -> ProgrammableFusionConfig:
        return ProgrammableFusionConfig(
            regions=tuple(self.regions), kernel_cache_dir=kernel_cache_dir
        )


def _op_name(node: fx.Node) -> str:
    if isinstance(node.target, torch._ops.OpOverload):
        return node.target.overloadpacket.__name__
    if isinstance(node.target, torch._ops.HigherOrderOperator):
        return node.target.name()
    return getattr(node.target, "__name__", str(node.target))


def op_paths(gm: fx.GraphModule) -> dict[fx.Node, str]:
    """The op path of every addressable node, in graph order."""
    counts: Counter[str] = Counter()
    paths: dict[fx.Node, str] = {}
    for node in gm.graph.nodes:
        if node.op != "call_function" or node.target is operator.getitem:
            continue
        if (
            isinstance(node.target, torch._ops.OpOverload)
            and node.target.namespace == "_c10d_functional"
        ):
            continue
        # Nodes outside any module (e.g. gradient-accumulation adds) take their
        # first input's module.
        module = _get_module_fqn(node) or next(
            (m for m in map(_get_module_fqn, node.all_input_nodes) if m), ""
        )
        phase = "bwd" if _is_backward_node(node) else "fwd"
        key = ".".join(p for p in (phase, module, _op_name(node)) if p)
        paths[node] = f"{key}#{counts[key]}"
        counts[key] += 1
    return paths


def _resolve(index: dict[str, list[fx.Node]], path: str, region: str) -> list[fx.Node]:
    nodes = index.get(path)
    if not nodes:
        closest = difflib.get_close_matches(path, index.keys(), n=3)
        raise ValueError(
            f"fusion region {region}: {path!r} matched no op; closest: {closest}"
        )
    return nodes


def _region_nodes(
    index: dict[str, list[fx.Node]], paths: dict[fx.Node, str], region: FusionRegion
) -> list[fx.Node]:
    nodes = {n: None for t in region.targets for n in _resolve(index, t, region.name)}
    for path in region.exclude:
        removed = [n for n in _resolve(index, path, region.name) if n in nodes]
        if not removed:
            raise ValueError(
                f"fusion region {region.name}: exclude {path!r} removes nothing"
            )
        for n in removed:
            del nodes[n]
    phases = {paths[n].split(".", 1)[0] for n in nodes}
    if len(phases) > 1:
        raise ValueError(f"fusion region {region.name} mixes forward and backward ops")
    for n in list(nodes):
        nodes.update((u, None) for u in n.users if u.target is operator.getitem)
    return sorted(nodes)


def _check_outlinable(nodes: list[fx.Node], region: str) -> None:
    """Raise if one call at the region's position could change results.

    The call runs where the region's last input is ready, so every output must
    be used after that, and no other op in the region's span may write storage
    the region reads (in-place ops are ordered by position, not edges).
    """
    members = set(nodes)
    inputs = [i for n in nodes for i in n.all_input_nodes if i not in members]
    users = [u for n in nodes for u in n.users if u not in members]
    if not users:
        raise ValueError(f"fusion region {region} has no outputs")
    if inputs and max(inputs) > min(users):
        raise ValueError(
            f"fusion region {region}: output used by {min(users)} before input "
            f"{max(inputs)} exists"
        )
    for n in nodes:
        for target in mutation_target_nodes(n):
            if base_tensor_for_mutation_target(target) not in members:
                raise ValueError(
                    f"fusion region {region}: {n} writes storage it does not own"
                )
    read = {base_tensor_for_mutation_target(i) for i in inputs}
    node = nodes[0]
    while node is not nodes[-1]:
        node = node.next
        if node not in members and any(
            base_tensor_for_mutation_target(t) in read
            for t in mutation_target_nodes(node)
        ):
            raise ValueError(
                f"fusion region {region}: {node} writes an input of the region inside its span"
            )


def _outline(gm: fx.GraphModule, nodes: list[fx.Node], name: str) -> fx.Node:
    """Replace ``nodes`` by one ``call_module`` inside their span, moving no other node."""
    members = set(nodes)
    inputs = list(
        dict.fromkeys(i for n in nodes for i in n.all_input_nodes if i not in members)
    )
    outputs = [n for n in nodes if any(u not in members for u in n.users)]
    sub = fx.Graph()
    env: dict[fx.Node, fx.Node] = {}
    for i in inputs:
        env[i] = sub.placeholder(i.name)
        env[i].meta = dict(i.meta)
    for n in nodes:
        env[n] = sub.node_copy(n, env.__getitem__)
    sub.output(env[outputs[0]] if len(outputs) == 1 else tuple(env[o] for o in outputs))
    gm.add_submodule(name, fx.GraphModule(gm, sub, class_name=name))

    anchor = max(inputs) if inputs and max(inputs) > nodes[0] else nodes[0].prev
    with gm.graph.inserting_after(anchor):
        call = gm.graph.call_module(name, tuple(inputs))
    if len(outputs) == 1:
        call.meta = dict(outputs[0].meta)
        replacements = [call]
    else:
        call.meta = {"val": tuple(o.meta.get("val") for o in outputs)}
        replacements = []
        previous = call
        for k, o in enumerate(outputs):
            with gm.graph.inserting_after(previous):
                previous = gm.graph.call_function(operator.getitem, (call, k))
            previous.meta = dict(o.meta)
            replacements.append(previous)
    for o, new in zip(outputs, replacements):
        o.replace_all_uses_with(new, delete_user_cb=lambda u: u not in members)
    for n in reversed(nodes):
        gm.graph.erase_node(n)
    return call


def programmable_fusion_pass(
    gm: fx.GraphModule,
    example_inputs: tuple | None = None,
    *,
    config: ProgrammableFusionConfig,
) -> fx.GraphModule:
    paths = op_paths(gm)
    index: dict[str, list[fx.Node]] = defaultdict(list)
    for node, path in paths.items():
        parts = path.split(".")
        for i in range(1, len(parts) + 1):
            index[".".join(parts[:i])].append(node)

    # Resolve and check every region before changing the graph.
    by_name: dict[str, list[list[fx.Node]]] = defaultdict(list)
    claimed: set[fx.Node] = set()
    for region in config.regions:
        nodes = _region_nodes(index, paths, region)
        if overlap := claimed.intersection(nodes):
            raise ValueError(
                f"fusion region {region.name} overlaps an earlier region at "
                f"{paths.get(next(iter(overlap)))}"
            )
        _check_outlinable(nodes, region.name)
        claimed.update(nodes)
        by_name[region.name].append(nodes)

    for name, instances in by_name.items():
        if hasattr(gm, name):
            raise ValueError(
                f"fusion region name {name!r} is already a graph attribute"
            )
        # Graph order: instances share the first one's submodule, whose node
        # metadata (e.g. module_fqn) is part of the kernel-cache key.
        instances.sort(key=lambda nodes: nodes[0])
        calls = [
            _outline(gm, nodes, f"{name}_i{k}") for k, nodes in enumerate(instances)
        ]
        variants: dict[str, list[fx.Node]] = defaultdict(list)
        for call in calls:
            variants[normalize_names(gm.get_submodule(call.target)).code].append(call)
        for v, group in enumerate(variants.values()):
            target = name if len(variants) == 1 else f"{name}_v{v}"
            gm.add_submodule(target, gm.get_submodule(group[0].target))
            for call in group:
                gm.delete_submodule(call.target)
                call.target = target
    gm.graph.lint()
    gm.recompile()
    return substitute_with_kernels_pass(gm, cache_dir=config.kernel_cache_dir)
