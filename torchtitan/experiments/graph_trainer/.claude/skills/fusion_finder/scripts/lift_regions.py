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
holding the matched nodes: the computation a kernel replaces. The call takes the
place of the match's last node. The nodes between the match's first and last node
that depend on it, through a tensor or through a storage they share
(``storage_effects``), move after the call in their order; nothing else moves. A
pass raises if it finds no match, or if a path leaves a match and comes back (it
cannot become one call).
"""

import json
import logging
import os
from typing import Any

import torch.fx as fx

from fusion_finder_capture import match_key, target_name
from regions import key_skeleton, Pattern

from torchtitan.experiments.graph_trainer.auto_fuser.region import RegionOutliner
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


def _lift_region(gm: fx.GraphModule, region: dict, pattern: Pattern) -> fx.GraphModule:
    pass_name = f"lift_fusion_region_{region['id']}"
    matches = _find(pattern, gm.graph)
    if not matches:
        raise ValueError(f"{pass_name}: no instance of the region in this graph")

    outliner = RegionOutliner(gm)
    for k, members in enumerate(matches):
        outliner.outline(set(members), name=f"fusion_{region['id']}_i{k}")
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
