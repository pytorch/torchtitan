# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Registered graph pass that dumps per-node and per-storage info of the final
joint graph (after memory policy, remat and bucketing, before Inductor).

The dump lets the offline analysis attribute allocator trace events (which
carry FX node names via ``_snapshot(augment_with_fx_traces=True)``) to
forward/backward phase and layer, and gives a graph-only liveness estimate.
"""

import json
import os

import torch
from torch._inductor.fx_passes.memory_estimator import GraphAliasTracker
from torch._subclasses.fake_tensor import is_fake_tensor
from torch.utils._pytree import tree_leaves

from torchtitan.experiments.graph_trainer.common_utils import (
    _get_layer_id,
    _get_module_fqn,
    _is_backward_node,
)
from torchtitan.experiments.graph_trainer.registry import register_graph_pass


def _out_dir() -> str:
    d = os.environ.get("MEMPLAN_OUT", "/tmp/memplan")
    os.makedirs(d, exist_ok=True)
    return d


@register_graph_pass
def memplan_dump_graph_info_pass(gm: torch.fx.GraphModule, example_inputs=None):
    nodes = list(gm.graph.nodes)
    node_idx = {n: i for i, n in enumerate(nodes)}
    node_rows = []
    for i, n in enumerate(nodes):
        outs = []
        val = n.meta.get("val")
        if val is not None:
            for t in tree_leaves(val):
                if is_fake_tensor(t) or isinstance(t, torch.Tensor):
                    try:
                        outs.append(
                            [
                                list(t.shape),
                                str(t.dtype),
                                t.device.type,
                                int(t.untyped_storage().nbytes()),
                            ]
                        )
                    except Exception:
                        outs.append([None, str(getattr(t, "dtype", None)), None, 0])
        node_rows.append(
            {
                "i": i,
                "name": n.name,
                "op": n.op,
                "target": str(n.target),
                "bwd": bool(_is_backward_node(n)),
                "fqn": _get_module_fqn(n),
                "layer": _get_layer_id(n),
                "outs": outs,
            }
        )

    tracker = GraphAliasTracker(nodes)
    storages = []
    for key, alloc_node in tracker.storage_to_allocator.items():
        last = tracker.storage_to_last_user.get(key)
        storages.append(
            {
                "nbytes": int(key.storage.nbytes()),
                "device": key.device.type,
                "alloc": node_idx[alloc_node],
                "alloc_op": alloc_node.op,
                "last": node_idx[last] if last is not None else None,
                "last_op": last.op if last is not None else None,
            }
        )

    out = _out_dir()
    with open(os.path.join(out, "graph_info.json"), "w") as f:
        json.dump({"nodes": node_rows, "storages": storages}, f)
    with open(os.path.join(out, "graph_code.py"), "w") as f:
        f.write(gm.code)
    print(
        f"[memplan] dumped graph info: {len(node_rows)} nodes, "
        f"{len(storages)} storages -> {out}",
        flush=True,
    )
    return gm
