# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Assign MinimalAsyncEP buffer sets after graph EP chunking."""

import torch
import torch.fx as fx


def _buffer_set_arg(node: fx.Node) -> int | None:
    if node.op != "call_function" or not isinstance(node.target, torch._ops.OpOverload):
        return None
    target = node.target
    if target.namespace != "minimal_async_ep":
        return None
    for index, argument in enumerate(target._schema.arguments):
        if argument.name == "buffer_set":
            return index
    return None


def assign_minimal_async_ep_buffer_sets_pass(
    gm: fx.GraphModule, example_inputs=None
) -> fx.GraphModule:
    """Set each MinimalAsyncEP launch op's buffer set to its chunk ID.

    Graph chunking duplicates a trace whose launches all select buffer set zero.
    The duplicated chunks need distinct symmetric-memory receive buffers before
    the overlap scheduler can reorder them.
    """
    del example_inputs

    assigned = 0
    for node in gm.graph.nodes:
        index = _buffer_set_arg(node)
        if index is None:
            continue
        chunk_id = node.meta.get("chunk_id")
        if node.meta.get("chunked_region_role") != "body" or chunk_id not in (0, 1):
            raise ValueError(
                f"MinimalAsyncEP launch {node.name} must be a chunk body with "
                f"chunk_id 0 or 1, got role={node.meta.get('chunked_region_role')!r} "
                f"and chunk_id={chunk_id!r}."
            )

        args, kwargs = list(node.args), dict(node.kwargs)
        if "buffer_set" in kwargs or len(args) <= index:
            kwargs["buffer_set"] = chunk_id
        else:
            args[index] = chunk_id
        node.args, node.kwargs = tuple(args), kwargs
        assigned += 1

    if assigned:
        gm.graph.lint()
        gm.recompile()
    return gm
