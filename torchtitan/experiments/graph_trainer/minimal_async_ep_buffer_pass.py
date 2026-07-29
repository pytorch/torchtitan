# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Assign MinimalAsyncEP buffer sets after graph EP chunking."""

from __future__ import annotations

import torch
import torch.fx as fx

from torchtitan.tools.logging import logger


def _buffer_set_arg(node: fx.Node) -> tuple[str, int] | None:
    if node.op != "call_function" or not isinstance(node.target, torch._ops.OpOverload):
        return None
    target = node.target
    if target.namespace != "minimal_async_ep":
        return None
    for index, argument in enumerate(target._schema.arguments):
        if argument.name == "buffer_set":
            return target._schema.name.rsplit("::", 1)[-1], index
    return None


def _set_buffer_set(node: fx.Node, index: int, buffer_set: int) -> None:
    args = list(node.args)
    kwargs = dict(node.kwargs)
    if "buffer_set" in kwargs:
        kwargs["buffer_set"] = buffer_set
    elif len(args) > index:
        args[index] = buffer_set
    else:
        kwargs["buffer_set"] = buffer_set
    node.args = tuple(args)
    node.kwargs = kwargs


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
        op_arg = _buffer_set_arg(node)
        if op_arg is None:
            continue
        name, index = op_arg
        if node.meta.get("chunked_region_role") != "body":
            raise ValueError(
                f"MinimalAsyncEP launch {node.name} ({name}) is not a chunk body "
                "node under EP overlap; every dispatch/combine launch in a "
                "chunked region must carry chunk metadata."
            )
        chunk_id = node.meta.get("chunk_id")
        if chunk_id not in (0, 1):
            raise ValueError(
                f"MinimalAsyncEP launch {node.name} ({name}) has invalid "
                f"chunk_id={chunk_id!r}; expected 0 or 1."
            )
        _set_buffer_set(node, index, chunk_id)
        assigned += 1

    if assigned:
        gm.graph.lint()
        gm.recompile()
        logger.info("Assigned MinimalAsyncEP buffer sets to %d launch op(s).", assigned)
    return gm
