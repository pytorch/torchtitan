# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from contextlib import contextmanager

import torch
from torch.distributed.tensor import DTensor, Replicate
from torch.utils._pytree import register_pytree_node, tree_map

from torchtitan.config import JobConfig


@contextmanager
def disable_compile(job_config: JobConfig):
    """Context manager to temporarily disable compilation."""
    original_value = job_config.compile.enable
    job_config.compile.enable = False
    try:
        yield
    finally:
        job_config.compile.enable = original_value


def parallelize_inputs(world_mesh, args, kwargs):
    def to_dtensor(tensor):
        if isinstance(tensor, torch.Tensor):
            return DTensor.from_local(tensor, world_mesh["tp"], [Replicate()])
        return tensor

    dt_args = tree_map(to_dtensor, args)

    # TODO: When using flex_attention, BlockMask would show up in kwargs,
    # and it's unclear how to convert it to DTensor. If I use to_dtensor,
    # it would fail with Dynamo Error: P2011360347
    # dt_kwargs = tree_map(to_dtensor, kwargs)

    dt_kwargs = kwargs

    return dt_args, dt_kwargs


def register_blockmask_pytree_node():
    from torch.nn.attention.flex_attention import BlockMask

    if BlockMask not in torch.utils._pytree.SUPPORTED_NODES:
        register_pytree_node(
            BlockMask,
            BlockMask._flatten,
            BlockMask._unflatten,
            flatten_with_keys_fn=BlockMask._flatten_with_keys,
            serialized_type_name="torch.nn.attention.flex_attention.BlockMask",
        )


def validate_flex_attention_annotation(joint_with_descriptors):
    """Verify user annotations show up in the graph."""
    for node in joint_with_descriptors.graph_module.graph.nodes:
        if node.target in {
            torch.ops.higher_order.flex_attention,
            torch.ops.higher_order.flex_attention_backward,
        }:
            assert "compile_with_inductor" in node.meta.get("custom", {})
