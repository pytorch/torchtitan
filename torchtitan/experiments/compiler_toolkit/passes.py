# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Compiler passes for the compiler toolkit.

This module provides various compiler passes that can be applied to graph modules
during compilation. Passes can be selected and configured via job config.
"""

from typing import Any, Sequence

import torch
from torch._inductor.fx_passes.overlap_manual_scheduling import manual_overlap_bucketing
from torch._inductor.fx_passes.overlap_scheduling import schedule_overlap_bucketing
from torch.fx.passes.regional_inductor import regional_inductor
from torchtitan.experiments.compiler_toolkit.cudagraph import (
    CUDAGraphWrapper,
    get_static_input_indices,
)
from torchtitan.experiments.simple_fsdp.reshard_after_forward import (
    annotate_fsdp_all_gather,
)


def autobucketing_reordering_pass(
    gm: torch.fx.GraphModule, example_inputs=None
) -> torch.fx.GraphModule:
    """
    Apply autobucketing and reordering optimization.

    This pass applies schedule_overlap_bucketing with collective_bucketing enabled
    to optimize comm/compute overlap patterns in the graph.
    """
    schedule_overlap_bucketing(gm, collective_bucketing=True)
    gm.recompile()
    return gm


def transformer_block_bucketing_reordering_pass(
    gm: torch.fx.GraphModule, example_inputs, fsdp_manual_buckets
) -> torch.fx.GraphModule:
    """
    Apply aten-level manual bucketing and reordering optimization.
    """
    manual_overlap_bucketing(
        gm, module_bucket_plans=fsdp_manual_buckets, insert_overlap_deps=False
    )
    gm.recompile()
    return gm


def regional_inductor_pass(
    gm: torch.fx.GraphModule, example_inputs
) -> torch.fx.GraphModule:
    """
    Apply regional inductor compilation based on user annotation.
    """
    return regional_inductor(gm, example_inputs)


def cudagraph_pass(
    gm: torch.fx.GraphModule, example_inputs: Sequence[Any], is_forward: bool
) -> torch.fx.GraphModule:
    """
    Apply cudagraph.

    This pass wraps the forward function with cudagraph during compilation and does
    not record cudagraph until runtime.
    - For the first run, it will warm up operators such as nccl.
    - For the second run, it will record cudagraph and replay cudagraph.
    - For the following runs, it will replay cudagraph.
    """
    static_input_indices = get_static_input_indices(gm, is_forward)
    gm.forward = CUDAGraphWrapper(gm.forward, example_inputs, static_input_indices)
    return gm


def validate_flex_attn_annotation_pass(
    gm: torch.fx.GraphModule,
) -> torch.fx.GraphModule:
    """Verify user annotations show up in the graph."""
    for node in gm.graph.nodes:
        if node.target in {
            torch.ops.higher_order.flex_attention,
            torch.ops.higher_order.flex_attention_backward,
        }:
            assert "compile_with_inductor" in node.meta.get("custom", {})
    return gm


# Apply activation checkpointing on joint graph before partitioner
def fsdp_reshard_after_fwd_pass(
    gm: torch.fx.GraphModule, reshard_after_forward: bool
) -> torch.fx.GraphModule:
    # this pass implements simplefsdp's fsdp_reshard_after_forward behavior
    # when fsdp_reshard_after_forward set to True, it will annotate simple_fsdp AG
    #   to CheckpointPolicy.MUST_RECOMPUTE.
    # when fsdp_reshard_after_forward set to False, it will annotate simple_fsdp AG
    #   to CheckpointPolicy.MUST_SAVE.
    gm = annotate_fsdp_all_gather(gm, reshard_after_forward)
    gm.recompile()
    return gm


# Registry mapping pass names to pass functions
AVAILABLE_COMPILER_PASSES = {
    "autobucketing_reordering": autobucketing_reordering_pass,
    "transformer_block_bucketing": transformer_block_bucketing_reordering_pass,
    "regional_inductor": regional_inductor_pass,
    "cudagraph": cudagraph_pass,
}
