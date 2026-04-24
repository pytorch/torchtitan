# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any, Callable

import torch
from torch._functorch._aot_autograd.graph_compile import (
    _aot_stage2a_partition,
    _apply_tensorify_python_scalars,
)
from torch._functorch.aot_autograd import (
    AOTConfig,
    AOTGraphCapture,
    AOTState,
    JointWithDescriptors,
    OutputType,
    ViewAndMutationMeta,
    boxed_nop_preserve_node_meta,
    default_partition,
)


def partition_joint_with_descriptors(
    jd: JointWithDescriptors,
    *,
    partition_fn: Callable = default_partition,
    fw_compiler: Callable = boxed_nop_preserve_node_meta,
    bw_compiler: Callable = boxed_nop_preserve_node_meta,
) -> tuple[
    torch.fx.GraphModule,
    torch.fx.GraphModule,
    int,
    int,
    int,
    int,
    int,
    list[int],
    list[Any],
]:
    aot_state: AOTState = jd._aot_state
    aot_graph_capture: AOTGraphCapture = jd._aot_graph_capture
    # Update the AOTState with the provided compilers
    aot_state.aot_config.partition_fn = partition_fn
    aot_state.aot_config.fw_compiler = fw_compiler
    aot_state.aot_config.bw_compiler = bw_compiler
    aot_state.aot_config.inference_compiler = fw_compiler

    fx_g: torch.fx.GraphModule = aot_graph_capture.graph_module
    maybe_subclass_meta: Any = aot_graph_capture.maybe_subclass_meta
    fw_metadata: ViewAndMutationMeta = aot_state.fw_metadata
    aot_config: AOTConfig = aot_state.aot_config

    # AOTAutogradStage2a: Partition the graph into forward and backward graphs and
    # return the some metadata about the partitioning.

    _apply_tensorify_python_scalars(fx_g)

    (
        fw_module,
        bw_module,
        num_fw_outs_saved_for_bw,
        num_symints_saved_for_bw,
        _indices_of_inps_to_detach,
        adjusted_flat_args,
    ) = _aot_stage2a_partition(
        fx_g,
        aot_graph_capture.updated_flat_args,
        maybe_subclass_meta,
        fw_metadata,
        aot_config,
    )

    num_user_outputs = (
        len(
            [
                x
                for x in fw_metadata.output_info
                if x.output_type
                in (OutputType.non_alias, OutputType.alias_of_intermediate)
            ]
        )
        + fw_metadata.num_intermediate_bases
    )

    num_mutate_inputs = len(
        [x for x in fw_metadata.input_info if x.mutates_data or x.mutates_metadata]
    )
    num_params_buffers = aot_config.num_params_buffers
    return (
        fw_module,
        bw_module,
        num_params_buffers,
        num_user_outputs,
        num_mutate_inputs,
        num_fw_outs_saved_for_bw,
        num_symints_saved_for_bw,
        _indices_of_inps_to_detach,
        adjusted_flat_args,
    )
