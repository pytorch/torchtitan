# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from .common import (
    get_input_generating_fns as get_input_generating_fns,
    get_shape_inference_fns as get_shape_inference_fns,
    ModelWithLoss as ModelWithLoss,
)
from .graph_multiplex import multiplex_fw_bw_graph as multiplex_fw_bw_graph
from .graph_partition import (
    partition_joint_with_descriptors as partition_joint_with_descriptors,
)
from .graph_pp_runner import (
    GraphCallables as GraphCallables,
    GraphMeta as GraphMeta,
    GraphPipelineStage as GraphPipelineStage,
    GraphPPRunner as GraphPPRunner,
    get_multiplexed_graph_callables as get_multiplexed_graph_callables,
    overlap_fw_bw as overlap_fw_bw,
    stage_backward_input as stage_backward_input,
    stage_backward_weight as stage_backward_weight,
    stage_forward as stage_forward,
    stage_full_backward as stage_full_backward,
    stage_reduce_grad as stage_reduce_grad,
    stage_reshard as stage_reshard,
    stage_unshard as stage_unshard,
)
from .split_di_dw_graph import split_di_dw_graph as split_di_dw_graph
from .split_fsdp_collectives import (
    split_fsdp_prefetch as split_fsdp_prefetch,
    split_fsdp_reduce_scatters_epilogue as split_fsdp_reduce_scatters_epilogue,
)
