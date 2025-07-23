# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# This schedule will eventually be upstreamed to PyTorch.

import copy
import csv
import itertools
import logging
import re
from abc import ABC, abstractmethod
from collections import Counter, defaultdict
from enum import Enum
from typing import Any, Callable, NamedTuple, Optional, Union

import torch
import torch.distributed as dist
from torch._dynamo import OptimizedModule
from torch.distributed.fsdp import FSDPModule, UnshardHandle
from torch.nn.modules.loss import _Loss
from torch.profiler import record_function

import torch
import torch.nn as nn
from torch.distributed.pipelining import PipelineStage, stage
from torch.distributed.pipelining.schedules import (
    _PipelineSchedule,
    get_schedule_class,
    PipelineScheduleSingle,
    ScheduleZBVZeroBubble,
    PipelineScheduleMulti,
    _Action,
    _ComputationType,
    _validate_schedule
)

from torch.distributed.pipelining._utils import generate_stage_to_rank_mapping, generate_rank_to_stage_mapping
from torch.distributed.pipelining.microbatch import merge_chunks, split_args_kwargs_into_chunks, TensorChunkSpec
from torch.distributed.pipelining.stage import _PipelineStageBase

from torch.distributed.pipelining._schedule_visualizer import visualize_schedule, get_schedule_ops, visualize_schedule

class DualPipeV(PipelineScheduleMulti):
    """
    The DualPipeV schedule. A more efficient schedule variant based on the 
    DualPipe schedule introduced by DeepSeek in https://arxiv.org/pdf/2412.19437
    """
    def __init__(
        self,
        stages: list[_PipelineStageBase],
        n_microbatches: int,
        loss_fn: Optional[Callable] = None,
        args_chunk_spec: Optional[tuple[TensorChunkSpec, ...]] = None,
        kwargs_chunk_spec: Optional[dict[str, TensorChunkSpec]] = None,
        output_merge_spec: Optional[Union[dict[str, Any], tuple[Any]]] = None,
        scale_grads: bool = True,
    ):
        self.pp_group_size = stages[0].group_size
        super().__init__(
            stages=stages,
            n_microbatches=n_microbatches,
            loss_fn=loss_fn,
            args_chunk_spec=args_chunk_spec,
            kwargs_chunk_spec=kwargs_chunk_spec,
            output_merge_spec=output_merge_spec,
            scale_grads=scale_grads,
        )
        self.stage_index_to_group_rank = generate_stage_to_rank_mapping(
            self.pp_group_size, self._num_stages, style="v"
        )
        for stage in self._stages:
            stage.stage_index_to_group_rank = self.stage_index_to_group_rank

        self.n_local_stages = len(stages)
        if self.n_local_stages != 2:
            raise ValueError(
                "ZBV requires exactly 2 stages per rank, but got "
                f"{self.n_local_stages}."
            )

        self.rank = stages[0].group_rank
        self.num_stages = stages[0].num_stages

        # 1. Create the pipeline_order (all ranks do this calculation)
        # This will be used to keep track of the current state of the entire pipeline
        # pipeline_order[rank] = [Action(computation_type, microbatch_index, stage_index), ...]
        self.pipeline_order: dict[int, list[Optional[_Action]]] = {}
        for rank in range(self.pp_group_size):
            rank_ops = self._calculate_single_rank_operations(rank)
            self.pipeline_order[rank] = rank_ops

    def _calculate_single_rank_operations(self, rank) -> list[Optional[_Action]]:
        actions = []
        counters = {}  # (stage_index, computation_type) -> mb_index
        weight_queue = []  # Queue of (stage_index, mb_index) for pending weight actions

        num_ranks = self.pp_group_size
        num_chunks = self._n_microbatches

        rank_to_stages = generate_rank_to_stage_mapping(num_ranks, num_ranks * 2, style="v")
        stage0_index, stage1_index = rank_to_stages[rank]
        
        def add_action(actions: list, stage_index: int, computation_type: _ComputationType, counters: dict, forward_stage: Optional[int] = None, backward_stage: Optional[int] = None):
            """Helper method to add an action with automatic microbatch index tracking."""
            if computation_type == _ComputationType.OVERLAP_F_B:
                assert forward_stage is not None and backward_stage is not None
                # Create overlapped forward+backward action with sub_actions
                forward_key = (forward_stage, _ComputationType.FORWARD)
                backward_key = (backward_stage, _ComputationType.FULL_BACKWARD)
                
                forward_mb = counters.get(forward_key, 0)
                backward_mb = counters.get(backward_key, 0)
                
                sub_actions = [
                    _Action(forward_stage, _ComputationType.FORWARD, forward_mb),
                    _Action(backward_stage, _ComputationType.FULL_BACKWARD, backward_mb)
                ]
                
                # Update counters for sub_actions
                counters[forward_key] = forward_mb + 1
                counters[backward_key] = backward_mb + 1
                
                # Handle FULL_BACKWARD counter updates
                input_key = (backward_stage, _ComputationType.BACKWARD_INPUT)
                weight_key = (backward_stage, _ComputationType.BACKWARD_WEIGHT)
                counters[input_key] = counters.get(input_key, 0) + 1
                counters[weight_key] = counters.get(weight_key, 0) + 1
                
                # Add the overlapped action
                actions.append(_Action(-1, _ComputationType.OVERLAP_F_B, None, sub_actions))
            else:
                # Regular single action
                key = (stage_index, computation_type)
                mb_index = counters.get(key, 0)
                actions.append(_Action(stage_index, computation_type, mb_index))
                counters[key] = mb_index + 1
                
                # If FULL_BACKWARD is used, also increment the separate BACKWARD_INPUT and BACKWARD_WEIGHT counters
                # so they skip this microbatch since it's already covered by FULL_BACKWARD
                if computation_type == _ComputationType.FULL_BACKWARD:
                    input_key = (stage_index, _ComputationType.BACKWARD_INPUT)
                    weight_key = (stage_index, _ComputationType.BACKWARD_WEIGHT)
                    counters[input_key] = counters.get(input_key, 0) + 1
                    counters[weight_key] = counters.get(weight_key, 0) + 1
                
                # If BACKWARD_INPUT is updated, add corresponding weight action to queue and increment FULL_BACKWARD counter
                if computation_type == _ComputationType.BACKWARD_INPUT:
                    # Add weight action to queue for later processing
                    weight_queue.append((stage_index, mb_index))
                    full_backward_key = (stage_index, _ComputationType.FULL_BACKWARD)
                    counters[full_backward_key] = counters.get(full_backward_key, 0) + 1

        def add_weight_action(actions: list, counters: dict):
            """Helper method to add a weight action from the queue."""
            if not weight_queue:
                return  # No pending weight actions, skip
            # Pop the oldest weight action from the queue
            actual_stage_index, weight_mb_index = weight_queue.pop(0)
            actions.append(_Action(actual_stage_index, _ComputationType.BACKWARD_WEIGHT, weight_mb_index))
            # Update the counter for the actual stage that was processed
            weight_key = (actual_stage_index, _ComputationType.BACKWARD_WEIGHT)
            counters[weight_key] = counters.get(weight_key, 0) + 1

        # Step 1: F0
        step_1 = (num_ranks - rank - 1) * 2
        for _ in range(step_1):
            add_action(actions, stage0_index, _ComputationType.FORWARD, counters)

        # Step 2: F0F1
        step_2 = rank + 1
        for _ in range(step_2):
            add_action(actions, stage0_index, _ComputationType.FORWARD, counters)
            add_action(actions, stage1_index, _ComputationType.FORWARD, counters)

        # Step 3: I1W1F1 (Use zero bubble)
        step_3 = num_ranks - rank - 1
        for _ in range(step_3):
            add_action(actions, stage1_index, _ComputationType.BACKWARD_INPUT, counters)
            add_weight_action(actions, counters)
            add_action(actions, stage1_index, _ComputationType.FORWARD, counters)

        # Step 4 (Main step): F0B1-F1B0 (combined, overlapped forward+backward)
        step_4 = num_chunks - num_ranks * 2 + rank + 1
        for i in range(step_4):
            if i == 0 and rank == num_ranks - 1:
                # NOTE: We don't overlap these two chunks to further reduce bubble size.
                add_action(actions, stage0_index, _ComputationType.FORWARD, counters)
                add_action(actions, stage1_index, _ComputationType.FULL_BACKWARD, counters)
            else:
                add_action(actions, -1, _ComputationType.OVERLAP_F_B, counters, forward_stage=stage0_index, backward_stage=stage1_index)
            add_action(actions, -1, _ComputationType.OVERLAP_F_B, counters, forward_stage=stage1_index, backward_stage=stage0_index)

        # Step 5: B1-F1B0
        step_5 = num_ranks - rank - 1
        for _ in range(step_5):
            add_action(actions, stage1_index, _ComputationType.FULL_BACKWARD, counters)
            add_action(actions, -1, _ComputationType.OVERLAP_F_B, counters, forward_stage=stage1_index, backward_stage=stage0_index)

        # Step 6: B1B0 (The second half of the chunks use zero bubble)
        step_6 = rank + 1
        enable_zb = False
        for i in range(step_6):
            if i == step_6 // 2 and rank % 2 == 1:
                enable_zb = True
            comp_type = _ComputationType.BACKWARD_INPUT if enable_zb else _ComputationType.FULL_BACKWARD
            add_action(actions, stage1_index, comp_type, counters)
            if i == step_6 // 2 and rank % 2 == 0:
                enable_zb = True
            comp_type = _ComputationType.BACKWARD_INPUT if enable_zb else _ComputationType.FULL_BACKWARD
            add_action(actions, stage0_index, comp_type, counters)

        # Step 7: W0B0
        step_7 = num_ranks - rank - 1
        for _ in range(step_7):
            add_weight_action(actions, counters)
            comp_type = _ComputationType.BACKWARD_INPUT if enable_zb else _ComputationType.FULL_BACKWARD
            add_action(actions, stage0_index, comp_type, counters)

        # Step 8: W0
        step_8 = rank + 1
        for _ in range(step_8):
            add_weight_action(actions, counters)

        return actions

if __name__ == "__main__":
    ops = get_schedule_ops(DualPipeV, 4, 10, 2)
    
    # Add None padding for visualization: rank i gets i None values at the beginning
    padded_ops = []
    for rank, op_list in enumerate(ops):
        padded_list = [None] * rank + op_list
        padded_ops.append(padded_list)
        
    for op_list in padded_ops:
        print(op_list)
    actions = {i: op for i, op in enumerate(ops)}
    _validate_schedule(actions, 4, 8, 10)
    visualize_schedule(padded_ops, "dualpipev.png")
