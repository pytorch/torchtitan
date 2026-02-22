# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
import threading
from typing import cast, Optional

import torch
import torch.nn as nn
from torch import Tensor
from torch.distributed.pipelining.schedules import (
    _Action,
    _PipelineContext,
    _PipelineScheduleRuntime,
    _wait_batch_p2p,
)
from torch.distributed.pipelining.stage import _PipelineStageBase
from torch.distributed.tensor import DeviceMesh, distribute_module
from torch.profiler import record_function

from torchtitan.config import ActivationCheckpointConfig, ParallelismConfig
from torchtitan.distributed.expert_parallel import BaseExpertParallel
from torchtitan.tools.utils import get_device_info

from .parallel_dims import ParallelDims

"""
Below are optimizations related to pipeline parallelism with expert parallelism
"""


def get_dual_pipe_v_flag(
    *,
    parallelism: ParallelismConfig,
    ac_config: ActivationCheckpointConfig,
    parallel_dims: ParallelDims,
) -> bool:
    """
    Determine if DualPipeV should be enabled based on config and
    validates that incompatible features (EP + DualPipeV + AC) are not used together.
    """
    if not parallel_dims.ep_enabled or not parallel_dims.pp_enabled:
        return False

    dual_pipe_v = (
        parallelism.pipeline_parallel_expert_parallel_overlap
        and parallelism.pipeline_parallel_schedule.lower() == "dualpipev"
    )

    if dual_pipe_v and ac_config.mode != "none":
        raise NotImplementedError(
            "Expert Parallel with DualPipeV and Activation Checkpointing "
            "cannot be used together. Please disable one of them."
        )

    return dual_pipe_v


class DualPipeExpertParallel(BaseExpertParallel):
    """
    Wrapper that adds dual-pipe synchronization hooks to any BaseExpertParallel.
    Wraps dispatch/combine with sync hooks for overlapping EP communication
    with PP computation in DualPipe scheduling.

    The execution order becomes:
    A -> dispatch -> B -> module -> C -> combine -> D
    """

    def __init__(self, inner_ep: BaseExpertParallel):
        super().__init__()
        self.inner_ep = inner_ep

    def _partition_fn(self, name: str, mod: nn.Module, device_mesh: DeviceMesh) -> None:
        return self.inner_ep._partition_fn(name, mod, device_mesh)

    def _token_dispatch(
        self, mod: nn.Module, inputs: tuple, device_mesh: DeviceMesh
    ) -> tuple[Tensor, Tensor]:
        """A -> dispatch -> B"""
        inputs = (cast(Tensor, SyncHook.apply(inputs[0], "A")),) + inputs[1:]
        outputs = self.inner_ep._token_dispatch(mod, inputs, device_mesh)
        outputs = (cast(Tensor, SyncHook.apply(outputs[0], "B")),) + outputs[1:]
        return outputs

    def _token_combine(
        self, mod: nn.Module, routed_output: Tensor, device_mesh: DeviceMesh
    ) -> Tensor:
        """C -> combine -> D"""
        routed_output = cast(Tensor, SyncHook.apply(routed_output, "C"))
        combine_output = self.inner_ep._token_combine(mod, routed_output, device_mesh)
        combine_output = cast(Tensor, SyncHook.apply(combine_output, "D"))
        return combine_output

    def _apply(self, module: nn.Module, device_mesh: DeviceMesh) -> nn.Module:
        return distribute_module(
            module,
            device_mesh,
            partition_fn=self._partition_fn,
            # pyrefly: ignore [bad-argument-type]
            input_fn=self._token_dispatch,
            # pyrefly: ignore [bad-argument-type]
            output_fn=self._token_combine,
        )


class HookCoordinator:
    def __init__(self):
        # Barrier for 2 threads (forward and backward) to synchronize
        # This ensures that we always alternate at executing one compute and one comm op together
        self._execution_barrier = threading.Barrier(2)

        self._coordination_enabled = False
        self._cycle_count = 0
        self._num_layers = None

    def barrier(self):
        """Barrier for 2 threads to synchronize"""
        if not self.is_coordination_enabled():
            return

        try:
            self._execution_barrier.wait()
        except threading.BrokenBarrierError:
            pass

    def enable_coordination(self, num_layers: Optional[int] = None):
        if num_layers is not None and num_layers > 0:
            self._coordination_enabled = True
            self._cycle_count = 0

            # Reset barrier
            self._execution_barrier = threading.Barrier(2)
            self._num_layers = num_layers

    def disable_coordination(self):
        self._coordination_enabled = False
        self._cycle_count = 0
        self._execution_barrier.abort()  # Break barrier to unblock threads

    def check_should_continue_coordination(self):
        if self._num_layers is not None and self._cycle_count >= self._num_layers:
            return False
        return True

    def is_coordination_enabled(self):
        return self._coordination_enabled


# Global coordinator
_hook_coordinator = HookCoordinator()


class SyncHook(torch.autograd.Function):
    @staticmethod
    # pyrefly: ignore [bad-override]
    def forward(ctx, x, hook_name=""):
        ctx.hook_name = hook_name
        # handle edge case for transformer level boundary
        if _hook_coordinator._coordination_enabled and hook_name == "D":
            _hook_coordinator._cycle_count += 1
            if not _hook_coordinator.check_should_continue_coordination():
                _hook_coordinator.disable_coordination()
                return x

        _hook_coordinator.barrier()
        return x

    @staticmethod
    # pyrefly: ignore [bad-override]
    def backward(ctx, grad_output):
        hook_name = ctx.hook_name

        # Edge case, skip initial barrier, all subsequent backward hooks will acquire
        if hook_name == "D" and _hook_coordinator._cycle_count == 0:
            return grad_output, None

        _hook_coordinator.barrier()
        return grad_output, None


def _count_moe_modules(model):
    """Count MoE modules directly"""
    from torchtitan.models.common.moe import MoE

    moe_count = 0
    for _, module in model.named_modules():
        if isinstance(module, MoE):
            moe_count += 1
    return moe_count


device_type, device_module = get_device_info()


def overlap_callback(action: _Action, ctx: _PipelineContext):
    """
    Custom callback for OVERLAP_F_B computation that allows expert parallel communication
    and pipeline parallel computation to overlap.
    """
    schedule = ctx.schedule_ref
    assert isinstance(schedule, _PipelineScheduleRuntime)
    stage_index_to_stage: dict[int, _PipelineStageBase] = {
        stage.stage_index: stage for stage in schedule._stages
    }
    assert action.sub_actions is not None
    fwd_action = action.sub_actions[0]
    bwd_action = action.sub_actions[1]

    # Get stages
    forward_stage_index = fwd_action.stage_index
    forward_mb_index = fwd_action.microbatch_index
    assert forward_mb_index is not None
    backward_stage_index = bwd_action.stage_index
    backward_stage = stage_index_to_stage[backward_stage_index]

    # Forward setup
    arg_mbs = ctx.arg_mbs
    kwarg_mbs = ctx.kwarg_mbs
    assert arg_mbs is not None and kwarg_mbs is not None
    fwd_recv_ops = schedule.fwd_recv_ops
    forward_stage = stage_index_to_stage[forward_stage_index]
    forward_is_next_stage_on_this_rank = forward_stage_index + 1 in stage_index_to_stage
    forward_is_prev_stage_on_this_rank = forward_stage_index - 1 in stage_index_to_stage

    # Backward setup
    backward_is_next_stage_on_this_rank = (
        backward_stage.stage_index + 1 in stage_index_to_stage
    )
    backward_is_prev_stage_on_this_rank = (
        backward_stage.stage_index - 1 in stage_index_to_stage
    )
    backward_mb_index = bwd_action.microbatch_index
    assert backward_mb_index is not None
    bwd_recv_ops = schedule.bwd_recv_ops

    # Fwd receives
    if (
        not forward_stage.is_first
        # no recv op expected for V-schedule special case
        and not forward_is_prev_stage_on_this_rank
    ):
        assert (
            forward_stage_index,
            forward_mb_index,
        ) in fwd_recv_ops, f"Computing {action=} before receiving input"
        _wait_batch_p2p(fwd_recv_ops.pop((forward_stage_index, forward_mb_index)))

    # Bwd receives
    if (
        not backward_stage.is_last
        # no recv op expected for V-schedule special case
        and not backward_is_next_stage_on_this_rank
    ):
        assert (
            backward_stage_index,
            backward_mb_index,
        ) in bwd_recv_ops, f"Attempted to run compute {action=} before receiving input"
        _wait_batch_p2p(bwd_recv_ops.pop((backward_stage_index, backward_mb_index)))

    # We count num layers in case the stage layers differ
    # If they differ than we only want coordination to happen for the min amount of layers
    min_num_layers = min(
        _count_moe_modules(forward_stage.submod),
        _count_moe_modules(backward_stage.submod),
    )
    # PP computation ========================================================
    _hook_coordinator.enable_coordination(num_layers=min_num_layers)
    main_stream = torch.accelerator.current_stream(device_type)

    # Shared container for exception from backward thread
    def run_backward():
        # pyrefly: ignore [missing-attribute]
        schedule._assert_unsharded(backward_stage)
        # Set the backward thread to use the same stream as forward
        device_module.set_stream(main_stream)
        with record_function(
            f"backward_stage_{backward_stage_index}_mb_{backward_mb_index}"
        ):
            loss = schedule._maybe_get_loss(backward_stage, backward_mb_index)
            # pyrefly: ignore [missing-attribute]
            schedule.backward_counter[backward_stage_index] += 1
            last_backward = (
                # pyrefly: ignore [missing-attribute]
                schedule.backward_counter[backward_stage_index]
                == schedule._n_microbatches
            )
            backward_stage.backward_one_chunk(
                # pyrefly: ignore [bad-argument-type]
                backward_mb_index,
                loss=loss,
                full_backward=True,
                last_backward=last_backward,
            )

            if backward_is_prev_stage_on_this_rank:
                stage_index_to_stage[backward_stage_index - 1].set_local_bwd_input(
                    backward_stage.get_local_bwd_output(backward_mb_index),
                    # pyrefly: ignore [bad-argument-type]
                    backward_mb_index,
                )

    def run_forward():
        # pyrefly: ignore [missing-attribute]
        schedule._assert_unsharded(forward_stage)
        output = forward_stage.forward_one_chunk(
            # pyrefly: ignore [bad-argument-type]
            forward_mb_index,
            # pyrefly: ignore[bad-index, unsupported-operation]
            arg_mbs[forward_mb_index],
            # pyrefly: ignore[bad-index, unsupported-operation]
            kwarg_mbs[forward_mb_index],
        )
        schedule._maybe_compute_loss(
            forward_stage, output, ctx.target_mbs, forward_mb_index
        )
        if forward_is_next_stage_on_this_rank:
            stage_index_to_stage[forward_stage_index + 1].set_local_fwd_input(
                output,
                # pyrefly: ignore [bad-argument-type]
                forward_mb_index,
            )

    # Run forward and backward in parallel
    thread = threading.Thread(target=run_backward, daemon=True)
    thread.start()
    run_forward()
    thread.join()

    _hook_coordinator.disable_coordination()
