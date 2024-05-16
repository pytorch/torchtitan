# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# from torch.distributed.pipelining import Schedule1F1B, ScheduleGPipe
from typing import List, Optional

import torch.distributed as dist
from torch.distributed.pipelining import ScheduleGPipe

# imports related to local copy of Schedule1F1B with local fix
from torch.distributed.pipelining.PipelineSchedule import (
    PipelineScheduleSingle,
    sorted_batch_isend_irecv,
)
from torch.profiler import record_function

from torchtitan.logging_utils import logger


class Schedule1F1B(PipelineScheduleSingle):
    """
    The 1F1B schedule.
    Will perform one forward and one backward on the microbatches in steady state.
    """

    def _step_microbatches(
        self,
        arg_mbs: Optional[List] = None,
        kwarg_mbs: Optional[List] = None,
        target_mbs: Optional[List] = None,
        losses: Optional[List] = None,
    ):
        """
        Run one iteration of the pipeline schedule with list of microbatches.
        Will go through all the microbatches according to the 1F1B schedule.

        Args:
            microbatches: list of microbatch args.
        """
        arg_mbs, kwarg_mbs = self._check_inputs(arg_mbs, kwarg_mbs, target_mbs, losses)

        # forward for num_microbatches + backward for num_microbatches
        total_ops = self._n_microbatches * 2

        # Example, 4 GPUs, 8 microbatches
        # Stage 0: 6 warmup, 2 1f1b, 6 cooldown
        # Stage 1: 4 warmup, 4 1f1b, 4 cooldown
        # Stage 2: 2 warmup, 6 1f1b, 2 cooldown
        # Stage 3: 0 warmup, 8 1f1b, 0 cooldown
        # fwd only
        warmup_steps = min(
            self._n_microbatches,
            2 * (self._num_stages - self._stage.stage_index - 1),
        )
        # fwd + bwd
        main_1f1b_steps = self._n_microbatches - warmup_steps
        # bwd only
        cooldown_steps = total_ops - (warmup_steps + (2 * main_1f1b_steps))
        total_steps = warmup_steps + main_1f1b_steps + cooldown_steps
        logger.debug(
            f"Stage {self._stage.stage_index}: "  # noqa: G004
            f"Warmup steps: {warmup_steps}, "
            f"Main 1F1B steps: {main_1f1b_steps}, "
            f"Cooldown steps: {cooldown_steps}, "
            f"Total steps: {total_steps}"
        )

        # Delay send waits
        fwd_sends_to_wait: List[dist.Work] = []
        bwd_sends_to_wait: List[dist.Work] = []

        def is_forward_step(i):
            return i < self._n_microbatches

        def is_backward_step(i):
            return i >= warmup_steps and self._has_backward

        def is_1f1b_step(i):
            return is_forward_step(i) and is_backward_step(i)

        def is_warmup_step(i):
            return is_forward_step(i) and not is_backward_step(i)

        def is_cooldown_step(i):
            return not is_forward_step(i) and is_backward_step(i)

        def should_coalesce_fwd_send_bwd_recv(fwd_send_i):
            return is_1f1b_step(fwd_send_i) or (
                is_warmup_step(fwd_send_i) and is_cooldown_step(fwd_send_i + 1)
            )

        def should_coalesce_bwd_send_fwd_recv(bwd_send_i):
            # The backward send to prev stage should be coalesced with the fwd recv from the previous stage
            return bwd_send_i >= warmup_steps and is_1f1b_step(bwd_send_i + 1)

        # bwd chunk counter
        bwd_mb_index = 0
        self._stage._configure_data_parallel_mode(last_backward=False)
        for i in range(total_steps):
            if is_forward_step(i):
                with record_function(f"Forward {i}"):
                    ops = self._stage.get_fwd_recv_ops()
                    if should_coalesce_bwd_send_fwd_recv(i - 1):
                        ops.extend(self._stage.get_bwd_send_ops())

                    works = sorted_batch_isend_irecv(ops)
                    for work in works.values():
                        work.wait()

                    output = self._stage.forward_one_chunk(arg_mbs[i], kwarg_mbs[i])  # type: ignore[index]

                    if not should_coalesce_fwd_send_bwd_recv(i):
                        ops = self._stage.get_fwd_send_ops()
                        works = sorted_batch_isend_irecv(ops)
                        fwd_sends_to_wait.extend(works.values())

                self._maybe_compute_loss(self._stage, output, target_mbs, i)

            if is_backward_step(i):
                self._stage._configure_data_parallel_mode(
                    last_backward=(i == total_steps - 1)
                )
                with record_function(f"Backward {bwd_mb_index}"):
                    ops = self._stage.get_bwd_recv_ops()

                    if should_coalesce_fwd_send_bwd_recv(i - 1):
                        ops.extend(self._stage.get_fwd_send_ops())

                    works = sorted_batch_isend_irecv(ops)
                    for work in works.values():
                        work.wait()

                    loss = self._maybe_get_loss(self._stage, bwd_mb_index)
                    self._stage.backward_one_chunk(loss=loss)

                    if should_coalesce_bwd_send_fwd_recv(i):
                        # see Note: coalesced bwd-send/fwd-recv
                        ops = self._stage.get_bwd_send_ops()
                        works = sorted_batch_isend_irecv(ops)
                        bwd_sends_to_wait.extend(works.values())

                    bwd_mb_index += 1

        # Wait for all forward sends to finish
        for work in fwd_sends_to_wait:
            work.wait()

        # Wait for all backward sends to finish
        for work in bwd_sends_to_wait:
            work.wait()

        # Return losses if there is a container passed in
        self._update_losses(self._stage, losses)


def build_pipeline_schedule(job_config, parallel_dims, stage, loss_fn):
    if job_config.experimental.pipeline_parallel_schedule == "1f1b":
        schedule_class = Schedule1F1B
    elif job_config.experimental.pipeline_parallel_schedule == "gpipe":
        schedule_class = ScheduleGPipe
    else:
        raise NotImplementedError(
            f"{job_config.experimental.pipeline_parallel_schedule} is not implemented"
        )
    logger.info(
        f"Using pipeline schedule {job_config.experimental.pipeline_parallel_schedule}"
    )
    return schedule_class(
        stage,
        n_microbatches=parallel_dims.pp,
        loss_fn=loss_fn,
    )


def split_stage_fqns(fqns, split_points, stage_id):
    """Helper for splitting ordered list of layer names into layers per stage.

    split_points is a list of layer names, each layer will be the first layer in a stage
    """
    stages = []
    cur = []

    for name in fqns:
        if name in split_points:
            assert len(
                cur
            ), f"{name} is not a valid split point, do not specify the first layer of stage 0"
            stages.append(cur)
            cur = []
        cur.append(name)

    stages.append(cur)
    return stages[stage_id]
