# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Monarch actor adapter for the RL batcher."""

from __future__ import annotations

import dataclasses
import socket
from pathlib import Path

from monarch.actor import Actor, endpoint

from torchtitan.observability import structured_logger as sl
from torchtitan.observability.logging import init_logger
from torchtitan.rl.actors.monarch.rdma_microbatch import (
    create_cuda_context_to_silence_monarch_log,
    MonarchRDMAMicrobatch,
)
from torchtitan.rl.observability import metrics as m
from torchtitan.rl.rollout.types import RolloutGroup
from torchtitan.rl.trainer.batcher import Batcher, TrainerStepBatch


class _BatcherActorEndpoints:
    def __init__(
        self,
        config: Batcher.Config,
        *,
        num_tokens_per_microbatch_per_dp_rank: int,
        max_context_length: int,
        num_prompts_per_train_step: int,
        dp_degree: int,
        num_pp_microbatches: int = 1,
        pad_id: int,
        temperature: float,
        num_topk_logprobs: int = 0,
        output_dir: str,
        enable_structured_logging: bool,
        replay_routed_experts: bool = False,
    ) -> None:
        init_logger()
        sl.init_structured_logger(
            source="rl_batcher",
            output_dir=output_dir,
            rank=0,
            enable=enable_structured_logging,
        )
        if config.rdma_min_tensor_bytes is not None:
            create_cuda_context_to_silence_monarch_log()
        super().__init__(
            config,
            num_tokens_per_microbatch_per_dp_rank=num_tokens_per_microbatch_per_dp_rank,
            max_context_length=max_context_length,
            num_prompts_per_train_step=num_prompts_per_train_step,
            dp_degree=dp_degree,
            num_pp_microbatches=num_pp_microbatches,
            pad_id=pad_id,
            temperature=temperature,
            num_topk_logprobs=num_topk_logprobs,
            register_rdma_microbatch=MonarchRDMAMicrobatch.register,
            replay_routed_experts=replay_routed_experts,
        )
        self._gc_timer = m.GCTimer("batcher")
        self._loop_lag = m.EventLoopLag("batcher")

    @endpoint
    async def add_group(
        self, *, rollout_group: RolloutGroup
    ) -> tuple[TrainerStepBatch | None, bool]:
        """Plain `@endpoint`, not `@concurrent_endpoint`: calls run one at a time, so groups reach the
        stateful batcher in arrival order. `async` because Monarch rejects mixing sync and async endpoints."""
        batch, has_samples = super().add_group(rollout_group=rollout_group)
        if batch is not None:
            # This process's GC time and event-loop lag since the previous batch, and its memory
            # (mostly the batches it holds), logged with the step it trains. The controller's
            # `host_memory/*` scan sees only its own host, and the batcher may run on another.
            batch = dataclasses.replace(
                batch,
                metrics=[
                    *batch.metrics,
                    *self._gc_timer.flush(),
                    *self._loop_lag.flush(),
                    m.Metric("perf/batcher/rss_anon_gib", m.NoReduce(_rss_anon_gib())),
                ],
            )
        return batch, has_samples

    @endpoint
    async def maybe_release_oldest_batch(self) -> None:
        await super().maybe_release_oldest_batch()

    @endpoint
    async def get_hostname(self) -> str:
        return socket.gethostname()

    @endpoint
    async def sync_log_step(self, step: int) -> None:
        """Plain `@endpoint` like `add_group`, so the step is set before the next group's spans."""
        sl.set_step(step)


class BatcherActor(Actor, _BatcherActorEndpoints, Batcher):
    """Expose a :class:`Batcher` through Monarch endpoints, in its own process.

    Why not batch in the controller, whose busy event loop holds the GIL:
        on the loop thread:      blocks every rollout while it runs
        in `asyncio.to_thread`:  each GIL release waits up to 5 ms to get it back (the switch interval)

    Example:
        batcher = this_host().spawn_procs(per_host={"cpus": 1}).spawn(
            "batcher",
            BatcherActor,
            Batcher.Config(rdma_min_tensor_bytes=1024),
            num_prompts_per_train_step=2,
            **other_init_kwargs,  # see `_BatcherActorEndpoints.__init__`
        )
        await batcher.add_group.call_one(rollout_group=group0)  # -> (None, True)
        await batcher.add_group.call_one(rollout_group=group1)  # -> (batch of MonarchRDMAMicrobatches, True)
        batcher.maybe_release_oldest_batch.broadcast()  # once every trainer rank has read group1's batch
    """


def _rss_anon_gib() -> float:
    """This process's anonymous memory, read as `compute_host_memory_metrics` reads each process's."""
    rollup = Path("/proc/self/smaps_rollup").read_text()
    # "Anonymous:   123456 kB"
    rss_anon_kib = next(
        int(line.split()[1])
        for line in rollup.splitlines()
        if line.startswith("Anonymous:")
    )
    return rss_anon_kib / 2**20
