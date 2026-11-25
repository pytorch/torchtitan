# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""How each trainer rank gets its microbatches: the DP replica's leader packs its column of the
step plan, and broadcasts each packed microbatch group to the replica's other ranks over gloo.

    controller --plan column + the rollouts it names, packed--> leader (pipeline stage 0, CP and TP rank 0)
    leader, reader thread:  unpack + materialize -> pack group i -> broadcast -> preprocess_images
    other ranks, reader thread:                         receive   -> preprocess_images
    every rank, main thread: engine.forward_backward(reader), one group per pipeline step

Every rank of a replica trains on the same microbatches, so packing once per replica instead
of once per rank saves the pp * cp * tp - 1 other ranks' CPU, and they need neither the rollouts
nor the builder. Image files travel packed; every rank turns them into pixel values itself.

Rollout groups that stay in their rollout workers (`BaseRolloutWorker.Config.rdma_min_group_bytes`)
reach the leader as handles; before packing the first microbatch group, the leader reads the
regions its column names from every worker at once (`read_packed_rollout_group_handles`).

When the next step's plan is already queued as this step's forward/backward starts (off-policy
runs), the controller starts the next step's readers right after it (`MicrobatchGroupReader.start`),
so the leader's reads, materialize and first packs overlap the optimizer step instead of delaying
the next forward/backward.
"""

from __future__ import annotations

import contextvars
import itertools
import time
from collections import deque
from collections.abc import Iterator
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch.distributed as dist

from torchtitan.observability import structured_logger as sl
from torchtitan.rl.rollout.packed_group import unpack_rollout_group
from torchtitan.rl.rollout.packed_group_handle import read_packed_rollout_group_handles

if TYPE_CHECKING:
    from torchtitan.rl.rollout.packed_group import PackedRolloutGroup
    from torchtitan.rl.rollout.packed_group_handle import PackedRolloutGroupHandle
    from torchtitan.rl.rollout.types import RolloutTurnID
    from torchtitan.rl.torch_transport_rdma import RDMAByteBuffers
    from torchtitan.rl.trainer.batcher import MicrobatchPacker, TrainingMicrobatch
    from torchtitan.rl.trainer.training_sample_builder import (
        TrainingSample,
        TrainingSampleBuilder,
        TrainingSampleSpec,
    )

# Groups made ahead of the one the engine waits on. A rank holds at most 4 microbatch groups
# on the host: these 2, the one waited on, and the previous one, still in the engine's loop.
_NUM_MICROBATCH_GROUPS_READ_AHEAD = 2


def _new_read_pool() -> ThreadPoolExecutor:
    """The one thread that makes a pass's microbatch groups."""
    return ThreadPoolExecutor(max_workers=1, thread_name_prefix="microbatch_group_read")


@dataclass(frozen=True, slots=True)
class ReplicaColumn:
    """What a DP replica's leader packs: its column of the step plan and the rollouts it names.

    Example:
        # DP rank 1 of 2, one microbatch per group; its specs come from rollouts 0 and 2 of
        # group 7 (by value) and rollout 1 of group 9 (in its rollout worker)
        ReplicaColumn(
            microbatch_groups=[[[spec_7_0, spec_7_2]], [[spec_9_1]]],
            # The controller stores each trainable group's packed bytes (`pack_rollout_group`).
            rollout_groups=[packed_group_7.select({0, 2})],
            rollout_group_handles=[handle_9.select({1})],
        )
    """

    microbatch_groups: list[list[list[TrainingSampleSpec]]]
    """[num_groups][num_microbatches_per_group] -> the microbatch's specs, in packing order:
    column d of `StepPlan.microbatch_groups`."""

    rollout_groups: list[PackedRolloutGroup]
    """Each group the column names, packed, with only the rollouts it names
    (`PackedRolloutGroup.select`)."""

    rollout_group_handles: list[PackedRolloutGroupHandle]
    """Each group the column names that stays in its rollout worker: where the rollouts it names
    and their images sit there (`PackedRolloutGroupHandle.select`)."""


@dataclass(frozen=True, slots=True)
class _LeaderError:
    """What the leader broadcasts instead of a microbatch group when packing fails, so the
    other ranks raise instead of waiting for a group that never comes."""

    message: str


class MicrobatchGroupReader:
    """Yields this rank's microbatch groups for one step in order, made on a background thread
    up to two groups ahead of the one the caller waits on: packed and broadcast by the leader,
    received by the other ranks of the replica.

    Implements only what `TrainingEngine.forward_backward` uses: `len()` and one in-order
    pass. How much of the reading training hides is up to the engine. With
    `training.preprocess_microbatch_groups_lazily`, it reads each group just before that
    group's forward, so only the first group's wait (which includes materializing the whole
    column) is exposed. Without it (the default), it reads every group before the first
    forward, so all of the reading is exposed.

    Example:
        reader = MicrobatchGroupReader(
            num_microbatch_groups=2,
            replica_group=replica_group,  # None when the replica is this rank alone
            column=column,  # on the leader; None on the other ranks
            training_sample_builder=builder,
            microbatch_packer=packer,
            rdma_byte_buffers=None,  # the leader's buffers when its column carries handles
        )
        len(reader)  # 2
        for microbatch_group in reader:
            train(microbatch_group)

        lazily, 2 groups:
        leader thread:  pack 0, send 0 | pack 1, send 1 |
        other threads:  receive 0      | receive 1      |
        main threads:   wait           | train 0        | train 1
    """

    def __init__(
        self,
        *,
        num_microbatch_groups: int,
        replica_group: dist.ProcessGroup | None,
        column: ReplicaColumn | None,
        training_sample_builder: TrainingSampleBuilder,
        microbatch_packer: MicrobatchPacker,
        rdma_byte_buffers: RDMAByteBuffers | None,
    ) -> None:
        # The replica's rank 0 is its leader.
        assert column is not None or replica_group is not None
        self._num_microbatch_groups = num_microbatch_groups
        self._replica_group = replica_group
        # The leader's column. The reader keeps the rollout groups only until it materializes
        # them, so the caller must not hold `column` past this call.
        self._microbatch_groups = None if column is None else column.microbatch_groups
        self._rollout_groups = [] if column is None else list(column.rollout_groups)
        self._rollout_group_handles = (
            [] if column is None else list(column.rollout_group_handles)
        )
        self._training_sample_builder = training_sample_builder
        self._microbatch_packer = microbatch_packer
        # Reads the column's handles; set on leaders when groups go by RDMA.
        self._rdma_byte_buffers = rdma_byte_buffers
        # The leader's materialized samples, [group][microbatch] -> samples in packing order;
        # each group's leave as it is packed.
        self._samples_by_group: deque[list[list[TrainingSample]]] | None = None
        self._iterated = False
        # Set once the leader fails to pack, on every rank: later groups raise it without a
        # broadcast, so no rank waits for a group the others never send.
        self._leader_error: str | None = None
        # The pass's thread and reads, when `start` began them before the pass.
        self._started: tuple[
            ThreadPoolExecutor, deque[Future[list[TrainingMicrobatch]]]
        ] | None = None
        # Total times, hidden or not; final once a pass ends. `pack_s` and `read_s` (with
        # `read_bytes`) are the leader's, on every rank of the replica; `broadcast_s` is this
        # rank's.
        self.pack_s = 0.0
        self.read_s = 0.0
        self.read_bytes = 0
        self.broadcast_s = 0.0

    def __len__(self) -> int:
        return self._num_microbatch_groups

    def start(self) -> None:
        """Begin the pass's first groups now, on the background thread: the groups made ahead of
        the first one the engine waits on, as if the pass had started. Every rank of the replica
        must call it, since the leader's groups are broadcasts.

        Example:
            reader.start()        # during the previous step's optimizer step
            for group in reader:  # the first groups are made or under way
                train(group)
        """
        assert self._started is None and not self._iterated
        pool = _new_read_pool()
        reads: deque[Future[list[TrainingMicrobatch]]] = deque()
        for index in range(
            min(self._num_microbatch_groups, _NUM_MICROBATCH_GROUPS_READ_AHEAD)
        ):
            reads.append(self._submit(pool, index))
        self._started = (pool, reads)

    def close(self) -> None:
        """Stop a pass `start` began but no one iterated: cancel its queued reads and leave the
        running one to finish (as when a pass is closed early)."""
        if self._started is not None:
            pool, _ = self._started
            self._started = None
            pool.shutdown(wait=False, cancel_futures=True)

    def __iter__(self) -> Iterator[list[TrainingMicrobatch]]:
        # One pass: the leader drops each group's samples once it is packed.
        assert not self._iterated
        self._iterated = True

        def wait_for_microbatch_group(
            read: Future[list[TrainingMicrobatch]],
        ) -> list[TrainingMicrobatch]:
            # The span is the read time that training did not hide.
            with sl.log_trace_span("wait_for_microbatch"):
                return read.result()

        if self._started is None:
            pool, reads = _new_read_pool(), deque()
        else:
            (pool, reads), self._started = self._started, None
        try:
            for index in range(len(reads), self._num_microbatch_groups):
                reads.append(self._submit(pool, index))
                if len(reads) > _NUM_MICROBATCH_GROUPS_READ_AHEAD:
                    yield wait_for_microbatch_group(reads.popleft())
            while reads:
                yield wait_for_microbatch_group(reads.popleft())
        finally:
            # The pass ended, or the generator was closed (e.g. a failed step). Cancel the queued
            # reads, and don't wait for the running one: a rank receiving from a leader that
            # failed would wait out the gloo timeout.
            pool.shutdown(wait=False, cancel_futures=True)

    def _submit(
        self, pool: ThreadPoolExecutor, index: int
    ) -> Future[list[TrainingMicrobatch]]:
        # Copy the caller's context: in a bare thread an actor framework can start a second
        # client (Monarch does).
        return pool.submit(contextvars.copy_context().run, self._read, index)

    def _read(self, index: int) -> list[TrainingMicrobatch]:
        if self._leader_error is not None:
            raise RuntimeError(
                f"The DP replica's leader failed to pack a microbatch group: {self._leader_error}"
            )
        if self._microbatch_groups is None:
            microbatch_group = self._receive()
        else:
            try:
                microbatch_group = self._pack(index, self._microbatch_groups)
            except Exception as error:
                self._leader_error = f"{type(error).__name__}: {error}"
                self._send(_LeaderError(self._leader_error))
                raise
            # With the leader's packing and read times so far, so every rank reports them
            # (under PP, only the last stage reports, and the leader is in stage 0).
            self._send((microbatch_group, self.pack_s, self.read_s, self.read_bytes))
        # On this thread too, so it overlaps training when the engine reads lazily.
        return [microbatch.preprocess_images() for microbatch in microbatch_group]

    def _pack(
        self, index: int, microbatch_groups: list[list[list[TrainingSampleSpec]]]
    ) -> list[TrainingMicrobatch]:
        if self._samples_by_group is None:
            self._read_rollout_group_handles()
        start = time.perf_counter()
        if self._samples_by_group is None:
            samples = iter(self._materialize(microbatch_groups))
            self._samples_by_group = deque(
                [
                    [list(itertools.islice(samples, len(specs))) for specs in group]
                    for group in microbatch_groups
                ]
            )
        # Groups are packed in order, on one thread.
        assert len(self._samples_by_group) == len(microbatch_groups) - index
        microbatch_group = [
            self._microbatch_packer.pack_planned(microbatch_specs, samples)
            for microbatch_specs, samples in zip(
                microbatch_groups[index], self._samples_by_group.popleft(), strict=True
            )
        ]
        self.pack_s += time.perf_counter() - start
        return microbatch_group

    def _read_rollout_group_handles(self) -> None:
        """Read the regions the column's handles name from their rollout workers, all reads in
        flight together; the groups then unpack like those sent by value."""
        if not self._rollout_group_handles:
            return
        assert self._rdma_byte_buffers is not None
        start = time.perf_counter()
        self._rollout_groups.extend(
            read_packed_rollout_group_handles(
                self._rollout_group_handles, self._rdma_byte_buffers
            )
        )
        self.read_s += time.perf_counter() - start
        self.read_bytes += sum(
            handle.num_bytes for handle in self._rollout_group_handles
        )
        self._rollout_group_handles = []

    def _materialize(
        self, microbatch_groups: list[list[list[TrainingSampleSpec]]]
    ) -> list[TrainingSample]:
        """The column's samples, in column order. One rollout group at a time, dropping its
        bytes as soon as its samples exist, so the column's bytes and its samples (copies) are
        never all alive together."""
        specs = [
            spec
            for group in microbatch_groups
            for microbatch_specs in group
            for spec in microbatch_specs
        ]
        specs_by_group_id: dict[int, list[TrainingSampleSpec]] = {}
        for spec in specs:
            specs_by_group_id.setdefault(spec.rollout_id.group_id, []).append(spec)
        samples_by_id: dict[RolloutTurnID, TrainingSample] = {}
        rollout_groups, self._rollout_groups = self._rollout_groups, []
        rollout_groups.reverse()
        while rollout_groups:
            # The rollouts' arrays are views into the packed regions, so they go with them.
            rollout_group = unpack_rollout_group(rollout_groups.pop())
            group_specs = specs_by_group_id[rollout_group.group_id]
            samples = self._training_sample_builder.materialize(
                training_sample_specs=group_specs, rollout_group=rollout_group
            )
            del rollout_group
            for spec, sample in zip(group_specs, samples, strict=True):
                samples_by_id[spec.rollout_id] = sample
        return [samples_by_id.pop(spec.rollout_id) for spec in specs]

    def _send(
        self, value: tuple[list[TrainingMicrobatch], float, float, int] | _LeaderError
    ) -> None:
        if self._replica_group is None:
            return
        start = time.perf_counter()
        dist.broadcast_object_list([value], group=self._replica_group, group_src=0)
        self.broadcast_s += time.perf_counter() - start

    def _receive(self) -> list[TrainingMicrobatch]:
        start = time.perf_counter()
        received: list[
            tuple[list[TrainingMicrobatch], float, float, int] | _LeaderError | None
        ] = [None]
        dist.broadcast_object_list(received, group=self._replica_group, group_src=0)
        self.broadcast_s += time.perf_counter() - start
        value = received[0]
        if isinstance(value, _LeaderError):
            self._leader_error = value.message
            raise RuntimeError(
                f"The DP replica's leader failed to pack a microbatch group: {value.message}"
            )
        assert value is not None
        microbatch_group, self.pack_s, self.read_s, self.read_bytes = value
        return microbatch_group
