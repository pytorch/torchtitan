# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from collections import Counter
from contextlib import contextmanager
from contextvars import ContextVar
from typing import TYPE_CHECKING

import torch
import torch.distributed as dist

if TYPE_CHECKING:
    from torchtitan.distributed.nccl_ep import RoutingSlots


routing_capture_plan: ContextVar["NCCLEPRoutingCapturePlan | None"] = ContextVar(
    "nccl_ep_routing_capture_plan", default=None
)


class NCCLEPRoutingCapturePlan:
    """Reserve measured routing demand and own leases until graph teardown."""

    def __init__(self):
        self.demands: Counter[tuple["RoutingSlots", "torch.Size", str]] = Counter()
        self.captured: list[tuple["RoutingSlots", int, int]] = []
        self.prepared = False

    @contextmanager
    def observe(self):
        if not self.prepared:
            self.demands.clear()
        token = routing_capture_plan.set(self)
        try:
            yield
        finally:
            routing_capture_plan.reset(token)

    def run(self, fn, *args, **kwargs):
        with self.observe():
            return fn(*args, **kwargs)

    def prepare_for_capture(self, *args, **kwargs):
        self.prepare()

    def invoke(self, fn, *args, **kwargs):
        # Only leases captured during this call can belong to a capture that
        # failed; earlier captures are still live graphs and keep theirs.
        num_captured = len(self.captured)
        try:
            return fn(*args, **kwargs)
        except BaseException:
            if len(self.captured) > num_captured:
                self._release(self.captured[num_captured:])
                del self.captured[num_captured:]
                self.demands.clear()
                self.prepared = False
            raise

    def prepare(self):
        demands = list(self.demands.items())
        num_missing = []
        for (pool, shape, layout), num_leases in demands:
            with pool.lock:
                num_free = sum(
                    not leased
                    for routing, leased, _ in pool.slots
                    if routing.topk_idx.shape == shape and routing.layout == layout
                )
            num_missing.append(max(0, num_leases - num_free))
        if demands and dist.is_initialized():
            # Creating a handle is collective, but the free-slot count is
            # local (leases are freed by garbage collection), so every rank
            # creates the largest count any rank needs. Demand keys are
            # inserted in the same order on every rank.
            counts = torch.tensor(
                num_missing,
                dtype=torch.int64,
                device=torch.accelerator.current_accelerator(),
            )
            dist.all_reduce(counts, op=dist.ReduceOp.MAX, group=demands[0][0][0].group)
            num_missing = counts.tolist()
        for ((pool, shape, layout), _), num_create in zip(demands, num_missing):
            with pool.lock:
                template = next(
                    routing.topk_idx
                    for routing, _, _ in pool.slots
                    if routing.topk_idx.shape == shape and routing.layout == layout
                )
                for _ in range(num_create):
                    pool._create(template, layout)
        self.demands.clear()
        self.prepared = True

    @staticmethod
    def _release(captured):
        for pool, index, generation in captured:
            with pool.lock:
                if pool.generations[index] == generation:
                    pool.slots[index][2] = False
                    pool.release(index, generation)

    def release(self):
        self._release(self.captured)
        self.captured.clear()
        self.demands.clear()
        self.prepared = False


_buffer_owner: ContextVar["NCCLEPBufferOwner | None"] = ContextVar(
    "nccl_ep_buffer_owner", default=None
)


class NCCLEPBufferOwner:
    """Own dispatchers from initialization until graph-safe terminal cleanup."""

    def __init__(self):
        self.dispatchers = []

    @contextmanager
    def initialize(self):
        token = _buffer_owner.set(self)
        try:
            yield
        finally:
            _buffer_owner.reset(token)

    def close(self):
        for dispatcher in self.dispatchers:
            dispatcher.close_buffer()
        self.dispatchers.clear()


def get_buffer_owner():
    owner = _buffer_owner.get()
    if owner is None:
        raise ValueError(
            "NCCL EP requires an explicit native-buffer owner before initialization; "
            "use GraphTrainer or NCCLEPBufferOwner.initialize() and close the owner "
            "after graph teardown, before destroying the process group."
        )
    return owner
