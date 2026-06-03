# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Per-direction point-to-point communicators for pipeline parallelism.

Why
---
A single PP communicator serializes every send/recv in submission (FIFO) order.
Coalescing (``batch_isend_irecv`` -> one ``ncclGroupStart/End``) makes a *single*
mixed send+recv batch deadlock-free, but it does **not** remove ordering hazards
*across* batches: when stages issue their per-neighbour exchanges in different
relative orders -- pipeline skew, looped / V schedules (``ScheduleZBVZeroBubble``,
``ScheduleDualPipeV``), or skip connections -- the single-comm FIFO can form a
dependency cycle and deadlock even though every individual batch is coalesced.

``torch.distributed.pipelining`` today threads a single ``group`` through every
P2P op (``PipelineStage`` uses ``group=self.group`` for both directions), and the
schedules lean on ``_sorted_batch_p2p`` (sorting ops by peer *within* one call)
plus coalescing to stay deadlock-free. Neither addresses the cross-batch FIFO
cycle.

What
----
This module builds **two** communicators over the same PP ranks:

* ``down`` -- carries downstream transfers (data flowing ``r -> r+1``: a *send*
  at ``r``, a *recv* at ``r+1``); i.e. forward activations.
* ``up``   -- carries upstream transfers (data flowing ``r -> r-1``: a *send* at
  ``r``, a *recv* at ``r-1``); i.e. backward gradients.

Forward and backward P2P then run on independent NCCL streams and can never
block each other in a shared FIFO, which removes the cross-batch deadlock and
restores full-duplex bandwidth. This is the same direction-split Megatron-LM
uses.

It is opt-in behind the ``TORCHTITAN_PP_PER_DIRECTION_P2P`` feature flag. The
helpers here are the building block; wiring them into ``PipelineStage`` /
schedule actions is an upstream (pytorch) change.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import torch
import torch.distributed as dist
from torch.distributed import ProcessGroup

__all__ = [
    "FEATURE_FLAG_ENV",
    "per_direction_p2p_enabled",
    "PPDirectionGroups",
    "build_pp_direction_groups",
    "bidirectional_exchange",
]

FEATURE_FLAG_ENV = "TORCHTITAN_PP_PER_DIRECTION_P2P"


def per_direction_p2p_enabled() -> bool:
    """True if the per-direction PP P2P feature flag is set (``1/true/yes/on``)."""
    return os.environ.get(FEATURE_FLAG_ENV, "0").lower() in ("1", "true", "yes", "on")


@dataclass(frozen=True)
class PPDirectionGroups:
    """A pair of communicators over the PP ranks, split by transfer direction.

    ``down`` carries ``r -> r+1`` traffic, ``up`` carries ``r -> r-1`` traffic.
    Both groups contain the full set of PP ranks; using two distinct groups gives
    two distinct communicators (and thus two NCCL streams) so the two directions
    do not serialize against each other.
    """

    down: ProcessGroup
    up: ProcessGroup


def build_pp_direction_groups(pp_global_ranks: list[int]) -> PPDirectionGroups:
    """Create the down/up communicators over ``pp_global_ranks``.

    Collective: like any ``new_group`` call, **every** rank in the default group
    must call this with the same ``pp_global_ranks`` (non-members included), in
    the same order, so the two new communicators are created consistently.
    """
    # Two new_group calls -> two independent communicators over the same ranks.
    down = dist.new_group(ranks=pp_global_ranks)
    up = dist.new_group(ranks=pp_global_ranks)
    return PPDirectionGroups(down=down, up=up)


def bidirectional_exchange(
    groups: PPDirectionGroups,
    *,
    next_peer: int | None,
    prev_peer: int | None,
    send_next: torch.Tensor | None = None,
    recv_prev: torch.Tensor | None = None,
    send_prev: torch.Tensor | None = None,
    recv_next: torch.Tensor | None = None,
) -> None:
    """Issue forward+backward P2P as two per-direction coalesced batches.

    Downstream batch (on ``groups.down``): ``send_next -> next_peer`` and
    ``recv_prev <- prev_peer``. Upstream batch (on ``groups.up``):
    ``send_prev -> prev_peer`` and ``recv_next <- next_peer``.

    Peers are global ranks. ``None`` peers/tensors are skipped, so first/last
    pipeline stages (which lack a prev/next neighbour) work without special
    casing. Each batch is a mixed send+recv issued via ``batch_isend_irecv`` --
    so it still relies on backend coalescing for *intra*-batch safety; the point
    of the direction split is *inter*-batch safety.
    """
    down_ops: list[dist.P2POp] = []
    if send_next is not None and next_peer is not None:
        down_ops.append(
            dist.P2POp(dist.isend, send_next, next_peer, group=groups.down)
        )
    if recv_prev is not None and prev_peer is not None:
        down_ops.append(
            dist.P2POp(dist.irecv, recv_prev, prev_peer, group=groups.down)
        )

    up_ops: list[dist.P2POp] = []
    if send_prev is not None and prev_peer is not None:
        up_ops.append(dist.P2POp(dist.isend, send_prev, prev_peer, group=groups.up))
    if recv_next is not None and next_peer is not None:
        up_ops.append(dist.P2POp(dist.irecv, recv_next, next_peer, group=groups.up))

    # Issue both batches before waiting: they live on different communicators /
    # streams, so they overlap (full-duplex) instead of serializing.
    works: list[dist.Work] = []
    if down_ops:
        works += dist.batch_isend_irecv(down_ops)
    if up_ops:
        works += dist.batch_isend_irecv(up_ops)
    for w in works:
        w.wait()
