# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
"""Transport-agnostic gather plan for selective K/V gather.

The plan answers "which K/V blocks, owned by which CP rank, does this rank
need" as a flat, deduplicated, capacity-padded table. It is independent of the
transport and of the kernel that consumes it -- the same plan drives the
intra-node and inter-node legs.

A "block" is ``block_numel`` contiguous elements of the K (or V) shard, i.e.
``block_size_tokens * num_kv_heads * head_dim``. Block ids are local to the
owning rank: global position is affine, ``rank * blocks_per_rank + local_block``,
so no rank/offset lookup table is needed (both producers shard affinely).
"""

from dataclasses import dataclass

import torch
import torch.distributed as dist


@dataclass(frozen=True, eq=False)
class BlockGatherPlan:
    """Which blocks to pull from which ranks, per batch element.

    ``eq=False``: the generated ``__eq__`` would compare the tensor fields
    elementwise and raise.

    Fields (all int32 CUDA tensors unless noted):
        block_numel: elements per block (Python int; a block is the transport
            granularity, ``block_size_tokens * num_kv_heads * head_dim``).
        src_rank: ``(B, capacity)`` owning CP rank of each needed block; padded
            entries hold ``-1``.
        src_block: ``(B, capacity)`` local block id on ``src_rank``; padding
            ``-1``.
        num_valid: ``(B,)`` number of real entries per batch; the rest of
            ``capacity`` is padding (kept fixed for shape stability).

    Invariant: within a batch the valid ``(src_rank, src_block)`` pairs are
    unique (no block requested twice). All builders guarantee this and the
    backward relies on it -- a duplicate would send and sum a block's gradient
    twice, double-counting the reduction. ``build_plan_metadata`` checks it, so
    a hand-built plan is caught at setup.

    Destination position is affine and derived by the consumer:
    ``dst_block = src_rank * blocks_per_rank + src_block`` for the full-sized
    (drop-in all-gather) layout. A compacted layout would carry an explicit
    ``dst_block`` instead.
    """

    block_numel: int
    src_rank: torch.Tensor
    src_block: torch.Tensor
    num_valid: torch.Tensor

    @property
    def batch_size(self) -> int:
        return self.src_rank.shape[0]

    @property
    def capacity(self) -> int:
        return self.src_rank.shape[1]


def full_plan(
    batch_size: int,
    cp_size: int,
    blocks_per_rank: int,
    block_numel: int,
    device: torch.device,
) -> BlockGatherPlan:
    """Plan that requests every block from every rank (== a plain all-gather).

    This is the verification baseline: gathering with this plan reproduces a
    plain all-gather of the shards, concatenated rank-major, per batch element
    -- bitwise (verified for B==1 in the tests). It also exercises the full
    capacity path (no padding), so the kernel is tested at its widest.
    """
    capacity = cp_size * blocks_per_rank
    ranks = torch.arange(cp_size, device=device, dtype=torch.int32)
    local = torch.arange(blocks_per_rank, device=device, dtype=torch.int32)
    # (cp_size, blocks_per_rank) -> flat (capacity,), same order as all-gather
    # concat (rank-major), then broadcast across the batch.
    src_rank = ranks.repeat_interleave(blocks_per_rank)
    src_block = local.repeat(cp_size)
    src_rank = src_rank.unsqueeze(0).expand(batch_size, capacity).contiguous()
    src_block = src_block.unsqueeze(0).expand(batch_size, capacity).contiguous()
    num_valid = torch.full((batch_size,), capacity, device=device, dtype=torch.int32)
    return BlockGatherPlan(block_numel, src_rank, src_block, num_valid)


def sliding_window_plan(
    cp_rank: int,
    cp_size: int,
    blocks_per_rank: int,
    block_numel: int,
    window_blocks: int,
    device: torch.device,
    include_own: bool = True,
) -> BlockGatherPlan:
    """Selective plan for contiguous causal sliding-window attention (B == 1).

    Contiguous sharding: rank ``r`` owns tokens ``[r*shard, (r+1)*shard)``. A
    causal window spanning ``window_blocks`` blocks means rank ``r``'s queries
    need their own blocks plus the last ``window_blocks`` blocks of rank
    ``r-1``. Rank 0 has no predecessor.

    ``include_own`` controls whether the rank's own blocks are part of the
    gather. With own blocks (default) the gathered buffer is self-contained;
    without them (``include_own=False``) only the remote transfer is planned --
    useful to isolate the transport that a plain all-gather actually saves
    (attention would read own K/V from the shard directly). Such a plan measures
    transport only; ``selective_gather`` rejects it.

    Capacity is fixed across ranks (rank 0 is padded with -1) so the kernel
    compiles to a single shape for the whole group.

    ``window_blocks`` must fit within one shard (``<= blocks_per_rank``): this
    builder only pulls from the immediately previous rank, so a window spanning
    more than one rank (needing blocks from ``r-2``) is not supported.
    """
    if not 0 <= window_blocks <= blocks_per_rank:
        raise ValueError(
            f"window_blocks ({window_blocks}) must be in [0, blocks_per_rank="
            f"{blocks_per_rank}]; this builder only supports a window that fits "
            "within the previous rank's shard (no spanning to r-2). A larger "
            "value yields negative block ids that collide with the -1 padding "
            "sentinel and wrap around during the P2P send."
        )
    own = blocks_per_rank if include_own else 0
    capacity = own + window_blocks
    src_rank = torch.full((1, capacity), -1, dtype=torch.int32, device=device)
    src_block = torch.full((1, capacity), -1, dtype=torch.int32, device=device)
    n = 0
    if include_own:
        # Own blocks -- a local read, no transfer.
        src_rank[0, :blocks_per_rank] = cp_rank
        src_block[0, :blocks_per_rank] = torch.arange(
            blocks_per_rank, dtype=torch.int32, device=device
        )
        n = blocks_per_rank
    if cp_rank > 0:
        # Last window_blocks of the previous rank -- the remote transfer.
        src_rank[0, n : n + window_blocks] = cp_rank - 1
        src_block[0, n : n + window_blocks] = torch.arange(
            blocks_per_rank - window_blocks,
            blocks_per_rank,
            dtype=torch.int32,
            device=device,
        )
        n += window_blocks
    num_valid = torch.tensor([n], dtype=torch.int32, device=device)
    return BlockGatherPlan(block_numel, src_rank, src_block, num_valid)


def _agree_plan_is_gatherable(
    plan: BlockGatherPlan, group, *, blocks_per_rank: int
) -> None:
    """Agree across ranks that every plan can be gathered, before gathering it.

    Checked here rather than locally: the plan collectives need the same dtype
    and shape everywhere, so a local raise would stop the bad rank while the
    valid ones wait in ``all_gather`` forever. This descriptor is fixed-size, so
    gathering it is always safe.

    The ids must be int32 because they are gathered and read as integers;
    casting instead would truncate a float plan into a different, possibly
    valid-looking one. ``block_numel`` and ``blocks_per_rank`` ride along
    because they set the P2P message size and bound the ids: ranks that disagree
    would validate against different bounds and then send different sizes.
    """
    dtypes_ok = all(
        getattr(plan, name).dtype == torch.int32
        for name in ("src_rank", "src_block", "num_valid")
    )
    shapes_ok = (
        plan.src_block.shape == plan.src_rank.shape
        and plan.num_valid.shape == (plan.batch_size,)
    )
    descriptor = torch.tensor(
        [
            int(dtypes_ok),
            int(shapes_ok),
            plan.batch_size,
            plan.capacity,
            plan.block_numel,
            blocks_per_rank,
        ],
        dtype=torch.int64,
        device=plan.src_rank.device,
    )
    gathered = [torch.empty_like(descriptor) for _ in range(group.size())]
    dist.all_gather(gathered, descriptor, group=group)
    rows = [t.tolist() for t in gathered]

    bad_dtype = [rank for rank, row in enumerate(rows) if not row[0]]
    if bad_dtype:
        raise ValueError(
            f"ranks {bad_dtype} have a plan whose src_rank, src_block, or "
            "num_valid is not int32."
        )
    bad_shape = [rank for rank, row in enumerate(rows) if not row[1]]
    if bad_shape:
        raise ValueError(
            f"ranks {bad_shape} have a plan whose src_block or num_valid does "
            "not match the shape of src_rank."
        )
    sizes = {tuple(row[2:]) for row in rows}
    if len(sizes) > 1:
        raise ValueError(
            "ranks disagree on (batch_size, capacity, block_numel, "
            f"blocks_per_rank): {sorted(sizes)}. They decide the message sizes "
            "and the layout, so every rank must use the same values."
        )


def _validate_plans(
    all_src_rank: list,
    all_src_block: list,
    *,
    all_num_valid: list,
    cp_size: int,
    blocks_per_rank: int,
) -> None:
    """Check every rank's plan once, at setup, before any gather runs.

    Every rank checks every plan, including the ``num_valid`` counts, so a
    malformed one fails the whole group instead of leaving the other ranks
    waiting in a collective. The P2P path turns these entries straight into
    sends and receives: an out-of-range id posts an operation no peer matches,
    and a duplicate pair sends and accumulates that block's gradient twice.
    """
    for owner, (ranks, blocks) in enumerate(
        zip(all_src_rank, all_src_block, strict=True)
    ):
        for batch, (rank_row, block_row) in enumerate(zip(ranks, blocks, strict=True)):
            seen = set()
            for src_rank, src_block in zip(rank_row, block_row, strict=True):
                if src_rank == -1 and src_block == -1:
                    continue
                if not 0 <= src_rank < cp_size or not 0 <= src_block < blocks_per_rank:
                    raise ValueError(
                        f"rank {owner} batch {batch} has entry (rank={src_rank}, "
                        f"block={src_block}); expected a rank below {cp_size} and "
                        f"a block below {blocks_per_rank}, or (-1, -1) padding."
                    )
                if (src_rank, src_block) in seen:
                    raise ValueError(
                        f"rank {owner} batch {batch} requests block {src_block} "
                        f"of rank {src_rank} twice; the backward would "
                        "double-count its gradient."
                    )
                seen.add((src_rank, src_block))
            if len(seen) != all_num_valid[owner][batch]:
                raise ValueError(
                    f"rank {owner} batch {batch} has {len(seen)} valid entries "
                    f"but num_valid says {all_num_valid[owner][batch]}."
                )


def _all_gather_int(t: torch.Tensor, group) -> list:
    cp = group.size()
    out = [torch.empty_like(t) for _ in range(cp)]
    dist.all_gather(out, t.contiguous(), group=group)
    return out


def _consumer_slot_map(all_src_rank: list, group, plan: BlockGatherPlan):
    """Pure transpose of the plan (no collective): from every rank's src_rank,
    compute this rank's (dst_slot, max_consumers, consumers). See
    ``backward_staging_map`` for field meanings."""
    cp = group.size()
    my = group.rank()
    device = plan.src_rank.device
    reads = [
        {int(x) for x in all_src_rank[R].unique().tolist() if x >= 0 and int(x) != R}
        for R in range(cp)
    ]
    consumers = {P: sorted(R for R in range(cp) if P in reads[R]) for P in range(cp)}
    max_consumers = max(1, max((len(v) for v in consumers.values()), default=0))

    sr = plan.src_rank.to(torch.int64)
    dst_slot = torch.zeros(
        plan.batch_size, plan.capacity, dtype=torch.int32, device=device
    )
    for P in range(cp):
        if my in consumers[P]:
            dst_slot[sr == P] = consumers[P].index(my)
    my_consumers = torch.tensor(consumers[my], dtype=torch.int32, device=device)
    return dst_slot, max_consumers, my_consumers


def backward_staging_map(plan: BlockGatherPlan, group, *, blocks_per_rank: int):
    """Consumer/slot map for the staging backward (transpose of the plan).

    The backward reduces, for each producer block, the grad contributions from
    every consumer that read it. To avoid atomics, each consumer writes into its
    own slot of the producer's staging buffer; this computes, per rank:

      * ``dst_slot`` -- ``(B, capacity)`` slot this rank writes on the producer
        of each plan entry (its index in that producer's sorted consumer list).
        0 for own / padding entries.
      * ``max_consumers`` -- global max consumers over all producers (staging
        depth; size the context with this).
      * ``consumers`` -- 1-D int32 tensor of this rank's consumer ranks (who
        read it), same as the forward reverse-ack ``dst_ranks``.

    All-gathers every rank's ``src_rank`` once; call at setup, not per step.
    ``blocks_per_rank`` is checked for agreement across ranks, as in
    ``build_plan_metadata``.
    """
    _agree_plan_is_gatherable(plan, group, blocks_per_rank=blocks_per_rank)
    all_sr = _all_gather_int(plan.src_rank, group)
    return _consumer_slot_map(all_sr, group, plan)


@dataclass(frozen=True, eq=False)
class PlanMetadata:
    """Everything both backends need, computed once at setup.

    ``plan`` is the plan this metadata was built from, and ``blocks_per_rank``
    is the bound its block ids were validated against. Every P2P path takes this
    object rather than a plan, so there is one validated source for both.

    ``group_ranks`` are the global ranks of the group the plans were gathered
    over; the backends check them against the context, since local rank ids only
    mean anything against the group they came from.

    ``ranks_missing_own`` lists the ranks whose plan does not name every block
    they own; it is empty for a plan the backward can use, and it is the same on
    every rank, so they all reject such a plan together instead of one raising
    while the others wait in a collective.

    ``all_src_rank`` / ``all_src_block`` are host-side nested lists (indexed
    ``[cp_rank][batch][entry]``) driving the P2P path: each rank builds its
    send/recv lists from every rank's plan, and keeping them on the host makes
    the per-step path sync-free. ``dst_slot`` / ``consumers`` (device tensors)
    are the plan transpose: the staging slot this rank writes on each producer,
    and the ranks that read this rank's blocks. ``max_consumers`` is the staging
    depth a backend sizes its buffer with.
    """

    plan: BlockGatherPlan
    blocks_per_rank: int
    group_ranks: tuple
    ranks_missing_own: tuple
    all_src_rank: list
    all_src_block: list
    dst_slot: torch.Tensor
    max_consumers: int
    consumers: torch.Tensor


def build_plan_metadata(
    plan: BlockGatherPlan, group, *, blocks_per_rank: int
) -> PlanMetadata:
    """Backend-agnostic setup: all-gather the plan once, derive the transpose.

    Use with the single ``selective_gather`` API -- size the context with
    ``metadata.max_consumers`` and pass the metadata every forward.
    ``blocks_per_rank`` is what bounds a valid block id; it is
    ``shard_numel // (batch_size * block_numel)``, the same value the context
    computes. It is kept on the metadata and rechecked against the context, so a
    bound that disagrees with the shard cannot slip through.

    Validates every rank's plan here, so the per-step path can trust the entries
    and stays free of device-to-host syncs. Every fact it checks or records is
    derived from the gathered plans, so all ranks reach the same verdict.
    """
    _agree_plan_is_gatherable(plan, group, blocks_per_rank=blocks_per_rank)
    all_sr = _all_gather_int(plan.src_rank, group)
    all_sb = _all_gather_int(plan.src_block, group)
    all_nv = _all_gather_int(plan.num_valid, group)
    dst_slot, max_consumers, consumers = _consumer_slot_map(all_sr, group, plan)
    # P2P builds its send/recv lists on the host; move the (static) plans to host
    # once here so the per-step path has no device-to-host sync.
    all_sr_host = [t.tolist() for t in all_sr]
    all_sb_host = [t.tolist() for t in all_sb]
    _validate_plans(
        all_sr_host,
        all_sb_host,
        all_num_valid=[t.tolist() for t in all_nv],
        cp_size=group.size(),
        blocks_per_rank=blocks_per_rank,
    )
    owned = set(range(blocks_per_rank))
    ranks_missing_own = tuple(
        owner
        for owner, (ranks, blocks) in enumerate(
            zip(all_sr_host, all_sb_host, strict=True)
        )
        if any(
            {b for r, b in zip(rank_row, block_row, strict=True) if r == owner} != owned
            for rank_row, block_row in zip(ranks, blocks, strict=True)
        )
    )
    return PlanMetadata(
        plan,
        blocks_per_rank,
        tuple(dist.get_process_group_ranks(group)),
        ranks_missing_own,
        all_sr_host,
        all_sb_host,
        dst_slot,
        max_consumers,
        consumers,
    )


@dataclass(frozen=True, eq=False)
class GINMetadata:
    """Per-rank push/own lists for the GIN (push) gather -- the plan transpose.

    GIN's device API is push-only, so a producer must know which of its blocks
    each consumer needs. That is the transpose of the plan, computed once from
    every rank's all-gathered ``(src_rank, src_block)``.

    All tensors are 1-D CUDA tensors on the plan's device -- int32, except the
    ``own_`` and ``copyin_`` fields, which are int64 (used for torch indexing).

    Forward gather:
        send_peer: consumer cp-rank for each remote block this rank pushes.
        send_src_block: this rank's local block id for each push (same length).
        send_batch: batch index for each push (same length).
        own_src_block / own_batch: this rank's own blocks (``src_rank == self``),
            copied locally into the gathered buffer (no network). int64 for
            torch indexing.
        copyin_src_block / copyin_batch: the UNIQUE local blocks this rank pushes
            (the only ones copy-in must stage into the registered window; a
            sliding-window plan pushes a few, so this avoids copying the whole
            shard). int64 for torch indexing.
        num_recv: remote blocks this rank receives per gather (the per-step
            increment of its forward GIN signal; the wait threshold accumulates).

    Backward gather (the transpose of the forward transpose):
        grad_send_peer: producer cp-rank this rank pushes a grad block to (one
            per remote entry in this rank's own plan), i.e. ``src_rank``.
        grad_send_src_block: the producer's local block id for that grad push.
        grad_send_batch: batch index for that grad push.
        grad_send_slot: this rank's staging slot on that producer (its index in
            the producer's sorted consumer list; from ``backward_staging_map``).
        max_consumers: global max consumers over all producers -- sizes the
            context's staging buffer (build the context with this).

    The forward ``send_peer`` count equals this rank's backward RECEIVE count
    (its consumers push grads back), so the backward wait accumulates that count.
    """

    send_peer: torch.Tensor
    send_src_block: torch.Tensor
    send_batch: torch.Tensor
    own_src_block: torch.Tensor
    own_batch: torch.Tensor
    num_recv: int
    grad_send_peer: torch.Tensor
    grad_send_src_block: torch.Tensor
    grad_send_batch: torch.Tensor
    grad_send_slot: torch.Tensor
    max_consumers: int
    copyin_src_block: torch.Tensor
    copyin_batch: torch.Tensor


def build_gin_metadata(plan: BlockGatherPlan, group) -> GINMetadata:
    """Setup-once transpose of the plan for the GIN push gather (fwd + bwd).

    All-gathers every rank's plan, then builds THIS rank's forward push list
    (blocks its consumers named from it), its own-block list, its receive count,
    and its backward grad-push list (grads for the remote blocks it read, with
    the staging slot on each producer). Host-side loops over the gathered plans,
    like ``p2p.all_gather_plans`` -- run at setup, not per step.
    """
    cp = group.size()
    me = group.rank()
    device = plan.src_rank.device
    all_sr = _all_gather_int(plan.src_rank.to(torch.int32), group)
    all_sb = _all_gather_int(plan.src_block.to(torch.int32), group)
    dst_slot, max_consumers, _ = _consumer_slot_map(all_sr, group, plan)
    dst_slot = dst_slot.tolist()
    B, cap = plan.batch_size, plan.capacity

    send_peer, send_src_block, send_batch = [], [], []
    for consumer in range(cp):
        if consumer == me:
            continue
        c_sr = all_sr[consumer].tolist()
        c_sb = all_sb[consumer].tolist()
        for b in range(B):
            for e in range(cap):
                if c_sr[b][e] == me:  # this consumer needs my block c_sb[b][e]
                    send_peer.append(consumer)
                    send_src_block.append(c_sb[b][e])
                    send_batch.append(b)

    own_src_block, own_batch = [], []
    grad_send_peer, grad_send_src_block, grad_send_batch, grad_send_slot = (
        [],
        [],
        [],
        [],
    )
    num_recv = 0
    my_sr = all_sr[me].tolist()
    my_sb = all_sb[me].tolist()
    for b in range(B):
        for e in range(cap):
            sr = my_sr[b][e]
            if sr == me:
                own_src_block.append(my_sb[b][e])
                own_batch.append(b)
            elif sr >= 0:
                num_recv += 1
                # Backward: push the grad for this remote block back to producer
                # sr, into this rank's staging slot dst_slot on that producer.
                grad_send_peer.append(sr)
                grad_send_src_block.append(my_sb[b][e])
                grad_send_batch.append(b)
                grad_send_slot.append(dst_slot[b][e])

    # Unique local blocks this rank pushes -- the only ones copy-in must stage.
    seen = set()
    copyin_batch, copyin_src_block = [], []
    for b, sb in zip(send_batch, send_src_block, strict=True):
        if (b, sb) not in seen:
            seen.add((b, sb))
            copyin_batch.append(b)
            copyin_src_block.append(sb)

    def _i32(xs):
        return torch.tensor(xs, dtype=torch.int32, device=device)

    def _i64(xs):
        return torch.tensor(xs, dtype=torch.int64, device=device)

    return GINMetadata(
        send_peer=_i32(send_peer),
        send_src_block=_i32(send_src_block),
        send_batch=_i32(send_batch),
        own_src_block=_i64(own_src_block),
        own_batch=_i64(own_batch),
        num_recv=num_recv,
        grad_send_peer=_i32(grad_send_peer),
        grad_send_src_block=_i32(grad_send_src_block),
        grad_send_batch=_i32(grad_send_batch),
        grad_send_slot=_i32(grad_send_slot),
        max_consumers=max_consumers,
        copyin_src_block=_i64(copyin_src_block),
        copyin_batch=_i64(copyin_batch),
    )
