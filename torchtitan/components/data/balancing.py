# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Cross-rank microbatch balancing for gradient accumulation.

Every microbatch of an optimizer step synchronizes all data-parallel ranks (FSDP
all-gathers and reduce-scatters run per microbatch), so each microbatch runs at
the speed of its slowest rank. With document masking, attention cost grows with
the sum of squared document lengths, so two equal-length packed rows can differ
several times in cost. The balancer reassigns a step's microbatches across
data-parallel ranks and accumulation slots so that each slot holds microbatches
of similar cost:

  1. each rank computes the cost of its local microbatches for the step;
  2. ranks all-gather the costs inside a balancing group (gloo, CPU);
  3. every rank computes the same plan: sort the group's S * G microbatches
     heaviest first and give accumulation slot g the next S of them, keeping a
     microbatch on its owner when that rank's slot is free;
  4. moved microbatches are exchanged in one all-to-all.

The step's set of microbatches is unchanged and the loss is normalized by the
step's global token count, so the gradient is unchanged up to summation order.

Balancing groups are built from the data-parallel mesh, so ranks that share
data along non-data-parallel axes (cp, tp) form separate, identical groups:
every cp and tp peer computes the same plan and receives the same microbatches.
"""

import io
import pickle
from abc import ABC, abstractmethod
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field

import torch
import torch.distributed as dist
from torch.distributed.device_mesh import DeviceMesh

from torchtitan.components.data.types import (
    TokenizedTrainingMicrobatch,
    TrainingMicrobatch,
)
from torchtitan.config import Configurable


HAS_PIN_MEMORY = torch.accelerator.is_available()

# plan[g][r] = (source rank index, source microbatch index) for slot g on rank r.
BalancePlan = list[list[tuple[int, int]]]


def document_attention_cost(microbatch: TrainingMicrobatch) -> float:
    """Sum of squared document lengths over non-padding tokens.

    Documents are maximal runs of consecutive positions, so a position reset
    starts a new document. Proportional to document-masked attention FLOPs.
    """
    if not isinstance(microbatch, TokenizedTrainingMicrobatch):
        raise ValueError(
            "document_attention cost requires TokenizedTrainingMicrobatch, got "
            f"{type(microbatch).__name__}"
        )
    positions = microbatch.positions.reshape(-1)
    valid = ~microbatch.padding_mask.reshape(-1)
    if positions.numel() == 0:
        return 0.0
    # A token starts a document if it is valid and does not continue the
    # previous valid token's position run.
    prev = torch.cat([positions.new_full((1,), -2), positions[:-1]])
    prev_valid = torch.cat([valid.new_zeros(1), valid[:-1]])
    starts = valid & ~(prev_valid & (positions == prev + 1))
    doc_ids = torch.cumsum(starts.to(torch.int64), dim=0)[valid]
    if doc_ids.numel() == 0:
        return 0.0
    lengths = torch.bincount(doc_ids).to(torch.float64)
    return float(torch.dot(lengths, lengths))


class MicrobatchCost(Configurable, ABC):
    """Estimated compute cost of one microbatch; only relative values matter."""

    @dataclass(kw_only=True, slots=True)
    class Config(Configurable.Config):
        pass

    @abstractmethod
    def __call__(self, microbatch: TrainingMicrobatch) -> float:
        ...


class DocumentAttentionCost(MicrobatchCost):
    """Sum of squared document lengths: document-masked attention FLOPs."""

    @dataclass(kw_only=True, slots=True)
    class Config(MicrobatchCost.Config):
        pass

    def __init__(self, config: Config) -> None:
        del config

    def __call__(self, microbatch: TrainingMicrobatch) -> float:
        return document_attention_cost(microbatch)


class VisionEncoderCost(MicrobatchCost):
    """Vision-encoder plus document-attention cost of a multimodal microbatch.

    ``cost = patch_weight * num_patches
           + patch_attention_weight * sum(segment_patches ** 2)
           + document_attention_weight * sum(document_length ** 2)``

    Patches come from ``grid_thw`` / ``grid_thw_videos`` in ``model_kwargs``
    (rows of [T, H, W] in patches). Encoders that attend within each temporal
    frame of an item (Qwen3.5) use one attention segment of H * W patches per
    frame, so video cost is linear in duration. With block-diagonal attention
    the quadratic term dominates for large images: with the Qwen3.5-9B weights
    below, one 4096x4096 image (65,536 16-px patches) costs about as much as
    900 images of 1,024 patches.

    The weights are relative and depend on the model and hardware; fit them from
    a profile (encoder time per microbatch against patches and sum of squared
    segments). Qwen3.5-9B on GB300, in seconds: ``patch_weight=8.6e-6``,
    ``patch_attention_weight=2.2e-9``, encoder fit R^2 = 0.9998.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(MicrobatchCost.Config):
        patch_weight: float
        """Cost per vision patch (encoder linear layers)."""
        patch_attention_weight: float
        """Cost per squared attention-segment length in patches."""
        document_attention_weight: float = 0.0
        """Cost per squared document length in tokens (LLM attention)."""

        def __post_init__(self) -> None:
            if min(self.patch_weight, self.patch_attention_weight) < 0:
                raise ValueError("vision cost weights must be non-negative")
            if self.document_attention_weight < 0:
                raise ValueError("document_attention_weight must be non-negative")

    def __init__(self, config: Config) -> None:
        self._config = config

    def __call__(self, microbatch: TrainingMicrobatch) -> float:
        if not isinstance(microbatch, TokenizedTrainingMicrobatch):
            raise ValueError(
                "VisionEncoderCost requires TokenizedTrainingMicrobatch, got "
                f"{type(microbatch).__name__}"
            )
        c = self._config
        cost = 0.0
        if c.document_attention_weight:
            cost += c.document_attention_weight * document_attention_cost(microbatch)
        for key in ("grid_thw", "grid_thw_videos"):
            grid = microbatch.model_kwargs.get(key)
            if grid is None:
                continue
            grid = grid.to(torch.float64).reshape(-1, 3)
            frame = grid[:, 1] * grid[:, 2]
            cost += c.patch_weight * float((grid[:, 0] * frame).sum())
            cost += c.patch_attention_weight * float((grid[:, 0] * frame * frame).sum())
        return cost


def plan_balanced_assignment(costs: Sequence[Sequence[float]]) -> BalancePlan:
    """Assign a step's microbatches to (slot, rank) pairs.

    Args:
        costs: costs[r][i] is the cost of rank r's i-th microbatch; every rank
            holds the same number G of microbatches.

    Returns:
        plan[g][r] = (src_rank, src_index): the microbatch rank r runs in slot g.
        The result depends only on ``costs``, so all ranks compute the same plan.
    """
    num_ranks = len(costs)
    num_slots = len(costs[0]) if num_ranks else 0
    if any(len(c) != num_slots for c in costs):
        raise ValueError("every rank must provide the same number of microbatches")
    # Heaviest first; ties broken by (rank, index) so the order is deterministic.
    order = sorted(
        ((r, i) for r in range(num_ranks) for i in range(num_slots)),
        key=lambda ri: (-costs[ri[0]][ri[1]], ri[0], ri[1]),
    )
    plan: BalancePlan = []
    for g in range(num_slots):
        chosen = order[g * num_ranks : (g + 1) * num_ranks]
        row: list[tuple[int, int] | None] = [None] * num_ranks
        moved = []
        for src, idx in chosen:
            if row[src] is None:
                row[src] = (src, idx)
            else:
                moved.append((src, idx))
        free = [r for r in range(num_ranks) if row[r] is None]
        for r, item in zip(free, moved, strict=True):
            row[r] = item
        plan.append(row)  # type: ignore[arg-type]
    return plan


def _serialize(microbatch: TrainingMicrobatch) -> torch.Tensor:
    buf = io.BytesIO()
    torch.save(microbatch, buf, pickle_protocol=pickle.HIGHEST_PROTOCOL)
    return torch.frombuffer(bytearray(buf.getbuffer()), dtype=torch.uint8)


def _deserialize(data: torch.Tensor) -> TrainingMicrobatch:
    # Microbatches come from peer ranks of this job, which run the same code.
    microbatch = torch.load(io.BytesIO(data.numpy().tobytes()), weights_only=False)
    if HAS_PIN_MEMORY and isinstance(microbatch, TokenizedTrainingMicrobatch):
        # Keep host-to-device copies asynchronous, like collated microbatches.
        for name in ("input", "labels", "positions", "padding_mask"):
            setattr(microbatch, name, getattr(microbatch, name).pin_memory())
        microbatch.model_kwargs = {
            k: v.pin_memory() if isinstance(v, torch.Tensor) else v
            for k, v in microbatch.model_kwargs.items()
        }
    return microbatch


class MicrobatchBalancer(Configurable):
    """Reassigns each optimizer step's microbatches across data-parallel ranks."""

    @dataclass(kw_only=True, slots=True)
    class Config(Configurable.Config):
        cost: MicrobatchCost.Config = field(
            default_factory=DocumentAttentionCost.Config
        )
        """Per-microbatch cost to balance."""
        group_size: int = 0
        """Data-parallel ranks per balancing group; 0 balances across the whole
        data-parallel mesh. Groups are consecutive data-parallel ranks, so with
        HSDP a group no larger than dp_shard stays within one replica."""

        def __post_init__(self) -> None:
            if self.group_size < 0:
                raise ValueError("group_size must be non-negative")

    def __init__(
        self,
        config: Config,
        *,
        dp_mesh: DeviceMesh,
        num_microbatches_per_step: int,
    ) -> None:
        if dp_mesh.ndim != 1:
            raise ValueError(
                f"MicrobatchBalancer expects a 1D data-parallel mesh, got {dp_mesh.ndim}D"
            )
        if num_microbatches_per_step < 1:
            raise ValueError("num_microbatches_per_step must be positive")
        dp_size = dp_mesh.size()
        group_size = config.group_size or dp_size
        if dp_size % group_size:
            raise ValueError(
                f"group_size={group_size} must divide the data-parallel degree {dp_size}"
            )
        self.num_microbatches_per_step = num_microbatches_per_step
        self._cost_fn = config.cost.build()

        # Every rank learns every data-parallel slice (one per coordinate of the
        # non-data-parallel axes), then all ranks create the same gloo groups in
        # the same order. Only ranks in one slice ever exchange microbatches.
        my_slice = dp_mesh.mesh.flatten().tolist()
        slices: list[list[int] | None] = [None] * dist.get_world_size()
        dist.all_gather_object(slices, my_slice)
        unique_slices = sorted({tuple(s) for s in slices if s is not None})
        groups = [
            list(s[start : start + group_size])
            for s in unique_slices
            for start in range(0, dp_size, group_size)
        ]
        group, _ = dist.new_subgroups_by_enumeration(groups, backend="gloo")
        assert isinstance(group, dist.ProcessGroup), "every rank is in one group"
        self._group: dist.ProcessGroup = group
        my_rank = dist.get_rank()
        self._group_ranks = next(g for g in groups if my_rank in g)
        self._me = self._group_ranks.index(my_rank)

    def balance(
        self, microbatches: list[TrainingMicrobatch]
    ) -> list[TrainingMicrobatch]:
        """Return this rank's microbatches for one step, in slot order."""
        num_slots = len(microbatches)
        if num_slots != self.num_microbatches_per_step:
            raise ValueError(
                f"expected {self.num_microbatches_per_step} microbatches, got {num_slots}"
            )
        num_ranks = len(self._group_ranks)
        local = torch.tensor(
            [self._cost_fn(mb) for mb in microbatches], dtype=torch.float64
        )
        gathered = [torch.empty_like(local) for _ in range(num_ranks)]
        dist.all_gather(gathered, local, group=self._group)
        plan = plan_balanced_assignment([g.tolist() for g in gathered])

        me = self._me
        sends: list[list[int]] = [[] for _ in range(num_ranks)]
        recvs: list[list[int]] = [[] for _ in range(num_ranks)]
        for g, row in enumerate(plan):
            for r, (src, idx) in enumerate(row):
                if src == r:
                    continue
                if src == me:
                    sends[r].append(idx)
                if r == me:
                    recvs[src].append(g)

        out: list[TrainingMicrobatch | None] = [None] * num_slots
        for g, row in enumerate(plan):
            src, idx = row[me]
            if src == me:
                out[g] = microbatches[idx]
        # Every rank derives the same move counts from the plan, so all ranks of
        # the group agree on whether, and with what shapes, to exchange.
        num_moves = torch.zeros(num_ranks, num_ranks, dtype=torch.int64)
        for row in plan:
            for r, (src, _) in enumerate(row):
                if src != r:
                    num_moves[src, r] += 1
        if num_moves.any():
            self._exchange(microbatches, sends, recvs, out, int(num_moves.max()))
        assert all(mb is not None for mb in out)
        return out  # type: ignore[return-value]

    def _exchange(
        self,
        microbatches: list[TrainingMicrobatch],
        sends: list[list[int]],
        recvs: list[list[int]],
        out: list[TrainingMicrobatch | None],
        max_moves: int,
    ) -> None:
        """All-to-all the moved microbatches: byte sizes first, then bytes."""
        num_ranks = len(self._group_ranks)
        payloads = [[_serialize(microbatches[i]) for i in dst] for dst in sends]
        size_send = torch.zeros(num_ranks, max_moves, dtype=torch.int64)
        for r, dst in enumerate(payloads):
            for k, p in enumerate(dst):
                size_send[r, k] = p.numel()
        size_recv = torch.empty_like(size_send)
        dist.all_to_all_single(size_recv, size_send, group=self._group)

        flat_send = torch.cat(
            [p for dst in payloads for p in dst] or [torch.empty(0, dtype=torch.uint8)]
        )
        in_splits = size_send.sum(dim=1).tolist()
        out_splits = size_recv.sum(dim=1).tolist()
        flat_recv = torch.empty(sum(out_splits), dtype=torch.uint8)
        dist.all_to_all_single(
            flat_recv,
            flat_send,
            output_split_sizes=out_splits,
            input_split_sizes=in_splits,
            group=self._group,
        )
        offset = 0
        for src in range(num_ranks):
            for k, g in enumerate(recvs[src]):
                n = int(size_recv[src, k])
                out[g] = _deserialize(flat_recv[offset : offset + n])
                offset += n


def balanced_step_iterator(
    iterator: Iterator[TrainingMicrobatch], balancer: MicrobatchBalancer
) -> Iterator[TrainingMicrobatch]:
    """Yield microbatches from ``iterator`` balanced one optimizer step at a time.

    Balancing runs when the trainer requests a step's first microbatch, so no
    microbatch is held between steps and dataloader checkpoints taken at step
    boundaries stay exact.
    """
    while True:
        step = []
        for _ in range(balancer.num_microbatches_per_step):
            try:
                step.append(next(iterator))
            except StopIteration:
                return
        yield from balancer.balance(step)
