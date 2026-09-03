# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Static block-layout algebra for AttnRes under Interleaved1F1B.

Given a schedule shape ``(P, V, num_blocks, n_layers, layers_per_block)``
this module enumerates, offline and deterministically, which block each
stage commits, which blocks each rank's shared cache holds at every
virtual-stage entry, and which subset a stage must ship on its outgoing
P2P (the "delta"). The adapter reads these tables at runtime so no
metadata ever travels over the wire.
"""

from __future__ import annotations

import torch


class BlockLayoutTables:
    """Precomputed per-microbatch Interleaved1F1B block-propagation tables.

    Given the tuple ``(P, V, num_blocks, n_layers, layers_per_block)``, this
    helper simulates the full single-microbatch forward in the schedule's
    execution order and materializes deterministic lookups:

    * ``commits_at(S)``            -> list[int] of block indices stage ``S`` commits.
    * ``rank_cache_at_entry(R, v)``-> ``frozenset[int]`` of block indices held in
      rank ``R``'s cache at the moment its ``v``-th virtual stage calls forward.
    * ``delta_to_send(S)``         -> list[int] of block indices stage ``S``
      ships on its P2P send to stage ``S+1`` (``[]`` for the last stage).
    * ``producer_stage_of_block(b)`` -> int, the stage that commits block ``b``.
    * ``cache_consumers_of_block(b)`` -> list[int] of stages that pull block ``b``
      out of THEIR rank-cache (not via the delta buffer).

    A stage may commit more than one block: that happens whenever its layer span
    is wider than ``layers_per_block`` (e.g. 96 layers over P=2, V=2 with
    ``attn_res_block_size=12`` puts two boundaries on every stage). Everything
    here is keyed by the commit's index WITHIN its producer stage, and so is the
    runtime -- the rank cache stores ``(rank, stage, block_idx_in_producer)`` and
    the producer installs one augment hook per commit.

    Expected delta sizes for the canonical config
    ``(P=8, V=2, num_blocks=8, n_layers=16, layers_per_block=2)``:

    * v=0 hops: sizes = [1, 1, 2, 2, 3, 3, 4, 3]
    * v=1 hops: sizes = [4, 3, 4, 3, 4, 3, 4]
    """

    def __init__(
        self,
        *,
        pp_size: int,
        virtual_stages_per_rank: int,
        num_blocks: int,
        n_layers: int,
        layers_per_block: int,
        layer_to_stage: dict[int, int] | None = None,
    ) -> None:
        if pp_size < 1 or virtual_stages_per_rank < 1:
            raise ValueError("pp_size and virtual_stages_per_rank must be >= 1")
        if n_layers <= 0 or layers_per_block <= 0:
            raise ValueError("n_layers and layers_per_block must be positive")
        # A partial final block is legal: K3 uses attn_res_block_size=12 over
        # 93 layers (report sec 2.2), so the last block holds 9 layers and
        # never reaches a commit. num_blocks is therefore the CEIL.
        expected_blocks = -(-n_layers // layers_per_block)
        if num_blocks != expected_blocks:
            raise ValueError(
                f"num_blocks ({num_blocks}) must equal ceil(n_layers / "
                f"layers_per_block) = {expected_blocks} for n_layers="
                f"{n_layers}, layers_per_block={layers_per_block}"
            )

        self.P = pp_size
        self.V = virtual_stages_per_rank
        self.num_stages = pp_size * virtual_stages_per_rank
        self.num_blocks = num_blocks
        self.n_layers = n_layers
        self.layers_per_block = layers_per_block

        if layer_to_stage is None:
            if n_layers % self.num_stages != 0:
                raise ValueError(
                    f"Default layer_to_stage requires n_layers ({n_layers}) "
                    f"to be divisible by num_stages ({self.num_stages}). "
                    f"Pass an explicit layer_to_stage map."
                )
            layers_per_stage = n_layers // self.num_stages
            layer_to_stage = {ell: ell // layers_per_stage for ell in range(n_layers)}
        self._layer_to_stage = dict(layer_to_stage)

        self._commits_at: dict[int, list[int]] = {}
        self._producer_stage_of_block: dict[int, int] = {}
        self._cache_at_entry: dict[tuple[int, int], frozenset[int]] = {}
        self._delta_to_send: dict[int, list[int]] = {}

        self._build()

    # ----- public lookups ---------------------------------------------- #

    def commits_at(self, stage_id: int) -> list[int]:
        return list(self._commits_at.get(stage_id, ()))

    def rank_cache_at_entry(self, rank: int, v: int) -> frozenset[int]:
        return self._cache_at_entry[(rank, v)]

    def delta_to_send(self, stage_id: int) -> list[int]:
        return list(self._delta_to_send.get(stage_id, ()))

    def producer_stage_of_block(self, block_idx: int) -> int:
        return self._producer_stage_of_block[block_idx]

    def cache_consumers_of_block(self, block_idx: int) -> list[int]:
        """Stages that consume ``block_idx`` via their shared rank cache."""
        return list(self._cache_consumers_of_block.get(block_idx, ()))

    def expected_same_rank_captures(
        self,
        producer_stage: int,
        block_idx_in_producer: int,
    ) -> int:
        """Count of later same-rank virtual stages that read producer
        ``producer_stage``'s ``block_idx_in_producer``-th commit from
        their shared rank cache.

        Each such consumer triggers exactly one
        :class:`pipeline_adapter._LocalCacheCapture.backward` deposit
        into the producer's captured-grad slot for the current mb. The
        producer-side hook uses this count to turn silent grad loss
        (a consumer backward that never ran) into an explicit warning
        at the moment its own backward fires.
        """
        commits = self._commits_at.get(producer_stage, [])
        if block_idx_in_producer < 0 or block_idx_in_producer >= len(commits):
            return 0
        b = commits[block_idx_in_producer]
        producer_rank = producer_stage % self.P
        return sum(
            1
            for c in self._cache_consumers_of_block.get(b, [])
            if c % self.P == producer_rank and c > producer_stage
        )

    # ----- the full simulation ----------------------------------------- #

    def _build(self) -> None:
        # 1) commits_at / producer_stage_of_block from the layer map.
        for stage_id in range(self.num_stages):
            self._commits_at[stage_id] = []
        for ell in range(self.n_layers):
            if ell % self.layers_per_block != 0:
                continue
            block_idx = ell // self.layers_per_block
            stage_id = self._layer_to_stage[ell]
            self._commits_at[stage_id].append(block_idx)
            self._producer_stage_of_block[block_idx] = stage_id

        if len(self._producer_stage_of_block) != self.num_blocks:
            raise ValueError(
                "Internal: not all blocks have a producer stage. "
                f"Expected {self.num_blocks}, got "
                f"{len(self._producer_stage_of_block)}."
            )

        # 2) Walk the mb forward stage-by-stage and track each rank's cache.
        # Interleaved1F1B: rank R owns stages R, R+P, ..., R+(V-1)P; forward
        # runs stage 0 -> num_stages-1, matching the autograd graph.
        rank_cache: dict[int, set[int]] = {r: set() for r in range(self.P)}
        accumulated: set[int] = set()
        for r in range(self.P):
            self._cache_at_entry[(r, 0)] = frozenset()

        for stage_id in range(self.num_stages):
            R = stage_id % self.P
            v = stage_id // self.P
            self._cache_at_entry.setdefault((R, v), frozenset(rank_cache[R]))

            for b in self._commits_at[stage_id]:
                accumulated.add(b)
                rank_cache[R].add(b)
            # Receiver cached what it just saw on the wire.
            rank_cache[R].update(accumulated)

            next_stage = stage_id + 1
            if next_stage < self.num_stages:
                next_R = next_stage % self.P
                next_v = next_stage // self.P
                receiver_cache = frozenset(rank_cache[next_R])
                self._cache_at_entry[(next_R, next_v)] = receiver_cache
                delta = sorted(accumulated - receiver_cache)
                self._delta_to_send[stage_id] = delta
            else:
                self._delta_to_send[stage_id] = []

        # 3) cache_consumers_of_block: later stages reading a block from their
        # RANK CACHE rather than the delta buffer. Each such read deposits one
        # grad into the producer's slot; expected_same_rank_captures counts them.
        cache_consumers_of_block: dict[int, list[int]] = {
            b: [] for b in range(self.num_blocks)
        }
        for stage_id in range(self.num_stages):
            R = stage_id % self.P
            v = stage_id // self.P
            for b in self._cache_at_entry[(R, v)]:
                cache_consumers_of_block[b].append(stage_id)
        self._cache_consumers_of_block = {
            b: list(stages) for b, stages in cache_consumers_of_block.items()
        }


def infer_block_layout_tables_from_stages(
    stages,
    *,
    pp_size: int,
    num_blocks: int,
    n_layers: int,
    layers_per_block: int,
) -> BlockLayoutTables:
    """Build :class:`BlockLayoutTables` from live ``PipelineStage`` objects.

    The layout itself is the contiguous default (layer ``ell`` on stage
    ``ell // layers_per_stage``). ``stages`` holds only the local rank's stages,
    so a complete layer-id -> stage-id map is not obtainable here without a
    collective; what the local stages DO expose is used to verify the default
    instead. A non-contiguous split raises rather than producing a layout that
    is wrong in a way only the gradients would show.

    Stages that expose no ``layers`` attribute (CPU unit tests) leave nothing to
    verify, which is not an error.
    """
    num_local_stages = len(stages)
    if num_local_stages < 1:
        raise ValueError("need at least one stage to infer layout")
    # Under Interleaved1F1B ``pp_schedule._stages`` returns only the local
    # rank's stages, so ``len(stages) == V``.
    V = num_local_stages
    num_stages = pp_size * V

    layer_to_stage: dict[int, int] = {}
    for stage in stages:
        submod = getattr(stage, "submod", None)
        inner = getattr(submod, "wrapped", submod)
        layers = getattr(inner, "layers", None)
        if layers is None:
            continue
        stage_idx = getattr(stage, "stage_index", None)
        if stage_idx is None:
            continue
        for key in layers.keys():
            try:
                layer_id = int(key)
            except (TypeError, ValueError):
                continue
            layer_to_stage[layer_id] = stage_idx

    # Verify, do not adopt: the map above covers this rank's layers only.
    # BlockLayoutTables raises on its own if the layer count is not divisible,
    # and its message is the clearer one, so leave that case to it.
    if layer_to_stage and n_layers % num_stages == 0:
        layers_per_stage = n_layers // num_stages
        for layer_id, stage_idx in sorted(layer_to_stage.items()):
            expected = layer_id // layers_per_stage
            if stage_idx != expected:
                raise ValueError(
                    f"layer {layer_id} sits on stage {stage_idx}, but the "
                    f"contiguous layout this adapter assumes puts it on stage "
                    f"{expected} (n_layers={n_layers}, num_stages={num_stages}). "
                    "A non-contiguous pipeline split is not supported: the "
                    "cross-stage cache would route block deltas to the wrong "
                    "stages."
                )
    layer_to_stage = None  # type: ignore[assignment]

    return BlockLayoutTables(
        pp_size=pp_size,
        virtual_stages_per_rank=V,
        num_blocks=num_blocks,
        n_layers=n_layers,
        layers_per_block=layers_per_block,
        layer_to_stage=layer_to_stage,
    )


def unstack_blocks(blocks_tensor: torch.Tensor) -> list[torch.Tensor]:
    """The columns of a ``[T, N, D]`` carrier, as a list of blocks.

    Returns ``[T, D]`` views, one per block. Views share storage with the input
    so autograd gradients flow back correctly.
    """
    return [blocks_tensor[:, i] for i in range(blocks_tensor.shape[1])]
