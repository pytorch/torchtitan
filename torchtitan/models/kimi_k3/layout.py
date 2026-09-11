# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Routing of the block attention residual across pipeline stages.

A block committed at stage ``S`` is read by every later stage. Torch's
pipeline stages only talk to their neighbours, so the block travels along the
chain: each hop carries the blocks the receiving rank does not hold yet, and a
rank keeps what it has seen for its later stages. ``BlockLayoutTables``
simulates one micro-batch's forward in stage order and tabulates, per stage,
the blocks it commits, the blocks its rank holds when it runs, the blocks its
hop carries, and the stages that read a block from their rank's store. The
tables are a pure function of the split and the stage-to-rank map, so every
rank computes the same ones and nothing but the blocks travels on the wire.

With ``cache=False`` no rank keeps anything and every hop carries the whole
stack: the plain transport, kept for comparison.
"""


class BlockLayoutTables:
    """Per-stage routing of the block stack for one micro-batch."""

    def __init__(
        self,
        *,
        stage_to_rank: dict[int, int],
        num_blocks: int,
        n_layers: int,
        layers_per_block: int,
        layer_to_stage: dict[int, int],
        cache: bool = True,
    ) -> None:
        if n_layers <= 0 or layers_per_block <= 0:
            raise ValueError("n_layers and layers_per_block must be positive")
        # A partial final block is legal (K3: 93 layers over blocks of 12), so
        # num_blocks is the ceiling.
        expected_blocks = -(-n_layers // layers_per_block)
        if num_blocks != expected_blocks:
            raise ValueError(
                f"num_blocks ({num_blocks}) must equal ceil(n_layers / "
                f"layers_per_block) = {expected_blocks} for n_layers="
                f"{n_layers}, layers_per_block={layers_per_block}"
            )
        self.num_stages = len(stage_to_rank)
        if sorted(stage_to_rank) != list(range(self.num_stages)):
            raise ValueError(
                f"stage_to_rank must cover stages 0..{self.num_stages - 1}; "
                f"got {sorted(stage_to_rank)}"
            )
        self.stage_to_rank = dict(stage_to_rank)
        self.num_blocks = num_blocks
        self.n_layers = n_layers
        self.layers_per_block = layers_per_block
        self.cache = cache
        self._layer_to_stage = dict(layer_to_stage)
        self._commits_at: dict[int, list[int]] = {}
        self._producer_stage_of_block: dict[int, int] = {}
        self._cache_at_entry: dict[int, frozenset[int]] = {}
        self._delta_to_send: dict[int, list[int]] = {}
        self._cache_readers: dict[int, list[int]] = {}
        self._build()

    # ----- lookups ------------------------------------------------------- #
    def commits_at(self, stage_id: int) -> list[int]:
        """Blocks stage ``stage_id`` opens, in order."""
        return list(self._commits_at.get(stage_id, ()))

    def cache_at_entry(self, stage_id: int) -> frozenset[int]:
        """Blocks the stage's rank holds when the stage runs."""
        return self._cache_at_entry[stage_id]

    def delta_to_send(self, stage_id: int) -> list[int]:
        """Blocks the hop from ``stage_id`` to ``stage_id + 1`` carries."""
        return list(self._delta_to_send.get(stage_id, ()))

    def producer_stage_of_block(self, block_idx: int) -> int:
        return self._producer_stage_of_block[block_idx]

    def cache_readers_of_block(self, block_idx: int) -> list[int]:
        """Stages that take ``block_idx`` from their rank's store."""
        return list(self._cache_readers.get(block_idx, ()))

    def deposits_expected(self, block_idx: int, owner_stage: int) -> int:
        """How many later stages on ``owner_stage``'s rank read ``block_idx``
        from the store.

        The owner is the stage that brought the block onto the rank, by
        committing or by receiving it. Each such reader deposits the block's
        gradient into the rank store for the owner's backward to collect, so
        the owner compares the deposits it finds against this count: a
        missing one is a lost gradient no loss curve shows.
        """
        rank = self.stage_to_rank[owner_stage]
        return sum(
            1
            for reader in self._cache_readers.get(block_idx, ())
            if self.stage_to_rank[reader] == rank and reader > owner_stage
        )

    # ----- the simulation ----------------------------------------------- #
    def _build(self) -> None:
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

        # One micro-batch's forward in stage order; with the cache on, every
        # rank remembers what it has seen.
        held: dict[int, set[int]] = {r: set() for r in set(self.stage_to_rank.values())}
        accumulated: set[int] = set()
        for stage_id in range(self.num_stages):
            rank = self.stage_to_rank[stage_id]
            self._cache_at_entry[stage_id] = frozenset(held[rank])
            accumulated.update(self._commits_at[stage_id])
            if self.cache:
                held[rank].update(accumulated)
            next_stage = stage_id + 1
            if next_stage < self.num_stages:
                receiver = held[self.stage_to_rank[next_stage]]
                self._delta_to_send[stage_id] = sorted(accumulated - receiver)
            else:
                self._delta_to_send[stage_id] = []

        readers: dict[int, list[int]] = {b: [] for b in range(self.num_blocks)}
        for stage_id in range(self.num_stages):
            for b in sorted(self._cache_at_entry[stage_id]):
                readers[b].append(stage_id)
        self._cache_readers = readers


def infer_block_layout_tables_from_stages(
    stages,
    *,
    stage_to_rank: dict[int, int],
    num_blocks: int,
    n_layers: int,
    layers_per_block: int,
    layer_to_stage: dict[int, int],
    cache: bool = True,
) -> BlockLayoutTables:
    """Build :class:`BlockLayoutTables` for the stages a rank holds.

    ``layer_to_stage`` is the global map, layer id to stage id, that
    :func:`layer_to_stage_from_split` reads off the split. Any split is accepted
    as long as every layer sits on exactly one stage and each stage holds a
    contiguous run of layers in stage order, which is what the routing
    assumes; anything else raises rather than producing tables that are wrong
    in a way only the gradients would show.
    """
    if len(stages) < 1:
        raise ValueError("need at least one stage to infer layout")
    num_stages = len(stage_to_rank)
    if sorted(layer_to_stage) != list(range(n_layers)):
        raise ValueError(
            f"layer_to_stage must cover layers 0..{n_layers - 1} exactly once; "
            f"got {sorted(layer_to_stage)}"
        )
    previous = -1
    for layer_id in range(n_layers):
        stage_idx = layer_to_stage[layer_id]
        if not 0 <= stage_idx < num_stages:
            raise ValueError(
                f"layer {layer_id} sits on stage {stage_idx}, outside the "
                f"{num_stages} stages of this pipeline"
            )
        if stage_idx < previous:
            raise ValueError(
                f"layer {layer_id} sits on stage {stage_idx} after layer "
                f"{layer_id - 1} on stage {previous}. A non-contiguous "
                "pipeline split is not supported: the block routing would "
                "carry deltas to the wrong stages."
            )
        previous = stage_idx
    return BlockLayoutTables(
        stage_to_rank=stage_to_rank,
        num_blocks=num_blocks,
        n_layers=n_layers,
        layers_per_block=layers_per_block,
        layer_to_stage=layer_to_stage,
        cache=cache,
    )


def layer_to_stage_from_split(module_fqns_per_model_part) -> dict[int, int]:
    """The global layer-to-stage map, read off the split.

    ``module_fqns_per_model_part`` is the split core applies (the config's, or
    the generated one), a pure function of the config that every rank computes
    identically, so no collective is needed to learn where a layer sits.
    """
    layer_to_stage: dict[int, int] = {}
    for stage_idx, names in enumerate(module_fqns_per_model_part):
        for name in names:
            prefix, _, layer = name.partition(".")
            if prefix != "layers" or not layer.isdigit():
                continue
            layer_to_stage[int(layer)] = stage_idx
    return layer_to_stage
