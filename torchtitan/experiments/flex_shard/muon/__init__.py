# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Distributed Muon + MuonClip test bed for FlexShard.

Three distributed-Muon implementations for the dense 2D matrices, all bit-exact with
single-device Muon -- they differ only in how the matrix is distributed and gathered:

1. Owned Muon (the main result, communication-efficient): each whole 2D matrix lives on
   one owner rank (``Owned`` placement), so its Newton-Schulz runs locally with **no
   collectives in the step**. Owners are balanced across ranks by NS FLOPs.
2. Gather Muon (baseline): each 2D matrix is sharded and all-gathered before NS, in two
   sharding layouts -- ``Shard(0)`` (per-tensor row split) or ``GroupedRaggedShard``
   (byte-balanced, the cut crosses matrix boundaries).
3. fsdp2 / DTensor Muon (baseline): core ``fully_shard`` shards the matrices and DTensor
   does the all-gather inside ``opt.step`` (``full_tensor()``).

MoE expert stacks get Muon too, but only the whole-expert case is collective-free: when
``efsdp <= num_local_experts`` (``world_size <= num_experts``) each rank holds whole
experts (``Shard(0)``) -> ``GroupedMuon`` (local NS, no step collectives). When
``efsdp > num_local_experts`` the experts are sharded *within* the matrix (``Shard(>=1)``)
and **fall back to** ``GatherGroupedMuon``, which all-gathers each expert stack over the
efsdp mesh before NS -- so in that regime the step is no longer fully collective-free (the
dense 2D Owned Muon still is). MuonClip (Kimi K2) QK-clip (``qkclip``) layers on any
recipe, rescaling per-head q/k rows after each step (this adds a small per-head
``all_reduce(MAX)``) so the max attention logit stays ``<= tau``. Everything else (1D
norms, embeddings, LM head) -> AdamW.

The code is organized in three layers:

| Layer | Concern | Modules |
|---|---|---|
| 1. Partition | tensor -> ranks (which slice each rank holds) | ``bucketing``, ``owned`` |
| 2. Routing | placement -> optimizer (which param uses which) | ``containers`` |
| 3. Optimizer step | the Newton-Schulz math | ``newton_schulz`` + ``{gather,grouped,dtensor}_muon`` + ``qkclip`` |

Benchmarking: ``bench._BenchMixin`` (enabled by ``FLEX_SHARD_MUON_BENCH=1``) reports a
per-step breakdown -- total_iter / opt_step / fwd_bwd and comm bytes (in-step vs fwd/bwd,
via the ``comm_counter`` collective-byte counter) -- so the recipes above compare on the
same axes (e.g. Owned's zero in-step comm vs the gather / DTensor all-gather cost).

Built on the FlexShard engine and its ``example`` placement primitives (``Owned`` /
``Shard`` / ``RaggedShard``); kept separate from ``example/`` so it does not collide with
that folder's own muon. The public API is re-exported here (``from ...muon import X``).
"""

from .bench import _BenchMixin  # noqa: F401
from .containers import (  # noqa: F401
    _build_grouped_expert_optimizers,
    _discover_dense_gather_buckets,
    _discover_expert_buckets,
    BenchInstrumentedOptimizers,
    build_comm_efficient_muon_optimizer,
    FlexShardGatherMuonOptimizers,
    FlexShardMuonOptimizers,
    FSDP2MuonOptimizers,
)
from .dtensor_muon import DTensorMuon  # noqa: F401
from .gather_muon import GatherGroupedMuon, GatherMuon  # noqa: F401
from .grouped_muon import GroupedMuon  # noqa: F401
from .newton_schulz import (  # noqa: F401
    _zeropower_via_newtonschulz_batched,
    assign_layer_owners_lpt,
    layer_newton_schulz_cost,
    newton_schulz_flops,
)
from .owned import (  # noqa: F401
    _group_params_by_layer,
    comm_efficient_muon_buckets,
    is_owned_2d,
    make_owned_placement_fn,
)
from .qkclip import (  # noqa: F401
    _is_mla_attention,
    _max_logit_per_head,
    _scale_rows,
    QKClip,
)
