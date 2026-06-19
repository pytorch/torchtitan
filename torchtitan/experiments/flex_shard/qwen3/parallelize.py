# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""FlexShard + communication-efficient Muon parallelizers for Qwen3 (dense + MoE).

A training test bed for benchmarking distributed Muon across model sizes: shards the
Qwen3 ladder with FlexShard so every transformer layer's 2D weights run communication-
efficient Muon (one ``Owned`` bucket per layer, owner balanced by Newton-Schulz FLOPs).
The model-agnostic bucketing/parallelizer logic lives in ``..muon.bucketing``; this module
just binds it to Qwen3. Two families:

- Dense Qwen3 (0.6B .. 32B): the ``parallelize_qwen3_muon*`` parallelizers. Qwen3's deep
  layer counts (28..64) keep ``num_layers >= dp_shard`` for large world sizes -- a clean
  vehicle for the model-size / GPU-count scaling studies.
- Qwen3-MoE (30B-A3B): the ``parallelize_qwen3_moe_muon*`` parallelizers are EP-capable --
  dense 2D matrices run Owned/Muon on the dp mesh; MoE expert stacks Shard on the EP
  (efsdp) mesh and run GroupedMuon / GatherGroupedMuon.
"""

from __future__ import annotations

from torchtitan.experiments.flex_shard.muon.bucketing import (
    build_gather_muon_buckets,
    build_muon_buckets,
    build_permatrix_muon_buckets,
    build_twolevel_muon_buckets,
    make_gather_muon_parallelize_fn as _make_gather_muon_parallelize_fn,
    make_muon_parallelize_fn,
)

# Comm-efficient Owned Muon. Dense-only: EP is rejected (support_ep=False).
# whole-layer allocation (case 1) -- the default.
parallelize_qwen3_muon = make_muon_parallelize_fn(model_name="Qwen3", support_ep=False)
# per-2D-tensor allocation (case 2): finer LPT balance, fills more ranks.
parallelize_qwen3_muon_permatrix = make_muon_parallelize_fn(
    model_name="Qwen3", support_ep=False, granularity="matrix"
)
# two-level allocation: per-layer rank groups, LPT within each (fewer broadcasts).
parallelize_qwen3_muon_twolevel = make_muon_parallelize_fn(
    model_name="Qwen3", support_ep=False, granularity="twolevel"
)
# whole-layer but allowing idle ranks -- the case-1 reference when num_layers < world_size.
parallelize_qwen3_muon_idle = make_muon_parallelize_fn(
    model_name="Qwen3", support_ep=False, allow_idle_ranks=True
)
# unified system: auto-select the shallowest allocation level for the regime.
parallelize_qwen3_muon_auto = make_muon_parallelize_fn(
    model_name="Qwen3", support_ep=False, granularity="auto"
)

# --- Qwen3-MoE (EP-capable): dense 2D matrices run comm-efficient Owned/Muon on the dp
# mesh; MoE expert stacks Shard on the EP (efsdp) mesh -> GroupedMuon. ---
# Whole-layer allocation: one Owned dense bucket per layer (requires num_layers >=
# world_size, e.g. 235B's 94 layers on 64 GPUs).
parallelize_qwen3_moe_muon = make_muon_parallelize_fn(
    model_name="Qwen3-MoE", support_ep=True
)
# Auto allocation: per-tensor / two-level for the dense matrices when num_layers <
# world_size (e.g. 30B-A3B's 48 layers on 64 GPUs, 235B's 94 on 128) -- fills every
# rank that whole-layer cannot. Experts still on the EP mesh.
parallelize_qwen3_moe_muon_auto = make_muon_parallelize_fn(
    model_name="Qwen3-MoE", support_ep=True, granularity="auto"
)
# Two-level allocation (per-layer rank groups), EP-capable.
parallelize_qwen3_moe_muon_twolevel = make_muon_parallelize_fn(
    model_name="Qwen3-MoE", support_ep=True, granularity="twolevel"
)


def make_gather_muon_parallelize_fn(
    dense_kind: str, *, reshard_after_forward: bool = False
):
    """Gather-for-NS Muon baseline parallelize_fn for Qwen3 (EP-capable: experts on the EP mesh).

    ``reshard_after_forward=True`` enables RAF for the sharded gather buckets (the engine
    supports it for ``Shard(0)``; GatherMuon maps params by canonical FQN), so the gather
    baselines stop holding the full model resident.
    """
    return _make_gather_muon_parallelize_fn(
        dense_kind, model_name="Qwen3", reshard_after_forward=reshard_after_forward
    )


__all__ = [
    "build_gather_muon_buckets",
    "build_muon_buckets",
    "build_permatrix_muon_buckets",
    "build_twolevel_muon_buckets",
    "make_gather_muon_parallelize_fn",
    "parallelize_qwen3_moe_muon",
    "parallelize_qwen3_moe_muon_auto",
    "parallelize_qwen3_moe_muon_twolevel",
    "parallelize_qwen3_muon",
    "parallelize_qwen3_muon_auto",
    "parallelize_qwen3_muon_idle",
    "parallelize_qwen3_muon_permatrix",
    "parallelize_qwen3_muon_twolevel",
]
