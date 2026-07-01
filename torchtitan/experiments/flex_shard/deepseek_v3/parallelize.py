# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""FlexShard + communication-efficient Muon parallelizer for DeepSeek V3.

A training test bed: shards DeepSeek V3 with FlexShard so the dense 2D weights run
communication-efficient Muon (each transformer layer is one ``Owned`` bucket, owner
balanced by Newton-Schulz FLOPs), while MoE expert stacks stay ``Shard``-ed (and
run GroupedMuon / GatherGroupedMuon). The model-agnostic bucketing/parallelizer logic lives in
``..example.bucketing``; this module just binds it to DeepSeek V3 (EP-capable).
"""

from __future__ import annotations

from collections.abc import Callable

import torch
import torch.nn as nn
from torch.distributed.device_mesh import DeviceMesh

from torchtitan.config import (
    CompileConfig,
    ParallelismConfig,
    TORCH_DTYPE_MAP,
    TrainingConfig,
)
from torchtitan.distributed import ParallelDims
from torchtitan.distributed.activation_checkpoint import ActivationCheckpointingConfig
from torchtitan.distributed.fsdp import get_fsdp_reshard_after_forward_policy
from torchtitan.experiments.flex_shard import BucketSpec, flex_shard
from torchtitan.experiments.flex_shard.example.shard import Shard
from torchtitan.experiments.flex_shard.flex_shard.bucket_storage import (
    MixedPrecisionPolicy,
)

from torchtitan.experiments.flex_shard.muon.bucketing import (
    build_gather_muon_buckets,
    build_muon_buckets,
    make_gather_muon_parallelize_fn as _make_gather_muon_parallelize_fn,
    make_muon_parallelize_fn,
)
from torchtitan.models.deepseek_v3 import DeepSeekV3Model
from torchtitan.tools.logging import logger


PlacementFn = Callable[
    [list[tuple[str, nn.Parameter]], DeviceMesh],
    dict[str, tuple[Shard, ...]],
]


def parallelize_deepseekv3(
    model: DeepSeekV3Model,
    *,
    parallel_dims: ParallelDims,
    training: TrainingConfig,
    parallelism: ParallelismConfig,
    compile_config: CompileConfig,
    ac_config: ActivationCheckpointingConfig,
    dump_folder: str,
):
    """Apply the experimental eager FlexShard path to DeepSeek V3."""
    _validate_supported_parallelisms(
        parallel_dims=parallel_dims,
        training=training,
        compile_config=compile_config,
    )

    assert (
        training.seq_len % parallel_dims.seq_len_divisor == 0
    ), f"""
        Sequence length {training.seq_len} must be divisible by the product of TP degree
        ({parallel_dims.tp}) and 2 * CP degree ({parallel_dims.cp}).
        """

    # EP/TP application moved into the model's parallelize protocol on current main
    # (was the standalone llama4 apply_moe_ep_tp on weifengpy's older base).
    if parallel_dims.ep_enabled:
        model.parallelize(parallel_dims)

    if ac_config is not None:
        ac_config.build(dump_folder=dump_folder).apply(model)

    dp_mesh = parallel_dims.get_mesh("fsdp")
    efsdp_mesh = parallel_dims.get_mesh("efsdp") if parallel_dims.ep_enabled else None

    _apply_flex_shard(
        model,
        dp_mesh=dp_mesh,
        efsdp_mesh=efsdp_mesh,
        param_dtype=TORCH_DTYPE_MAP[training.mixed_precision_param],
        reduce_dtype=TORCH_DTYPE_MAP[training.mixed_precision_reduce],
        pp_enabled=parallel_dims.pp_enabled,
        reshard_after_forward_policy=parallelism.fsdp_reshard_after_forward,
    )

    logger.info("Applied experimental eager FlexShard to the DeepSeek V3 model")
    return model


def _validate_supported_parallelisms(
    *,
    parallel_dims: ParallelDims,
    training: TrainingConfig,
    compile_config: CompileConfig,
) -> None:
    if parallel_dims.pp_enabled:
        raise NotImplementedError(
            "The experimental FlexShard DeepSeek V3 path does not support PP yet."
        )
    if parallel_dims.tp_enabled:
        raise NotImplementedError(
            "The experimental FlexShard DeepSeek V3 path does not support TP yet."
        )
    if parallel_dims.cp_enabled:
        raise NotImplementedError(
            "The experimental FlexShard DeepSeek V3 path does not support CP yet."
        )
    if parallel_dims.dp_replicate_enabled:
        raise NotImplementedError(
            "The experimental FlexShard DeepSeek V3 path does not support HSDP yet."
        )
    if training.enable_cpu_offload:
        raise NotImplementedError(
            "FlexShard eager training does not support CPU offload yet."
        )
    if compile_config.enable and "model" in compile_config.components:
        raise NotImplementedError(
            "This FlexShard training entry point is eager-only; disable model compile."
        )


def _placement_fn(dim: int) -> PlacementFn:
    def placements(
        named_params: list[tuple[str, nn.Parameter]],
        mesh: DeviceMesh,
    ) -> dict[str, tuple[Shard, ...]]:
        return {fqn: (Shard(dim),) for fqn, _ in named_params}

    return placements


def _expert_placement_fn(
    named_params: list[tuple[str, nn.Parameter]],
    mesh: DeviceMesh,
) -> dict[str, tuple[Shard, ...]]:
    if not named_params:
        return {}
    num_local_experts = named_params[0][1].shape[0]
    dim = 1 if mesh.size() > num_local_experts else 0
    return {fqn: (Shard(dim),) for fqn, _ in named_params}


def _apply_flex_shard(
    model: nn.Module,
    *,
    dp_mesh: DeviceMesh,
    efsdp_mesh: DeviceMesh | None,
    param_dtype: torch.dtype,
    reduce_dtype: torch.dtype,
    pp_enabled: bool,
    reshard_after_forward_policy: str,
) -> None:
    mp_policy = MixedPrecisionPolicy(
        param_dtype=param_dtype,
        reduce_dtype=reduce_dtype,
    )
    reshard_after_forward = get_fsdp_reshard_after_forward_policy(
        reshard_after_forward_policy,
        pp_enabled,
    )
    reshard_last = reshard_after_forward_policy == "always"
    expert_mesh = efsdp_mesh if efsdp_mesh is not None else dp_mesh

    buckets: list[BucketSpec] = [
        BucketSpec(
            ["tok_embeddings.*"],
            placement_fn=_placement_fn(0),
            mesh=dp_mesh,
            mp_policy=mp_policy,
            reshard_after_forward=reshard_after_forward,
        ),
    ]

    for layer_id in model.layers.keys():
        buckets.append(
            BucketSpec(
                [
                    f"layers.{layer_id}.*attention.*",
                    f"layers.{layer_id}.*attention_norm.*",
                    f"layers.{layer_id}.*ffn_norm.*",
                    f"layers.{layer_id}.*feed_forward.*",
                    f"layers.{layer_id}.*moe.router.*",
                    f"layers.{layer_id}.*moe.shared_experts.*",
                ],
                placement_fn=_placement_fn(0),
                mesh=dp_mesh,
                mp_policy=mp_policy,
                reshard_after_forward=reshard_after_forward,
            )
        )
        buckets.append(
            BucketSpec(
                [f"layers.{layer_id}.*moe.experts.*"],
                placement_fn=_expert_placement_fn,
                mesh=expert_mesh,
                mp_policy=mp_policy,
                reshard_after_forward=reshard_after_forward,
            )
        )

    buckets.extend(
        [
            BucketSpec(
                ["norm.*"],
                placement_fn=_placement_fn(0),
                mesh=dp_mesh,
                mp_policy=mp_policy,
                reshard_after_forward=reshard_last,
            ),
            BucketSpec(
                ["lm_head.*"],
                placement_fn=_placement_fn(0),
                mesh=dp_mesh,
                mp_policy=mp_policy,
                reshard_after_forward=reshard_last,
            ),
        ]
    )

    flex_shard(model, buckets=buckets)


# Comm-efficient Owned Muon (EP-capable: dense weights run Owned/Muon on the dp mesh,
# MoE expert stacks Shard on the expert mesh).
# Whole-layer allocation -- one Owned bucket per layer. dsv3-16B is 27 layers, run
# single-node (world_size <= num_layers), so every rank owns whole layers. Finer per-matrix
# / two-level allocations are available via ``make_muon_parallelize_fn(granularity=...)``
# for the world_size > num_layers regimes; the dsv3 configs don't use them (on 8 GPUs both
# would reduce to this whole-layer placement anyway).
parallelize_deepseekv3_muon = make_muon_parallelize_fn(
    model_name="DeepSeek V3", support_ep=True
)


def make_gather_muon_parallelize_fn(dense_kind: str):
    """Gather-for-NS Muon baseline parallelize_fn for DeepSeek V3 (EP-capable)."""
    return _make_gather_muon_parallelize_fn(dense_kind, model_name="DeepSeek V3")


__all__ = [
    "build_gather_muon_buckets",
    "build_muon_buckets",
    "make_gather_muon_parallelize_fn",
    "parallelize_deepseekv3_muon",
]
