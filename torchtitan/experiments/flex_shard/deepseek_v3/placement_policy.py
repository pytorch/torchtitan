# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass, field
from typing import Any

from torch.distributed.device_mesh import DeviceMesh

from torchtitan.experiments.flex_shard import BucketSpec
from torchtitan.experiments.flex_shard.example.owned import (
    make_grouped_owned_expert_block_placement_fn,
)
from torchtitan.experiments.flex_shard.example.shard import make_shard_placement_fn
from torchtitan.experiments.flex_shard.flex_shard.bucket_storage import (
    MixedPrecisionPolicy,
    PlacementFn,
)


@dataclass(frozen=True)
class DeepSeekV3FlexShardPolicy:
    """Placement policy for the experimental DeepSeek V3 FlexShard path."""

    common_placement_fn: PlacementFn = field(
        default_factory=lambda: make_shard_placement_fn(0)
    )
    routed_expert_placement_fn: PlacementFn = field(
        default_factory=make_grouped_owned_expert_block_placement_fn
    )
    output_placement_fn: PlacementFn = field(
        default_factory=lambda: make_shard_placement_fn(0)
    )

    def __post_init__(self) -> None:
        for name in (
            "common_placement_fn",
            "routed_expert_placement_fn",
            "output_placement_fn",
        ):
            if not callable(getattr(self, name)):
                raise TypeError(f"{name} must be callable.")

    def build_buckets(
        self,
        model: Any,
        *,
        dp_mesh: DeviceMesh,
        efsdp_mesh: DeviceMesh | None,
        mp_policy: MixedPrecisionPolicy,
        reshard_after_forward: bool,
        reshard_last: bool,
    ) -> list[BucketSpec]:
        return _build_flex_shard_buckets(
            model,
            dp_mesh=dp_mesh,
            efsdp_mesh=efsdp_mesh,
            mp_policy=mp_policy,
            reshard_after_forward=reshard_after_forward,
            reshard_last=reshard_last,
            policy=self,
        )


def _embedding_bucket(
    *,
    dp_mesh: DeviceMesh,
    mp_policy: MixedPrecisionPolicy,
    reshard_after_forward: bool,
    placement_fn: PlacementFn,
) -> BucketSpec:
    return BucketSpec(
        ["tok_embeddings.*"],
        placement_fn=placement_fn,
        mesh=dp_mesh,
        mp_policy=mp_policy,
        reshard_after_forward=reshard_after_forward,
    )


def _layer_common_bucket(
    layer_id: str,
    *,
    dp_mesh: DeviceMesh,
    mp_policy: MixedPrecisionPolicy,
    reshard_after_forward: bool,
    placement_fn: PlacementFn,
) -> BucketSpec:
    return BucketSpec(
        [
            f"layers.{layer_id}.*attention.*",
            f"layers.{layer_id}.*attention_norm.*",
            f"layers.{layer_id}.*ffn_norm.*",
            f"layers.{layer_id}.*feed_forward.*",
            f"layers.{layer_id}.*moe.router.*",
            f"layers.{layer_id}.*moe.shared_experts.*",
        ],
        placement_fn=placement_fn,
        mesh=dp_mesh,
        mp_policy=mp_policy,
        reshard_after_forward=reshard_after_forward,
    )


def _layer_routed_expert_bucket(
    layer_id: str,
    *,
    expert_mesh: DeviceMesh,
    mp_policy: MixedPrecisionPolicy,
    reshard_after_forward: bool,
    placement_fn: PlacementFn,
) -> BucketSpec:
    return BucketSpec(
        [f"layers.{layer_id}.*moe.experts.*"],
        placement_fn=placement_fn,
        mesh=expert_mesh,
        mp_policy=mp_policy,
        reshard_after_forward=reshard_after_forward,
    )


def _output_buckets(
    *,
    dp_mesh: DeviceMesh,
    mp_policy: MixedPrecisionPolicy,
    reshard_after_forward: bool,
    placement_fn: PlacementFn,
) -> list[BucketSpec]:
    return [
        BucketSpec(
            ["norm.*"],
            placement_fn=placement_fn,
            mesh=dp_mesh,
            mp_policy=mp_policy,
            reshard_after_forward=reshard_after_forward,
        ),
        BucketSpec(
            ["lm_head.*"],
            placement_fn=placement_fn,
            mesh=dp_mesh,
            mp_policy=mp_policy,
            reshard_after_forward=reshard_after_forward,
        ),
    ]


def _build_flex_shard_buckets(
    model: Any,
    *,
    dp_mesh: DeviceMesh,
    efsdp_mesh: DeviceMesh | None,
    mp_policy: MixedPrecisionPolicy,
    reshard_after_forward: bool,
    reshard_last: bool,
    policy: DeepSeekV3FlexShardPolicy,
) -> list[BucketSpec]:
    expert_mesh = efsdp_mesh if efsdp_mesh is not None else dp_mesh

    buckets: list[BucketSpec] = [
        _embedding_bucket(
            dp_mesh=dp_mesh,
            mp_policy=mp_policy,
            reshard_after_forward=reshard_after_forward,
            placement_fn=policy.common_placement_fn,
        )
    ]

    for layer_id in model.layers.keys():
        buckets.append(
            _layer_common_bucket(
                layer_id,
                dp_mesh=dp_mesh,
                mp_policy=mp_policy,
                reshard_after_forward=reshard_after_forward,
                placement_fn=policy.common_placement_fn,
            )
        )
        buckets.append(
            _layer_routed_expert_bucket(
                layer_id,
                expert_mesh=expert_mesh,
                mp_policy=mp_policy,
                reshard_after_forward=reshard_after_forward,
                placement_fn=policy.routed_expert_placement_fn,
            )
        )

    buckets.extend(
        _output_buckets(
            dp_mesh=dp_mesh,
            mp_policy=mp_policy,
            reshard_after_forward=reshard_last,
            placement_fn=policy.output_placement_fn,
        )
    )
    return buckets
