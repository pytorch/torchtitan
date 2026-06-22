# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass
from typing import Any, Literal

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


_SHARD_DIM0_PLACEMENT_FN = make_shard_placement_fn(0)
_GROUPED_OWNED_EXPERT_BLOCK_PLACEMENT_FN = (
    make_grouped_owned_expert_block_placement_fn()
)


@dataclass(frozen=True)
class DeepSeekV3FlexShardPolicy:
    """Placement policy for the experimental DeepSeek V3 FlexShard path."""

    common: Literal["shard"] = "shard"
    routed_experts: Literal["shard", "grouped_owned"] = "grouped_owned"
    output: Literal["shard"] = "shard"

    def __post_init__(self) -> None:
        if self.common != "shard":
            raise ValueError(f"Unsupported common placement policy: {self.common}")
        if self.routed_experts not in {"shard", "grouped_owned"}:
            raise ValueError(
                "Unsupported routed expert placement policy: "
                f"{self.routed_experts}"
            )
        if self.output != "shard":
            raise ValueError(f"Unsupported output placement policy: {self.output}")

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


def _routed_expert_placement_fn(policy: DeepSeekV3FlexShardPolicy) -> PlacementFn:
    if policy.routed_experts == "shard":
        return _SHARD_DIM0_PLACEMENT_FN
    if policy.routed_experts == "grouped_owned":
        return _GROUPED_OWNED_EXPERT_BLOCK_PLACEMENT_FN
    raise AssertionError(f"Unhandled routed expert policy: {policy.routed_experts}")


def _embedding_bucket(
    *,
    dp_mesh: DeviceMesh,
    mp_policy: MixedPrecisionPolicy,
    reshard_after_forward: bool,
    policy: DeepSeekV3FlexShardPolicy,
) -> BucketSpec:
    assert policy.common == "shard"
    return BucketSpec(
        ["tok_embeddings.*"],
        placement_fn=_SHARD_DIM0_PLACEMENT_FN,
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
    policy: DeepSeekV3FlexShardPolicy,
) -> BucketSpec:
    assert policy.common == "shard"
    return BucketSpec(
        [
            f"layers.{layer_id}.*attention.*",
            f"layers.{layer_id}.*attention_norm.*",
            f"layers.{layer_id}.*ffn_norm.*",
            f"layers.{layer_id}.*feed_forward.*",
            f"layers.{layer_id}.*moe.router.*",
            f"layers.{layer_id}.*moe.shared_experts.*",
        ],
        placement_fn=_SHARD_DIM0_PLACEMENT_FN,
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
    policy: DeepSeekV3FlexShardPolicy,
) -> BucketSpec:
    return BucketSpec(
        [f"layers.{layer_id}.*moe.experts.*"],
        placement_fn=_routed_expert_placement_fn(policy),
        mesh=expert_mesh,
        mp_policy=mp_policy,
        reshard_after_forward=reshard_after_forward,
    )


def _output_buckets(
    *,
    dp_mesh: DeviceMesh,
    mp_policy: MixedPrecisionPolicy,
    reshard_after_forward: bool,
    policy: DeepSeekV3FlexShardPolicy,
) -> list[BucketSpec]:
    assert policy.output == "shard"
    return [
        BucketSpec(
            ["norm.*"],
            placement_fn=_SHARD_DIM0_PLACEMENT_FN,
            mesh=dp_mesh,
            mp_policy=mp_policy,
            reshard_after_forward=reshard_after_forward,
        ),
        BucketSpec(
            ["lm_head.*"],
            placement_fn=_SHARD_DIM0_PLACEMENT_FN,
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
            policy=policy,
        )
    ]

    for layer_id in model.layers.keys():
        buckets.append(
            _layer_common_bucket(
                layer_id,
                dp_mesh=dp_mesh,
                mp_policy=mp_policy,
                reshard_after_forward=reshard_after_forward,
                policy=policy,
            )
        )
        buckets.append(
            _layer_routed_expert_bucket(
                layer_id,
                expert_mesh=expert_mesh,
                mp_policy=mp_policy,
                reshard_after_forward=reshard_after_forward,
                policy=policy,
            )
        )

    buckets.extend(
        _output_buckets(
            dp_mesh=dp_mesh,
            mp_policy=mp_policy,
            reshard_after_forward=reshard_last,
            policy=policy,
        )
    )
    return buckets
