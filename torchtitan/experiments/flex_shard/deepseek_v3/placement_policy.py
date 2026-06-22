# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Literal

import torch.nn as nn
from torch.distributed.device_mesh import DeviceMesh

from torchtitan.experiments.flex_shard import BucketSpec
from torchtitan.experiments.flex_shard.example.owned import (
    GroupedOwned,
    GroupedOwnedSegmentSpec,
)
from torchtitan.experiments.flex_shard.example.shard import Shard
from torchtitan.experiments.flex_shard.flex_shard.bucket_storage import (
    MixedPrecisionPolicy,
)
from torchtitan.experiments.flex_shard.flex_shard.placement_contract import Placement


PlacementFn = Callable[
    [list[tuple[str, nn.Parameter]], DeviceMesh],
    dict[str, tuple[Placement, ...]],
]


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


def _placement_fn(dim: int) -> PlacementFn:
    def placements(
        named_params: list[tuple[str, nn.Parameter]],
        mesh: DeviceMesh,
    ) -> dict[str, tuple[Shard, ...]]:
        return {fqn: (Shard(dim),) for fqn, _ in named_params}

    return placements


def _is_grouped_expert_weight(fqn: str, param: nn.Parameter) -> bool:
    return (
        param.dim() == 3
        and fqn.endswith((".moe.experts.w1", ".moe.experts.w2", ".moe.experts.w3"))
    )


def _expert_block_order_key(fqn: str) -> int:
    if fqn.endswith(".w1"):
        return 0
    if fqn.endswith(".w3"):
        return 1
    if fqn.endswith(".w2"):
        return 2
    raise ValueError(f"Unexpected routed expert weight FQN: {fqn}")


def _make_expert_block_grouped_owned_segments(
    named_params: list[tuple[str, nn.Parameter]],
    world_size: int,
) -> dict[str, list[GroupedOwnedSegmentSpec]]:
    if not named_params:
        return {}
    bad = [
        (fqn, tuple(param.shape))
        for fqn, param in named_params
        if not _is_grouped_expert_weight(fqn, param)
    ]
    if bad:
        raise ValueError(f"GroupedOwned expert block expects packed w1/w2/w3: {bad}")
    num_experts = named_params[0][1].shape[0]
    if any(param.shape[0] != num_experts for _, param in named_params):
        raise ValueError("GroupedOwned expert block requires matching expert counts.")
    ordered_params = sorted(
        named_params, key=lambda item: _expert_block_order_key(item[0])
    )
    segments_by_fqn: dict[str, list[GroupedOwnedSegmentSpec]] = {
        fqn: [] for fqn, _ in ordered_params
    }
    # Fill owner rows in contiguous equal-capacity expert ranges. Since the
    # all-gather input is padded to the max owner row length, this keeps the
    # gathered w1/w2/w3 expert slices evenly strided and enables view-out.
    experts_per_owner = max(1, (num_experts + world_size - 1) // world_size)
    for expert_idx in range(num_experts):
        owner = min(expert_idx // experts_per_owner, world_size - 1)
        for param_order, (fqn, param) in enumerate(ordered_params):
            expert_numel = param[0].numel()
            segments_by_fqn[fqn].append(
                GroupedOwnedSegmentSpec(
                    name=f"{fqn}#expert{expert_idx}",
                    fqn=fqn,
                    param_offset=expert_idx * expert_numel,
                    numel=expert_numel,
                    owner_rank=owner,
                    storage_order=expert_idx * len(ordered_params) + param_order,
                )
            )
    return segments_by_fqn


def _grouped_owned_expert_placement_fn(
    named_params: list[tuple[str, nn.Parameter]],
    mesh: DeviceMesh,
) -> dict[str, tuple[Placement, ...]]:
    segments_by_fqn = _make_expert_block_grouped_owned_segments(
        named_params,
        mesh.size(),
    )
    placement = GroupedOwned(segments_by_fqn)
    return {fqn: (placement,) for fqn, _ in named_params}


def _routed_expert_placement_fn(policy: DeepSeekV3FlexShardPolicy) -> PlacementFn:
    if policy.routed_experts == "shard":
        return _placement_fn(0)
    if policy.routed_experts == "grouped_owned":
        return _grouped_owned_expert_placement_fn
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
        placement_fn=_placement_fn(0),
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
        placement_fn=_placement_fn(0),
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
            placement_fn=_placement_fn(0),
            mesh=dp_mesh,
            mp_policy=mp_policy,
            reshard_after_forward=reshard_after_forward,
        ),
        BucketSpec(
            ["lm_head.*"],
            placement_fn=_placement_fn(0),
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
