# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""MoE token-dispatcher model transform."""

from dataclasses import dataclass, field, fields
from typing import Any

from torchtitan.models.common.moe import RoutedExperts
from torchtitan.models.common.token_dispatcher import LocalTokenDispatcher
from torchtitan.protocols.module import Module

from .base import convert_config_type, ModelConfigTransform, ModelConfigTransformContext

__all__ = ["TokenDispatcherTransform"]


@dataclass(kw_only=True, slots=True)
class TokenDispatcherTransform(ModelConfigTransform):
    """Replace every MoE token dispatcher with ``dispatcher``.

    The transform fills the structural fields from each routed-expert config
    and derives persistent EP buffer capacity from the training token shape.
    Additional backend-specific constructor arguments belong in ``kwargs``.
    ``routed_experts`` also converts every routed-expert config, for a dispatcher
    that needs its own experts.
    """

    dispatcher: type[LocalTokenDispatcher]
    kwargs: dict[str, Any] = field(default_factory=dict)
    routed_experts: type[RoutedExperts] | None = None

    def __post_init__(self) -> None:
        if not issubclass(self.dispatcher, LocalTokenDispatcher):
            raise ValueError(
                f"{self.dispatcher.__qualname__} must inherit LocalTokenDispatcher."
            )

    def transform(
        self,
        model: Module.Config,
        *,
        context: ModelConfigTransformContext | None = None,
    ) -> Module.Config:
        if context is None:
            raise ValueError("TokenDispatcherTransform requires training context.")
        config_fields = {item.name for item in fields(self.dispatcher.Config)}
        num_max_tokens_per_rank = None
        if "num_max_tokens_per_rank" in config_fields:
            parallelism = context.parallelism
            num_token_shards = (
                parallelism.context_parallel_degree * parallelism.tensor_parallel_degree
            )
            num_tokens = context.training.num_tokens_per_microbatch_per_dp_rank
            if num_tokens % num_token_shards != 0:
                raise ValueError(
                    "training.num_tokens_per_microbatch_per_dp_rank "
                    f"({num_tokens}) must be divisible by context_parallel_degree "
                    f"* tensor_parallel_degree ({num_token_shards})."
                )
            num_max_tokens_per_rank = num_tokens // num_token_shards

        for _, routed_experts, parent, name in list(
            model.traverse(RoutedExperts.Config)
        ):
            if self.routed_experts is not None:
                converted = convert_config_type(routed_experts, self.routed_experts)
                assert isinstance(converted, RoutedExperts.Config)
                assert isinstance(name, str)
                setattr(parent, name, converted)
                routed_experts = converted
            existing = routed_experts.token_dispatcher
            values: dict[str, Any] = {
                "num_experts": existing.num_experts,
                "top_k": existing.top_k,
                **self.kwargs,
            }
            if "hidden_dim" in config_fields:
                values["hidden_dim"] = routed_experts.w13.in_features
            if "num_max_tokens_per_rank" in config_fields:
                values["num_max_tokens_per_rank"] = num_max_tokens_per_rank
            routed_experts.token_dispatcher = self.dispatcher.Config(**values)
        return model
