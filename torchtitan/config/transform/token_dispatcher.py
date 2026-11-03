# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""MoE token-dispatcher model transform."""

from dataclasses import dataclass, field, fields
from typing import Any

from torchtitan.models.base.module import Module

from torchtitan.models.common.moe import RoutedExperts
from torchtitan.models.common.moe.token_dispatcher import LocalTokenDispatcher

from .base import ModelConfigTransform

__all__ = ["TokenDispatcherTransform"]


@dataclass(kw_only=True, slots=True)
class TokenDispatcherTransform(ModelConfigTransform):
    """Replace every MoE token dispatcher with ``dispatcher``.

    The transform fills the structural fields from each routed-expert config.
    Dispatchers that preallocate persistent EP buffers (e.g. DeepEP, HybridEP)
    need ``num_max_tokens_per_rank``: the most tokens one EP rank dispatches
    per forward. It depends on how the model is run, e.g. a trainer's
    microbatch size divided by its CP x TP degree, or an inference engine's
    per-step token budget divided by its TP degree. Additional
    backend-specific constructor arguments belong in ``kwargs``.
    """

    dispatcher: type[LocalTokenDispatcher]
    num_max_tokens_per_rank: int | None = None
    kwargs: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not issubclass(self.dispatcher, LocalTokenDispatcher):
            raise ValueError(
                f"{self.dispatcher.__qualname__} must inherit LocalTokenDispatcher."
            )
        config_fields = {item.name for item in fields(self.dispatcher.Config)}
        needs_capacity = "num_max_tokens_per_rank" in config_fields
        if needs_capacity and self.num_max_tokens_per_rank is None:
            raise ValueError(
                f"{self.dispatcher.__qualname__} preallocates EP buffers; set "
                "TokenDispatcherTransform.num_max_tokens_per_rank."
            )
        if not needs_capacity and self.num_max_tokens_per_rank is not None:
            raise ValueError(
                f"{self.dispatcher.__qualname__} does not take "
                "num_max_tokens_per_rank."
            )

    def transform(self, model: Module.Config) -> Module.Config:
        config_fields = {item.name for item in fields(self.dispatcher.Config)}
        for _, routed_experts, _, _ in model.traverse(RoutedExperts.Config):
            existing = routed_experts.token_dispatcher
            values: dict[str, Any] = {
                "num_experts": existing.num_experts,
                "top_k": existing.top_k,
                **self.kwargs,
            }
            if "hidden_dim" in config_fields:
                values["hidden_dim"] = routed_experts.w13.in_features
            if self.num_max_tokens_per_rank is not None:
                values["num_max_tokens_per_rank"] = self.num_max_tokens_per_rank
            routed_experts.token_dispatcher = self.dispatcher.Config(**values)
        return model
