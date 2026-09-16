# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Tensor-parallel model transforms."""

# TODO: Make this transform part of every TP model-construction path, then
# remove the legacy enclosing-module redistributions from decoder sharding.
# Common dense ``feed_forward`` configs are already supported across Llama 3,
# Qwen 3/3.5, DeepSeek V3/V4 dense layers, Muse Glimmer, Kimi K2, and common
# MoE shared experts. Remaining work is wiring all TP recipes through the
# transform and migrating model-specific attention and feed-forward paths.

from dataclasses import dataclass
from typing import cast

from torchtitan.models.common.async_linear import (
    AsyncColumnParallelLinear,
    AsyncRowParallelLinear,
)

from torchtitan.models.common.attention import GQAttention, QKVLinear
from torchtitan.models.common.feed_forward import FeedForward
from torchtitan.models.common.linear import (
    ColumnParallelLinear,
    Linear,
    RowParallelLinear,
    specialize_column_parallel_linear,
    specialize_row_parallel_linear,
)
from torchtitan.protocols.module import Module

from .base import convert_config_type, ModelConfigTransform
from .lora import LoRATransform

__all__ = ["AsyncTensorParallelTransform", "TensorParallelTransform"]


def _convert_linear(
    config: Linear.Config,
    replacement: type[Linear],
    *,
    async_tp: bool,
    projection_name: str,
) -> Linear.Config:
    if async_tp:
        if type(config) is Linear.Config:
            return cast(Linear.Config, convert_config_type(config, replacement))
        raise ValueError(
            "Async tensor parallelism does not support converted "
            f"{projection_name} projections"
        )

    assert config._owner is not None
    if replacement is ColumnParallelLinear:
        specialized = specialize_column_parallel_linear(config._owner, type(config))
    else:
        assert replacement is RowParallelLinear
        specialized = specialize_row_parallel_linear(config._owner, type(config))
    return cast(Linear.Config, convert_config_type(config, specialized))


def _transform_attention(model: Module.Config, *, async_tp: bool) -> Module.Config:
    """Select TP projection roles for common GQA blocks.

    Both modes put the input collective on ``wqkv`` and the output collective
    on ``wo``. Synchronous modules call explicit ``spmd.redistribute``
    collectives, while async modules overlap those collectives with their GEMMs.
    Model-specific ``GQAttention`` subclasses retain their existing behavior.
    """
    output_linear = AsyncRowParallelLinear if async_tp else RowParallelLinear
    for _, traversed, parent, attr in model.traverse(GQAttention.Config):
        existing = cast(GQAttention.Config, traversed)
        if existing._owner is not GQAttention:
            continue

        if existing.qkv_linear._owner is not QKVLinear:
            if async_tp:
                raise ValueError(
                    "Async tensor parallelism requires the common QKVLinear "
                    "implementation"
                )
            continue
        input_linear = AsyncColumnParallelLinear if async_tp else ColumnParallelLinear
        existing.qkv_linear.wqkv = _convert_linear(
            existing.qkv_linear.wqkv,
            input_linear,
            async_tp=async_tp,
            projection_name="QKV",
        )
        existing.wo = _convert_linear(
            existing.wo,
            output_linear,
            async_tp=async_tp,
            projection_name="attention output",
        )
    return model


def _transform_feed_forward(
    model: Module.Config,
    *,
    async_tp: bool,
) -> Module.Config:
    """Select TP projection roles for transformer-block dense FFNs.

    Configs stored as ``feed_forward`` fields and common shared experts are
    transformed, along with a root FFN config. The synchronous transform
    specializes the final projection implementation, so the collective
    encloses converted compute such as quantization or LoRA. Async TP leaves
    shared experts unchanged because their input layout depends on EP.
    """
    input_linear = AsyncColumnParallelLinear if async_tp else ColumnParallelLinear
    output_linear = AsyncRowParallelLinear if async_tp else RowParallelLinear
    for _, traversed, parent, attr in model.traverse(FeedForward.Config):
        is_root = parent is None
        is_dense_block = attr == "feed_forward"
        is_shared_expert = attr == "shared_experts" and not async_tp
        if not is_root and not is_dense_block and not is_shared_expert:
            continue
        existing = cast(FeedForward.Config, traversed)
        if existing._owner is not FeedForward:
            continue

        existing.w13 = _convert_linear(
            existing.w13,
            input_linear,
            async_tp=async_tp,
            projection_name="w13",
        )
        existing.w2 = _convert_linear(
            existing.w2,
            output_linear,
            async_tp=async_tp,
            projection_name="w2",
        )
    return model


def _transform_tensor_parallel(
    model: Module.Config,
    *,
    async_tp: bool,
) -> Module.Config:
    model = _transform_attention(model, async_tp=async_tp)
    return _transform_feed_forward(model, async_tp=async_tp)


@dataclass(kw_only=True, slots=True)
class TensorParallelTransform(ModelConfigTransform):
    """Select synchronous TP projections for common attention and dense FFNs."""

    def transform(self, model: Module.Config) -> Module.Config:
        return _transform_tensor_parallel(model, async_tp=False)


@dataclass(kw_only=True, slots=True)
class AsyncTensorParallelTransform(ModelConfigTransform):
    """Select async tensor-parallel attention and dense FFN projections."""

    def transform(self, model: Module.Config) -> Module.Config:
        return _transform_tensor_parallel(model, async_tp=True)


# TP wraps the final projection implementation. Quantization uses legacy model
# converters, which run before transforms; this dependency ensures LoRA also
# runs before TP specializes the resulting projection config.
TensorParallelTransform.run_after = (LoRATransform,)
AsyncTensorParallelTransform.run_after = (LoRATransform,)
TensorParallelTransform.conflicts_with = (AsyncTensorParallelTransform,)
AsyncTensorParallelTransform.conflicts_with = (TensorParallelTransform,)
