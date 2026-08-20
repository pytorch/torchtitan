# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass
from typing import Any

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor import DTensor, Replicate, Shard
from torch.nn.attention.flex_attention import AuxRequest

from torchtitan.components.optimizer import OptimizersContainer
from torchtitan.config import Configurable
from torchtitan.distributed import ParallelDims
from torchtitan.models.common.attention import FlexAttention
from torchtitan.models.deepseek_v3.model import Attention


@dataclass(kw_only=True, slots=True)
class QKClipConfig(Configurable.Config):
    threshold: float = 100.0
    alpha: float = 0.5

    def __post_init__(self) -> None:
        if self.threshold <= 0:
            raise ValueError("QK clip threshold must be positive.")
        if not 0.0 <= self.alpha <= 1.0:
            raise ValueError("QK clip alpha must be in [0, 1].")


class QKClipFlexAttention(FlexAttention):
    """FlexAttention that records the maximum score for each query head."""

    @dataclass(kw_only=True, slots=True)
    class Config(FlexAttention.Config):
        pass

    def __init__(self, config: Config) -> None:
        super().__init__(config)
        self.max_attention_logits_N: list[torch.Tensor] = []

    def _get_aux_request(self, *, return_lse: bool) -> AuxRequest:
        return AuxRequest(lse=return_lse, max_scores=self.training)

    def _process_aux(self, aux: Any) -> None:
        if self.training:
            max_scores_BNL = aux.max_scores
            assert max_scores_BNL is not None
            # Record gradient-accumulation and PP microbatches, plus AC recomputation.
            self.max_attention_logits_N.append(max_scores_BNL.amax(dim=(0, 2)).detach())


def _local_head_range(
    param: torch.Tensor,
    scales_N: torch.Tensor,
    *,
    head_extent: int,
) -> tuple[int, int]:
    global_shape = param.shape
    if len(global_shape) != 2 or global_shape[0] != scales_N.numel() * head_extent:
        raise ValueError("QK clip scales do not match the MLA weight shape.")

    if isinstance(param, DTensor):
        local_param = param.to_local()
        local_rows = global_shape[0]
        row_offset = 0
        for mesh_axis, (mesh_axis_size, placement) in enumerate(
            zip(param.device_mesh.shape, param.placements, strict=True)
        ):
            if type(placement) is Replicate:
                continue
            if type(placement) is not Shard or placement.dim % param.ndim != 0:
                raise ValueError(
                    "QK clipping requires MLA weights sharded only on tensor "
                    "dimension 0."
                )
            local_rows, local_offset = Shard.local_shard_size_and_offset(
                local_rows,
                mesh_axis_size,
                param.device_mesh.get_local_rank(mesh_axis),
            )
            row_offset += local_offset
    else:
        local_param = param
        local_rows = global_shape[0]
        row_offset = 0

    if (
        local_rows % head_extent
        or row_offset % head_extent
        or tuple(local_param.shape) != (local_rows, global_shape[1])
    ):
        raise ValueError("QK clip storage shards must align to complete MLA heads.")

    num_local_heads = local_rows // head_extent
    first_local_head = row_offset // head_extent
    return first_local_head, num_local_heads


@torch.no_grad()
def _clip_mla_weights(
    attention: Attention,
    scales_N: torch.Tensor,
    *,
    alpha: float,
) -> None:
    q_projection = attention.wq if attention.q_lora_rank == 0 else attention.wq_b
    first_local_head, num_local_heads = _local_head_range(
        q_projection.weight,
        scales_N,
        head_extent=attention.qk_head_dim,
    )
    q_local = (
        q_projection.weight.to_local()
        if isinstance(q_projection.weight, DTensor)
        else q_projection.weight
    )
    q_weight_NDI = q_local.view(
        num_local_heads,
        attention.qk_head_dim,
        q_projection.weight.shape[1],
    )
    local_scales_N = scales_N.narrow(
        0,
        first_local_head,
        num_local_heads,
    )
    local_scales_N11 = local_scales_N.view(-1, 1, 1)
    q_weight_NDI[:, : attention.qk_nope_head_dim].mul_(local_scales_N11.pow(alpha))
    q_weight_NDI[:, attention.qk_nope_head_dim :].mul_(local_scales_N11)

    first_local_head, num_local_heads = _local_head_range(
        attention.wkv_b.weight,
        scales_N,
        head_extent=attention.qk_nope_head_dim + attention.v_head_dim,
    )
    kv_local = (
        attention.wkv_b.weight.to_local()
        if isinstance(attention.wkv_b.weight, DTensor)
        else attention.wkv_b.weight
    )
    kv_weight_NDI = kv_local.view(
        num_local_heads,
        attention.qk_nope_head_dim + attention.v_head_dim,
        attention.wkv_b.weight.shape[1],
    )
    local_scales_N = scales_N.narrow(
        0,
        first_local_head,
        num_local_heads,
    )
    kv_weight_NDI[:, : attention.qk_nope_head_dim].mul_(
        local_scales_N.view(-1, 1, 1).pow(1.0 - alpha)
    )

    if isinstance(q_projection.weight, DTensor):
        torch.autograd.graph.increment_version(q_projection.weight)
    if isinstance(attention.wkv_b.weight, DTensor):
        torch.autograd.graph.increment_version(attention.wkv_b.weight)


@torch.no_grad()
def qk_clip(
    model_parts: list[nn.Module],
    *,
    config: QKClipConfig,
    reduction_mesh: DeviceMesh,
) -> None:
    attention_layers = [
        module
        for model_part in model_parts
        for module in model_part.modules()
        if isinstance(module, Attention)
        and isinstance(module.inner_attention, QKClipFlexAttention)
    ]
    if not attention_layers:
        return

    inner_attentions = [layer.inner_attention for layer in attention_layers]
    # Each entry holds one layer's local maximum logit per query head.
    layer_max_logits_N = [
        torch.stack(inner_attention.max_attention_logits_N).amax(dim=0)
        for inner_attention in inner_attentions
    ]
    num_heads_per_layer = [logits.numel() for logits in layer_max_logits_N]
    max_logits_N = torch.cat(layer_max_logits_N)
    if reduction_mesh.size() > 1:
        dist.all_reduce(
            max_logits_N,
            op=dist.ReduceOp.MAX,
            group=reduction_mesh.get_group(),
        )
    scales_N = config.threshold / max_logits_N.clamp_min(config.threshold)

    for attention, layer_scales_N in zip(
        attention_layers,
        scales_N.split(num_heads_per_layer),
        strict=True,
    ):
        _clip_mla_weights(attention, layer_scales_N, alpha=config.alpha)
        attention.inner_attention.max_attention_logits_N.clear()


def register_qk_clip_hook(
    optimizers: OptimizersContainer,
    model_parts: list[nn.Module],
    parallel_dims: ParallelDims,
    *,
    config: QKClipConfig,
) -> None:
    """Apply QK clipping after each ordinary optimizer step."""
    reduction_mesh = parallel_dims.get_mesh("loss")

    def _qk_clip_hook(
        _optimizer: torch.optim.Optimizer,
        _args: tuple[Any, ...],
        _kwargs: dict[str, Any],
    ) -> None:
        qk_clip(
            model_parts,
            config=config,
            reduction_mesh=reduction_mesh,
        )

    optimizers.register_step_post_hook(_qk_clip_hook)


__all__ = [
    "QKClipConfig",
    "QKClipFlexAttention",
    "register_qk_clip_hook",
]
