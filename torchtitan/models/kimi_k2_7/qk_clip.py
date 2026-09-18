# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass
from typing import Any, cast

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor import DTensor, Replicate, Shard
from torch.distributed.tensor.placement_types import _StridedShard
from torch.nn.attention.flex_attention import AuxRequest

from torchtitan.components.optimizer import OptimizersContainer
from torchtitan.distributed import ParallelDims
from torchtitan.models.common.attention import FlexInnerAttention
from torchtitan.models.deepseek_v3.model import Attention

# Shape suffixes:
# T = packed tokens, H = attention heads, D = projection rows per head,
# I = input features.


class QKClipFlexInnerAttention(FlexInnerAttention):
    """FlexInnerAttention that records the maximum score for each query head."""

    @dataclass(kw_only=True, slots=True)
    class Config(FlexInnerAttention.Config):
        pass

    def __init__(self, config: Config) -> None:
        super().__init__(config)
        self.max_attention_logits_H: list[torch.Tensor] = []

    def _get_aux_request(self, *, return_lse: bool) -> AuxRequest:
        return AuxRequest(lse=return_lse, max_scores=self.training)

    def _process_aux(self, aux: Any) -> None:
        if self.training:
            max_scores_1HT = aux.max_scores
            assert max_scores_1HT is not None
            # Record gradient-accumulation and PP microbatches, plus AC recomputation.
            self.max_attention_logits_H.append(max_scores_1HT.amax(dim=(0, 2)).detach())


def _distributed_row_scales(
    local_row_scales_R1: torch.Tensor,
    weight: DTensor,
) -> DTensor:
    """Wrap TP-local row scales as TP-sharded and DP-replicated without communication."""
    mesh_axis_names = weight.device_mesh.mesh_dim_names
    assert mesh_axis_names is not None

    tp_axis = mesh_axis_names.index("tp") if "tp" in mesh_axis_names else None
    scale_placements = []
    for mesh_axis, placement in enumerate(weight.placements):
        is_tp_axis = mesh_axis == tp_axis
        if is_tp_axis:
            assert type(placement) is Shard and placement.dim == 0
            scale_placements.append(Shard(0))
        else:
            if type(placement) is Replicate:
                scale_placements.append(Replicate())
                continue
            assert isinstance(placement, (Shard, _StridedShard)) and placement.dim == 0
            scale_placements.append(Replicate())

    return DTensor.from_local(
        local_row_scales_R1,
        weight.device_mesh,
        tuple(scale_placements),
        run_check=False,
    )


@torch.no_grad()
def _scale_mla_heads(
    weight: DTensor,
    scales_H: torch.Tensor,
    *,
    rows_per_head: int,
    nope_rows_per_head: int,
    nope_scale_exponent: float,
    remaining_scale_exponent: float | None,
) -> None:
    """Scale each MLA head's NoPE and trailing rows in place.

    ``scales_H`` contains one scale per TP-local head. Trailing rows are
    unchanged when ``remaining_scale_exponent`` is ``None``. The weight remains
    2D because an FSDP storage shard may split a head.
    """
    if weight.ndim != 2 or weight.shape[0] % rows_per_head:
        raise ValueError("QK clip scales do not match the MLA weight shape.")
    scales_H1 = scales_H.view(-1, 1)
    nope_scales_HD = scales_H1.pow(nope_scale_exponent).expand(-1, nope_rows_per_head)
    num_remaining_rows = rows_per_head - nope_rows_per_head
    remaining_scales_HD = (
        torch.ones_like(scales_H1).expand(-1, num_remaining_rows)
        if remaining_scale_exponent is None
        else scales_H1.pow(remaining_scale_exponent).expand(-1, num_remaining_rows)
    )
    local_row_scales_R1 = torch.cat(
        (nope_scales_HD, remaining_scales_HD), dim=1
    ).reshape(-1, 1)
    row_scales_R1 = _distributed_row_scales(
        local_row_scales_R1,
        weight,
    )
    if row_scales_R1.shape != (weight.shape[0], 1):
        raise ValueError("QK clip scales do not match the MLA weight shape.")
    weight.mul_(row_scales_R1)


@torch.no_grad()
def _clip_mla_weights(
    attention: Attention,
    scales_H: torch.Tensor,
    *,
    alpha: float,
) -> None:
    q_projection = attention.wq if attention.q_lora_rank == 0 else attention.wq_b
    # Query: NoPE rows take ``scale ** alpha``, RoPE rows take the full scale.
    _scale_mla_heads(
        cast(DTensor, q_projection.weight),
        scales_H,
        rows_per_head=attention.qk_head_dim,
        nope_rows_per_head=attention.qk_nope_head_dim,
        nope_scale_exponent=alpha,
        remaining_scale_exponent=1.0,
    )
    # Key/value: the K rows take the remaining ``scale ** (1 - alpha)``, and the
    # V rows stay unchanged.
    _scale_mla_heads(
        cast(DTensor, attention.wkv_b.weight),
        scales_H,
        rows_per_head=attention.qk_nope_head_dim + attention.v_head_dim,
        nope_rows_per_head=attention.qk_nope_head_dim,
        nope_scale_exponent=1.0 - alpha,
        remaining_scale_exponent=None,
    )


@torch.no_grad()
def qk_clip(
    model_parts: list[nn.Module],
    *,
    reduction_mesh: DeviceMesh,
) -> None:
    """Apply the Kimi K2 QK clipping update with the report defaults."""
    threshold = 100.0
    alpha = 0.5
    attention_layers = [
        module
        for model_part in model_parts
        for module in model_part.modules()
        if isinstance(module, Attention)
        and isinstance(module.inner_attention, QKClipFlexInnerAttention)
    ]
    if not attention_layers:
        return

    inner_attentions = [layer.inner_attention for layer in attention_layers]
    # Each entry holds one layer's local maximum logit per query head.
    layer_max_logits_H = [
        torch.stack(inner_attention.max_attention_logits_H).amax(dim=0)
        for inner_attention in inner_attentions
    ]
    num_heads_per_layer = [logits.numel() for logits in layer_max_logits_H]
    max_logits_H = torch.cat(layer_max_logits_H)
    if reduction_mesh.size() > 1:
        dist.all_reduce(
            max_logits_H,
            op=dist.ReduceOp.MAX,
            group=reduction_mesh.get_group(),
        )
    scales_H = threshold / max_logits_H.clamp_min(threshold)

    for attention, layer_scales_H in zip(
        attention_layers,
        scales_H.split(num_heads_per_layer),
        strict=True,
    ):
        _clip_mla_weights(attention, layer_scales_H, alpha=alpha)
        attention.inner_attention.max_attention_logits_H.clear()


def register_qk_clip_hook(
    optimizers: OptimizersContainer,
    model_parts: list[nn.Module],
    parallel_dims: ParallelDims,
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
            reduction_mesh=reduction_mesh,
        )

    optimizers.register_step_post_hook(_qk_clip_hook)


__all__ = [
    "QKClipFlexInnerAttention",
    "register_qk_clip_hook",
]
