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
import torch_remat as remat
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor import DTensor, Replicate
from torch.nn.attention.flex_attention import AuxRequest

from torchtitan.components.optim import OptimizersContainer
from torchtitan.distributed import ParallelismContext
from torchtitan.distributed.parallelism_context import MeshAxisName
from torchtitan.models.common.attention import FlexInnerAttention, MLAAttention
from torchtitan.models.deepseek_v3.attention import DeepSeekMLAAttention

# Shape suffixes:
# T = packed tokens, H = attention heads, D = projection rows per head,
# I = input features, R = weight rows (H * D).


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
        if self.training and not remat.is_recomputing():
            max_scores_1HT = aux.max_scores
            assert max_scores_1HT is not None
            # Record gradient-accumulation and PP microbatches once, excluding
            # activation-checkpoint replay.
            self.max_attention_logits_H.append(max_scores_1HT.amax(dim=(0, 2)).detach())


def _distributed_row_scales(row_scales_R1: torch.Tensor, weight: DTensor) -> DTensor:
    """Represent per-row scales on the same distributed mesh as ``weight``.

    The MAX all-reduce leaves identical scales across DP and CP, but TP
    records only its local heads. Preserve the TP row shard and replicate
    over the reduction axes; DTensor aligns these rows with FSDP's storage
    shards without broadcasting or gathering the head scales.
    """
    mesh_dim_names = weight.device_mesh.mesh_dim_names
    return DTensor.from_local(
        row_scales_R1,
        weight.device_mesh,
        # DP/CP scales stay replicated after MAX, even if FSDP shards weight rows.
        tuple(
            placement
            if mesh_dim_names is not None
            and mesh_dim_names[axis] == MeshAxisName.TP.value
            and placement.is_shard(dim=0)
            else Replicate()
            for axis, placement in enumerate(weight.placements)
        ),
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
    remaining_scale_exponent: float,
) -> None:
    """Scale the NoPE and remaining rows of every MLA head in place.

    Rows ``[h * rows_per_head, (h + 1) * rows_per_head)`` of ``weight`` belong to
    head ``h``, and ``scales_H`` contains one scale per TP-local head. A
    zero ``remaining_scale_exponent`` leaves the remaining rows unchanged.
    """
    # One scale per row: Shard(0) can split a head across ranks when num_heads
    # is not a multiple of the shard count.
    row_scales_R1 = torch.cat(
        [
            scales_H.pow(nope_scale_exponent)[:, None].expand(-1, nope_rows_per_head),
            scales_H.pow(remaining_scale_exponent)[:, None].expand(
                -1, rows_per_head - nope_rows_per_head
            ),
        ],
        dim=1,
    ).view(-1, 1)
    distributed_scales_R1 = _distributed_row_scales(row_scales_R1, weight)
    if weight.ndim != 2 or distributed_scales_R1.shape[0] != weight.shape[0]:
        raise ValueError("QK clip scales do not match the MLA weight shape.")
    weight.mul_(distributed_scales_R1)


@torch.no_grad()
def _clip_mla_weights(
    attention: DeepSeekMLAAttention,
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
        cast(DTensor, cast(Any, attention.inner_attention.wkv_b).weight),
        scales_H,
        rows_per_head=attention.qk_nope_head_dim + attention.v_head_dim,
        nope_rows_per_head=attention.qk_nope_head_dim,
        nope_scale_exponent=1.0 - alpha,
        remaining_scale_exponent=0.0,
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
        if isinstance(module, DeepSeekMLAAttention)
        and isinstance(module.inner_attention, MLAAttention)
        and isinstance(module.inner_attention.inner_attention, QKClipFlexInnerAttention)
    ]
    if not attention_layers:
        return

    inner_attentions = [
        cast(QKClipFlexInnerAttention, layer.inner_attention.inner_attention)
        for layer in attention_layers
    ]
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
        cast(
            QKClipFlexInnerAttention, attention.inner_attention.inner_attention
        ).max_attention_logits_H.clear()


def register_qk_clip_hook(
    optimizers: OptimizersContainer,
    model_parts: list[nn.Module],
    parallelism_context: ParallelismContext,
) -> None:
    """Apply QK clipping after each ordinary optimizer step."""
    reduction_mesh = parallelism_context.get_mesh("loss")

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
