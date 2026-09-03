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
from torch.nn.attention.flex_attention import AuxRequest

from torchtitan.components.optimizer import OptimizersContainer
from torchtitan.distributed import ParallelDims
from torchtitan.models.common.attention import FlexAttention
from torchtitan.models.deepseek_v3.model import Attention

# Shape suffixes:
# T = packed tokens, H = attention heads, D = projection rows per head,
# I = input features.


class QKClipFlexAttention(FlexAttention):
    """FlexAttention that records the maximum score for each query head."""

    @dataclass(kw_only=True, slots=True)
    class Config(FlexAttention.Config):
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


def _validate_head_sharding(weight: DTensor) -> None:
    """Require MLA weights to be replicated or contiguously sharded on dim 0."""
    for placement in weight.placements:
        if type(placement) is Replicate:
            continue
        # ``_StridedShard`` subclasses ``Shard``, so an exact type check rejects
        # it: its non-default shard order would pair heads with the wrong rows.
        if type(placement) is not Shard or placement.dim != 0:
            raise ValueError(
                "QK clipping requires MLA weights sharded only on tensor "
                "dimension 0."
            )


def _replicated_scales(scales_H: torch.Tensor, weight: DTensor) -> DTensor:
    """Represent per-head scales on the same distributed mesh as ``weight``.

    The MAX all-reduce already leaves identical scales on every rank.
    ``from_local`` records the scales as replicated without communication,
    allowing DTensor dispatch to align them with sharded heads. Using
    ``distribute_tensor`` would add an unnecessary broadcast.
    """
    return DTensor.from_local(
        scales_H,
        weight.device_mesh,
        tuple(Replicate() for _ in weight.placements),
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
    """Scale the NoPE and remaining rows of every MLA head in place.

    ``weight`` is viewed as ``[num_heads, rows_per_head, in_features]``, and
    ``scales_H`` contains one scale per head. The remaining rows are unchanged
    when ``remaining_scale_exponent`` is ``None``.
    """
    num_heads = scales_H.numel()
    if weight.ndim != 2 or weight.shape[0] != num_heads * rows_per_head:
        raise ValueError("QK clip scales do not match the MLA weight shape.")
    _validate_head_sharding(weight)

    scales_H11 = _replicated_scales(scales_H, weight).view(-1, 1, 1)
    heads_HDI = weight.view(num_heads, rows_per_head, weight.shape[1])
    heads_HDI[:, :nope_rows_per_head].mul_(scales_H11.pow(nope_scale_exponent))
    if remaining_scale_exponent is not None:
        heads_HDI[:, nope_rows_per_head:].mul_(scales_H11.pow(remaining_scale_exponent))


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
        and isinstance(module.inner_attention, QKClipFlexAttention)
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
    "QKClipFlexAttention",
    "register_qk_clip_hook",
]
