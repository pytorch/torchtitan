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


def _validate_head_sharding(weight: torch.Tensor) -> None:
    """Reject storage layouts that would remap heads under a head-dim view."""
    if not isinstance(weight, DTensor):
        return
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


def _replicated_scales(scales_N: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """Return ``scales_N`` as an operand that broadcasts against ``weight``.

    ``qk_clip`` derives the scales from a MAX all-reduce, so every rank already
    holds identical values. Recording that with ``from_local`` avoids the
    broadcast that ``distribute_tensor`` performs to establish ``Replicate``,
    which would otherwise cost one collective per weight per step.
    """
    if not isinstance(weight, DTensor):
        return scales_N
    return DTensor.from_local(
        scales_N,
        weight.device_mesh,
        tuple(Replicate() for _ in weight.placements),
        run_check=False,
    )


@torch.no_grad()
def _scale_mla_heads(
    weight: torch.Tensor,
    scales_N: torch.Tensor,
    *,
    head_extent: int,
    head_split: int,
    leading_exponent: float,
    trailing_exponent: float | None,
) -> None:
    """Scale each head's leading and trailing row block of ``weight`` in place.

    ``weight`` is ``[num_heads * head_extent, in_features]``. Viewing it as
    ``[num_heads, head_extent, in_features]`` keeps the head dimension on the
    weight's own ``Shard(0)``, and a replicated ``scales_N`` is sliced locally
    against it, so every rank scales exactly the heads it stores without
    exchanging data. ``trailing_exponent`` of ``None`` leaves the trailing rows
    untouched.
    """
    num_heads = scales_N.numel()
    if weight.ndim != 2 or weight.shape[0] != num_heads * head_extent:
        raise ValueError("QK clip scales do not match the MLA weight shape.")
    _validate_head_sharding(weight)

    scales_N11 = _replicated_scales(scales_N, weight).view(-1, 1, 1)
    heads_NDI = weight.view(num_heads, head_extent, weight.shape[1])
    heads_NDI[:, :head_split].mul_(scales_N11.pow(leading_exponent))
    if trailing_exponent is not None:
        heads_NDI[:, head_split:].mul_(scales_N11.pow(trailing_exponent))


@torch.no_grad()
def _clip_mla_weights(
    attention: Attention,
    scales_N: torch.Tensor,
    *,
    alpha: float,
) -> None:
    q_projection = attention.wq if attention.q_lora_rank == 0 else attention.wq_b
    # Query: NoPE rows take ``scale ** alpha``, RoPE rows take the full scale.
    _scale_mla_heads(
        q_projection.weight,
        scales_N,
        head_extent=attention.qk_head_dim,
        head_split=attention.qk_nope_head_dim,
        leading_exponent=alpha,
        trailing_exponent=1.0,
    )
    # Key/value: the K rows take the remaining ``scale ** (1 - alpha)``, and the
    # V rows stay unchanged.
    _scale_mla_heads(
        attention.wkv_b.weight,
        scales_N,
        head_extent=attention.qk_nope_head_dim + attention.v_head_dim,
        head_split=attention.qk_nope_head_dim,
        leading_exponent=1.0 - alpha,
        trailing_exponent=None,
    )


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
