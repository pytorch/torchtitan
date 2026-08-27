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
from torchtitan.models.common.attention import FlexAttention
from torchtitan.models.deepseek_v3.model import Attention


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
            max_scores_BNT = aux.max_scores
            assert max_scores_BNT is not None
            # Record gradient-accumulation and PP microbatches, plus AC recomputation.
            self.max_attention_logits_N.append(max_scores_BNT.amax(dim=(0, 2)).detach())


def _distributed_head_scales(
    scales_N: torch.Tensor,
    weight: DTensor,
    *,
    num_heads: int,
) -> DTensor:
    """Represent TP-local head scales on the weight's storage mesh.

    The MAX all-reduce makes scales identical across data-parallel axes, while
    FlexAttention produces only the heads local to each TP rank. Recording
    those placements lets DTensor locally align the scales with finer-grained
    FlexShard ownership without another collective.
    """
    mesh_axis_names = weight.device_mesh.mesh_dim_names
    if mesh_axis_names is None:
        raise ValueError("QK clipping requires named MLA weight mesh axes.")

    tp_axis = mesh_axis_names.index("tp") if "tp" in mesh_axis_names else None
    scale_placements = []
    for mesh_axis, placement in enumerate(weight.placements):
        is_tp_axis = mesh_axis == tp_axis and weight.device_mesh.size(mesh_axis) > 1
        if is_tp_axis:
            if type(placement) is not Shard or placement.dim != 0:
                raise ValueError(
                    "QK clipping requires the TP mesh axis to shard MLA head rows."
                )
            scale_placements.append(Shard(0))
        else:
            if type(placement) is Replicate:
                scale_placements.append(Replicate())
                continue
            if type(placement) not in (Shard, _StridedShard):
                raise ValueError(
                    "QK clipping requires MLA weights sharded only on tensor "
                    "dimension 0."
                )
            sharded_placement = cast(Shard | _StridedShard, placement)
            if sharded_placement.dim != 0:
                raise ValueError(
                    "QK clipping requires MLA weights sharded only on tensor "
                    "dimension 0."
                )
            scale_placements.append(Replicate())

    expected_local_heads = num_heads
    if tp_axis is not None and weight.device_mesh.size(tp_axis) > 1:
        expected_local_heads = Shard.local_shard_size_and_offset(
            num_heads,
            weight.device_mesh.size(tp_axis),
            weight.device_mesh.get_local_rank(tp_axis),
        )[0]
    if scales_N.numel() != expected_local_heads:
        raise ValueError("QK clip scales do not match the TP-local MLA heads.")

    return DTensor.from_local(
        scales_N,
        weight.device_mesh,
        tuple(scale_placements),
        run_check=False,
        shape=torch.Size((num_heads,)),
        stride=(1,),
    )


@torch.no_grad()
def _scale_mla_heads(
    weight: DTensor,
    scales_N: torch.Tensor,
    *,
    rows_per_head: int,
    nope_rows_per_head: int,
    nope_scale_exponent: float,
    remaining_scale_exponent: float | None,
) -> None:
    """Scale the NoPE and remaining rows of every MLA head in place.

    ``weight`` is viewed as ``[num_heads, rows_per_head, in_features]``, and
    ``scales_N`` contains one scale per TP-local head. The remaining rows are
    unchanged when ``remaining_scale_exponent`` is ``None``.
    """
    if weight.ndim != 2 or weight.shape[0] % rows_per_head:
        raise ValueError("QK clip scales do not match the MLA weight shape.")
    num_heads = weight.shape[0] // rows_per_head

    scales_N11 = _distributed_head_scales(
        scales_N,
        weight,
        num_heads=num_heads,
    ).view(-1, 1, 1)
    heads_NDI = weight.view(num_heads, rows_per_head, weight.shape[1])
    heads_NDI[:, :nope_rows_per_head].mul_(scales_N11.pow(nope_scale_exponent))
    if remaining_scale_exponent is not None:
        heads_NDI[:, nope_rows_per_head:].mul_(scales_N11.pow(remaining_scale_exponent))


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
        cast(DTensor, q_projection.weight),
        scales_N,
        rows_per_head=attention.qk_head_dim,
        nope_rows_per_head=attention.qk_nope_head_dim,
        nope_scale_exponent=alpha,
        remaining_scale_exponent=1.0,
    )
    # Key/value: the K rows take the remaining ``scale ** (1 - alpha)``, and the
    # V rows stay unchanged.
    _scale_mla_heads(
        cast(DTensor, attention.wkv_b.weight),
        scales_N,
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
    scales_N = threshold / max_logits_N.clamp_min(threshold)

    for attention, layer_scales_N in zip(
        attention_layers,
        scales_N.split(num_heads_per_layer),
        strict=True,
    ):
        _clip_mla_weights(attention, layer_scales_N, alpha=alpha)
        attention.inner_attention.max_attention_logits_N.clear()


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
