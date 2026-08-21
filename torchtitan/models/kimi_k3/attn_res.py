# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Block Attention Residuals (AttnRes).

Implements Block AttnRes from "Attention Residuals" (Kimi Team, 2026),
https://arxiv.org/abs/2603.15031. AttnRes replaces fixed residual accumulation
with softmax attention over preceding layer outputs, using a per-layer learned
pseudo-query vector. Block AttnRes partitions layers into N blocks, applies
standard residuals within a block, and uses attention only across block
boundaries to keep memory and cross-stage communication at O(Nd).

Pseudocode reference: paper Figure 2.
"""

from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.distributed.tensor import DTensor
from torch.nn import functional as F

from torchtitan.models.common.linear import Linear as _TTLinear
from torchtitan.protocols.module import Module


@dataclass(kw_only=True, slots=True)
class AttnResConfig:
    """Configuration for Block Attention Residuals.

    Attributes:
        enabled: Master switch. When False, the model uses standard residuals
            and all AttnRes code paths are skipped.
        num_blocks: Number of blocks to partition layers into (N in the paper).
            Sweet spot is ~8; N=2,4,8 all perform similarly, N>=16 degrades.
        norm_eps: Epsilon for the RMSNorm applied to keys.
    """

    enabled: bool = False
    num_blocks: int = 8
    norm_eps: float = 1e-5


def block_attn_res(
    blocks: list[torch.Tensor],
    partial_block: torch.Tensor,
    proj: nn.Linear,
    norm: nn.Module,
) -> torch.Tensor:
    """Inter-block attention: attend over completed blocks + current partial.

    Follows paper Figure 2. Pseudo-query is ``proj.weight`` (shape [1, D]),
    values are the stacked blocks (including the current partial). Keys are
    RMSNorm-ed values. Softmax over the block axis produces mixing weights.

    Args:
        blocks: List of completed block representations, each [B, T, D].
        partial_block: Current intra-block partial sum [B, T, D].
        proj: Linear(D, 1, bias=False). Its weight vector is the pseudo-query
            w_l. MUST be zero-initialized so softmax weights start uniform.
        norm: RMSNorm over D, applied to keys.

    Returns:
        Aggregated hidden state [B, T, D].
    """
    V = torch.stack(blocks + [partial_block], dim=0)  # [N+1, B, T, D]
    # ORDER IS LOAD-BEARING under FSDP2: proj.weight is read directly below,
    # never through proj.forward, so nothing all-gathers it except this call
    # to norm, which shares proj's FSDP param group. See apply_fsdp's
    # attn_res_tail note.
    #
    # Float BEFORE the norm so variance and rsqrt are fp32 too, matching the
    # release. proj is zero-initialized, so the pseudo-query gradient is a
    # difference of nearly equal terms (6x to 15x cancellation here) and bf16
    # is where that costs: normalizing in the stream dtype leaves 3.6e-3
    # relative error against the release form.
    K = norm(V.float())
    # Under TP, proj is NoParallel-wrapped so proj.weight is DTensor(Replicate)
    # and the einsum below would mix it with the plain K. to_local unwraps it;
    # its default Replicate grad placement is the correct spec, because K and V
    # are replicated on the tp axis in the forward and the rowwise projections
    # feeding them use local_output_grad_placements=(Replicate(),), so every tp
    # rank already computes the full gradient. Requesting Partial instead sums
    # tp identical copies and inflates proj.weight.grad by exactly tp
    # (measured: 1/tp on both AttnRes projections at tp2 and tp4, all other
    # parameters unaffected).
    # No to_local: with the residual stream a DTensor, K is one too, and
    # unwrapping only the query is what makes the einsum mixed. Both operands are
    # Replicate on the tp axis, so the contraction is local either way.
    query = proj.weight.squeeze(0).float()
    logits = torch.einsum("d,nbtd->nbt", query, K)
    weights = F.softmax(logits, dim=0)
    h = torch.einsum("nbt,nbtd->btd", weights, V.float())
    return h.to(V.dtype)


def block_attn_res_tensor(
    prefix_sum_BLD: torch.Tensor,
    block_residual_TND: torch.Tensor,
    proj: nn.Linear,
    norm: nn.Module,
) -> torch.Tensor:
    """``block_attn_res`` with the block history as one ``[T, N, D]`` tensor.

    Same computation, different container: the values are the committed blocks
    followed by the current partial, which is exactly what the list form stacks.
    Bitwise equality with ``block_attn_res`` is the gate.
    """
    B, L, D = prefix_sum_BLD.shape
    values_TND = torch.cat(
        (block_residual_TND, prefix_sum_BLD.reshape(-1, 1, D)), dim=1
    )
    # Same order as block_attn_res: float BEFORE the norm, and the norm call is
    # what all-gathers proj.weight under FSDP2 (see the note there).
    keys_TND = norm(values_TND.float())
    # No to_local here. With the residual stream a DTensor, keys_TND is one too,
    # and unwrapping only the query is what makes the einsum mixed. Both operands
    # are Replicate on the tp axis, so the contraction is local either way -- the
    # difference is purely whether DTensor's dispatcher can see it.
    query_D = proj.weight.squeeze(0).float()
    probs_TN = F.softmax(torch.einsum("d,tnd->tn", query_D, keys_TND), dim=-1)
    out_TD = torch.einsum("tn,tnd->td", probs_TN, values_TND.float())
    return out_TD.to(values_TND.dtype).view(B, L, D)


class AttnResProjection(_TTLinear):
    """Pseudo-query projection for AttnRes (D -> 1, no bias).

    Inherits from ``torchtitan.models.common.linear.Linear`` (which is
    ``nn.Linear + Module``) so instances satisfy
    ``Float8LinearConverter.verify_module_protocol``. The weight IS the
    per-layer pseudo-query vector ``w_l`` from the paper.
    ``param_init`` must zero-initialize the weight for training stability.

    NOTE: filter via ``filter_fqns`` to keep AttnRes pseudo-queries in
    high precision -- the zero-init carrier story relies on small
    deltas accumulating, which rowwise FP8 quantization noise would
    destroy.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        dim: int

    def __init__(self, config: Config):
        nn.Linear.__init__(self, config.dim, 1, bias=False)


def stack_blocks(blocks: list[torch.Tensor]) -> torch.Tensor:
    """Stack per-block tensors into the ``[T, N, D]`` carrier.

    ``T`` is ``B * L`` flattened and ``N`` is the block axis, which is the
    layout the upstream K3 model threads through its block signature. Nothing
    downstream needs B or L back: the pipeline adapter slices and stacks along
    the block axis and reasons about block INDICES, never about batch or
    sequence extents.

    Used when crossing a pipeline stage boundary, where the list has to become
    one tensor for P2P send/recv.
    """
    if not blocks:
        raise ValueError("stack_blocks needs at least one block to infer D")
    D = blocks[0].shape[-1]
    return torch.stack([b.reshape(-1, D) for b in blocks], dim=1)


def unstack_blocks(blocks_tensor: torch.Tensor) -> list[torch.Tensor]:
    """Inverse of ``stack_blocks``: the columns of a ``[T, N, D]`` carrier.

    Returns ``[T, D]`` views, one per block. Views share storage with the input
    so autograd gradients flow back correctly.
    """
    return [blocks_tensor[:, i] for i in range(blocks_tensor.shape[1])]
