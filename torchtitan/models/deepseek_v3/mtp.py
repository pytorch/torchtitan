# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass

import spmd_types as spmd
import torch
import torch_remat as remat

from torchtitan.distributed.parallelism_context import MeshAxisName
from torchtitan.distributed.spmd_types import spmd_dense_sp_enabled, spmd_mesh_group
from torchtitan.models.common.attention import (
    FlexAttentionMetadata,
    VarlenAttentionMetadata,
)
from torchtitan.models.common.decoder import TransformerBlock
from torchtitan.models.common.linear import Linear, maybe_gather_tp_input
from torchtitan.models.common.moe import MoE
from torchtitan.models.common.nn_modules import RMSNorm


class DeepSeekV3MTPBlock(TransformerBlock):
    """DeepSeek-V3 MTP block.

    The block implements the DeepSeek-V3 fusion:

    ``eh_proj(cat(enorm(shifted_embedding), hnorm(previous_hidden)))``

    followed by one regular transformer block.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(TransformerBlock.Config):
        enorm: RMSNorm.Config
        hnorm: RMSNorm.Config
        eh_proj: Linear.Config
        mtp_norm: RMSNorm.Config

    def __init__(self, config: Config):
        super().__init__()
        self.attention = config.attention.build()
        self.attention_norm = config.attention_norm.build()
        self.ffn_norm = config.ffn_norm.build()
        self.enorm = config.enorm.build()
        self.hnorm = config.hnorm.build()
        self.eh_proj = config.eh_proj.build()
        self.mtp_norm = config.mtp_norm.build()

        self.feed_forward = config.feed_forward.build()

    def forward(
        self,
        mtp_input_embed: torch.Tensor,
        prev_embed: torch.Tensor,
        mtp_input_valid_mask: torch.Tensor,
        attention_metadata: FlexAttentionMetadata | VarlenAttentionMetadata | None,
        positions: torch.Tensor | None = None,
        *,
        padding_mask: torch.Tensor | None = None,
        aux_loss_denominator: torch.Tensor | None = None,
    ):
        mtp_padding_mask_T = ~mtp_input_valid_mask
        if padding_mask is not None:
            mtp_padding_mask_T = mtp_padding_mask_T | padding_mask
        # Under SP, prev_embed already arrives Shard(0) from the preceding
        # decoder or MTP block, while the validity mask arrives replicated.
        # The old module boundary implicitly sharded only this mask; do that
        # explicitly here. The MoE owns padding-mask sharding for its branch.
        local_mtp_input_valid_mask_T = self._maybe_shard_mtp_valid_mask_across_tp(
            mtp_input_valid_mask
        )
        prev_embed = prev_embed * local_mtp_input_valid_mask_T.unsqueeze(-1).to(
            dtype=prev_embed.dtype
        )
        h = self.eh_proj(
            torch.cat([self.enorm(mtp_input_embed), self.hnorm(prev_embed)], dim=-1)
        )
        # attention_norm and the residual add read the eh_proj output with bare ops.
        remat.recompute_needs_tensor(h)
        attn_out = self.attention(self.attention_norm(h), attention_metadata, positions)
        # The residual add reads the attention output with a bare op.
        remat.recompute_needs_tensor(attn_out)
        h = h + attn_out
        if isinstance(self.feed_forward, MoE):
            ffn_out = self.feed_forward(
                self.ffn_norm(h),
                padding_mask_T=mtp_padding_mask_T,
                aux_loss_denominator=aux_loss_denominator,
            )
        else:
            ffn_out = self.feed_forward(self.ffn_norm(h))
        # The residual add reads the feed-forward output with a bare op.
        remat.recompute_needs_tensor(ffn_out)
        h = h + ffn_out
        return maybe_gather_tp_input(self.mtp_norm, self.mtp_norm(h))

    def _maybe_shard_mtp_valid_mask_across_tp(
        self, mtp_input_valid_mask_T: torch.Tensor
    ) -> torch.Tensor:
        """Shard the validity mask to match sequence-parallel MTP activations."""
        if not spmd_dense_sp_enabled():
            return mtp_input_valid_mask_T
        tp_group = spmd_mesh_group(MeshAxisName.TP)
        if tp_group is None:
            return mtp_input_valid_mask_T
        return spmd.redistribute(
            mtp_input_valid_mask_T,
            tp_group,
            src=spmd.R,
            dst=spmd.S(0),
            backward_options={"op_dtype": mtp_input_valid_mask_T.dtype},
        )
