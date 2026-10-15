# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Compressed main KV for DeepSeek V4.1 (CSA2).

Shape legend for this file:
    B = batch, L = sequence length, D = model dimension, R = ``compress_ratio``,
    Dk = head dimension of a compressed KV entry (``head_dim``).

Every layer owns a :class:`Compressor` instance, but only the *source* layers
carry parameters: a layer whose compressed KV is produced elsewhere holds no
weights and simply returns the tensor it was handed.  ``is_source`` encodes that
contract and is asserted in ``forward``, so a misconfigured layer fails loudly
instead of silently compressing or silently passing through.

Entry ``j`` of the compressed KV stands for the ``R`` tokens of group ``j``, so
it takes the position of the group's first token for RoPE.  The compressed KV is
rotated here rather than in the attention module, because the indexer needs the
*un-rotated* latent: CSA2 projects indexer keys from the main KV latent, so both
consumers need a different view of the same pooling result.

The pooling is a softmax-gated sum over each group; the norm and the rope are the
modules the config builds, with the rope's prefix width set by the site.
"""

from dataclasses import dataclass

import torch
from torchtitan.models.common.linear import Linear
from torchtitan.models.common.nn_modules import RMSNorm
from torchtitan.models.common.rope import RoPE
from torchtitan.models.base.module import Module


class Compressor(Module):
    """Pool ``compress_ratio`` tokens into one main-KV entry.

    ``ratio == 1`` is the uncompressed case: the entry is a plain projection of the
    token.  ``ratio > 1`` pools each group of ``R`` tokens with a softmax gate over the
    group.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        compress_ratio: int
        is_source: bool
        # Present only on source layers with ``compress_ratio > 0``:
        rope: RoPE.Config | None = None
        wkv: Linear.Config | None = None
        # Gate of the softmax pooling; only used when ``compress_ratio > 1``.
        wgate: Linear.Config | None = None
        norm: RMSNorm.Config | None = None

    def __init__(self, config: Config):
        super().__init__()
        self.compress_ratio = config.compress_ratio
        self.is_source = config.is_source
        if not self.is_source:
            return
        if config.rope is None or config.wkv is None or config.norm is None:
            raise ValueError("A Compressor source layer requires rope, wkv and norm configs.")
        self.wkv = config.wkv.build()
        if config.compress_ratio > 1:
            # The softmax gate only exists when there is more than one token to pool.
            if config.wgate is None:
                raise ValueError("A Compressor with compress_ratio > 1 requires a wgate config.")
            self.wgate = config.wgate.build()
        self.norm = config.norm.build()
        self.rope = config.rope.build()

    def forward(
        self,
        x_BLD: torch.Tensor,
        positions_BL: torch.Tensor,
        cmp_k: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        """Args:
            x: Hidden states of shape ``[B, L, D]``.
            positions: Position ids of shape ``[B, L]``.
            cmp_k: The shared compressed KV when this layer is not a source.

        Returns:
            ``(cmp_k, latent)``. A source layer returns its newly compressed,
            RoPE-rotated KV ``[B, L // R, head_dim]`` together with the pre-RoPE latent
            of the same shape that the indexer projects its keys from. A non-source layer
            returns the tensor it was handed and ``latent=None``.
        """
        # A source supersedes whatever compressed KV was in flight: the first source of
        # the stack sees ``None``, and a later group's source replaces the previous
        # group's tensor rather than consuming it. What must hold is that no *reusing*
        # layer is ever handed nothing.
        if self.compress_ratio > 0 and not self.is_source:
            assert cmp_k is not None, (
                "A layer that reuses the compressed KV must receive it: no source "
                f"layer precedes this one (compress_ratio={self.compress_ratio})."
            )
        if not self.is_source:
            return cmp_k, None

        ratio = self.compress_ratio
        seqlen = x_BLD.size(1)
        if seqlen % ratio != 0:
            raise ValueError(f"token count ({seqlen}) must be divisible by compress_ratio ({ratio})")

        if ratio == 1:
            latent_BLD = self.norm(self.wkv(x_BLD))
        else:
            # The softmax pooling runs in fp32; the projection itself stays in the model
            # dtype, as every other projection in the model does.
            kv_BLrD = self.wkv(x_BLD).unflatten(1, (-1, ratio))
            gate_BLrD = self.wgate(x_BLD).unflatten(1, (-1, ratio))  # pyrefly: ignore [not-callable]
            pooled_BLD = (kv_BLrD.float() * gate_BLrD.float().softmax(dim=2)).sum(dim=2)
            latent_BLD = self.norm(pooled_BLD.to(x_BLD.dtype))

        # Entry j stands for the group starting at token j * R, so it rotates at that
        # token's position; the packed positions reset per document, so the stride
        # selects each group's first token.  The latent is one rank-2 head; the rope
        # rotates rank-3 [B, N, 1, H].
        rotated_BLD = self.rope(latent_BLD.unsqueeze(2), positions=positions_BL[..., ::ratio]).squeeze(2)
        return rotated_BLD, latent_BLD
