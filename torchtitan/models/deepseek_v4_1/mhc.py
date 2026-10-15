# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Single-Pass mHC (manifold-constrained Hyper-Connections) for DeepSeek V4.1.

The V4.1 stack carries one stream mix across the whole model: the first
sub-layer collapses with the identity mix (``HcPre.identity_pre_mix``) and
every following sub-layer collapses with the previous one's freshly computed
``pre``; the final hidden state is collapsed once by the last block
(``HcPre.collapse``).  ``HcPre`` and ``HcPost`` follow the reference inference
implementation (``Block.hc_mixes`` / ``Block.hc_post``): the coefficient
derivation, the Sinkhorn balancing and the residual update are the same
operations in the same order.

Shape legend: ``B`` batch, ``L`` sequence length, ``D`` model dimension, ``hc``
``hc_mult`` residual branches.  The pinned torchtitan keeps a batch dim on
these tensors, so every contraction below names the branch axis of the
batch-first rank (``[B, L, hc, D]``); ``T`` (tokens) is the folded
batch-into-token form and does not appear here.
"""

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn
from torchtitan.models.base.module import Module


class HcPre(Module):
    """Head-collaboration pre step; owns its mixing parameters."""

    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        hc_mult: int = 4
        dim: int
        sinkhorn_iters: int = 20
        hc_eps: float = 1e-6
        norm_eps: float = 1e-6

    def __init__(self, config: Config):
        super().__init__()
        hc_mult = config.hc_mult
        mix_hc = (2 + hc_mult) * hc_mult
        hc_dim = hc_mult * config.dim
        self.hc_mult = config.hc_mult
        self.sinkhorn_iters = config.sinkhorn_iters
        self.hc_eps = config.hc_eps
        self.norm_eps = config.norm_eps
        self.hc_fn = nn.Parameter(torch.empty(mix_hc, hc_dim, dtype=torch.float32))
        self.hc_base = nn.Parameter(torch.empty(mix_hc, dtype=torch.float32))
        self.hc_scale = nn.Parameter(torch.empty(3, dtype=torch.float32))

    @staticmethod
    def identity_pre_mix(x_BLHcD: torch.Tensor, hc_mult: int) -> torch.Tensor:
        """Return the one-hot stream mix used at the input of the stack.

        Args:
            x: Residual branches of shape ``[B, L, hc, D]``; its leading dims are
                reused for the mix.
            hc_mult: Number of residual branches ``hc``.

        Returns:
            Mixing coefficients ``[B, L, hc]`` selecting branch 0.
        """
        pre_mix_BLHc = torch.zeros(
            (*x_BLHcD.shape[:-2], hc_mult),
            device=x_BLHcD.device,
            dtype=torch.float32,
        )
        pre_mix_BLHc[..., 0] = 1.0
        return pre_mix_BLHc

    @staticmethod
    def collapse(x_BLHcD: torch.Tensor, pre_mix_BLHc: torch.Tensor) -> torch.Tensor:
        """Collapse the branches into one sublayer input.

        ``[B, L, hc, D] x [B, L, hc] -> [B, L, D]``
        """
        y_BLD = torch.sum(pre_mix_BLHc.unsqueeze(-1) * x_BLHcD.float(), dim=2)
        return y_BLD.to(x_BLHcD.dtype)

    def _split_sinkhorn(self, mixes_BLM: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Split the mixing logits and balance ``comb`` to be doubly stochastic.

        Args:
            mixes: Mixing logits of shape ``[B, L, (2 + hc) * hc]``.

        Returns:
            ``(pre, post, comb)`` of shapes ``[B, L, hc]``, ``[B, L, hc]`` and
            ``[B, L, hc, hc]``: a row softmax followed by alternating row and
            column normalizations (Sinkhorn), so ``comb`` is doubly stochastic
            over the branches.
        """
        hc_mult = self.hc_mult
        pre_BLHc, post_BLHc, comb_BLMcc = mixes_BLM.split([hc_mult, hc_mult, hc_mult * hc_mult], dim=-1)
        comb_BLHcHc = comb_BLMcc.unflatten(-1, (hc_mult, hc_mult))

        pre_BLHc = torch.sigmoid(pre_BLHc * self.hc_scale[0] + self.hc_base[:hc_mult]) + self.hc_eps
        post_BLHc = 2 * torch.sigmoid(post_BLHc * self.hc_scale[1] + self.hc_base[hc_mult : 2 * hc_mult])
        comb_BLHcHc = comb_BLHcHc * self.hc_scale[2] + self.hc_base[2 * hc_mult :].view(hc_mult, hc_mult)

        comb_BLHcHc = torch.exp(comb_BLHcHc - comb_BLHcHc.amax(dim=-1, keepdim=True))
        comb_BLHcHc = comb_BLHcHc / comb_BLHcHc.sum(dim=-1, keepdim=True) + self.hc_eps
        comb_BLHcHc = comb_BLHcHc / (comb_BLHcHc.sum(dim=-2, keepdim=True) + self.hc_eps)
        for _ in range(self.sinkhorn_iters - 1):
            comb_BLHcHc = comb_BLHcHc / (comb_BLHcHc.sum(dim=-1, keepdim=True) + self.hc_eps)
            comb_BLHcHc = comb_BLHcHc / (comb_BLHcHc.sum(dim=-2, keepdim=True) + self.hc_eps)
        return pre_BLHc, post_BLHc, comb_BLHcHc

    def forward(
        self, x_BLHcD: torch.Tensor, pre_mix_BLHc: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Collapse the branches and emit the coefficients the next step consumes.

        Args:
            x: Residual branches of shape ``[B, L, hc, D]``.
            pre_mix: Input-mixing coefficients ``[B, L, hc]`` produced by the
                previous sub-layer (``identity_pre_mix`` at the stack input).

        Returns:
            ``(y, pre, post, comb)``: the sublayer input ``[B, L, D]``, the
            input-mixing coefficients the next sublayer collapses with
            ``[B, L, hc]``, and the ``post`` ``[B, L, hc]`` / ``comb``
            ``[B, L, hc, hc]`` consumed by :class:`HcPost`.
        """
        # One RMS statistic per token over the whole flattened hc * D stream.
        flat_BLN = x_BLHcD.flatten(2).float()
        rsqrt_BL1 = torch.rsqrt(flat_BLN.square().mean(-1, keepdim=True) + self.norm_eps)
        mixes_BLM = F.linear(flat_BLN, self.hc_fn) * rsqrt_BL1
        pre_BLHc, post_BLHc, comb_BLHcHc = self._split_sinkhorn(mixes_BLM)
        return self.collapse(x_BLHcD, pre_mix_BLHc), pre_BLHc, post_BLHc, comb_BLHcHc


class HcPost(Module):
    """Expand a sublayer output back to the residual branches and mix the residual in."""

    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        pass

    def __init__(self, config: Config):
        super().__init__()

    def forward(
        self,
        y_BLD: torch.Tensor,
        residual_BLHcD: torch.Tensor,
        post_BLHc: torch.Tensor,
        comb_BLHcHc: torch.Tensor,
    ) -> torch.Tensor:
        """Args:
            y: Sublayer output of shape ``[B, L, D]``.
            residual: Residual branches entering the sublayer, ``[B, L, hc, D]``.
            post: Output-mixing coefficients ``[B, L, hc]``, scaling ``y`` onto each branch.
            comb: Branch-mixing coefficients ``[B, L, hc, hc]``; entry ``(m, q)`` weights
                residual branch ``m`` into output branch ``q``.

        Returns:
            Updated residual branches of shape ``[B, L, hc, D]``.

        ``comb * residual`` has shape ``[B, L, hc, hc, D]`` with axes
        ``(b, l, m, q, d)``, so the contraction is over ``m`` (``dim=2``, the
        third-from-last); the batch dim is present here, so ``dim=-2`` would
        contract ``q`` and reduce the update to ``residual * comb.sum(-1)``: a
        per-branch rescaling instead of a mixing.
        """
        out_BLHcD = post_BLHc.unsqueeze(-1) * y_BLD.unsqueeze(-2) + torch.sum(
            comb_BLHcHc.unsqueeze(-1) * residual_BLHcD.unsqueeze(-2), dim=2
        )
        return out_BLHcD.type_as(y_BLD)
