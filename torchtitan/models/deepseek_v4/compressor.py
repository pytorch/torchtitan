# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass
from functools import cache

import torch
import torch.nn.functional as F
import torch_remat as remat
from attn_gym.sparse import lightning_indexer
from torch import nn
from torch.distributed.tensor import DTensor, Replicate

from torchtitan.models.common.linear import Linear
from torchtitan.models.common.nn_modules import RMSNorm
from torchtitan.models.common.rope import RoPE
from torchtitan.protocols.module import Module
from torchtitan.tools.utils import has_cuda_capability

# Shape suffixes: G = compressed groups, R = compression ratio.


@cache
def _hadamard(dim: int, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    if dim & (dim - 1) != 0:
        raise ValueError("Hadamard dim must be a power of two")
    h = torch.ones((1, 1), dtype=dtype, device=device)
    while h.shape[0] < dim:
        h = torch.cat([torch.cat([h, h], 1), torch.cat([h, -h], 1)], 0)
    return h


def compressed_cu_seqlens(cu_seqlens: torch.Tensor, ratio: int) -> torch.Tensor:
    """Offsets of each document's compressed entries: ``len_d // ratio`` per document.

    A document's trailing partial group forms no entry, matching Attention Gym's
    packed ``cu_seqlens_k`` convention.
    """
    counts = torch.diff(cu_seqlens) // ratio
    return F.pad(counts.cumsum(0, dtype=torch.int32), (1, 0))


class Compressor(Module):
    """Compress local hidden states into lower-rate KV tokens.

    The compressor scores each token inside a compression group, forms a
    weighted sum, normalizes the result, and applies RoPE to the rope slice.
    For ``compress_ratio == 4`` it also includes the previous group's value as
    the overlapping candidate used by CSA.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        rope: RoPE.Config
        wkv: Linear.Config
        wgate: Linear.Config
        norm: RMSNorm.Config
        head_dim: int = 512
        rope_head_dim: int = 64
        compress_ratio: int = 4

    def __init__(self, config: Config):
        super().__init__()
        cfg = config
        self.head_dim = cfg.head_dim
        self.rope_head_dim = cfg.rope_head_dim
        self.compress_ratio = cfg.compress_ratio
        self.overlap = cfg.compress_ratio == 4
        self.rope = cfg.rope.build()

        self.wkv = cfg.wkv.build()
        self.wgate = cfg.wgate.build()
        self.norm = cfg.norm.build()
        self.ape = nn.Parameter(torch.empty(cfg.compress_ratio, self.wkv.out_features))

    def _overlap_transform(self, tensor, value=0, first_G=None):
        """Append previous-token overlap candidates along the ratio dimension.

        Args:
            tensor: Grouped tensor of shape ``[G, R, D]``.
            value: Fill value for a group with no previous group.
            first_G: Optional bool mask of groups that start a packed document;
                they take ``value`` like the sequence's first group.

        Returns:
            Tensor of shape ``[G, 2 * R, D]``.
        """
        d = self.head_dim
        prev = torch.cat(
            [
                torch.full_like(tensor[:1, :, :d], value),
                tensor[:-1, :, :d],
            ],
            dim=0,
        )
        if first_G is not None:
            prev = prev.masked_fill(first_G[:, None, None], value)
        curr = tensor[:, :, d:]
        return torch.cat([prev, curr], dim=1)

    def forward(self, x, positions, cu_seqlens=None):
        """Compress hidden states into compressed KV states.

        Args:
            x: Hidden states of shape ``[L, D_model]``.
            positions: Position IDs of shape ``[L]``.
            cu_seqlens: Optional packed document offsets ``[N + 1]``. Groups then
                restart at each document start, a document's trailing partial
                group forms no entry, and the overlap candidate never crosses a
                document boundary.

        Returns:
            Compressed KV tensor of shape ``[L // compress_ratio, head_dim]``. With
            ``cu_seqlens``, entries are packed by document at
            ``compressed_cu_seqlens(cu_seqlens, compress_ratio)`` and later slots
            are unused.
        """
        seqlen = x.size(0)
        rd = self.rope_head_dim
        ratio = self.compress_ratio
        dtype = x.dtype
        with torch.autocast(device_type=x.device.type, dtype=torch.float32):
            kv = self.wkv(x)
            score = self.wgate(x)
        # The softmax pooling below reads the wkv and wgate projection outputs
        # with bare ops.
        remat.recompute_needs_tensor(kv, score)
        # Compressed entry j summarizes tokens [j * ratio, (j + 1) * ratio) of its
        # document and takes the position of its first token, as in the
        # DeepSeek-V4 reference.
        first_G = None
        if cu_seqlens is None:
            if seqlen % ratio != 0:
                raise ValueError(
                    f"seqlen ({seqlen}) must be divisible by compress_ratio ({ratio})"
                )
            comp_positions = (
                positions[::ratio]
                if positions is not None
                else torch.arange(0, seqlen, ratio, device=x.device)
            )
            kv = kv.unflatten(0, (-1, ratio))
            score = score.unflatten(0, (-1, ratio))
        else:
            if seqlen < ratio:
                # Shorter than one group: no document has a compressed entry.
                return x.new_zeros(0, self.head_dim)
            # Entries are packed by document at compressed_cu_seqlens offsets in
            # a fixed pool of seqlen // ratio slots, so no host sync is needed.
            # Slots past the last complete group are unused and read token 0.
            cmp_cu = compressed_cu_seqlens(cu_seqlens, ratio)
            slot_G = torch.arange(seqlen // ratio, dtype=torch.int32, device=x.device)
            doc_G = torch.searchsorted(cmp_cu[1:], slot_G, right=True, out_int32=True)
            group_G = slot_G - cmp_cu[doc_G]
            start_G = torch.where(
                slot_G < cmp_cu[-1], cu_seqlens[doc_G] + group_G * ratio, 0
            )
            token_GR = start_G.unsqueeze(1) + torch.arange(ratio, device=x.device)
            kv, score = kv[token_GR], score[token_GR]
            comp_positions = (
                positions[start_G] if positions is not None else group_G * ratio
            )
            first_G = group_G == 0
        score = score + self.ape
        if self.overlap:
            kv = self._overlap_transform(kv, 0, first_G)
            score = self._overlap_transform(score, float("-inf"), first_G)
        kv = (kv * score.softmax(dim=1)).sum(dim=1)
        kv = self.norm(kv.to(dtype))
        kv_nope, kv_rope = torch.split(kv, [self.head_dim - rd, rd], dim=-1)
        kv_rope = self.rope(kv_rope.unsqueeze(1), positions=comp_positions)
        kv = torch.cat([kv_nope, kv_rope.squeeze(1)], dim=-1)
        return kv


class Indexer(Module):
    """Produce low-dimensional query/key features for CSA top-k selection."""

    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        rope: RoPE.Config
        wq_b: Linear.Config
        weights_proj: Linear.Config
        compressor: "Compressor.Config"
        num_index_heads: int = 64
        index_head_dim: int = 128
        rope_head_dim: int = 64

    def __init__(self, config: Config):
        super().__init__()
        cfg = config
        self.num_index_heads = cfg.num_index_heads
        self.head_dim = cfg.index_head_dim
        self.rope_head_dim = cfg.rope_head_dim
        self.softmax_scale = cfg.index_head_dim**-0.5
        self.rope = cfg.rope.build()

        self.wq_b = cfg.wq_b.build()
        self.weights_proj = cfg.weights_proj.build()
        self.compressor = cfg.compressor.build()

    @staticmethod
    def _rotate_activation(x):
        dim = x.size(-1)
        hadamard_mat = _hadamard(dim, dtype=x.dtype, device=x.device)
        if isinstance(x, DTensor):
            hadamard_mat = DTensor.from_local(
                hadamard_mat,
                x.device_mesh,
                [Replicate()] * x.device_mesh.ndim,
                run_check=False,
            )
        return F.linear(x, hadamard_mat) * (dim**-0.5)

    def forward(
        self,
        x,
        qr,
        *,
        positions,
        cu_seqlens=None,
    ):
        """Project raw indexer queries, keys, and per-head weights."""
        seqlen = x.size(0)
        rd = self.rope_head_dim
        q = self.wq_b(qr)
        q = q.view(seqlen, self.num_index_heads, self.head_dim)
        q_nope, q_rope = torch.split(q, [self.head_dim - rd, rd], dim=-1)
        # rope and the concat read the wq_b projection output with bare ops.
        remat.recompute_needs_tensor(q_nope, q_rope)
        q_rope = self.rope(q_rope, positions=positions)
        q = torch.cat([q_nope, q_rope], dim=-1)
        q = self._rotate_activation(q)
        k = self.compressor(x, positions=positions, cu_seqlens=cu_seqlens)
        k = self._rotate_activation(k)
        weights = self.weights_proj(x)
        # The scale reads the weights_proj output with bare ops.
        remat.recompute_needs_tensor(weights)
        weights = weights * (self.softmax_scale * self.num_index_heads**-0.5)
        return q, k, weights

    @staticmethod
    def select(
        idx_q,
        idx_k,
        idx_w,
        *,
        max_seqlen: int,
        ratio: int,
        topk: int,
        cu_seqlens: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Select top-k compressed positions per folded query token.

        Uses Attention Gym's ``lightning_indexer``: score[t, s] is proportional
        to sum_h(w[t, h] * relu(q[t, h] . k[s])) (it applies a further positive
        scale, so the ranking is unchanged), with query ``t`` limited to
        compressed positions ``s < (t + 1) // ratio``.

        ``max_seqlen`` is the longest sequence a query can attend: ``T``, or the
        longest document when packed, which bounds the selectable positions.

        Returns:
            int32 ``[T, min(topk, max_seqlen // ratio)]`` indices into
            ``idx_k``, in unspecified order. Slots a query cannot causally
            select are ``-1``. With packed ``cu_seqlens`` the indices are local
            to the query's document in the packed ``idx_k``.
        """
        # The fused kernels need fp16/bf16 on NVIDIA SM90 or newer and do not
        # fall back; CPU, fp32, ROCm, and older GPUs use the reference
        # implementation.
        fused = (
            idx_q.is_cuda
            and idx_q.dtype in (torch.float16, torch.bfloat16)
            and has_cuda_capability(9, 0)
        )
        topk_indices = lightning_indexer(
            idx_q.unsqueeze(0),
            idx_k.unsqueeze(0),
            idx_w.unsqueeze(0),
            min(topk, max_seqlen // ratio),
            causal=True,
            compress_ratio=ratio,
            cu_seqlens=cu_seqlens,
            cu_seqlens_k=(
                None if cu_seqlens is None else compressed_cu_seqlens(cu_seqlens, ratio)
            ),
            impl="fused" if fused else "reference",
        )
        return topk_indices.squeeze(0)
