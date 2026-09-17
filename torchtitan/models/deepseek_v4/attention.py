# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass

import spmd_types as spmd
import torch
from torch.nn.attention.flex_attention import BlockMask

from torchtitan.distributed.utils import get_spmd_backend
from torchtitan.models.common.attention import (
    apply_attention_sink_rescale,
    BaseAttention,
    FlexAttention,
)
from torchtitan.models.common.linear import Linear
from torchtitan.models.common.nn_modules import RMSNorm
from torchtitan.models.common.rope import RoPE
from torchtitan.tools.leaf_compile import leaf_compile

from .compressor import Compressor, Indexer


def _assert_spmd_attention_type(tensor, *, tp):
    if get_spmd_backend() == "spmd_types":
        spmd.assert_type(
            tensor,
            {"dp": spmd.S(0), "cp": spmd.S(1), "tp": tp},
        )



# ---------------------------------------------------------------------------
# Leaf-compiled attention pre/post-processing (see torchtitan/tools/leaf_compile.py).
#
# Eager, the q path is: per-head RMS normalisation (3 passes over the
# [T, H, 512] bf16 q), split, fp32 upcast + complex rotation of the rope tail,
# cast back, and a cat that copies the whole q again; the output path repeats
# the split / inverse rotation / cat on o. In the 8-node profile the cat and
# split copies alone were 5.6 % of kernel time. Each function below is one
# graph, so Inductor writes q (or o) exactly once. The rotation is the
# complex product written in real arithmetic -- (a + bi)(c + di) -- on the
# same adjacent-pair layout ComplexRoPE.apply_rotary_emb uses, so the math is
# unchanged; the complex cache is passed as its real view because Inductor
# does not lower complex tensors.
# ---------------------------------------------------------------------------


def _rotate_tail(x, cache_ri, *, rd, inverse):
    """Rotate the last ``rd`` features of ``x`` [T, H, D] by the RoPE cache
    given as ``view_as_real`` of the complex cache, shape [T, 1, rd // 2, 2]."""
    head, tail = x[..., :-rd], x[..., -rd:]
    pairs = tail.float().unflatten(-1, (-1, 2))
    a, b = pairs[..., 0], pairs[..., 1]
    c, d = cache_ri[..., 0], cache_ri[..., 1]
    if inverse:
        d = -d
    rotated = torch.stack((a * c - b * d, a * d + b * c), dim=-1)
    return torch.cat([head, rotated.flatten(-2).type_as(tail)], dim=-1)


@leaf_compile(group="attn")
def _q_norm_rope(q, cache_ri, *, rd, norm_eps):
    q = q * torch.rsqrt(q.square().mean(-1, keepdim=True) + norm_eps)
    return _rotate_tail(q, cache_ri, rd=rd, inverse=False)


@leaf_compile(group="attn")
def _o_rope_inverse(o, cache_ri, *, rd):
    return _rotate_tail(o, cache_ri, rd=rd, inverse=True)


class DSV4FlexAttention(FlexAttention):
    """DeepSeek sparse attention core for DeepSeek-V4.

    The core attends over the concatenated KV sequence ``[0, L + n_cmp)``,
    where the first ``L`` positions are the uncompressed sliding-window KV
    (``swa_k``) and the next ``n_cmp`` positions are the compressed KV
    (``cmp_k``):

    - sliding window: fixed pattern over ``swa_k``, expressed as a
      ``mask_mod`` predicate (no indices);
    - compressed blocks: for HCA (``compress_ratio=128``) all causal blocks
      are attendable, also a fixed ``mask_mod`` pattern; for CSA
      (``compress_ratio=4``) each query attends only its top-k selected
      compressed positions, which is the only dynamic (index-based) part.
    The learned per-head attention sink is applied to the output as
    ``out * sigmoid(lse - sink)`` -- identical to a zero-valued sink token in
    the softmax denominator, without a sink column in the kernel or a
    ``score_mod`` (PR #18, C. Lowman).

    The ``mask_mod`` is evaluated at token granularity inside flex_attention;
    the per-query-block KV block listing (``BlockMask.from_kv_blocks``) only
    restricts which blocks the kernel loads.

    Overrides can replace ``_build_block_mask`` (e.g. NPU varlen kernels) or
    the whole ``forward`` (e.g. fused SMLA/CSA kernels, which consume the raw
    ``q / swa_k / cmp_k / idx_q / idx_k / idx_w`` tensors). Under context
    parallelism, all-gathering ``idx_k`` and ``cmp_k`` at this module boundary
    enables global sparse selection.

    TODO: the indexer auxiliary loss is intentionally dropped for now; it will
    be re-added as a carrier-injected aux loss (see the NPU fork) once the
    general aux-loss mechanism lands.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(FlexAttention.Config):
        window_size: int
        compress_ratio: int
        softmax_scale: float
        index_topk: int

    def __init__(self, config: Config) -> None:
        super().__init__(config)
        self.window_size = config.window_size
        self.compress_ratio = config.compress_ratio
        self.softmax_scale = config.softmax_scale
        self.index_topk = config.index_topk
        self.block_size = config.block_size

    def get_window_topk_idxs(
        self,
        *,
        bsz: int,
        seqlen: int,
        device,
    ) -> torch.Tensor:
        """Return sliding-window KV indices in the concatenated KV space.

        Args:
            bsz: Batch size.
            seqlen: Query sequence length and uncompressed KV length.
            device: Device used for the generated index tensor.

        Returns:
            Tensor of shape ``[B, L, W]``. Valid entries are uncompressed KV
            positions in ``[0, L)`` and padded entries are ``-1``.
        """
        window = min(seqlen, self.window_size)
        q_idx = torch.arange(seqlen, device=device).unsqueeze(1)
        idxs = (q_idx - window + 1).clamp_min(0) + torch.arange(window, device=device)
        idxs = torch.where(idxs <= q_idx, idxs, -1)
        return idxs.unsqueeze(0).expand(bsz, -1, -1)

    def get_compress_topk_idxs(
        self,
        *,
        bsz: int,
        seqlen: int,
        n_cmp: int,
        device,
    ) -> torch.Tensor:
        """Return causal compressed KV indices in the concatenated KV space.

        Args:
            bsz: Batch size.
            seqlen: Query sequence length and uncompressed KV length.
            n_cmp: Number of compressed KV tokens.
            device: Device used for the generated index tensor.

        Returns:
            Tensor of shape ``[B, L, n_cmp]``. Valid entries are compressed KV
            positions offset by ``seqlen`` and padded entries are ``-1``.
        """
        if n_cmp == 0:
            return torch.empty((bsz, seqlen, 0), dtype=torch.int64, device=device)

        cmp_idx = torch.arange(n_cmp, device=device).repeat(seqlen, 1)
        causal_limit = torch.arange(1, seqlen + 1, device=device).unsqueeze(1)
        causal_limit = causal_limit // self.compress_ratio
        cmp_idx = torch.where(cmp_idx < causal_limit, seqlen + cmp_idx, -1)
        return cmp_idx.unsqueeze(0).expand(bsz, -1, -1)

    # Kept out of torch.compile on purpose. Under the per-block compile the
    # dense ``selected_mask`` this builds becomes an Inductor intermediate, and
    # the flex template asserts "convert FlexibleLayout to FixedLayout first"
    # while rendering the ``mask_mod`` that indexes it (job 395). Built eagerly,
    # it enters the flex region as a graph input with a fixed layout. The index
    # math here is small; the compile target is the HC-branch elementwise work.
    @torch.compiler.disable
    def _build_block_mask(
        self,
        bsz: int,
        seqlen: int,
        kv_len: int,
        selected_indices: torch.Tensor,
        device,
    ) -> BlockMask:
        """Build a FlexAttention block mask from selected KV indices.

        Args:
            bsz: Batch size.
            seqlen: Query sequence length.
            kv_len: Length of the concatenated KV sequence.
            selected_indices: Tensor of shape ``[B, L, K]`` containing final KV
                positions in ``[0, kv_len)``; ``-1`` entries are ignored.
            device: Device used for mask tensors.

        Returns:
            ``BlockMask`` whose block list and token-level predicate encode
            exactly the selected KV positions.
        """
        bs = self.block_size
        bq, bk = bs if isinstance(bs, tuple) else (bs, bs)
        assert (
            seqlen % bq == 0
        ), f"seqlen ({seqlen}) must be divisible by Q block size ({bq})"
        n_kv_blocks = (kv_len + bk - 1) // bk
        n_q_blocks = seqlen // bq

        valid = selected_indices >= 0
        safe_indices = selected_indices.clamp(0, kv_len - 1)

        selected_blocks = (safe_indices // bk).reshape(
            bsz, n_q_blocks, bq * selected_indices.size(-1)
        )
        block_values = valid.reshape(selected_blocks.shape).to(torch.int32)
        bm = torch.zeros(
            bsz, 1, n_q_blocks, n_kv_blocks, dtype=torch.int32, device=device
        )
        bm[:, 0].scatter_add_(-1, selected_blocks, block_values)
        bm = (bm > 0).to(torch.int32)
        kv_num_blocks = bm.sum(dim=-1).to(torch.int32)
        kv_indices = torch.argsort(bm, dim=-1, descending=True, stable=True).to(
            torch.int32
        )

        selected_count = torch.zeros(
            bsz, seqlen, kv_len, dtype=torch.int32, device=device
        )
        selected_count.scatter_add_(2, safe_indices, valid.to(torch.int32))
        selected_mask = selected_count > 0

        def dsa_mask_mod(b, h, q_idx, kv_idx):
            return selected_mask[b, q_idx, kv_idx]

        return BlockMask.from_kv_blocks(
            kv_num_blocks,
            kv_indices,
            BLOCK_SIZE=(bq, bk),
            mask_mod=dsa_mask_mod,
            seq_lengths=(seqlen, kv_len),
        )

    # Kept out of torch.compile on purpose, and called OUTSIDE spmd.no_typecheck():
    # job 413 (TORCH_LOGS=graph_breaks) showed the only graph break in the block
    # was this construction, and it broke *inside* the no_typecheck context
    # manager -- "Attempted to graph break in an active context manager that
    # doesn't support graph breaking" -- so Dynamo abandoned the whole block
    # frame and nothing compiled (job 408's profile: eager launches unchanged).
    # Built eagerly here, the mask enters the flex region as a graph input and
    # the rest of the block is compilable.
    @torch.compiler.disable
    def _eager_block_mask(self, *, seqlen, n_cmp, kv_len, idx_q, idx_k, idx_w, device):
        with spmd.no_typecheck():
            selected_indices = [
                self.get_window_topk_idxs(bsz=1, seqlen=seqlen, device=device)
            ]
            if self.compress_ratio == 4:
                if idx_q is None or idx_k is None or idx_w is None:
                    raise ValueError(
                        "DSV4FlexAttention requires idx_q, idx_k, "
                        "and idx_w when compress_ratio=4"
                    )
                cmp_topk = Indexer.select(
                    idx_q,
                    idx_k,
                    idx_w,
                    seqlen=seqlen,
                    ratio=self.compress_ratio,
                    topk=self.index_topk,
                ).unsqueeze(0)
                causal_limit = (
                    torch.arange(1, seqlen + 1, device=device).unsqueeze(1)
                    // self.compress_ratio
                )
                cmp_topk = torch.where(
                    cmp_topk < causal_limit.unsqueeze(0), seqlen + cmp_topk, -1
                )
                selected_indices.append(cmp_topk)
            elif self.compress_ratio > 1:
                selected_indices.append(
                    self.get_compress_topk_idxs(
                        bsz=1, seqlen=seqlen, n_cmp=n_cmp, device=device
                    )
                )
            selected_indices = torch.cat(selected_indices, dim=-1)

            block_mask = self._build_block_mask(
                1, seqlen, kv_len, selected_indices, device
            )
            return block_mask

    def _forward_impl(
        self,
        q,
        swa_k,
        attn_sink,
        *,
        cmp_k=None,
        idx_q=None,
        idx_k=None,
        idx_w=None,
        attention_masks=None,
    ) -> torch.Tensor:
        """Run DSV4 sparse attention over a folded token stream."""
        if attention_masks is not None:
            raise ValueError(
                "DSV4FlexAttention does not accept attention_masks; "
                "the DSA block mask is built internally."
            )
        if attn_sink is None:
            raise ValueError("DSV4FlexAttention requires attn_sink")

        seqlen = q.size(0)
        n_cmp = 0 if cmp_k is None else cmp_k.size(0)

        kv = swa_k.unsqueeze(1)
        if cmp_k is not None:
            kv = torch.cat([kv, cmp_k.unsqueeze(1)], dim=0)
        kv = kv.expand(-1, q.size(1), -1)

        block_mask = self._eager_block_mask(
            seqlen=seqlen,
            n_cmp=n_cmp,
            kv_len=kv.size(0),
            idx_q=idx_q,
            idx_k=idx_k,
            idx_w=idx_w,
            device=q.device,
        )

        with spmd.no_typecheck():

            def apply_sink(out_THV, lse_TH):
                return apply_attention_sink_rescale(out_THV, lse_TH, attn_sink)

            return super().forward(
                q,
                kv,
                kv,
                attention_masks=block_mask,
                scale=self.softmax_scale,
                out_transform=apply_sink,
            )


class SlidingWindowAttention(DSV4FlexAttention):
    @dataclass(kw_only=True, slots=True)
    class Config(DSV4FlexAttention.Config):
        pass

    def forward(  # pyrefly: ignore[bad-param-name-override]
        self,
        q,
        swa_k,
        attn_sink,
        *,
        attention_masks=None,
    ) -> torch.Tensor:
        return self._forward_impl(
            q,
            swa_k,
            attn_sink,
            attention_masks=attention_masks,
        )


class HeavilyCompressedAttention(DSV4FlexAttention):
    @dataclass(kw_only=True, slots=True)
    class Config(DSV4FlexAttention.Config):
        pass

    def forward(  # pyrefly: ignore[bad-param-name-override]
        self,
        q,
        swa_k,
        cmp_k,
        attn_sink,
        *,
        attention_masks=None,
    ) -> torch.Tensor:
        return self._forward_impl(
            q,
            swa_k,
            attn_sink,
            cmp_k=cmp_k,
            attention_masks=attention_masks,
        )


class CompressedSparseAttention(DSV4FlexAttention):
    @dataclass(kw_only=True, slots=True)
    class Config(DSV4FlexAttention.Config):
        pass

    def forward(  # pyrefly: ignore[bad-param-name-override]
        self,
        q,
        swa_k,
        cmp_k,
        idx_q,
        idx_k,
        idx_w,
        attn_sink,
        *,
        attention_masks=None,
    ) -> torch.Tensor:
        return self._forward_impl(
            q,
            swa_k,
            attn_sink,
            cmp_k=cmp_k,
            idx_q=idx_q,
            idx_k=idx_k,
            idx_w=idx_w,
            attention_masks=attention_masks,
        )


class Attention(BaseAttention):
    """DeepSeek V4 attention wrapper around sparse inner attention.

    The module projects Q/KV, applies pre- and post-phase RoPE, prepares
    optional compressed/indexer tensors, and delegates sparse attention to
    ``DSV4FlexAttention``.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(BaseAttention.Config):
        dim: int
        n_heads: int
        inner_attention: DSV4FlexAttention.Config  # pyrefly: ignore [bad-override]
        rope: RoPE.Config
        head_dim: int = 512
        rope_head_dim: int = 64
        q_lora_rank: int = 1024
        o_lora_rank: int = 1024
        n_groups: int = 8
        compress_ratio: int = 1
        norm_eps: float = 1e-6
        index_n_heads: int = 64
        index_head_dim: int = 128
        n_layers: int = 4
        layer_id: int = 0
        mask_type: str = "causal"

        # Sub-module configs — declared as fields so the sharding system can
        # set sharding_config on them before build().
        wq_a: Linear.Config
        q_norm: RMSNorm.Config
        wq_b: Linear.Config
        wkv: Linear.Config
        kv_norm: RMSNorm.Config
        wo_a: Linear.Config
        wo_b: Linear.Config
        attn_sink: Linear.Config

        # Compressor/indexer are conditional, so keep them here too.
        compressor: Compressor.Config | None = None
        compressor_128: Compressor.Config | None = None
        indexer: Indexer.Config | None = None

    def __init__(self, config: Config):
        super().__init__()
        cfg = config
        self.n_heads = cfg.n_heads
        self.head_dim = cfg.head_dim
        self.rope_head_dim = cfg.rope_head_dim
        self.q_lora_rank = cfg.q_lora_rank
        self.o_lora_rank = cfg.o_lora_rank
        self.n_groups = cfg.n_groups
        self.compress_ratio = cfg.compress_ratio
        self.norm_eps = cfg.norm_eps
        self.softmax_scale = cfg.head_dim**-0.5
        self.layer_id = cfg.layer_id
        self.n_layers = cfg.n_layers
        self.rope = cfg.rope.build()

        # Build all sub-modules from their configs.
        self.wq_a = cfg.wq_a.build()
        self.q_norm = cfg.q_norm.build()
        self.wq_b = cfg.wq_b.build()
        self.wkv = cfg.wkv.build()
        self.kv_norm = cfg.kv_norm.build()
        self.wo_a = cfg.wo_a.build()
        self.wo_b = cfg.wo_b.build()
        self.attn_sink = cfg.attn_sink.build()

        if cfg.compressor is not None:
            self.compressor = cfg.compressor.build()
        if cfg.indexer is not None:
            self.indexer = cfg.indexer.build()
        if cfg.compressor_128 is not None:
            self.compressor_128 = cfg.compressor_128.build()

        self.inner_attention = cfg.inner_attention.build()

    def forward(self, x, attention_masks=None, positions=None):
        """Apply one DeepSeek V4 attention layer over folded tokens."""
        num_tokens = x.size(0)
        rd = self.rope_head_dim

        qr = self.q_norm(self.wq_a(x))
        q = self.wq_b(qr)
        with spmd.local():
            q = q.view(num_tokens, -1, self.head_dim)
            _assert_spmd_attention_type(q, tp=spmd.S(1))
        # Complex cache for these positions, as a real [T, 1, rd/2, 2] view;
        # shared by the q path here and the inverse rotation of o below.
        rope_cache_ri = torch.view_as_real(
            self.rope._reshape_cache(q[..., -rd:], positions)
        )
        q = _q_norm_rope(q, rope_cache_ri, rd=rd, norm_eps=self.norm_eps)

        kv = self.kv_norm(self.wkv(x))
        kv_nope, kv_rope = torch.split(kv, [self.head_dim - rd, rd], dim=-1)
        kv_rope = self.rope(kv_rope.unsqueeze(1), positions=positions)
        kv = torch.cat([kv_nope, kv_rope.squeeze(1)], dim=-1)

        cmp_k = idx_q = idx_k = idx_w = None
        if self.compress_ratio > 1 and hasattr(self, "indexer"):
            idx_q, idx_k, idx_w = self.indexer(
                x.detach(), qr.detach(), positions=positions
            )
        if self.compress_ratio == 4:
            cmp_k = self.compressor(x, positions=positions)
        elif self.compress_ratio > 1:
            cmp_k = self.compressor_128(x, positions=positions)

        attn_sink_param = self.attn_sink.weight.squeeze(-1)
        if self.compress_ratio == 4:
            o = self.inner_attention(
                q,
                kv,
                cmp_k,
                idx_q,
                idx_k,
                idx_w,
                attn_sink_param,
                attention_masks=attention_masks,
            )
        elif self.compress_ratio > 1:
            o = self.inner_attention(
                q,
                kv,
                cmp_k,
                attn_sink_param,
                attention_masks=attention_masks,
            )
        else:
            o = self.inner_attention(
                q,
                kv,
                attn_sink_param,
                attention_masks=attention_masks,
            )

        o = _o_rope_inverse(o, rope_cache_ri, rd=rd)

        with spmd.local():
            n_local_heads = o.shape[1]
            n_local_groups = self.n_groups // (self.n_heads // n_local_heads)
            o = o.view(num_tokens, n_local_groups, -1)
            _assert_spmd_attention_type(o, tp=spmd.S(1))
            wo_a = self.wo_a.weight.view(n_local_groups, self.o_lora_rank, -1)
            if get_spmd_backend() == "spmd_types" and spmd.is_type_checking():
                spmd.assert_type(
                    wo_a,
                    {"dp": spmd.R, "cp": spmd.R, "tp": spmd.S(0)},
                )
        o = torch.einsum("tgd,grd->tgr", o, wo_a)
        with spmd.local():
            o = o.reshape(num_tokens, -1)
            _assert_spmd_attention_type(o, tp=spmd.S(1))
        return self.wo_b(o)
