# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""CSA2 with explicit, differentiable state shared by Full/Reindex/Reuse layers.

T: tokens, S: compressed tokens, H: heads, D: latent width, K: selected keys,
C: candidate keys, G: output groups, R: output LoRA rank, P: pooling positions.
"""

from dataclasses import dataclass
from typing import Literal

import spmd_types as spmd
import torch
import torch.nn.functional as F
import torch_remat as remat

from torchtitan.distributed.spmd_types import spmd_dense_sp_enabled, spmd_mesh_group
from torchtitan.models.base.module import Module
from torchtitan.models.common.attention import Attention
from torchtitan.models.common.attn_gym import AuxRequest, gather_attn, lightning_indexer
from torchtitan.models.common.linear import Linear
from torchtitan.models.common.nn_modules import RMSNorm
from torchtitan.models.common.rope import RoPE
from torchtitan.models.deepseek_v4.attention import DSV4InnerAttention
from torchtitan.models.deepseek_v4.compressor import (
    compressed_cu_seqlens,
    compressed_gather_indices,
)

from .quantization import fp4_activation, fp8_activation


def rms_norm(norm: RMSNorm, x_TD):
    return norm(x_TD.float()).to(x_TD.dtype)


def _rotate_shared_keys(rope: RoPE, k_SD, positions_S, *, rope_head_dim: int):
    if k_SD.size(0) == 0:
        return k_SD
    return torch.cat(
        (
            k_SD[:, :-rope_head_dim],
            rope(k_SD[:, None, -rope_head_dim:], positions=positions_S).squeeze(1),
        ),
        -1,
    )


class Compressor(Module):
    """Non-overlapping, per-feature softmax pooling; ratio one is a projection."""

    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        ratio: int
        wkv: Linear.Config
        norm: RMSNorm.Config
        wgate: Linear.Config | None = None

        def __post_init__(self):
            if self.ratio not in (1, 2) or (self.ratio == 2) != (
                self.wgate is not None
            ):
                raise ValueError(
                    "CSA2 uses ratio one without a gate or ratio two with a gate."
                )

    def __init__(self, config: Config):
        super().__init__()
        self.ratio = config.ratio
        self.wkv = config.wkv.build()
        self.norm = config.norm.build()
        self.wgate = config.wgate.build() if config.wgate is not None else None
        if self.ratio > 1:
            self.wkv.float()
            assert self.wgate is not None
            self.wgate.float()

    def forward(self, x_TD, positions_T, *, cu_seqlens=None):
        ratio = self.ratio
        if ratio == 1:
            latent_TD = self.wkv(x_TD)
            remat.recompute_needs_tensor(latent_TD)
            return rms_norm(self.norm, latent_TD), positions_T
        if cu_seqlens is None:
            starts_S = torch.arange(x_TD.size(0) // ratio, device=x_TD.device) * ratio
        else:
            compressed_cu = compressed_cu_seqlens(cu_seqlens, ratio)
            slots_S = torch.arange(
                x_TD.size(0) // ratio, dtype=torch.int32, device=x_TD.device
            )
            docs_S = torch.searchsorted(compressed_cu[1:], slots_S, right=True)
            starts_S = torch.where(
                slots_S < compressed_cu[-1],
                cu_seqlens[docs_S] + (slots_S - compressed_cu[docs_S]) * ratio,
                0,
            )
        gather_SP = starts_S[:, None] + torch.arange(ratio, device=x_TD.device)
        with torch.autocast(device_type=x_TD.device.type, enabled=False):
            latent_TD = self.wkv(x_TD.float())
            remat.recompute_needs_tensor(latent_TD)
            latent_SPD = latent_TD[gather_SP]
            assert self.wgate is not None
            score_TD = self.wgate(x_TD.float())
            remat.recompute_needs_tensor(score_TD)
            score_SPD = score_TD[gather_SP]
            latent_SD = (latent_SPD * score_SPD.softmax(-2)).sum(-2).to(x_TD.dtype)
        return rms_norm(self.norm, latent_SD), positions_T[starts_S]


def _query_geometry(positions_T, cu_seqlens, *, ratio: int):
    limits_T = (positions_T + 1) // ratio
    if cu_seqlens is None:
        return limits_T, torch.zeros_like(limits_T)
    docs_T = torch.searchsorted(
        cu_seqlens[1:],
        torch.arange(positions_T.numel(), device=positions_T.device, dtype=torch.int32),
        right=True,
    )
    return limits_T, compressed_cu_seqlens(cu_seqlens, ratio)[docs_T]


@torch.no_grad()
def hierarchical_select(
    q_THD,
    k_SD,
    weights_TH,
    positions_T,
    *,
    topk: int,
    ratio: int,
    cu_seqlens=None,
    candidates_TC=None,
    num_candidate_blocks: int = 0,
    block_size: int = 8,
    chunk_size: int = 128,
    max_seqlen: int
):
    """Bound working scores by query/key chunks or the carried candidate pool.

    Candidate IDs and selected IDs are local to the query's document. The
    candidate source pins the newest reachable block before selecting blocks.
    Later Reindex layers gather and score only those candidate IDs.
    """
    width = min(k_SD.size(0), max_seqlen // ratio)
    selected_width = min(topk, width)
    candidate_width = (
        min(num_candidate_blocks, (width + block_size - 1) // block_size) * block_size
    )
    limits_T, offsets_T = _query_geometry(positions_T, cu_seqlens, ratio=ratio)
    selected, candidates = [], []
    if width == 0:
        return torch.empty(
            q_THD.size(0), 0, device=q_THD.device, dtype=torch.int32
        ), torch.empty(q_THD.size(0), 0, device=q_THD.device, dtype=torch.int32)
    for start in range(0, q_THD.size(0), chunk_size):
        stop = min(start + chunk_size, q_THD.size(0))
        q_QHD, w_QH = q_THD[start:stop].float(), weights_TH[start:stop].float()
        limit_Q = limits_T[start:stop]
        offset_Q = offsets_T[start:stop]
        best_QK = torch.full((stop - start, 0), -torch.inf, device=q_THD.device)
        indices_QK = torch.full_like(best_QK, -1, dtype=torch.int64)
        block_scores_QC = torch.full(
            (stop - start, 0),
            -torch.inf,
            device=q_THD.device,
        )
        blocks_QC = torch.full_like(block_scores_QC, -1, dtype=torch.int64)
        # Full scans stream keys. Reindex scans only the bounded candidates.
        scan_width = width if candidates_TC is None else candidates_TC.size(1)
        key_chunk = max(block_size, 1024 // block_size * block_size)
        for key_start in range(0, scan_width, key_chunk):
            key_stop = min(key_start + key_chunk, scan_width)
            ids_QS = (
                torch.arange(key_start, key_stop, device=q_THD.device).expand(
                    stop - start, -1
                )
                if candidates_TC is None
                else candidates_TC[start:stop, key_start:key_stop].long()
            )
            valid_QS = (ids_QS >= 0) & (ids_QS < limit_Q[:, None])
            gather_QS = (ids_QS.clamp_min(0) + offset_Q[:, None]).clamp_max(
                k_SD.size(0) - 1
            )
            logits_QHS = torch.einsum("qhd,qsd->qhs", q_QHD, k_SD[gather_QS].float())
            scores_QS = (
                (logits_QHS.relu() * w_QH[:, :, None])
                .sum(1)
                .masked_fill(~valid_QS, -torch.inf)
            )
            joined_QS = torch.cat((best_QK, scores_QS), -1)
            joined_ids_QS = torch.cat((indices_QK, ids_QS), -1)
            # Seed with actual scores. Extra -inf slots change topk's tie order.
            best_QK, picks_QK = joined_QS.topk(
                min(selected_width, joined_QS.size(-1)), -1, sorted=False
            )
            indices_QK = joined_ids_QS.gather(-1, picks_QK)
            if candidate_width:
                scores_QB = (
                    F.pad(
                        scores_QS,
                        (0, -scores_QS.size(-1) % block_size),
                        value=-torch.inf,
                    )
                    .unflatten(-1, (-1, block_size))
                    .amax(-1)
                )
                block_ids_QB = torch.arange(
                    key_start // block_size,
                    (key_stop + block_size - 1) // block_size,
                    device=q_THD.device,
                ).expand(stop - start, -1)
                scores_QB = scores_QB.masked_fill(
                    (limit_Q[:, None] > 0)
                    & (block_ids_QB == (limit_Q[:, None] - 1) // block_size),
                    torch.inf,
                )
                joined_QB = torch.cat((block_scores_QC, scores_QB), -1)
                joined_ids_QB = torch.cat((blocks_QC, block_ids_QB), -1)
                block_scores_QC, picks_QC = joined_QB.topk(
                    min(candidate_width // block_size, joined_QB.size(-1)),
                    -1,
                    sorted=False,
                )
                blocks_QC = joined_ids_QB.gather(-1, picks_QC)
        valid_QK = torch.isfinite(best_QK)
        indices_QK = indices_QK.masked_fill(~valid_QK, width).sort(-1).values
        selected.append(torch.where(indices_QK < width, indices_QK, -1).int())
        if candidate_width:
            ids_QCS = blocks_QC.unsqueeze(-1) * block_size + torch.arange(
                block_size, device=q_THD.device
            )
            valid_QCS = (block_scores_QC > -torch.inf).unsqueeze(-1) & (
                ids_QCS < limit_Q[:, None, None]
            )
            ids_QC = (
                ids_QCS.flatten(-2)
                .masked_fill(~valid_QCS.flatten(-2), width)
                .sort(-1)
                .values
            )
            candidates.append(torch.where(ids_QC < width, ids_QC, -1).int())
    selected_TK = torch.cat(selected)
    candidates_TC = (
        torch.cat(candidates)
        if candidates
        else torch.empty(q_THD.size(0), 0, device=q_THD.device, dtype=torch.int32)
    )
    return selected_TK, candidates_TC


class Indexer(Module):
    """Own index queries, and own index keys only at a Full layer."""

    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        num_heads: int
        head_dim: int
        rope_head_dim: int
        ratio: int
        topk: int
        wq_b: Linear.Config
        weights_proj: Linear.Config
        rope: RoPE.Config
        wk: Linear.Config | None = None
        k_norm: RMSNorm.Config | None = None
        candidate_source: bool = False
        uses_candidates: bool = False
        num_candidate_blocks: int = 2048
        candidate_block_size: int = 8
        quantize: bool = True
        impl: Literal["fused", "reference"] = "fused"

    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        self.wq_b, self.weights_proj = config.wq_b.build(), config.weights_proj.build()
        self.wk = config.wk.build() if config.wk is not None else None
        self.k_norm = config.k_norm.build() if config.k_norm is not None else None
        self.rope = config.rope.build()

    def forward(
        self,
        x_TD,
        qr_TR,
        latent_SD,
        key_positions_S,
        positions_T,
        index_k_SD,
        candidates_TC,
        *,
        cu_seqlens=None,
        max_seqlen: int
    ):
        cfg, rd = self.config, self.config.rope_head_dim
        if self.wk is not None:
            assert self.k_norm is not None
            k_SD = self.wk(latent_SD.detach())
            remat.recompute_needs_tensor(k_SD)
            k_SD = rms_norm(self.k_norm, k_SD)
            index_k_SD = _rotate_shared_keys(
                self.rope, k_SD, key_positions_S, rope_head_dim=rd
            )
            if cfg.quantize:
                index_k_SD = fp4_activation(index_k_SD)
        q_TD = self.wq_b(qr_TR.detach())
        remat.recompute_needs_tensor(q_TD)
        q_THD = q_TD.unflatten(-1, (cfg.num_heads, cfg.head_dim))
        q_THD = torch.cat(
            (q_THD[..., :-rd], self.rope(q_THD[..., -rd:], positions=positions_T)), -1
        )
        if cfg.quantize:
            q_THD = fp4_activation(q_THD)
        weights_TH = self.weights_proj(x_TD.detach())
        remat.recompute_needs_tensor(weights_TH)
        weights_TH = weights_TH * (cfg.head_dim**-0.5 * cfg.num_heads**-0.5)
        if cfg.candidate_source or cfg.uses_candidates:
            indices_TK, new_candidates_TC = hierarchical_select(
                q_THD,
                index_k_SD,
                weights_TH,
                positions_T,
                topk=cfg.topk,
                ratio=cfg.ratio,
                cu_seqlens=cu_seqlens,
                candidates_TC=candidates_TC if cfg.uses_candidates else None,
                num_candidate_blocks=cfg.num_candidate_blocks
                if cfg.candidate_source
                else 0,
                block_size=cfg.candidate_block_size,
                max_seqlen=max_seqlen,
            )
            if cfg.candidate_source:
                candidates_TC = new_candidates_TC
        else:
            with torch.no_grad():
                indices_TK = lightning_indexer(
                    q_THD.unsqueeze(0),
                    index_k_SD.unsqueeze(0),
                    weights_TH.unsqueeze(0),
                    min(cfg.topk, max_seqlen // cfg.ratio),
                    causal=True,
                    compress_ratio=cfg.ratio,
                    cu_seqlens=cu_seqlens,
                    cu_seqlens_k=None
                    if cu_seqlens is None
                    else compressed_cu_seqlens(cu_seqlens, cfg.ratio),
                    impl=cfg.impl,
                ).squeeze(0)
            width = max_seqlen // cfg.ratio
            indices_TK = indices_TK.masked_fill(indices_TK < 0, width).sort(-1).values
            indices_TK = torch.where(indices_TK < width, indices_TK, -1).int()
        gather_TK = compressed_gather_indices(indices_TK, cu_seqlens, cfg.ratio)
        scores = []
        for start in range(0, x_TD.size(0), 128):
            logits_THK = torch.einsum(
                "thd,tkd->thk",
                q_THD[start : start + 128].float(),
                index_k_SD[gather_TK[start : start + 128]].float(),
            )
            scores.append(
                (logits_THK.relu() * weights_TH[start : start + 128, :, None]).sum(1)
            )
        scores_TK = torch.cat(scores).masked_fill(indices_TK < 0, -torch.inf)
        return index_k_SD, indices_TK, scores_TK, candidates_TC


class CSA2InnerAttention(DSV4InnerAttention):
    @dataclass(kw_only=True, slots=True)
    class Config(DSV4InnerAttention.Config):
        impl: Literal["fused", "reference"] = "fused"

    def __init__(self, config: Config):
        super().__init__(config)
        self.impl = config.impl

    def forward(
        self,
        q_THD,
        local_k_TD,
        global_k_SD,
        indices_TK,
        scores_TK,
        sink_H,
        *,
        attention_metadata=None,
        aux_loss_denominator=None,
        padding_mask_T=None
    ):
        cu_seqlens = None if attention_metadata is None else attention_metadata.cu_seq_q
        with spmd.no_typecheck():
            cu_seqlens_k = None
            if cu_seqlens is not None:
                cu_seqlens_k = (
                    compressed_cu_seqlens(cu_seqlens, self.compress_ratio)
                    if global_k_SD.size(0)
                    else torch.zeros_like(cu_seqlens)
                )
            aux_loss = self.aux_loss if self.training else None
            out_1HTD, aux = gather_attn(
                q_THD.transpose(0, 1).unsqueeze(0),
                local_k_TD[None, None],
                global_k_SD[None, None],
                indices_TK.unsqueeze(0),
                attention_sink=sink_H,
                sliding_window_size=self.window_size,
                cu_seqlens=cu_seqlens,
                cu_seqlens_k=cu_seqlens_k,
                scale=self.softmax_scale,
                impl=self.impl,
                return_aux=AuxRequest(lse=aux_loss is not None),
            )
            out_THD = out_1HTD.squeeze(0).transpose(0, 1)
            if aux_loss is not None:
                assert aux.lse is not None
                if aux_loss_denominator is None:
                    raise RuntimeError(
                        "SparseIndexerLoss requires aux_loss_denominator."
                    )
                out_THD = aux_loss(
                    q_THD.detach(),
                    global_k_SD.detach(),
                    indices_TK,
                    scores_TK,
                    aux.lse.squeeze(0).transpose(0, 1).detach(),
                    cu_seqlens,
                    carrier=out_THD,
                    denominator=aux_loss_denominator,
                    padding_mask_T=padding_mask_T,
                )
        if spmd.is_type_checking():
            spmd.assert_type_like(out_THD, q_THD)
        return out_THD


class DeepSeekV41Attention(Attention):
    """Local MLA plus a shared CSA2 pool. Every layer still follows the hidden chain."""

    @dataclass(kw_only=True, slots=True)
    class Config(Attention.Config):
        inner_attention: CSA2InnerAttention.Config  # pyrefly: ignore [bad-override]
        head_dim: int
        rope_head_dim: int
        num_groups: int
        o_lora_rank: int
        ratio: int
        rope: RoPE.Config
        wq_a: Linear.Config
        q_norm: RMSNorm.Config
        wq_b: Linear.Config
        wkv: Linear.Config
        kv_norm: RMSNorm.Config
        wo_a: Linear.Config
        wo_b: Linear.Config
        attn_sink: Linear.Config
        compressor: Compressor.Config | None = None
        indexer: Indexer.Config | None = None
        quantize: bool = True

    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        self.wq_a = config.wq_a.build()
        self.q_norm: RMSNorm = config.q_norm.build()
        self.wq_b = config.wq_b.build()
        self.wkv = config.wkv.build()
        self.kv_norm: RMSNorm = config.kv_norm.build()
        self.wo_a: Linear = config.wo_a.build()
        self.wo_b = config.wo_b.build()
        self.attn_sink: Linear = config.attn_sink.build()
        self.rope: RoPE = config.rope.build()
        # pyrefly: ignore [bad-override]
        self.inner_attention: CSA2InnerAttention = config.inner_attention.build()
        self.compressor = (
            config.compressor.build() if config.compressor is not None else None
        )
        self.indexer = config.indexer.build() if config.indexer is not None else None
        self.attn_sink.float()

    def _gather_input(self, x_TD):
        tp_group = spmd_mesh_group("tp")
        if tp_group is not None:
            x_TD = spmd.redistribute(
                x_TD,
                tp_group,
                src=spmd.S(0) if spmd_dense_sp_enabled() else spmd.I,
                dst=spmd.R,
                backward_options={"op_dtype": x_TD.dtype},
            )
        return x_TD

    def _query(self, x_TD, positions):
        cfg, rd = self.config, self.config.rope_head_dim
        qr_TR = self.wq_a(x_TD)
        remat.recompute_needs_tensor(qr_TR)
        qr_TR = rms_norm(self.q_norm, qr_TR)
        q_THD = self.wq_b(qr_TR)
        remat.recompute_needs_tensor(q_THD)
        with spmd.local():
            q_THD = q_THD.view(x_TD.size(0), -1, cfg.head_dim)
            if spmd.is_type_checking():
                spmd.assert_type(
                    q_THD, {"dp": spmd.S(0), "cp": spmd.S(0), "tp": spmd.S(1)}
                )
        # V4.1 deliberately has no per-head query RMS normalization here.
        q_THD = torch.cat(
            (q_THD[..., :-rd], self.rope(q_THD[..., -rd:], positions=positions)), -1
        )
        return qr_TR, q_THD

    def _local_kv(self, x_TD, positions):
        cfg, rd = self.config, self.config.rope_head_dim
        local_k_TD = self.wkv(x_TD)
        remat.recompute_needs_tensor(local_k_TD)
        local_k_TD = rms_norm(self.kv_norm, local_k_TD)
        local_k_TD = torch.cat(
            (
                local_k_TD[:, :-rd],
                self.rope(local_k_TD[:, None, -rd:], positions=positions).squeeze(1),
            ),
            -1,
        )
        if cfg.quantize:
            with spmd.no_typecheck():
                local_k_TD = fp8_activation(local_k_TD)
        return local_k_TD

    def _output(self, out_THD, positions):
        cfg, rd = self.config, self.config.rope_head_dim
        num_tokens = out_THD.size(0)
        out_THD = torch.cat(
            (
                out_THD[..., :-rd],
                self.rope(out_THD[..., -rd:], positions=positions, inverse=True),
            ),
            -1,
        )
        with spmd.local():
            num_local_groups = cfg.num_groups * out_THD.size(1) // cfg.n_heads
            out_TGD = out_THD.reshape(num_tokens, num_local_groups, -1)
            weight_GRD = self.wo_a.weight.view(num_local_groups, cfg.o_lora_rank, -1)
            if spmd.is_type_checking():
                spmd.assert_type(
                    out_TGD, {"dp": spmd.S(0), "cp": spmd.S(0), "tp": spmd.S(1)}
                )
                spmd.assert_type(
                    weight_GRD, {"dp": spmd.R, "cp": spmd.R, "tp": spmd.S(0)}
                )
        if cfg.quantize:
            with spmd.no_typecheck():
                out_TGD = fp8_activation(out_TGD)
            if spmd.is_type_checking():
                spmd.assert_type(
                    out_TGD, {"dp": spmd.S(0), "cp": spmd.S(0), "tp": spmd.S(1)}
                )
        out_TGR = torch.einsum("tgd,grd->tgr", out_TGD, weight_GRD)
        with spmd.local():
            out_TR = out_TGR.reshape(num_tokens, -1)
            if spmd.is_type_checking():
                spmd.assert_type(
                    out_TR, {"dp": spmd.S(0), "cp": spmd.S(0), "tp": spmd.S(1)}
                )
        return self.wo_b(out_TR)

    def forward(
        self,
        x_TD,
        global_k_SD,
        index_k_SD,
        indices_TK,
        scores_TK,
        candidates_TC,
        *,
        positions,
        attention_metadata=None,
        aux_loss_denominator=None,
        padding_mask=None
    ):
        cfg, rd = self.config, self.config.rope_head_dim
        x_TD = self._gather_input(x_TD)
        qr_TR, q_THD = self._query(x_TD, positions)
        local_k_TD = self._local_kv(x_TD, positions)
        cu = None if attention_metadata is None else attention_metadata.cu_seq_q
        if self.compressor is not None:
            latent_SD, key_positions_S = self.compressor(x_TD, positions, cu_seqlens=cu)
        else:
            latent_SD, key_positions_S = global_k_SD, None
        if self.indexer is not None:
            with spmd.no_typecheck():
                index_k_SD, indices_TK, scores_TK, candidates_TC = self.indexer(
                    x_TD,
                    qr_TR,
                    latent_SD,
                    key_positions_S,
                    positions,
                    index_k_SD,
                    candidates_TC,
                    cu_seqlens=cu,
                    max_seqlen=x_TD.size(0)
                    if attention_metadata is None
                    else attention_metadata.max_k,
                )
        if self.compressor is not None:
            global_k_SD = _rotate_shared_keys(
                self.rope, latent_SD, key_positions_S, rope_head_dim=rd
            )
            if cfg.quantize:
                with spmd.no_typecheck():
                    global_k_SD = fp4_activation(global_k_SD, compressed_kv=True)
        out_THD = self.inner_attention(
            q_THD,
            local_k_TD,
            global_k_SD,
            indices_TK,
            scores_TK,
            self.attn_sink.weight.squeeze(-1),
            attention_metadata=attention_metadata,
            aux_loss_denominator=aux_loss_denominator,
            padding_mask_T=padding_mask,
        )
        return (
            self._output(out_THD, positions),
            global_k_SD,
            index_k_SD,
            indices_TK,
            scores_TK,
            candidates_TC,
        )
