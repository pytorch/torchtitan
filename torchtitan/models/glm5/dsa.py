# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""GLM sparse attention with Attention Gym indexing and gathered attention.

T/Q/K are token/query/key counts; S is selected keys per query; N is heads.
D/H/R/P are model/head/latent-or-rotary/pass-through widths; V is value width.
Document metadata is linear in tokens. Attention delegates to the common
gathered backend; Attention Gym owns the weighted-ReLU top-k selection.
"""

from dataclasses import dataclass
from typing import Literal

import spmd_types as spmd
import torch
import torch_remat as remat

from torchtitan.distributed.context_parallel import cp_all_gather_tensors

from torchtitan.distributed.parallelism_context import MeshAxisName
from torchtitan.distributed.spmd_types import spmd_mesh_group
from torchtitan.models.base.module import Module
from torchtitan.models.common import ComplexRoPE, LayerNorm, Linear
from torchtitan.models.common.attention import InnerAttention
from torchtitan.models.common.attention.cp_mla import AllGatherKVMLAFlexAttention
from torchtitan.models.common.attn_gym import (
    AuxRequest,
    gather_attn,
    lightning_indexer,
    require_attn_gym,
)
from torchtitan.models.deepseek_v4.compressor import SparseIndexerLoss
from torchtitan.tools.utils import has_cuda_capability


@dataclass(frozen=True)
class DsaMetadata:
    """Global keys in CP order and rank-local query indices; all tensors are 1D."""

    positions_T: torch.Tensor  # noqa: N815
    documents_T: torch.Tensor  # noqa: N815
    valid_T: torch.Tensor  # noqa: N815
    query_indices_Q: torch.Tensor  # noqa: N815
    order_T: torch.Tensor  # noqa: N815

    @classmethod
    def from_positions(cls, positions, padding_mask=None):
        valid = (
            torch.ones_like(positions, dtype=torch.bool)
            if padding_mask is None
            else ~padding_mask
        )
        order = torch.arange(positions.numel(), device=positions.device)
        return cls(positions, (positions == 0).cumsum(0), valid, order, order)


class Glm5CPMLAAttention(AllGatherKVMLAFlexAttention):
    """Reuse upstream compact KV communication with GLM document metadata."""

    @dataclass(kw_only=True, slots=True)
    class Config(AllGatherKVMLAFlexAttention.Config):
        def __post_init__(self) -> None:
            # Communication is shared; only the leaf attention/metadata differ.
            if not isinstance(self.inner_attention, Glm5SparseAttention.Config):
                raise ValueError("GLM CP MLA requires Glm5SparseAttention.Config")

    @staticmethod
    def prepare_cp_metadata(  # pyrefly: ignore[bad-override]
        attention_metadata: DsaMetadata, *, permutation: torch.Tensor | None
    ) -> DsaMetadata:
        metadata = attention_metadata
        group = spmd_mesh_group(MeshAxisName.CP)
        if group is None:
            return metadata
        count = metadata.positions_T.numel()
        if count % group.size():
            raise ValueError("DSA token count must divide evenly across CP ranks")
        order = (
            torch.arange(count, device=metadata.positions_T.device)
            if permutation is None
            else permutation.reshape(-1)
        )
        local = count // group.size()
        queries = torch.arange(
            group.rank() * local, (group.rank() + 1) * local, device=order.device
        )
        return DsaMetadata(
            metadata.positions_T[order],
            metadata.documents_T[order],
            metadata.valid_T[order],
            queries,
            metadata.order_T[order],
        )


class DSAIndexerTopK(Module):
    """Compute local-query top-k indices against a global key sequence.

    Attention Gym processes complete packed documents and returns indices local
    to each document. Compact valid tokens before selection, then map selections
    back to the global key sequence consumed by gathered attention. CP gathers query
    features as well as keys because the indexer API requires complete documents.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        index_topk: int

    def __init__(self, config: Config) -> None:
        super().__init__()
        require_attn_gym(lightning_indexer)
        self.index_topk = config.index_topk

    @torch.no_grad()
    def forward(
        self,
        q_QNH: torch.Tensor,
        k_KH: torch.Tensor,
        weights_QN: torch.Tensor,
        *,
        metadata: DsaMetadata,
    ) -> torch.Tensor:
        K = metadata.positions_T.numel()
        cp_group = spmd_mesh_group(MeshAxisName.CP)
        if cp_group is not None:
            # TODO: Keep queries local once Attention Gym supports query chunks.
            q_QNH, weights_QN = cp_all_gather_tensors(
                (q_QNH, weights_QN),
                reduce_dtype=torch.float32,
            )
        if q_QNH.shape[0] != K:
            raise ValueError("DSA indexing requires complete query and key sequences")

        # Keep token order within each document while moving padding to the end.
        # Fixed-capacity offsets use repeated endpoints for empty documents.
        # Restore original document/token order after any CP load-balancer permutation.
        token_order_K = metadata.positions_T.argsort(stable=True)
        token_order_K = token_order_K[
            metadata.documents_T[token_order_K].argsort(stable=True)
        ]
        token_order_K = token_order_K[
            (~metadata.valid_T[token_order_K]).to(torch.int32).argsort(stable=True)
        ]
        valid_K = metadata.valid_T[token_order_K]
        document_starts_K = metadata.documents_T[token_order_K]
        token_indices_K = torch.arange(K, device=q_QNH.device, dtype=torch.int32)
        starts_K = valid_K & (
            (token_indices_K == 0) | (document_starts_K != document_starts_K.roll(1))
        )
        num_valid = valid_K.sum(dtype=torch.int32)
        cu_seqlens = torch.cat(
            (
                torch.where(starts_K, token_indices_K, num_valid).sort().values,
                num_valid[None],
            )
        )
        document_offsets_K = torch.where(starts_K, token_indices_K, 0).cummax(0).values
        # Match DSV4: fused selection requires low-precision inputs on SM90+.
        fused = (
            q_QNH.is_cuda
            and q_QNH.dtype in (torch.float16, torch.bfloat16)
            and has_cuda_capability(9, 0)
        )
        q_QNH = q_QNH if fused else q_QNH.float()
        k_KH = k_KH if fused else k_KH.float()
        indices_KS = lightning_indexer(
            q_QNH[token_order_K].unsqueeze(0),
            k_KH[token_order_K].unsqueeze(0),
            weights_QN[token_order_K].to(q_QNH.dtype).unsqueeze(0),
            min(self.index_topk, K),
            causal=True,
            cu_seqlens=cu_seqlens,
            cu_seqlens_k=cu_seqlens,
            impl="fused" if fused else "reference",
        ).squeeze(0)
        packed_indices_KS = document_offsets_K[:, None] + indices_KS.clamp_min(0)
        global_indices_KS = token_order_K[packed_indices_KS.long()].to(torch.int32)
        global_indices_KS = global_indices_KS.masked_fill(indices_KS < 0, -1)
        indices_KS = torch.empty_like(global_indices_KS).index_copy_(
            0, token_order_K, global_indices_KS
        )
        if cp_group is not None:
            return spmd.redistribute(indices_KS, cp_group, src=spmd.R, dst=spmd.S(0))
        return indices_KS


class Glm5DsaIndexer(Module):
    """Query-to-key selector with detached backbone inputs for distillation.

    ``hidden_states_TD`` supplies both the shared key and per-head mixing
    weights. ``q_resid_TR`` reuses MLA's normalized Q LoRA representation so
    the main attention and indexer observe the same query state. RoPE is
    applied only to the configured rotary prefix; the remaining features pass
    through unchanged.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        n_heads: int
        head_dim: int
        qk_rope_head_dim: int
        wq_b: Linear.Config
        wk: Linear.Config
        k_norm: LayerNorm.Config
        weights_proj: Linear.Config
        rope: ComplexRoPE.Config
        topk: DSAIndexerTopK.Config
        training_stage: Literal["frozen", "warmup", "sparse"] = "frozen"

        def __post_init__(self):
            if self.training_stage not in ("frozen", "warmup", "sparse"):
                raise ValueError("Unknown GLM indexer training stage")

    def __init__(self, config: Config):
        super().__init__()
        self.n_heads = config.n_heads
        self.head_dim = config.head_dim
        self.qk_rope_head_dim = config.qk_rope_head_dim
        self.wq_b = config.wq_b.build()
        self.wk = config.wk.build()
        self.k_norm = config.k_norm.build()
        self.weights_proj = config.weights_proj.build().float()
        self.rope = config.rope.build()
        self.topk = config.topk.build()
        self.training_stage = config.training_stage
        self.requires_grad_(config.training_stage != "frozen")

    def forward(
        self,
        hidden_states_TD: torch.Tensor,
        q_resid_TR: torch.Tensor,
        positions_T: torch.Tensor,
        attention_mask_TK: DsaMetadata | None,
    ):
        # Indexer distillation must not optimize the backbone feature producer.
        remat.recompute_needs_tensor(hidden_states_TD, q_resid_TR)
        with torch.set_grad_enabled(
            torch.is_grad_enabled() and self.training_stage != "frozen"
        ):
            features = self._project_features(
                hidden_states_TD.detach(), q_resid_TR.detach(), positions_T
            )
        if attention_mask_TK is None:
            attention_mask_TK = DsaMetadata.from_positions(positions_T)
        indices = self.topk(*features, metadata=attention_mask_TK)
        return indices if self.training_stage == "frozen" else (indices, features)

    def _project_features(self, hidden_states_TD, q_resid_TR, positions_T):
        T = hidden_states_TD.shape[0]
        # Build query and key features in the same indexer space. The key has a
        # singleton head dimension because every query head searches one shared
        # global key sequence.
        q_TNH = self.wq_b(q_resid_TR)
        remat.recompute_needs_tensor(q_TNH)
        q_TNH = q_TNH.view(T, self.n_heads, self.head_dim)
        q_rot_TNR, q_pass_TNP = torch.split(
            q_TNH,
            [self.qk_rope_head_dim, self.head_dim - self.qk_rope_head_dim],
            dim=-1,
        )
        k_TH = self.wk(hidden_states_TD)
        remat.recompute_needs_tensor(k_TH)
        k_TH = self.k_norm(k_TH)
        remat.recompute_needs_tensor(k_TH)
        k_T1H = k_TH.unsqueeze(1)
        k_rot_T1R, k_pass_T1P = torch.split(
            k_T1H,
            [self.qk_rope_head_dim, self.head_dim - self.qk_rope_head_dim],
            dim=-1,
        )
        q_rot_TNR, k_rot_T1R = self.rope(q_rot_TNR, k_rot_T1R, positions_T)
        remat.recompute_needs_tensor(q_rot_TNR, k_rot_T1R)
        q_TNH = torch.cat((q_rot_TNR, q_pass_TNP), dim=-1)
        k_TH = torch.cat((k_rot_T1R, k_pass_T1P), dim=-1).squeeze(1)

        # Context parallelism keeps queries local but every query ranks the
        # global key sequence. Match the KV-all-gather attention contract by
        # gathering indexer keys over CP before the discrete top-k boundary.
        cp_group = spmd_mesh_group(MeshAxisName.CP)
        if cp_group is not None:
            k_TH = spmd.redistribute(
                k_TH,
                cp_group,
                src=spmd.S(0),
                dst=spmd.R,
            )

        # The learned per-token head weights turn N head-specific similarities
        # into the scalar [T,T] ranking consumed by top-k.
        with torch.autocast(device_type=hidden_states_TD.device.type, enabled=False):
            if hidden_states_TD.dtype != self.weights_proj.weight.dtype:
                # The cast is a bare read of the TP-gather region's output.
                remat.recompute_needs_tensor(hidden_states_TD)
            weights_TN = self.weights_proj(
                hidden_states_TD.to(self.weights_proj.weight.dtype)
            )
        remat.recompute_needs_tensor(weights_TN)
        # Attention Gym applies the 1/sqrt(N*H) indexer normalization.
        return q_TNH, k_TH, weights_TN


class Glm5IndexerLoss(SparseIndexerLoss):
    """Reuse DSV4 normalized KL and injection with GLM's per-head key pool."""

    @dataclass(kw_only=True, slots=True)
    class Config(SparseIndexerLoss.Config):
        compress_ratio: int = 1

    @torch.no_grad()
    def _teacher(
        self,
        q_THD: torch.Tensor,
        cmp_k_SD: torch.Tensor,
        topk_indices_TK: torch.Tensor,
        attn_lse_TH: torch.Tensor,
        cu_seqlens: torch.Tensor | None,
        padding_mask_T: torch.Tensor | None = None,
    ) -> torch.Tensor:
        from torch.distributed._functional_collectives import all_reduce, wait_tensor

        from torchtitan.distributed.spmd_types import spmd_mesh_size

        if q_THD.shape[1] * spmd_mesh_size("tp") != self.num_heads:
            raise ValueError("Teacher head count does not match TP-sharded GLM queries")
        targets = []
        with torch.autocast(device_type=q_THD.device.type, enabled=False):
            for start in range(0, q_THD.shape[0], self.chunk_size):
                stop = start + self.chunk_size
                indices_QS = topk_indices_TK[start:stop]
                keys_QSHD = cmp_k_SD[indices_QS.clamp_min(0).long()].float()
                logits_QHS = (
                    torch.einsum("qhd,qshd->qhs", q_THD[start:stop].float(), keys_QSHD)
                    * self.softmax_scale
                )
                logits_QHS = logits_QHS - attn_lse_TH[start:stop, :, None].float()
                valid_QS = indices_QS >= 0
                if padding_mask_T is not None:
                    valid_QS = valid_QS & ~padding_mask_T[start:stop, None]
                targets.append(
                    logits_QHS.masked_fill(~valid_QS[:, None], -torch.inf)
                    .exp()
                    .sum(dim=1)
                    / self.num_heads
                )
        targets_QS = torch.cat(targets)
        group = spmd_mesh_group("tp")
        if group is not None:
            targets_QS = wait_tensor(all_reduce(targets_QS, "sum", group))
        return targets_QS


def filter_dsa_indices(
    indices_QS: torch.Tensor,
    metadata: DsaMetadata,
) -> torch.Tensor:
    """Keep unique selected keys satisfying packed-document and padding rules."""
    _, S = indices_QS.shape
    if S == 0:
        raise ValueError("DSA selection width must be positive")
    K = metadata.positions_T.numel()
    valid_QS = (indices_QS >= 0) & (indices_QS < K)
    safe_QS = indices_QS.clamp(0, K - 1).long()
    queries_Q = metadata.query_indices_Q
    valid_QS = valid_QS & metadata.valid_T[queries_Q, None] & metadata.valid_T[safe_QS]
    valid_QS = valid_QS & (
        metadata.documents_T[queries_Q, None] == metadata.documents_T[safe_QS]
    )
    valid_QS = valid_QS & (
        metadata.positions_T[queries_Q, None] >= metadata.positions_T[safe_QS]
    )
    selected_QS = torch.where(valid_QS, indices_QS, K).sort(dim=-1).values
    first_QS = torch.cat(
        (
            torch.ones_like(selected_QS[:, :1], dtype=torch.bool),
            selected_QS[:, 1:] != selected_QS[:, :-1],
        ),
        dim=-1,
    )
    return torch.where((selected_QS < K) & first_QS, selected_QS, -1)


class Glm5SparseAttention(InnerAttention):
    """Gather exactly the selected expanded MLA keys and values.

    Attention Gym shares its K/V pool. Encoding query as [q, 0] and pool as
    [k, v] preserves q @ k and returns the weighted values in the output tail.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(InnerAttention.Config):
        training_stage: Literal["frozen", "warmup", "sparse"] = "frozen"
        indexer_loss: Glm5IndexerLoss.Config | None = None

        def __post_init__(self):
            if self.training_stage not in ("frozen", "warmup", "sparse"):
                raise ValueError("Unknown GLM indexer training stage")

        @property
        def is_full_attention(self) -> bool:
            return True

    def __init__(self, config):
        super().__init__()
        require_attn_gym(gather_attn)
        self.training_stage = config.training_stage
        self.indexer_loss = config.indexer_loss.build() if config.indexer_loss else None

    def forward(
        self,
        q_THK: torch.Tensor,
        k_THK: torch.Tensor,
        v_THV: torch.Tensor,
        *,
        attention_metadata: DsaMetadata,
        topk_indices_QS: torch.Tensor,
        scale: float,
        indexer_features: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None,
        aux_loss_denominator: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self.training_stage == "warmup":
            # Bound dense teacher indices/scores by query chunks, never Q x K.
            outputs = []
            keys_K = torch.arange(k_THK.shape[0], device=q_THK.device)
            for start in range(0, q_THK.shape[0], 128):
                stop = min(start + 128, q_THK.shape[0])
                queries_Q = attention_metadata.query_indices_Q[start:stop]
                valid_QK = (
                    attention_metadata.valid_T[queries_Q, None]
                    & attention_metadata.valid_T[None, :]
                    & (
                        attention_metadata.documents_T[queries_Q, None]
                        == attention_metadata.documents_T[None, :]
                    )
                    & (
                        attention_metadata.positions_T[queries_Q, None]
                        >= attention_metadata.positions_T[None, :]
                    )
                )
                indices_QK = keys_K.expand(stop - start, -1).masked_fill(~valid_QK, -1)
                features = (
                    None
                    if indexer_features is None
                    else (
                        indexer_features[0][start:stop],
                        indexer_features[1],
                        indexer_features[2][start:stop],
                    )
                )
                outputs.append(
                    self._attend(
                        q_THK[start:stop],
                        k_THK,
                        v_THV,
                        indices_QK,
                        scale,
                        features,
                        aux_loss_denominator,
                    )
                )
            return torch.cat(outputs)
        indices_QS = filter_dsa_indices(topk_indices_QS, attention_metadata)
        return self._attend(
            q_THK,
            k_THK,
            v_THV,
            indices_QS,
            scale,
            indexer_features,
            aux_loss_denominator,
        )

    def _attend(
        self,
        q_THK,
        k_THK,
        v_THV,
        indices_QS,
        scale,
        indexer_features,
        aux_loss_denominator,
    ):
        Q, H, _ = q_THK.shape
        K, _, V = v_THV.shape
        train_indexer = self.training and self.indexer_loss is not None
        with spmd.no_typecheck():
            query_1HQD = (
                torch.cat((q_THK, q_THK.new_zeros(Q, H, V)), dim=-1)
                .transpose(0, 1)
                .unsqueeze(0)
            )
            pool_1HKD = torch.cat((k_THK, v_THV), dim=-1).transpose(0, 1).unsqueeze(0)
            # No local window: real document/causal rules are already in indices.
            # Explicit offsets select the gathered (not dense) reference path.
            output_1HQD, aux = gather_attn(
                query_1HQD,
                query_1HQD.new_zeros(()).expand_as(query_1HQD),
                pool_1HKD,
                indices_QS.unsqueeze(0),
                sliding_window_size=0,
                cu_seqlens=torch.arange(2, dtype=torch.int32, device=q_THK.device) * Q,
                cu_seqlens_k=torch.arange(2, dtype=torch.int32, device=q_THK.device)
                * K,
                scale=scale,
                impl="fused" if q_THK.device.type == "cuda" else "reference",
                return_aux=AuxRequest(lse=train_indexer),
            )
            output_THV = output_1HQD[..., -V:].squeeze(0).transpose(0, 1)
            if train_indexer:
                assert self.indexer_loss is not None
                assert indexer_features is not None
                assert aux.lse is not None
                if aux_loss_denominator is None:
                    raise ValueError(
                        "Indexer training requires a global token denominator"
                    )
                qi_QNH, ki_KH, weights_QN = indexer_features
                scores = []
                for start in range(0, Q, self.indexer_loss.chunk_size):
                    stop = start + self.indexer_loss.chunk_size
                    selected_QSH = ki_KH[indices_QS[start:stop].clamp_min(0).long()]
                    with torch.autocast(device_type=q_THK.device.type, enabled=False):
                        scores_QNS = torch.einsum(
                            "qnh,qsh->qns",
                            qi_QNH[start:stop].float(),
                            selected_QSH.float(),
                        )
                    scores.append(
                        (
                            scores_QNS.relu()
                            * weights_QN[start:stop].float().unsqueeze(-1)
                        ).sum(1)
                    )
                scores_QS = torch.cat(scores)
                scores_QS = scores_QS / (qi_QNH.shape[1] * qi_QNH.shape[2]) ** 0.5
                scores_QS = scores_QS.masked_fill(indices_QS < 0, -torch.inf)
                output_THV = self.indexer_loss(
                    q_THK.detach(),
                    k_THK.detach(),
                    indices_QS,
                    scores_QS,
                    aux.lse.squeeze(0).transpose(0, 1).detach(),
                    None,
                    carrier=output_THV,
                    denominator=aux_loss_denominator,
                )
        if spmd.is_type_checking():
            spmd.assert_type(
                output_THV, spmd.get_local_type(q_THK), spmd.get_partition_spec(q_THK)
            )
        return output_THV
