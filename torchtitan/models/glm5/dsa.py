# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""GLM sparse attention with local CP selection and gathered attention.

T/Q/K are token/query/key counts; S is selected keys per query; N is heads.
D/H/R/P are model/head/latent-or-rotary/pass-through widths; V is value width.
Document metadata is linear in tokens. Attention delegates to the common
gathered backend; CP selection keeps query work local to each rank.
"""

from dataclasses import dataclass
from typing import Literal

import spmd_types as spmd
import torch
import torch_remat as remat

from torchtitan.distributed.parallelism_context import MeshAxisName
from torchtitan.distributed.context_parallel import cp_all_gather_tensors
from torchtitan.distributed.spmd_types import spmd_mesh_group
from torchtitan.models.base.module import Module
from torchtitan.models.common import ComplexRoPE, LayerNorm, Linear
from torchtitan.models.common.attention import InnerAttention, MLAAttention
from torchtitan.models.common.attention.cp_mla import AllGatherKVMLAFlexAttention
from torchtitan.models.common.attn_gym import (
    AuxRequest,
    gather_attn,
    lightning_indexer,
    require_attn_gym,
)
from torchtitan.models.deepseek_v4.compressor import SparseIndexerLoss
from torchtitan.tools.utils import has_cuda_capability, set_default_dtype

from .indexer import select_local_topk


@dataclass(frozen=True)
class DsaMetadata:
    """Global keys in CP order and rank-local query indices; all tensors are 1D."""

    positions_T: torch.Tensor  # noqa: N815
    documents_T: torch.Tensor  # noqa: N815
    valid_T: torch.Tensor  # noqa: N815
    query_indices_Q: torch.Tensor  # noqa: N815

    @classmethod
    def from_positions(cls, positions, padding_mask=None):
        valid = (
            torch.ones_like(positions, dtype=torch.bool)
            if padding_mask is None
            else ~padding_mask
        )
        order = torch.arange(positions.numel(), device=positions.device)
        return cls(positions, (positions == 0).cumsum(0), valid, order)


class Glm5MLAAttention(MLAAttention):
    """Absorb K into Q and project V after attention over one latent pool."""

    @dataclass(kw_only=True, slots=True)
    class Config(MLAAttention.Config):
        def __post_init__(self) -> None:
            if not isinstance(self.inner_attention, Glm5SparseAttention.Config):
                raise ValueError("GLM MLA requires Glm5SparseAttention.Config")
            if (
                type(self.wkv_b) is not Linear.Config
                or self.wkv_b.bias
                or self.wkv_b.num_linears != 1
            ):
                raise ValueError("Absorbed GLM MLA requires a dense bias-free wkv_b")

    def _prepare_latents(self, latent_TL, rope_TR):
        return latent_TL, rope_TR

    def prepare_absorbed(self, q_THK, kv_c_normed_TL, k_shared_TR):
        latent_KL, rope_KR = self._prepare_latents(kv_c_normed_TL, k_shared_TR)
        # The pool concatenation reads the saved CP communication outputs.
        remat.recompute_needs_tensor(latent_KL, rope_KR, q_THK)
        with spmd.local():
            heads = q_THK.shape[1]
            nope = q_THK.shape[-1] - rope_KR.shape[-1]
            weight_HPL = self.wkv_b.weight.view(heads, self.packed_kv_head_dim, -1)
            weight_HPL = weight_HPL.to(q_THK.dtype)
            key_weight_HNL, value_weight_HVL = weight_HPL.split(
                [nope, self.packed_kv_head_dim - nope], dim=1
            )
            q_nope_THN, q_rope_THR = q_THK.split([nope, rope_KR.shape[-1]], dim=-1)
        q_latent_THL = remat.region(
            torch.einsum, self.remat_region_name("q_absorb"),
            recompute=self.remat_should_recompute("q_absorb"),
        )("thn,hnl->thl", q_nope_THN, key_weight_HNL)
        remat.recompute_needs_tensor(q_latent_THL)
        q_abs_THD = torch.cat((q_latent_THL, q_rope_THR), dim=-1)
        pool_KD = torch.cat((latent_KL, rope_KR), dim=-1)
        return q_abs_THD, pool_KD, value_weight_HVL

    def forward(self, q_THK, kv_c_normed_TL, k_shared_TR, **kwargs):
        q_abs_THD, pool_KD, value_weight_HVL = self.prepare_absorbed(
            q_THK, kv_c_normed_TL, k_shared_TR
        )
        return self.attend_absorbed(q_abs_THD, pool_KD, value_weight_HVL, **kwargs)

    def attend_absorbed(self, q_abs_THD, pool_KD, value_weight_HVL, **kwargs):
        """Consume one prepared teacher pool without repeating projections."""
        output_THD = remat.region(
            self.inner_attention, self.remat_region_name("inner_attention"),
            recompute=self.remat_should_recompute("inner_attention"),
        )(q_abs_THD, pool_KD, **kwargs)
        remat.recompute_needs_tensor(output_THD)
        return remat.region(
            torch.einsum, self.remat_region_name("v_project"),
            recompute=self.remat_should_recompute("v_project"),
        )("thl,hvl->thv", output_THD[..., :value_weight_HVL.shape[-1]], value_weight_HVL)

    def warmup_teacher(self, q_THK, latent_TL, rope_TR, **kwargs):
        """Return dense output and detached inputs for KL-only block training.

        Unlike DSV4's joint-training output carrier, KL-only training must
        expose the already prepared teacher inputs without replaying projections.
        This is a GLM-local layout boundary, not a second teacher model.
        """
        function = self._warmup_teacher
        if spmd.is_type_checking():
            from torchtitan.models.common.decoder_sharding import (
                attention_activation_placement,
                dense_activation_placement,
            )

            head = attention_activation_placement()
            local_kv = dense_activation_placement(tp=spmd.R, cp=spmd.S(0))
            global_kv = dense_activation_placement(tp=spmd.R, cp=spmd.R)
            function = spmd.local_map(
                in_types=tuple((layout.local_type, layout.partition_spec)
                               for layout in (head, local_kv, local_kv)),
                out_types=tuple((layout.local_type, layout.partition_spec)
                                for layout in (head, head, global_kv)),
            )(function)
        return function(q_THK, latent_TL, rope_TR, **kwargs)

    def _warmup_teacher(self, q_THK, latent_TL, rope_TR, **kwargs):
        query, pool, value_weight = self.prepare_absorbed(q_THK, latent_TL, rope_TR)
        empty = q_THK.new_empty((q_THK.shape[0], 0), dtype=torch.int32)
        output = self.attend_absorbed(
            query, pool, value_weight, topk_indices_QS=empty, **kwargs
        )
        return output, query, pool


class Glm5CPMLAAttention(Glm5MLAAttention, AllGatherKVMLAFlexAttention):
    """Reuse upstream compact KV communication with GLM document metadata."""

    @dataclass(kw_only=True, slots=True)
    class Config(Glm5MLAAttention.Config, AllGatherKVMLAFlexAttention.Config):
        def __post_init__(self) -> None:
            Glm5MLAAttention.Config.__post_init__(self)

    def _prepare_latents(self, latent_TL, rope_TR):
        # Reuse the public compact-MLA communication primitive and remat policy;
        # the common forward would expand K/V, so only its gather is needed here.
        return remat.region(
            cp_all_gather_tensors, self.remat_region_name("cp_all_gather"),
            recompute=self.remat_should_recompute("inner_attention"),
        )((latent_TL, rope_TR), reduce_dtype=self.reduce_dtype)

    @staticmethod
    def prepare_cp_metadata(  # pyrefly: ignore[bad-override]
        attention_metadata: DsaMetadata, *, permutation: torch.Tensor | None
    ) -> DsaMetadata:
        if permutation is not None:
            raise ValueError("GLM DSA CP requires contiguous shards without a permutation")
        metadata = attention_metadata
        group = spmd_mesh_group(MeshAxisName.CP)
        if group is None:
            return metadata
        count = metadata.positions_T.numel()
        if count % group.size():
            raise ValueError("DSA token count must divide evenly across CP ranks")
        local = count // group.size()
        queries = torch.arange(
            group.rank() * local,
            (group.rank() + 1) * local,
            device=metadata.positions_T.device,
        )
        return DsaMetadata(
            metadata.positions_T,
            metadata.documents_T,
            metadata.valid_T,
            queries,
        )


class DSAIndexerTopK(Module):
    """Compute local-query top-k indices against a global key sequence.

    Compact valid tokens before selection, then map document-relative selections
    back to the global key sequence consumed by gathered attention. Under CP,
    explicit candidate bounds retain local queries against the global key pool.
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
        if cp_group is None and q_QNH.shape[0] != K:
            raise ValueError("DSA indexing requires complete query and key sequences")

        # Keep token order within each document while moving padding to the end.
        # Fixed-capacity offsets use repeated endpoints for empty documents.
        token_order_K = (~metadata.valid_T).to(torch.int32).argsort(stable=True)
        valid_K = metadata.valid_T[token_order_K]
        document_starts_K = metadata.documents_T[token_order_K]
        token_indices_K = torch.arange(K, device=q_QNH.device, dtype=torch.int32)
        starts_K = valid_K & (
            (token_indices_K == 0) | (document_starts_K != document_starts_K.roll(1))
        )
        document_offsets_K = torch.where(starts_K, token_indices_K, 0).cummax(0).values
        fused = (
            q_QNH.is_cuda
            and q_QNH.dtype in (torch.float16, torch.bfloat16)
            and has_cuda_capability(9, 0)
        )
        if cp_group is not None or not fused:
            # Local query rows retain their original order. In the packed key
            # pool, each valid row sees its document prefix through that query.
            packed_rows_K = torch.empty_like(token_indices_K).scatter_(
                0, token_order_K, token_indices_K
            )
            queries_Q = metadata.query_indices_Q
            packed_queries_Q = packed_rows_K[queries_Q].long()
            valid_Q = metadata.valid_T[queries_Q]
            starts_Q = torch.where(valid_Q, document_offsets_K[packed_queries_Q], 0)
            ends_Q = torch.where(valid_Q, packed_rows_K[queries_Q] + 1, 0)
            bounds_Q2 = torch.stack((starts_Q, ends_Q), dim=-1)
            kernel_weights_QN = (
                weights_QN.to(q_QNH.dtype)
                if cp_group is not None and q_QNH.is_cuda
                and q_QNH.dtype in (torch.float16, torch.bfloat16)
                else weights_QN
            )
            # The non-fused whole-document path uses the same bounded selector
            # as CP, avoiding the stock reference's full query/key score tensor.
            # Preserve the former FP32 reference arithmetic outside CP.
            reference = cp_group is None
            indices_QS = select_local_topk(
                q_QNH.float() if reference else q_QNH,
                k_KH[token_order_K].float() if reference else k_KH[token_order_K],
                kernel_weights_QN,
                bounds_Q2,
                min(self.index_topk, K),
            )
            return token_order_K[indices_QS.clamp_min(0).long()].to(torch.int32).masked_fill(
                indices_QS < 0, -1
            )
        # Whole-document offsets are only used by the CP=1 Attention Gym path.
        num_valid = valid_K.sum(dtype=torch.int32)
        cu_seqlens = torch.cat(
            (
                torch.where(starts_K, token_indices_K, num_valid).sort().values,
                num_valid[None],
            )
        )
        indices_KS = lightning_indexer(
            q_QNH[token_order_K].unsqueeze(0),
            k_KH[token_order_K].unsqueeze(0),
            weights_QN[token_order_K].to(q_QNH.dtype).unsqueeze(0),
            min(self.index_topk, K),
            causal=True,
            cu_seqlens=cu_seqlens,
            cu_seqlens_k=cu_seqlens,
            impl="fused",
        ).squeeze(0)
        # Packed causal top-k already guarantees unique legal keys and -1 for
        # invalid rows. This bijection preserves that contract in the global pool.
        packed_indices_KS = document_offsets_K[:, None] + indices_KS.clamp_min(0)
        global_indices_KS = token_order_K[packed_indices_KS.long()].to(torch.int32)
        global_indices_KS = global_indices_KS.masked_fill(indices_KS < 0, -1)
        indices_KS = torch.empty_like(global_indices_KS).index_copy_(
            0, token_order_K, global_indices_KS
        )
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
        # TrainingEngine may build the model with a low default parameter dtype.
        # Create mixing weights in FP32; config preserves this dtype under FSDP.
        with set_default_dtype(torch.float32):
            self.weights_proj = config.weights_proj.build()
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
        if self.training_stage == "warmup":
            # Dense teacher attention ignores selection. A zero-width support
            # preserves the block/index-sharing transport contract without ranking.
            return features[0].new_empty(
                (features[0].shape[0], 0), dtype=torch.int32
            ), features
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
                backward_options={"op_dtype": k_TH.dtype},
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
    """Use DSV4 distillation without compression or document-relative indices.

    GLM passes global pool IDs and cu_seqlens=None to the shared teacher.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(SparseIndexerLoss.Config):
        compress_ratio: int = 1

    def warmup_objective(self, query_QHD, pool_KD, features, metadata, denominator):
        """KL-only objective using the shared teacher, normalization and metrics.

        Teacher activations are detached. Only projected indexer features and
        the small query-chunk KL graphs participate in backward.
        """
        # The shared sparse teacher also treats gathered attention/KL as local
        # arithmetic; its explicit TP reduction handles the replicated student.
        with spmd.no_typecheck():
            return self._warmup_objective(query_QHD, pool_KD, features, metadata, denominator)

    def _warmup_objective(self, query_QHD, pool_KD, features, metadata, denominator):
        from torchtitan.distributed.spmd_types import spmd_mesh_size

        qi_QNH, ki_KH, weights_QN = features
        raw_sum = query_QHD.new_zeros((), dtype=torch.float32)
        keys_K = torch.arange(pool_KD.shape[0], device=pool_KD.device)
        for start in range(0, query_QHD.shape[0], self.chunk_size):
            stop = min(start + self.chunk_size, query_QHD.shape[0])
            queries_Q = metadata.query_indices_Q[start:stop]
            with torch.no_grad(), torch.autocast(device_type=pool_KD.device.type, enabled=False):
                valid_QK = (
                    metadata.valid_T[queries_Q, None] & metadata.valid_T[None, :]
                    & (metadata.documents_T[queries_Q, None] == metadata.documents_T[None, :])
                    & (metadata.positions_T[queries_Q, None] >= metadata.positions_T[None, :])
                )
                indices_QK = keys_K.expand(stop - start, -1).masked_fill(~valid_QK, -1)
                logits_QHK = torch.einsum(
                    "qhd,kd->qhk", query_QHD[start:stop].float(), pool_KD.float()
                ) * self.softmax_scale
                lse_QH = logits_QHK.masked_fill(~valid_QK[:, None], -torch.inf).logsumexp(-1)
                p_QK = self._teacher(query_QHD[start:stop], pool_KD, indices_QK, lse_QH, None)
                targets_QK = p_QK / p_QK.sum(-1, keepdim=True).clamp_min(torch.finfo(torch.float32).tiny)
            with torch.autocast(device_type=pool_KD.device.type, enabled=False):
                student_QNK = torch.einsum("qnh,kh->qnk", qi_QNH[start:stop].float(), ki_KH.float())
                scores_QK = (student_QNK.relu() * weights_QN[start:stop].float()[:, :, None]).sum(1)
                scores_QK = scores_QK / (qi_QNH.shape[1] * qi_QNH.shape[2]) ** 0.5
                raw_sum = raw_sum + self._logged_kl(p_QK, targets_QK, scores_QK, valid_QK).sum()
        # One retained aux-loss region per module invocation, not per query chunk.
        raw_sum = raw_sum.detach() + (raw_sum - raw_sum.detach()) / spmd_mesh_size("tp")
        carrier = (raw_sum * self.coeff / denominator.clamp_min(1)).detach()
        return self.inject(raw_sum, carrier=carrier, denominator=denominator)


class Glm5SparseAttention(InnerAttention):
    """Gather selected keys and values directly from one shared latent pool.
    The indexer supplies unique causal, document-local selections mapped to
    global pool IDs, with -1 padding. Consumers do not re-sort or revalidate them.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(InnerAttention.Config):
        training_stage: Literal["frozen", "warmup", "sparse"] = "frozen"
        indexer_loss: Glm5IndexerLoss.Config | None = None
        max_context_length: int | None = None

        def __post_init__(self):
            if self.training_stage not in ("frozen", "warmup", "sparse"):
                raise ValueError("Unknown GLM indexer training stage")
            if self.training_stage == "warmup" and (
                self.max_context_length is None or self.max_context_length <= 0
            ):
                raise ValueError("GLM warmup requires a positive max_context_length")

        @property
        def is_full_attention(self) -> bool:
            return True

    def __init__(self, config):
        super().__init__()
        require_attn_gym(gather_attn)
        self.training_stage = config.training_stage
        self.indexer_loss = config.indexer_loss.build() if config.indexer_loss else None
        self.max_context_length = config.max_context_length

    def forward(
        self,
        q_THK: torch.Tensor,
        k_KH: torch.Tensor,
        *,
        attention_metadata: DsaMetadata,
        topk_indices_QS: torch.Tensor,
        scale: float,
        indexer_features: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None,
        aux_loss_denominator: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self.training_stage == "warmup":
            # Packed documents do not attend across boundaries. Allocate one
            # document's key capacity, not the whole flattened microbatch.
            assert self.max_context_length is not None
            outputs = []
            offsets_S = torch.arange(
                min(self.max_context_length, k_KH.shape[0]),
                device=q_THK.device,
                dtype=torch.int32,
            )
            # Producers share the KL chunk budget. Shared-index consumers have
            # no KL module; cap their dense-support workspace at 128 query rows.
            chunk_size = self.indexer_loss.chunk_size if self.indexer_loss else 128
            for start in range(0, q_THK.shape[0], chunk_size):
                stop = min(start + chunk_size, q_THK.shape[0])
                queries_Q = attention_metadata.query_indices_Q[start:stop]
                document_starts_Q = (
                    queries_Q - attention_metadata.positions_T[queries_Q]
                )
                indices_QS = document_starts_Q[:, None] + offsets_S
                # Unlike top-k, warmup creates dense support itself. Each row
                # starts at its document boundary and ends at its query, so
                # only padding and the causal endpoint need masking; no dedup.
                safe_QS = indices_QS.clamp(0, k_KH.shape[0] - 1).long()
                valid_QS = (
                    (indices_QS <= queries_Q[:, None])
                    & attention_metadata.valid_T[queries_Q, None]
                    & attention_metadata.valid_T[safe_QS]
                )
                indices_QS = indices_QS.masked_fill(~valid_QS, -1)
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
                        k_KH,
                        indices_QS,
                        scale,
                        features,
                        aux_loss_denominator,
                    )
                )
            return torch.cat(outputs)
        return self._attend(
            q_THK,
            k_KH,
            topk_indices_QS,
            scale,
            indexer_features,
            aux_loss_denominator,
        )

    def _attend(
        self,
        q_THK,
        k_KH,
        indices_QS,
        scale,
        indexer_features,
        aux_loss_denominator,
    ):
        Q = q_THK.shape[0]
        K = k_KH.shape[0]
        train_indexer = self.training and torch.is_grad_enabled() and self.indexer_loss is not None
        with spmd.no_typecheck():
            query_1HQD = q_THK.transpose(0, 1).unsqueeze(0)
            pool_1HKD = k_KH[None, None]
            # No local window: real document/causal rules are already in indices.
            # Explicit offsets select the gathered (not dense) reference path.
            output_1HQD, aux = gather_attn(
                query_1HQD,
                k_KH.new_zeros(()).expand(1, 1, Q, k_KH.shape[-1]),
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
            output_THV = output_1HQD.squeeze(0).transpose(0, 1)
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
                    k_KH.detach(),
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
