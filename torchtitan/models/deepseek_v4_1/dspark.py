# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""DSpark's parallel draft positions, Markov correction, and confidence head.

Backbone features and shared embedding/head weights are detached at this API.
The caller supplies teacher-forced previous tokens for training, and may call
the Markov head successively on sampled tokens during generation. Verification
scheduling and engine throughput profiling belong to the serving engine.

T: context tokens, B: draft blocks, S: positions per draft block, C: HC branches,
D: hidden width, F: concatenated backbone width, V: vocabulary,
R: Markov rank, K: attended positions.
"""

from dataclasses import dataclass

import spmd_types as spmd
import torch
import torch.nn.functional as F
import torch_remat as remat

from torchtitan.distributed.spmd_types import spmd_dense_sp_enabled, spmd_mesh_group
from torchtitan.models.base.module import Module, ModuleDict
from torchtitan.models.common.decoder import Decoder
from torchtitan.models.common.embedding import Embedding
from torchtitan.models.common.flops import get_nparams_and_active_nparams
from torchtitan.models.common.linear import Linear
from torchtitan.models.common.nn_modules import RMSNorm

from .attention import DeepSeekV41Attention, rms_norm
from .mhc import initial_pre_mix, mix_branches
from .model import DeepSeekV41TransformerBlock


class DSparkAttention(DeepSeekV41Attention):
    @dataclass(kw_only=True, slots=True)
    class Config(DeepSeekV41Attention.Config):
        block_size: int
        context_window: int

    def __init__(self, config: Config):
        super().__init__(config)
        self.block_size = config.block_size
        self.context_window = config.context_window

    def forward(  # pyrefly: ignore [bad-override]
        self,
        x_TD,
        global_k_SD,
        index_k_SD,
        indices_TK,
        scores_TK,
        candidates_TC,
        *,
        positions,
        main_x_TD,
        main_positions_T,
        anchors_B,
        **kwargs,
    ):
        x_TD = self._gather_input(x_TD)
        _, q_THD = self._query(x_TD, positions)
        # The shared KV projection is invoked twice within one checkpoint.
        with remat.name_scope("draft_kv"):
            draft_k_TD = self._local_kv(x_TD, positions)
        with remat.name_scope("main_kv"):
            main_k_SD = self._local_kv(main_x_TD, main_positions_T)
        size, window = self.block_size, self.context_window
        lookback_K = torch.arange(window - 1, -1, -1, device=x_TD.device)
        context_BK = anchors_B[:, None] - lookback_K
        context_BK = torch.where(
            main_positions_T[anchors_B, None] >= lookback_K, context_BK, -1
        )
        drafts_BS = torch.arange(x_TD.size(0), device=x_TD.device).view(
            -1, size
        ) + main_k_SD.size(0)
        gather_TK = (
            torch.cat((context_BK, drafts_BS), -1).repeat_interleave(size, dim=0).int()
        )
        pool_SD = torch.cat((main_k_SD, draft_k_TD), 0)
        # The explicit selections include the whole draft block, bidirectionally.
        out_THD = self.inner_attention(
            q_THD,
            draft_k_TD,
            pool_SD,
            gather_TK,
            scores_TK,
            self.attn_sink.weight.squeeze(-1),
        )
        return (
            self._output(out_THD, positions),
            global_k_SD,
            index_k_SD,
            indices_TK,
            scores_TK,
            candidates_TC,
        )


class MarkovHead(Module):
    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        embed: Embedding.Config
        head: Linear.Config

    def __init__(self, config: Config):
        super().__init__()
        self.embed, self.head = config.embed.build(), config.head.build()

    def forward(self, previous_tokens_T):
        embedded_TR = self.embed(previous_tokens_T)
        return self.head(embedded_TR), embedded_TR


class ConfidenceHead(Module):
    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        proj: Linear.Config

    def __init__(self, config: Config):
        super().__init__()
        self.proj = config.proj.build().float()

    def forward(self, hidden_TD, markov_TR):
        return self.proj(torch.cat((hidden_TD, markov_TR), -1).float()).squeeze(-1)


class DSparkBlock(DeepSeekV41TransformerBlock):
    """Run each projection inside its owning block's FSDP forward boundary."""

    @dataclass(kw_only=True, slots=True)
    class Config(DeepSeekV41TransformerBlock.Config):
        attention: DSparkAttention.Config  # pyrefly: ignore [bad-override]
        main_proj: Linear.Config | None = None
        main_norm: RMSNorm.Config | None = None
        norm: RMSNorm.Config | None = None
        markov_head: MarkovHead.Config | None = None
        confidence_head: ConfidenceHead.Config | None = None

        def __post_init__(self):
            if (self.main_proj is None) != (self.main_norm is None):
                raise ValueError(
                    "DSpark main projection and norm must be configured together."
                )
            heads = (self.norm, self.markov_head, self.confidence_head)
            if any(head is None for head in heads) and any(
                head is not None for head in heads
            ):
                raise ValueError(
                    "DSpark final norm, Markov and confidence heads must be configured together."
                )

    def __init__(self, config: Config):
        super().__init__(config)
        self.main_proj = (
            config.main_proj.build() if config.main_proj is not None else None
        )
        self.main_norm: RMSNorm | None = (
            config.main_norm.build() if config.main_norm is not None else None
        )
        self.norm: RMSNorm | None = (
            config.norm.build() if config.norm is not None else None
        )
        self.markov_head = (
            config.markov_head.build() if config.markov_head is not None else None
        )
        self.confidence_head = (
            config.confidence_head.build()
            if config.confidence_head is not None
            else None
        )

    def forward(  # pyrefly: ignore [bad-override]
        self,
        x_TCD,
        pre_TC,
        global_k_SD,
        index_k_SD,
        indices_TK,
        scores_TK,
        candidates_TP,
        *,
        positions,
        main_features_TF,
        main_x_TD,
        main_positions_T,
        anchors_B,
        previous_tokens_T,
        output_weight_VD,
    ):
        if self.main_proj is not None:
            assert self.main_norm is not None
            main_x_TD = self.main_proj(main_features_TF)
            remat.recompute_needs_tensor(main_x_TD)
            main_x_TD = rms_norm(self.main_norm, main_x_TD)
        assert main_x_TD is not None
        x_TCD, pre_TC, *_ = super().forward(
            x_TCD,
            pre_TC,
            global_k_SD,
            index_k_SD,
            indices_TK,
            scores_TK,
            candidates_TP,
            positions=positions,
            main_x_TD=main_x_TD,
            main_positions_T=main_positions_T,
            anchors_B=anchors_B,
        )
        # A checkpoint's output is a flat tuple of tensors, including these blocks.
        logits_TV = x_TCD.new_empty(0, output_weight_VD.size(0))
        confidence_T = x_TCD.new_empty(0, dtype=torch.float32)
        if self.norm is not None:
            assert self.markov_head is not None and self.confidence_head is not None
            mixed_TD = mix_branches(x_TCD, pre_TC)
            hidden_TD = rms_norm(self.norm, mixed_TD)
            base_logits_TV = F.linear(hidden_TD, output_weight_VD.detach())
            bias_TV, markov_TR = self.markov_head(previous_tokens_T)
            remat.recompute_needs_tensor(bias_TV)
            logits_TV = base_logits_TV + bias_TV
            # The confidence head reads the residual before the final norm.
            confidence_T = self.confidence_head(mixed_TD, markov_TR)
        return x_TCD, pre_TC, main_x_TD, logits_TV, confidence_T


class DSpark(Decoder):
    """An auxiliary decoder with externally supplied, frozen shared projections."""

    @dataclass(kw_only=True, slots=True)
    class Config(Decoder.Config):
        layers: list[DSparkBlock.Config]
        hc_mult: int
        block_size: int
        noise_token_id: int
        target_layer_ids: tuple[int, ...]
        # These projections are owned by the frozen backbone and passed to forward.
        tok_embeddings: Embedding.Config | None = None  # pyrefly: ignore [bad-override]
        lm_head: Linear.Config | None = None  # pyrefly: ignore [bad-override]
        norm: RMSNorm.Config | None = None  # pyrefly: ignore [bad-override]

        def __post_init__(self):
            super(DSpark.Config, self).__post_init__()
            if (
                self.tok_embeddings is not None
                or self.lm_head is not None
                or self.norm is not None
                or self.enable_weight_tying
            ):
                raise ValueError(
                    "DSpark receives frozen backbone embeddings and head through forward."
                )
            if (
                not self.layers
                or self.layers[0].main_proj is None
                or self.layers[-1].norm is None
            ):
                raise ValueError(
                    "DSpark requires a first-block input projection and last-block heads."
                )

        def get_nparams_and_flops(self, model, seq_len):
            nparams, active = get_nparams_and_active_nparams(model)
            return nparams, 6 * active + sum(
                12
                * layer.attention.n_heads
                * layer.attention.head_dim
                * (layer.attention.context_window + self.block_size)
                for layer in self.layers
            )

    def __init__(self, config: Config):
        Module.__init__(self)
        self.config: DSpark.Config = config  # pyrefly: ignore [bad-override]
        # pyrefly: ignore [bad-override]
        self.layers: ModuleDict[DSparkBlock] = ModuleDict(
            {str(index): layer.build() for index, layer in enumerate(config.layers)}
        )
        self.tok_embeddings, self.norm, self.lm_head = None, None, None
        self.enable_weight_tying = False

    @classmethod
    def _register_optimizer_hooks(cls, optimizers, model_parts, parallelism_context):
        from torchtitan.models.common.moe import register_moe_load_balancing_hook

        register_moe_load_balancing_hook(optimizers, model_parts, parallelism_context)

    def forward(  # pyrefly: ignore [bad-param-name-override]
        self,
        main_features_TF,
        draft_embeddings_TD,
        previous_tokens_BS,
        anchors_B,
        *,
        main_positions_T,
        output_weight_VD,
    ):
        size = self.config.block_size
        if (
            previous_tokens_BS.shape != (anchors_B.numel(), size)
            or draft_embeddings_TD.size(0) != anchors_B.numel() * size
        ):
            raise ValueError(
                "DSpark requires one previous token and embedding per draft position."
            )
        main_features_TF = main_features_TF.detach()
        if (tp_group := spmd_mesh_group("tp")) is not None:
            main_features_TF = spmd.redistribute(
                main_features_TF,
                tp_group,
                src=spmd.S(0) if spmd_dense_sp_enabled() else spmd.I,
                dst=spmd.R,
            )
        hidden_TCD = draft_embeddings_TD.detach()[:, None, :].expand(
            -1, self.config.hc_mult, -1
        )
        pre_TC = initial_pre_mix(hidden_TCD)
        positions_T = (
            main_positions_T[anchors_B, None]
            + 1
            + torch.arange(size, device=anchors_B.device)
        ).flatten()
        empty_SD = main_features_TF.new_empty(
            0, self.config.layers[0].attention.head_dim
        )
        empty_TK = torch.empty(
            positions_T.numel(), 0, dtype=torch.int32, device=anchors_B.device
        )
        scores_TK = main_features_TF.new_empty(
            positions_T.numel(), 0, dtype=torch.float32
        )
        main_x_TD = logits_TV = confidence_T = None
        for layer in self.layers.values():
            hidden_TCD, pre_TC, main_x_TD, logits_TV, confidence_T = layer(
                hidden_TCD,
                pre_TC,
                empty_SD,
                empty_SD,
                empty_TK,
                scores_TK,
                empty_TK,
                positions=positions_T,
                main_features_TF=main_features_TF,
                main_x_TD=main_x_TD,
                main_positions_T=main_positions_T,
                anchors_B=anchors_B,
                previous_tokens_T=previous_tokens_BS.flatten(),
                output_weight_VD=output_weight_VD,
            )
        assert logits_TV is not None and confidence_T is not None
        return logits_TV.view(-1, size, logits_TV.size(-1)), confidence_T.view(-1, size)
