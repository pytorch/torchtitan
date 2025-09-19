# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Literal, Optional

# We reuse the Llama3 TransformerModelArgs as a base and extend it.
from torchtitan.models.llama3.model.args import TransformerModelArgs as LlamaTransformerModelArgs
from torchtitan.tools.logging import logger
from torch import nn


class SupportedLossFunctionIdentifiers(str, Enum):
    SFT_WITH_MOE_AUX_LOSS = "sft_with_moe_aux_loss"
    SFT = "sft"

@dataclass
class Qwen3TransformerModelArgs(LlamaTransformerModelArgs):
    """
    Model arguments for Qwen3, extending the Llama3 base arguments.
    """
    # --- Core model shape (override Llama defaults to match 235B config) ---
    dim: int = 4096  # hidden_size
    n_layers: int = 94  # num_hidden_layers
    n_heads: int = 64  # num_attention_heads
    n_kv_heads: int | None = 4  # num_key_value_heads
    vocab_size: int = 151936
    max_seq_len: int = 40960  # max_position_embeddings

    # --- MoE Specific Arguments ---
    num_experts: int = 128
    num_experts_per_tok: int = 8
    moe_intermediate_size: int = 1536
    router_aux_loss_coef: float = 0.001

    # --- Qwen3 Specific Arguments ---
    attention_bias: bool = False
    norm_eps: float = 1e-6
    # Optional MLP intermediate size for dense layers. 
    intermediate_size: int | None = 12288
    rope_theta: float = 1_000_000.0
    # Head dim allows to have a different effective hidden dim right before the attention.
    # I.e. we don't just divide the hidden dim by the number of attention heads.
    head_dim: int = 128
    
    decoder_sparse_step: int = 1
    mlp_only_layers: List[int] = field(default_factory=lambda: [])
    norm_topk_prob: bool = True
    use_grouped_mm: bool = True
    route_scale: float = 1.0
    tie_word_embeddings: bool = False

    def get_nparams_and_flops(
        self, model: nn.Module, seq_len: int
    ) -> tuple[int, float]:
        nparams_embedding = 0
        nparams_moe_router = 0
        nparams_shared_expert = 0
        nparams_experts = 0
        nparams_dense = 0

        for name, p in model.named_parameters():
            if "embedding" in name:
                nparams_embedding += p.numel()
                nparams_dense += p.numel()
            elif "mlp.shared_expert" in name:
                nparams_shared_expert += p.numel()
            elif "mlp.router" in name:
                nparams_moe_router += p.numel()
            elif "mlp.experts" in name:
                nparams_experts += p.numel()
            else:
                nparams_dense += p.numel()

        nparams_sparse = nparams_moe_router + nparams_shared_expert + nparams_experts
        nparams = nparams_dense + nparams_sparse
        nparams_sparse_active = (
            nparams_moe_router
            + nparams_shared_expert
            + nparams_experts * self.num_experts_per_tok // self.num_experts
        )

        logger.info(
            f"Total parameter count: dense {nparams_dense:,}, "
            f"sparse {nparams_sparse:,}, active {nparams_dense + nparams_sparse_active:,}"
        )

        l, h, q, t = (
            self.n_layers,
            self.n_heads,
            self.dim // self.n_heads,
            seq_len,
        )
        # Reasoning behind the factor of 12 for the self-attention part of the formula:
        # 1. each self-attention has 2 matmul in the forward and 4 in the backward (6)
        # 2. the flash attention does 1 more matmul recomputation in the backward
        #    but recomputation should not be counted in calculating MFU           (+0)
        # 3. each matmul performs 1 multiplication and 1 addition                 (*2)
        # 4. we follow the convention and do not account for sparsity in causal attention
        num_flops_per_token = (
            6 * (nparams_dense - nparams_embedding - nparams_embedding*self.tie_word_embeddings + nparams_sparse_active)
            + 12 * l * h * q * t
        )

        return nparams, num_flops_per_token

