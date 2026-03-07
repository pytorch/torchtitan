# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import re
from dataclasses import dataclass, field

from torch import nn

from torchtitan.config import JobConfig
from torchtitan.protocols.model import BaseModelArgs
from torchtitan.tools.logging import logger


@dataclass
class Nemotron3ModelArgs(BaseModelArgs):
    """
    Model arguments for NemotronH (Hybrid Mamba2 + Attention model).

    NemotronH is a hybrid architecture combining:
    - Mamba2 layers (M)
    - Attention layers (*)
    - MLP layers (-)
    - MoE layers (any other character, typically 'E')

    The layer pattern is defined by `hybrid_override_pattern`.

    Default values are for NemotronH-v0.1 configuration.
    """

    # Core model architecture
    vocab_size: int = 131072
    dim: int = 4096  # hidden_size
    hidden_dim: int = 21504  # intermediate_size
    n_layers: int = 52  # num_hidden_layers

    # Hybrid layer pattern: M=Mamba2, *=Attention, -=MLP, other (E/O)=MoE
    hybrid_override_pattern: str = (
        "M-M-M-M*-M-M-M-M-M*-M-M-M-M-M*-M-M-M-M-M*-M-M-M-M-M-"
    )

    # Attention configuration
    n_heads: int = 32  # num_attention_heads
    head_dim: int = 128
    n_kv_heads: int = 8  # num_key_value_heads (GQA)
    max_seq_len: int = 4096  # max_position_embeddings
    attn_dropout: float = 0.0  # attention_dropout

    # Activation and biases
    mlp_hidden_act: str = "relu2"
    attn_bias: bool = False  # attention_bias
    mlp_bias: bool = False

    # Initialization and normalization
    initializer_range: float = 0.02
    norm_eps: float = 1e-5  # layer_norm_epsilon
    residual_in_fp32: bool = False

    # Weight tying
    enable_weight_tying: bool = False  # tie_word_embeddings

    # Mamba2 configuration
    ssm_state_size: int = 128  # mamba_state_size
    mamba_num_heads: int = 128
    mamba_n_groups: int = 8
    mamba_head_dim: int = 64
    mamba_d_conv: int = 4
    mamba_hidden_act: str = "silu"
    mamba_dt_limit: tuple[float, float] = field(
        default_factory=lambda: (0.0, float("inf"))
    )
    mamba_conv_bias: bool = True
    mamba_proj_bias: bool = False
    mamba_chunk_size: int = 128

    # MoE (Mixture of Experts) configuration
    n_routed_experts: int = 8
    moe_intermediate_size: int = 7688
    moe_shared_expert_intermediate_size: int = 7688
    num_experts_per_tok: int = 2
    routed_scaling_factor: float = 1.0
    n_group: int = 1
    topk_group: int = 1
    norm_topk_prob: bool = True

    # Attention type for torchtitan
    attn_type: str = "sdpa"

    # Block initialization strategy
    depth_init: bool = True

    def __post_init__(self) -> None:
        """Validate configuration after initialization."""
        # Validate hybrid_override_pattern length matches num_hidden_layers
        assert len(self.hybrid_override_pattern) == self.n_layers, (
            f"hybrid_override_pattern length ({len(self.hybrid_override_pattern)}) "
            f"must match n_layers ({self.n_layers})"
        )

        # Validate hybrid_override_pattern contains only valid characters
        assert re.match(r"^[*M\-EO]+$", self.hybrid_override_pattern), (
            "hybrid_override_pattern must only contain characters 'M' (Mamba2), "
            "'*' (Attention), '-' (MLP), or 'E'/'O' (MoE)"
        )

        # Set n_kv_heads to n_heads if not specified (for backward compatibility)
        if self.n_kv_heads is None:
            self.n_kv_heads = self.n_heads

    @property
    def layers_block_type(self):
        return [
            "mamba"
            if self.hybrid_override_pattern[i] == "M"
            else "attention"
            if self.hybrid_override_pattern[i] == "*"
            else "mlp"
            if self.hybrid_override_pattern[i] == "-"
            else "moe"
            for i in range(self.n_layers)
        ]

    def update_from_config(self, job_config: JobConfig, **kwargs) -> None:
        """Update model args from job config."""
        seq_len = job_config.training.seq_len
        if seq_len > self.max_seq_len:
            logger.warning(
                f"Sequence length {seq_len} exceeds original maximum {self.max_seq_len}."
            )
        self.max_seq_len = seq_len

        if (
            job_config.parallelism.context_parallel_degree > 1
            and self.attn_type != "sdpa"
        ):
            raise NotImplementedError(
                "CP support for FlexAttention is still in progress."
            )

    def get_nparams_and_flops(
        self, model: nn.Module, seq_len: int
    ) -> tuple[int, int]:
        """
        Estimate parameters and FLOPs/token for Nemotron's hybrid stack.

        This follows the same active-parameter accounting pattern used for MoE
        models, while only applying the attention quadratic term to attention
        layers in the hybrid pattern.
        """
        nparams_embedding = 0
        nparams_moe_router = 0
        nparams_shared_experts = 0
        nparams_experts = 0
        nparams_dense = 0

        for name, p in model.named_parameters():
            if "tok_embeddings" in name:
                nparams_embedding += p.numel()
                nparams_dense += p.numel()
            elif ".mixer.shared_experts." in name:
                nparams_shared_experts += p.numel()
            elif ".mixer.router." in name:
                nparams_moe_router += p.numel()
            elif ".mixer.experts." in name:
                nparams_experts += p.numel()
            else:
                nparams_dense += p.numel()

        nparams_sparse = nparams_moe_router + nparams_shared_experts + nparams_experts
        nparams = nparams_dense + nparams_sparse

        if self.n_routed_experts > 0:
            nparams_sparse_active = (
                nparams_moe_router
                + nparams_shared_experts
                + nparams_experts * self.num_experts_per_tok // self.n_routed_experts
            )
        else:
            nparams_sparse_active = nparams_moe_router + nparams_shared_experts
        active_nparams = nparams_dense + nparams_sparse_active

        attention_layers = self.layers_block_type.count("attention")
        head_dims = 2 * self.head_dim
        num_flops_per_token = (
            6 * (active_nparams - nparams_embedding)
            + 6 * attention_layers * self.n_heads * head_dims * seq_len
        )

        logger.info(
            f"Nemotron hybrid parameter count: dense {nparams_dense:,}, "
            f"sparse {nparams_sparse:,}, active {active_nparams:,}, "
            f"attention_layers {attention_layers}"
        )

        if self.enable_weight_tying:
            nparams = nparams - nparams_embedding

        return nparams, num_flops_per_token
