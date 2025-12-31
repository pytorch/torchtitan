# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import re
from dataclasses import dataclass, field

from torch import nn

from torchtitan.config import JobConfig
from torchtitan.models.utils import get_dense_model_nparams_and_flops
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
    sliding_window: int | None = None
    max_seq_len: int = 4096  # max_position_embeddings
    attn_dropout: float = 0.0  # attention_dropout
    hidden_dropout: float = 0.0

    # Activation and biases
    mlp_hidden_act: str = "relu2"
    attn_bias: bool = False  # attention_bias
    mlp_bias: bool = False
    use_bias: bool = False

    # Initialization and normalization
    initializer_range: float = 0.02
    norm_eps: float = 1e-5  # layer_norm_epsilon
    residual_in_fp32: bool = False

    # Cache
    use_cache: bool = True
    num_logits_to_keep: int = 1

    # Special tokens
    pad_id: int = 0
    bos_id: int = 1
    eos_id: int = 2

    # Weight tying
    enable_weight_tying: bool = False  # tie_word_embeddings

    # Mamba2 configuration
    use_mamba_kernels: bool = True
    ssm_state_size: int = 128  # mamba_state_size
    mamba_num_heads: int = 128
    mamba_n_groups: int = 8
    mamba_head_dim: int = 64
    mamba_d_conv: int = 4
    mamba_expand: int = 2
    mamba_hidden_act: str = "silu"
    mamba_dt_min: float = 0.001
    mamba_dt_max: float = 0.1
    mamba_dt_limit: tuple[float, float] = field(
        default_factory=lambda: (0.0, float("inf"))
    )
    mamba_dt_init_floor: float = 1e-4
    mamba_conv_bias: bool = True
    mamba_proj_bias: bool = False
    mamba_chunk_size: int = 128

    # Residual scaling
    rescale_prenorm_residual: bool = True

    # MoE (Mixture of Experts) configuration
    n_routed_experts: int = 8
    n_shared_experts: int = 1
    moe_intermediate_size: int = 7688
    moe_shared_expert_intermediate_size: int = 7688
    num_experts_per_tok: int = 2
    routed_scaling_factor: float = 1.0
    n_group: int = 1
    topk_group: int = 1
    norm_topk_prob: bool = True

    # Attention type for torchtitan
    attn_type: str = "sdpa"
    attn_mask_type: str = "causal"

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
    ) -> tuple[int, float]:
        # TODO(aghilann): this number should accurately consider the {Mambda, Attention, MoE layers}
        return get_dense_model_nparams_and_flops(
            self,
            model,
            2 * self.head_dim,
            seq_len,
        )
