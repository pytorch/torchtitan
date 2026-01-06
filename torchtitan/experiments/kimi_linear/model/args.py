# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass, field

from torch import nn

from torchtitan.config import JobConfig
from torchtitan.models.moe import MoEArgs
from torchtitan.models.utils import get_moe_model_nparams_and_flops
from torchtitan.protocols.train_spec import BaseModelArgs
from torchtitan.tools.logging import logger


@dataclass
class KimiLinearModelArgs(BaseModelArgs):
    # Core model dimensions
    dim: int = 2304
    n_layers: int = 27
    vocab_size: int = 163840
    hidden_dim: int = 9216
    hidden_act: str = "silu"

    # MLA (Multi-Latent Attention) parameters
    n_heads: int = 32
    n_kv_heads: int = 32
    head_dim: int = 72  # q_head_dim for attention (qk_nope_head_dim + qk_rope_head_dim)
    v_head_dim: int = 128  # value head dimension (separate from head_dim for MLA)
    kv_lora_rank: int = 512
    qk_nope_head_dim: int = 128
    qk_rope_head_dim: int = 64
    mla_use_nope: bool = True

    # RMSNorm
    norm_eps: float = 1e-5

    # RoPE
    rope_theta: float = 10000.0
    max_seq_len: int = 8192

    # Attention settings
    use_flex_attn: bool = True
    attn_mask_type: str = "block_causal"
    depth_init: bool = True
    enable_weight_tying: bool = False

    # Linear Attention (KDA) parameters
    linear_attn_num_heads: int = 32
    linear_attn_head_dim: int = 128
    linear_attn_conv_kernel_size: int = 4

    # Hybrid attention configuration
    # full_attn_layers specifies which layers use full MLA attention.
    # All other layers use linear (KDA) attention.
    # Uses 1-based indexing for direct compatibility with HuggingFace configs.
    # (HF uses `(layer_idx + 1) in kda_layers` for 0-indexed layer_idx)
    full_attn_layers: list[int] = field(default_factory=lambda: [4, 8, 12, 16, 20, 24, 27])
    layer_types: list[str] = field(default_factory=lambda: [])

    # MoE parameters
    moe_enabled: bool = False
    moe_inter_dim: int = 1024
    first_k_dense_replace: int = 1
    moe_layer_freq: int = 1
    moe_args: MoEArgs = field(default_factory=MoEArgs)

    def update_from_config(self, job_config: JobConfig, **kwargs) -> None:
        seq_len = job_config.training.seq_len
        if seq_len > self.max_seq_len:
            logger.warning(
                f"Sequence length {seq_len} exceeds original maximum {self.max_seq_len}."
            )
        self.max_seq_len = seq_len

        # Build layer_types from full_attn_layers if not specified
        # Note: full_attn_layers uses 1-based indexing (layer_idx + 1)
        if self.layer_types == []:
            self.layer_types = []
            for i in range(self.n_layers):
                if (i + 1) in self.full_attn_layers:
                    self.layer_types.append("full_attention")
                else:
                    self.layer_types.append("linear_attention")

        if not self.use_flex_attn:
            raise ValueError("Kimi Linear requires FlexAttention")
        if (
            job_config.compile.enable
            and "model" in job_config.compile.components
            and job_config.compile.fullgraph
        ):
            raise ValueError("`compile.fullgraph` must be off for Kimi Linear")

    def get_nparams_and_flops(
        self, model: nn.Module, seq_len: int
    ) -> tuple[int, float]:
        return get_moe_model_nparams_and_flops(
            self, model, self.qk_nope_head_dim + self.qk_rope_head_dim, seq_len
        )

    def is_kda_layer(self, layer_idx: int) -> bool:
        """Check if a layer should use KDA (linear attention).

        Args:
            layer_idx: 0-indexed layer index

        Note: full_attn_layers uses 1-based indexing for HF compatibility.
        """
        if self.layer_types:
            return self.layer_types[layer_idx] == "linear_attention"
        # Convert 0-indexed layer_idx to 1-indexed for comparison
        return (layer_idx + 1) not in self.full_attn_layers

    def is_moe_layer(self, layer_idx: int) -> bool:
        """Check if a layer should use MoE."""
        if not self.moe_enabled:
            return False
        return layer_idx >= self.first_k_dense_replace and layer_idx % self.moe_layer_freq == 0
