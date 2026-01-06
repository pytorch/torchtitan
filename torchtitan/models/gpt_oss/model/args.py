# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


from dataclasses import dataclass, field
from typing import Literal

from torch import nn

from torchtitan.config.job_config import JobConfig
from torchtitan.models.moe import MoEArgs
from torchtitan.models.utils import get_moe_model_nparams_and_flops
from torchtitan.protocols.train_spec import BaseModelArgs
from torchtitan.tools.logging import logger
from torchtitan.tools.utils import has_cuda_capability


@dataclass
class GptOssModelArgs(BaseModelArgs):
    """
    Data class for defining model arguments and hyperparameters.

    Attributes:
        max_batch_size (int): Maximum batch size.
        max_seq_len (int): Maximum sequence length.
        dtype (Literal["bf16", "fp8"]): Data type for computations.
        vocab_size (int): Vocabulary size.
        dim (int): Model dimension.
        moe_inter_dim (int): Intermediate dimension for MoE layers.
        n_layers (int): Number of transformer layers.
        norm_eps (float): Epsilon used for RMSNorm.
        moe_args (MoEArgs): Arguments for Mixture of Experts (MoE) layers.
        swiglu_limit (float): SwiGLU activation limit.
        head_dim (int): Dimension of each attention head.
        n_heads (int): Number of attention heads.
        n_kv_heads (int): Number of key-value heads.
        sliding_window_size (int): Size of the sliding attention window.
        attn_mask_type (str): Type of basic attention mask.
        attn_type (bool): Attention type, only supports Flex.
        original_seq_len (int): Original sequence length.
        rope_theta (float): Base for rotary positional encoding.
        rope_factor (float): Scaling factor for extended sequence lengths.
        beta_fast (int): Fast beta correction factor.
        beta_slow (int): Slow beta correction factor.
    """

    max_batch_size: int = 8
    max_seq_len: int = 131072
    dtype: Literal["bf16", "fp8"] = "bf16"
    vocab_size: int = 201088
    dim: int = 2880
    moe_inter_dim: int = 2880
    n_layers: int = 24
    norm_eps: float = 1e-5  # eps used for RMSNorm
    # MoE
    moe_args: MoEArgs = field(default_factory=MoEArgs)
    swiglu_limit: float = 7.0
    # Multi-Head Latent Attention (MLA)
    head_dim: int = 64
    n_heads: int = 64
    n_kv_heads: int = 8
    sliding_window_size: int = 128
    attn_mask_type: str = "causal"
    attn_type: str = "flex"  # NOTE: gpt-oss only support FlexAttention
    # yarn
    original_seq_len: int = 4096
    rope_theta: float = 150000.0
    rope_factor: float = 32
    beta_fast: int = 32
    beta_slow: int = 1

    def update_from_config(self, job_config: JobConfig, **kwargs) -> None:
        seq_len = job_config.training.seq_len
        if seq_len > self.max_seq_len:
            logger.warning(
                f"Sequence length {seq_len} exceeds original maximum {self.max_seq_len}."
            )
        self.max_seq_len = seq_len

        if self.moe_args.use_grouped_mm and not has_cuda_capability(9, 0):
            logger.warning(
                "Failed to use grouped mm, which is only supported on SM90 or later",
            )
            self.moe_args.use_grouped_mm = False

        if job_config.parallelism.context_parallel_degree > 1:
            raise NotImplementedError(
                "CP support for gpt-oss model is still in progress."
            )

    def get_nparams_and_flops(
        self, model: nn.Module, seq_len: int
    ) -> tuple[int, float]:
        return get_moe_model_nparams_and_flops(self, model, 2 * self.head_dim, seq_len)
