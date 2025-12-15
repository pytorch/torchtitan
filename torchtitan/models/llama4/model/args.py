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
from torchtitan.protocols import BaseModelArgs
from torchtitan.tools.logging import logger
from torchtitan.tools.utils import has_cuda_capability


@dataclass
class RoPEScalingArgs:
    scaling_factor: float = 16.0
    low_freq_factor: float = 1.0
    high_freq_factor: float = 1.0
    original_max_position_embeddings: int = 8192


@dataclass
class TransformerModelArgs(BaseModelArgs):
    dim: int = 4096
    n_layers: int = 32
    n_heads: int = 32
    n_kv_heads: int | None = None
    vocab_size: int = 202048
    multiple_of: int = 256  # make SwiGLU hidden layer size multiple of large power of 2
    ffn_dim_multiplier: float | None = None
    norm_eps: float = 1e-5
    rope_theta: float = 10000
    rope_scaling_args: RoPEScalingArgs | None = None

    max_seq_len: int = 1048576
    # If `True`, then each transformer block init uses its layer ID, and if
    # `False`, each uses the total number of transformer blocks
    depth_init: bool = True

    attn_type: str = "sdpa"
    attn_mask_type: str = "causal"
    # iRoPE settings
    # When ``every_n_layers_nope`` is specified, NoPE (no positional embedding) is
    # used every n layers. Other layers uses RoPE (rotary positional embedding) and
    # the inner attention of those layer will use the fixed block size specified by
    # ``fixed_attn_block_size``. ``fixed_attn_block_size`` means that the query will
    # only attend to the tokens within the same block regardless how long is the
    # sequence.
    every_n_layers_nope: int | None = None
    fixed_attn_block_size: int = 8192

    # MoE
    moe_args: MoEArgs = field(default_factory=MoEArgs)
    auto_scale_hidden_dim: bool = True
    # frequency of using MoE layer instead of feedforward layer in a transformer block
    interleave_moe_layer_step: int = 2

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

        if (
            job_config.parallelism.context_parallel_degree > 1
            and self.attn_type != "sdpa"
        ):
            raise NotImplementedError("CP support is only supported for SDPA.")

        self.moe_args._debug_force_load_balance = (
            job_config.debug.moe_force_load_balance
        )

    def get_nparams_and_flops(
        self, model: nn.Module, seq_len: int
    ) -> tuple[int, float]:
        return get_moe_model_nparams_and_flops(
            self,
            model,
            2 * (self.dim // self.n_heads),
            seq_len,
        )
