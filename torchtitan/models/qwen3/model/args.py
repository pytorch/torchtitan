# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
#
# Copyright (c) Meta Platforms, Inc. All Rights Reserved.


from dataclasses import dataclass

from torch import nn

from torchtitan.components.tokenizer import Tokenizer
from torchtitan.config_manager import JobConfig
from torchtitan.protocols.train_spec import BaseModelArgs


@dataclass
class TransformerModelArgs(BaseModelArgs):
    # Changing the model parameters to qwen 3

    # Vocab size: 151936

    dim: int = 1024  # 1024
    n_layers: int = 36  # 36
    n_heads: int = 16  # 16 heads
    # n_kv_heads: int | None = None
    n_kv_heads: int = 8
    vocab_size: int = 151936  # defined later by tokenizer
    # Question: where mulltiple of is being used?
    # We need to change the multiple of
    hidden_dim: int = 2560
    # multiple_of: int = 256  # make SwiGLU hidden layer size multiple of large power of 2 #
    # ffn_dim_multiplier: float | None = None
    norm_eps: float = 1e-6  # 1e-6
    rope_theta: float = 1000000  # 1000000
    qk_norm: bool = True
    max_seq_len: int = 4096  # Question this one? it's 4096
    # If `True`, then each transformer block init uses its layer ID, and if
    # `False`, each uses the total number of transformer blocks
    depth_init: bool = True

    use_flex_attn: bool = False
    attn_mask_type: str = "causal"
    eos_id: int = 0

    def update_from_config(self, job_config: JobConfig, tokenizer: Tokenizer) -> None:
        self.vocab_size = tokenizer.n_words
        self.max_seq_len = job_config.training.seq_len
        self.eos_id = tokenizer.eos_id

        if job_config.activation_checkpoint.mode == "selective" and self.use_flex_attn:
            raise ValueError(
                "FlexAttention is not compatible with selective AC yet. "
                "See https://github.com/pytorch/pytorch/issues/147879"
            )

        if job_config.parallelism.context_parallel_degree > 1 and self.use_flex_attn:
            raise ValueError(
                "FlexAttention is not compatible with CP yet. "
                "We are still working on this."
            )

    def get_nparams_and_flops(self, model: nn.Module, seq_len: int) -> tuple[int, int]:
        nparams = sum(p.numel() for p in model.parameters())
        nparams_embedding = sum(
            sum(p.numel() for p in m.parameters())
            for m in model.children()
            if isinstance(m, nn.Embedding)
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
        # Question: is Nu_flops_per_token being used anywhere?
        num_flops_per_token = (
            6 * (nparams - nparams_embedding) + 12 * l * h * q * t
        )  # Going over the math of this formula for Qwen3

        return nparams, num_flops_per_token
