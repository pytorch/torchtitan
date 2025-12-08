# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
#
# Copyright (c) Meta Platforms, Inc. All Rights Reserved.


# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
#
# Copyright (c) Meta Platforms, Inc. All Rights Reserved.


from dataclasses import dataclass, field

from torch import nn

from torchtitan.config import JobConfig
from torchtitan.models.moe import MoEArgs
from torchtitan.models.utils import get_moe_model_nparams_and_flops
from torchtitan.protocols.train_spec import BaseModelArgs

from torchtitan.tools.logging import logger


@dataclass
class Qwen3NextModelArgs(BaseModelArgs):

    dim: int = 2048
    n_layers: int = 48
    n_heads: int = 16
    n_kv_heads: int = 2
    vocab_size: int = 151936
    head_dim: int = 256
    hidden_dim: int = 5120
    hidden_act: str = "silu"

    norm_eps: float = 1e-6
    rope_theta: float = 1000000
    partial_rotary_factor: float = 0.25
    max_seq_len: int = 4096
    depth_init: bool = True

    use_flex_attn: bool = False
    attn_mask_type: str = "causal"
    eos_id: int = 151645

    enable_weight_tying: bool = False

    # MoE params
    moe_enabled: bool = False
    moe_inter_dim: int = 512
    moe_args: MoEArgs = field(default_factory=MoEArgs)

    # Hybrid attention
    decoder_sparse_step: int = 1
    full_attention_interval: int = 4
    layer_types: list[str] = field(default_factory=lambda: [])
    linear_num_key_heads: int = 16
    linear_num_value_heads: int = 32
    linear_key_head_dim: int = 128
    linear_value_head_dim: int = 128
    linear_conv_kernel_dim: int = 4

    def update_from_config(self, job_config: JobConfig, **kwargs) -> None:
        seq_len = job_config.training.seq_len
        if seq_len > self.max_seq_len:
            logger.warning(
                f"Sequence length {seq_len} exceeds original maximum {self.max_seq_len}."
            )
        self.max_seq_len = seq_len

        self.moe_args._debug_force_load_balance = (
            job_config.training.debug_moe_force_load_balance
        )

        # Pass DeepEP config to MoE layer and validate
        self.moe_args.deepep_config = job_config.deepep
        self.moe_args.validate_deepep_config()

        if self.layer_types == []:
            self.layer_types = [
                (
                    "linear_attention"
                    if bool((i + 1) % self.full_attention_interval)
                    else "full_attention"
                )
                for i in range(self.n_layers)
            ]

    def get_nparams_and_flops(
        self, model: nn.Module, seq_len: int
    ) -> tuple[int, float]:
        return get_moe_model_nparams_and_flops(self, model, seq_len)
