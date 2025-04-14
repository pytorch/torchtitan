# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


from dataclasses import dataclass
from typing import Optional

from torch import nn
from torchtitan.components.tokenizer import Tokenizer
from torchtitan.config_manager import JobConfig

from torchtitan.protocols.train_spec import BaseModelArgs
from torchtitan.tools.logging import logger
from torchtitan.tools.utils import has_cuda_capability

# Model configuration registry
# Key is the model distribution ID on HuggingFace Hub
deepseek_config_registry = {
    "deepseek-ai/DeepSeek-V2-Lite": deepseek_v2_lite_config,
    "deepseek-ai/DeepSeek-V2-Lite-Chat": deepseek_v2_lite_config,
    "deepseek-ai/deepseek-v3": ModelArgs(),
}


# default to ds-v2lite
@dataclass
class DSV2ModelArgs(BaseModelArgs):
    use_grouped_mm: bool = True  # grouped mm or for-loop for the experts computation

    vocab_size = 10240
    hidden_size = 2048
    intermediate_size = 10944
    moe_intermediate_size = 1408
    num_hidden_layers = 27
    num_attention_heads = 16
    num_key_value_heads = 16
    n_shared_experts = 2
    n_routed_experts = 64
    routed_scaling_factor = 1.0
    kv_lora_rank = 512
    q_lora_rank = None
    qk_rope_head_dim = 64
    v_head_dim = 128
    qk_nope_head_dim = 128
    topk_method = "greedy"
    n_group = 1
    topk_group = 1
    num_experts_per_tok = 6
    first_k_dense_replace = 1
    norm_topk_prob = False
    scoring_func = "softmax"
    max_position_embeddings = 4096
    rope_scaling = {
        "beta_fast": 32,
        "beta_slow": 1,
        "factor": 40,
        "mscale": 0.707,
        "mscale_all_dim": 0.707,
        "original_max_position_embeddings": 4096,
        "type": "yarn",
    }

    # enforced from base class
    def update_from_config(self, job_config: JobConfig, tokenizer: Tokenizer) -> None:
        self.norm_type = job_config.model.norm_type
        self.vocab_size = tokenizer.n_words
        self.max_seq_len = job_config.training.seq_len
        self.use_flex_attn = job_config.model.use_flex_attn
        if self.use_grouped_mm and not has_cuda_capability(9, 0):
            logger.warning(
                "Failed to use grouped mm, which is only supported on SM90 or later",
            )
            self.use_grouped_mm = False
