# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass, field
from typing import Optional, Union
import os

from torch import nn
from torchtitan.config import JobConfig
from torchtitan.protocols import BaseModelArgs
from torchtitan.tools.logging import logger
from transformers.models.llama.configuration_llama import LlamaConfig


@dataclass
class HFTransformerModelArgs(BaseModelArgs):
    # Torchtitan naming
    dim: int = 4096
    n_layers: int = 32
    n_heads: int = 32
    n_kv_heads: Optional[int] = None
    vocab_size: int = 128256
    multiple_of: int = 256
    ffn_dim_multiplier: Optional[float] = None
    rope_theta: float = 10000
    max_seq_len: int = 2048
    
    # HF compatibility
    rms_norm_eps: float = 1e-6
    use_cache: bool = True
    depth_init: bool = True
    use_flex_attn: bool = False
    attn_mask_type: str = "causal"
    eos_id: int = 0

    _torchtitan_args: dict = field(init=False, repr=False, default_factory=dict)

    def update_from_config(self, job_config: JobConfig):
        #TODO(3outeille): what if we dont specify flavor? Should use full as default
        flavor = getattr(job_config.model, "flavor", None)
        
        if flavor == "full":
            model_name_or_config: Union[LlamaConfig, str, os.PathLike] = job_config.model.name
            hf_model_config = LlamaConfig.from_pretrained(model_name_or_config)

            #TODO(3outeille): use getattr to handle models that don't have all the attributes
            # Fill torchtitan args with HF ones
            self.dim = hf_model_config.hidden_size
            self.n_layers = hf_model_config.num_hidden_layers
            self.n_heads = hf_model_config.num_attention_heads
            self.n_kv_heads = hf_model_config.num_key_value_heads
            self.vocab_size = hf_model_config.vocab_size
            self.rope_theta = getattr(hf_model_config, "rope_theta", 10000.0)
            self.max_seq_len = hf_model_config.max_position_embeddings
            self.rms_norm_eps = getattr(hf_model_config, "rms_norm_eps", 1e-6)

            if hasattr(hf_model_config, "intermediate_size") and hf_model_config.intermediate_size:
                self.ffn_dim_multiplier = hf_model_config.intermediate_size / hf_model_config.hidden_size

        # Always update max_seq_len to match training seq_len, warn if exceeded
        seq_len = job_config.training.seq_len
        if seq_len > self.max_seq_len:
            logger.warning(f"Sequence length {seq_len} exceeds original maximum {self.max_seq_len}.")
        self.max_seq_len = seq_len

        if job_config.parallelism.context_parallel_degree > 1 and self.use_flex_attn:
            raise NotImplementedError("CP support for FlexAttention is still in progress.")

        self._torchtitan_args = {
            "dim": self.dim,
            "n_layers": self.n_layers,
            "n_heads": self.n_heads,
            "n_kv_heads": self.n_kv_heads,
            "vocab_size": self.vocab_size,
            "multiple_of": self.multiple_of,
            "ffn_dim_multiplier": self.ffn_dim_multiplier,
            "rope_theta": self.rope_theta,
            "max_seq_len": self.max_seq_len,
            "rms_norm_eps": self.rms_norm_eps,
            "use_cache": self.use_cache,
            "depth_init": self.depth_init,
            "use_flex_attn": self.use_flex_attn,
            "attn_mask_type": self.attn_mask_type,
            "eos_id": self.eos_id,
        }
        return self

    def convert_to_hf_config(self) -> LlamaConfig:
        if not self._torchtitan_args:
            raise RuntimeError(
                "`update_from_config` must be called before `convert_to_hf_config` to prepare the arguments."
            )
        return LlamaConfig(**self._torchtitan_args)

    def get_nparams_and_flops(self, model: nn.Module, seq_len: int) -> tuple[int, int]:
        nparams = sum(p.numel() for p in model.parameters())
        nparams_embedding = sum(
            sum(p.numel() for p in m.parameters())
            for m in model.children()
            if isinstance(m, nn.Embedding)
        )

        l, h, q, t = self.n_layers, self.n_heads, self.dim // self.n_heads, seq_len
        num_flops_per_token = 6 * (nparams - nparams_embedding) + 12 * l * h * q * t
        return nparams, num_flops_per_token
