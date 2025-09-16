# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
#
# Copyright (c) Meta Platforms, Inc. All Rights Reserved.


from dataclasses import dataclass

from torch import nn

from torchtitan.config import JobConfig
from torchtitan.protocols.train_spec import BaseModelArgs
from torchtitan.tools.logging import logger


@dataclass
class TransformerModelArgs(BaseModelArgs):
    dim: int = 4096
    n_layers: int = 32
    n_heads: int = 32
    n_kv_heads: int | None = None
    vocab_size: int = 128256
    multiple_of: int = 256  # make SwiGLU hidden layer size multiple of large power of 2
    ffn_dim_multiplier: float | None = None
    norm_eps: float = 1e-5
    rope_theta: float = 10000

    max_seq_len: int = 131072
    # If `True`, then each transformer block init uses its layer ID, and if
    # `False`, each uses the total number of transformer blocks
    depth_init: bool = True

    use_flex_attn: bool = False
    attn_mask_type: str = "causal"
    eos_id: int = 0

    def update_from_config(self, job_config: JobConfig, **kwargs) -> None:
        seq_len = job_config.training.seq_len
        if seq_len > self.max_seq_len:
            logger.warning(
                f"Sequence length {seq_len} exceeds original maximum {self.max_seq_len}."
            )
        self.max_seq_len = seq_len

        if job_config.parallelism.context_parallel_degree > 1 and self.use_flex_attn:
            raise NotImplementedError(
                "CP support for FlexAttention is still in progress."
            )

        self.max_seq_len = seq_len

    def get_nparams_and_flops(self, model: nn.Module, seq_len: int) -> tuple[int, int]:
        """
        Count parameters and estimate flops for a TT (TorchTitan) model.

        Args:
            model (nn.Module): The TT model (not HF).
            seq_len (int): Sequence length.

        Returns:
            tuple[int, int]: (nparams, num_flops_per_token)
        """
        nparams = sum(p.numel() for p in model.parameters())

        layer_params = {}  # layer_id -> int
        embedding_params = 0
        norm_params = 0
        lm_head_params = 0
        misc_params = {}

        # TT model: top-level modules are tok_embeddings, layers (ModuleDict), norm, output
        for name, p in model.named_parameters():
            if name.startswith("tok_embeddings."):
                embedding_params += p.numel()
            elif name.startswith("layers."):
                try:
                    # layers.<layer_id>.<rest>
                    layer_id = int(name.split(".")[1])
                    if layer_id not in layer_params:
                        layer_params[layer_id] = 0
                    layer_params[layer_id] += p.numel()
                except (ValueError, IndexError):
                    # Should not happen, but catch any oddities
                    component = "misc_layer_parts"
                    if component not in misc_params:
                        misc_params[component] = 0
                    misc_params[component] += p.numel()
            elif name.startswith("norm."):
                norm_params += p.numel()
            elif name.startswith("output."):
                lm_head_params += p.numel()
            else:
                # Catch anything else
                component = name.split(".")[0]
                if component not in misc_params:
                    misc_params[component] = 0
                misc_params[component] += p.numel()

        logger.info("Parameter breakdown:")
        logger.info(f"  - embedding: {embedding_params:,} parameters")
        for layer_num in sorted(layer_params.keys()):
            params = layer_params[layer_num]
            logger.info(f"  - layer_{layer_num}: {params:,} parameters")
        logger.info(f"  - final_norm: {norm_params:,} parameters")
        logger.info(f"  - lm_head: {lm_head_params:,} parameters")
        if misc_params:
            for name, params in misc_params.items():
                logger.info(f"  - {name} (misc): {params:,} parameters")

        # For TT, embedding is always model.tok_embeddings
        nparams_embedding = sum(
            p.numel() for p in getattr(model, "tok_embeddings", nn.Module()).parameters()
        )

        l, h, q, t = self.n_layers, self.n_heads, self.dim // self.n_heads, seq_len
        num_flops_per_token = 6 * (nparams - nparams_embedding) + 12 * l * h * q * t
        return nparams, num_flops_per_token
