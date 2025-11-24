# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
"""
TorchTitan Qwen3 Model for vLLM Integration.

This module provides a vLLM-compatible wrapper around TorchTitan's Qwen3 model,
enabling models trained with TorchTitan to be served through vLLM for inference.

Example:
    ```python
    from vllm import LLM

    llm = LLM(
        model="Qwen/Qwen3-0.6B",
        trust_remote_code=True,
    )
    ```

IMPORTANT: TorchTitan imports are deferred to avoid CUDA initialization
before vLLM's multiprocessing fork.
"""

import torch

# Import from local custom_models directory
from torchtitan.experiments.vllm.custom_models import (
    load_external_weights,
    replace_with_trainable_attention,
    store_positions_in_context,
    VLLMModelForCausalLM,
)

from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.model_executor.parallel_context import ParallelContext

logger = init_logger(__name__)


class TorchTitanQwen3ForCausalLM(VLLMModelForCausalLM):
    """
    vLLM-compatible wrapper for TorchTitan's Qwen3 model.

    This class integrates TorchTitan's Qwen3Model with vLLM by:
    1. Importing TorchTitan's model architecture
    2. Replacing attention with vLLM's TrainableFlashAttention
    3. Implementing the vLLM model interface

    The architecture uses standard multi-head attention (not MLA),
    with RoPE positional embeddings and optional QK normalization.
    """

    supports_pp = False  # Pipeline parallelism not supported yet
    supports_multimodal = False

    def __init__(
        self,
        *,
        vllm_config: VllmConfig,
        prefix: str = "",
        parallel_context: ParallelContext | None = None,
    ):
        super().__init__()

        # vLLM config is required
        assert vllm_config is not None, "vllm_config is required"

        # Import TorchTitan's Qwen3 model (deferred import to avoid CUDA init issues)
        from torchtitan.models.qwen3.model.args import Qwen3ModelArgs
        from torchtitan.models.qwen3.model.model import Qwen3Model

        # Map HuggingFace config to TorchTitan ModelArgs
        hf_config = vllm_config.hf_config
        print("hf_config: ", hf_config)
        model_args = Qwen3ModelArgs(
            vocab_size=getattr(hf_config, "vocab_size", 151936),
            dim=getattr(hf_config, "hidden_size", 2048),
            n_layers=getattr(hf_config, "num_hidden_layers", 4),
            n_heads=getattr(hf_config, "num_attention_heads", 16),
            n_kv_heads=getattr(hf_config, "num_key_value_heads", 2),
            head_dim=getattr(hf_config, "head_dim", 128),
            hidden_dim=getattr(hf_config, "intermediate_size", 11008),
            norm_eps=getattr(hf_config, "rms_norm_eps", 1e-6),
            max_seq_len=getattr(hf_config, "max_position_embeddings", 8192),
            rope_theta=getattr(hf_config, "rope_theta", 1000000.0),
            qk_norm=getattr(hf_config, "qk_norm", True),
        )

        # Create TorchTitan model
        self.model = Qwen3Model(model_args)
        self.config = model_args
        self.parallel_context = parallel_context

        # Replace attention with vLLM's TrainableFlashAttention
        # (This happens before TP so TP can shard the attention weights)
        replace_with_trainable_attention(self.model, use_mla=False)

    def get_input_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Convert input token IDs to embeddings."""
        return self.model.tok_embeddings(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        positions: torch.Tensor | None = None,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor:
        """
        Forward pass with vLLM interface.

        Args:
            input_ids: Token IDs [batch, seq_len] (optional if inputs_embeds provided)
            positions: Position indices from vLLM for RoPE
            inputs_embeds: Pre-computed embeddings (optional, used by vLLM)
            **kwargs: Additional vLLM kwargs

        Returns:
            hidden_states: Final hidden states before LM head
        """
        # Store positions in forward context for attention layers
        store_positions_in_context(positions)

        # Get embeddings
        h = (
            inputs_embeds
            if inputs_embeds is not None
            else self.model.tok_embeddings(input_ids)
        )

        # Get RoPE cache
        seqlen = h.shape[1] if h.dim() == 3 else h.shape[0]
        rope_cache = self.model.rope_cache[:seqlen]

        # Pass through transformer layers
        for layer in self.model.layers.values():
            h = layer(h, rope_cache, attention_masks=None)

        # Final norm
        return self.model.norm(h)

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
        sampling_metadata=None,
    ) -> torch.Tensor:
        """Compute logits from hidden states."""
        return self.model.output(hidden_states)

    def load_weights(self, weights_iter):
        """
        Load weights from HuggingFace checkpoint.

        Maps HF Qwen weight names → TorchTitan naming convention.
        This uses the same mapping as TorchTitan's Qwen3StateDictAdapter.

        Args:
            weights_iter: Iterator of (name, tensor) pairs from HF checkpoint

        Returns:
            Number of loaded and skipped parameters
        """
        # HF → TorchTitan name mapping (from Qwen3StateDictAdapter)
        hf_to_tt = {
            "model.embed_tokens.weight": "tok_embeddings.weight",
            "lm_head.weight": "output.weight",
            "model.norm.weight": "norm.weight",
            # Attention weights
            "model.layers.{}.self_attn.q_proj.weight": "layers.{}.attention.wq.weight",
            "model.layers.{}.self_attn.k_proj.weight": "layers.{}.attention.wk.weight",
            "model.layers.{}.self_attn.v_proj.weight": "layers.{}.attention.wv.weight",
            "model.layers.{}.self_attn.o_proj.weight": "layers.{}.attention.wo.weight",
            "model.layers.{}.self_attn.q_norm.weight": (
                "layers.{}.attention.q_norm.weight"
            ),
            "model.layers.{}.self_attn.k_norm.weight": (
                "layers.{}.attention.k_norm.weight"
            ),
            # Skip rotary_emb.inv_freq (not used in TorchTitan)
            "model.layers.{}.self_attn.rotary_emb.inv_freq": None,
            # MLP weights
            "model.layers.{}.mlp.gate_proj.weight": "layers.{}.feed_forward.w1.weight",
            "model.layers.{}.mlp.up_proj.weight": "layers.{}.feed_forward.w3.weight",
            "model.layers.{}.mlp.down_proj.weight": "layers.{}.feed_forward.w2.weight",
            # Layer norms
            "model.layers.{}.input_layernorm.weight": (
                "layers.{}.attention_norm.weight"
            ),
            "model.layers.{}.post_attention_layernorm.weight": (
                "layers.{}.ffn_norm.weight"
            ),
        }

        # Load weights using utility function
        loaded, skipped = load_external_weights(
            model=self.model, weights_iter=weights_iter, name_mapping=hf_to_tt
        )

        logger.info(f"✅ Loaded {loaded} parameters, skipped {skipped}")

        return loaded, skipped
