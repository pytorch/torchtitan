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
    # load_external_weights,
    store_positions_in_context,
    VLLMModelForCausalLM,
)
from torchtitan.experiments.vllm.model.attention import VLLMCompatibleFlashAttention

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
        logger.info("vllm config: ", vllm_config.__class__)
        hf_config = vllm_config.model_config.hf_config
        logger.info("hf_config: ", hf_config)
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

        # Replace inner_attention with vLLM compatible Flash Attention
        # NOTE: We replace `inner_attention` (the attention kernel), NOT the whole `Attention` module
        # The `Attention` module handles QKV projection, RoPE, etc., and calls `inner_attention`
        if not hasattr(self.model, "layers"):
            raise AttributeError(
                f"Model {type(self.model).__name__} must have .layers attribute"
            )

        for layer_name, layer in self.model.layers.items():
            if not hasattr(layer, "attention"):
                raise ValueError(f"Layer {layer_name} must have .attention attribute")

            if not hasattr(layer.attention, "inner_attention"):
                raise ValueError(
                    f"Layer {layer_name}.attention must have .inner_attention attribute"
                )

            # NOTE(jianiw): Attention implementation 1: Add backward for vllm FlashAttn
            # Replace only the inner attention kernel, not the whole Attention module
            layer.attention.inner_attention = VLLMCompatibleFlashAttention(
                hidden_size=model_args.dim,
                num_heads=model_args.n_heads,
                num_kv_heads=model_args.n_kv_heads,
                head_dim=model_args.head_dim,
                causal=True,
            )

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
            input_ids: Token IDs from vLLM [total_tokens] (1D varlen format)
            positions: Position indices from vLLM [total_tokens] (1D varlen format)
            inputs_embeds: Pre-computed embeddings (optional, used by vLLM)
            **kwargs: Additional vLLM kwargs

        Returns:
            hidden_states: Final hidden states [total_tokens, hidden_size]
        """
        # Handle inputs_embeds vs input_ids properly
        if inputs_embeds is not None:
            raise NotImplementedError(
                "inputs_embeds is not yet supported by TorchTitan Qwen3. "
                "The model expects token IDs and computes embeddings internally. "
                "Please provide input_ids instead."
            )

        if input_ids is None:
            raise ValueError("Either input_ids or inputs_embeds must be provided")

        # Convert vLLM interface to TorchTitan interface
        # vLLM passes input_ids as [total_tokens] but TorchTitan expects [batch_size, seq_len]
        # For now, reshape to [1, total_tokens] as a simple batch of 1
        # TODO: In future, use attn_metadata.seq_lens to properly reconstruct batch structure
        tokens_2d = input_ids.unsqueeze(0)  # [total_tokens] -> [1, total_tokens]

        # Store positions in forward context for attention layers
        # Also convert positions to 2D format
        if positions is not None:
            positions_2d = positions.unsqueeze(0)  # [total_tokens] -> [1, total_tokens]
            store_positions_in_context(positions_2d)

        # Get embeddings from 2D tokens
        h = self.model.tok_embeddings(tokens_2d)  # [1, total_tokens, hidden_size]

        # Get RoPE cache
        seqlen = h.shape[1]  # seq_len dimension
        rope_cache = self.model.rope_cache[:seqlen]

        # Pass through transformer layers
        for layer in self.model.layers.values():
            h = layer(h, rope_cache, attention_masks=None)

        # Final norm
        h = self.model.norm(h)  # [1, total_tokens, hidden_size]

        # Convert output format back to vLLM expectations
        # vLLM expects hidden_states in [total_tokens, hidden_size] format
        # TorchTitan returns [batch_size, seq_len, hidden_size], so we need to flatten
        if h.dim() == 3:  # [batch_size, seq_len, hidden_size]
            batch_size, seq_len, hidden_size = h.shape
            h = h.view(batch_size * seq_len, hidden_size)  # [total_tokens, hidden_size]

        return h

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
            Set of loaded parameter names (for vLLM compatibility)
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

        # Track loaded parameter names
        loaded_params = set()

        # Convert iterator to list for processing
        from vllm.model_executor.model_loader.weight_utils import default_weight_loader

        # Get parameters from model
        params_dict = dict(self.model.named_parameters())

        weights_list = list(weights_iter)

        for hf_name, loaded_weight in weights_list:
            # Try to find matching pattern in name_mapping
            target_name = None

            # Check if it's a layer-specific weight
            if "layers" in hf_name:
                # Extract layer number
                import regex as re

                layer_match = re.search(r"layers\.(\d+)\.", hf_name)
                if layer_match:
                    layer_num = layer_match.group(1)

                    # Try to find matching pattern
                    for hf_pattern, target_pattern in hf_to_tt.items():
                        if "{}" in hf_pattern and target_pattern is not None:
                            hf_concrete = hf_pattern.format(layer_num)
                            if hf_name == hf_concrete:
                                target_name = target_pattern.format(layer_num)
                                break
            else:
                # Non-layer weight (embeddings, norms, output)
                target_name = hf_to_tt.get(hf_name)

            # Skip if no mapping or explicitly marked as None
            if target_name is None:
                continue

            # Check if parameter exists in model
            if target_name not in params_dict:
                continue

            # Load the weight into model parameter
            param = params_dict[target_name]

            # Verify shapes match
            if param.shape != loaded_weight.shape:
                logger.warning(
                    f"Shape mismatch for {target_name}: "
                    f"Model: {param.shape}, Checkpoint: {loaded_weight.shape}"
                )
                continue

            # Load the weight
            default_weight_loader(param, loaded_weight)

            # Add the parameter name to loaded set
            # Since CallableModelWrapper overrides named_parameters(),
            # the names returned here already match what vLLM expects
            loaded_params.add(target_name)

        logger.info(
            f"✅ Loaded {len(loaded_params)} parameters, loaded weights are: {loaded_params}"
        )

        return loaded_params
