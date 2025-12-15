# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Base wrapper for TorchTitan models to work with vLLM V1 engine.

This module provides TorchTitanVLLMModel: Core model class that adapts
TorchTitan models for vLLM.
"""

from functools import partial
from typing import Callable, TypeAlias

import torch
import torch.nn as nn
from torch.distributed._tensor import DTensor, Replicate
from torch.distributed.checkpoint.state_dict import (
    set_model_state_dict,
    StateDictOptions,
)

from vllm.config import VllmConfig
from vllm.logger import init_logger

from torchtitan.experiments.deterministic_vllm_rl.unified.attention import VLLMAttention
from torchtitan.models.qwen3.model.model import precompute_rope_cache
from torchtitan.protocols.model import BaseModelArgs, ModelProtocol
from torchtitan.protocols.state_dict_adapter import BaseStateDictAdapter

from .utils import create_parallel_dims_from_vllm_config


logger = init_logger(__name__)

ParallelizeFunction: TypeAlias = Callable[..., nn.Module]


class TorchTitanVLLMModelWrapper(nn.Module):
    """
    Generic vLLM-compatible model wrapper for TorchTitan models.

    The wrapper handles:
    - Direct usage of TorchTitan model args (no HF config mapping needed)
    - Attention replacement with vLLM paged attention
    - Tensor parallelism setup
    - Weight loading from HF checkpoints
    - vLLM forward/compute_logits interface
    """

    is_text_generation_model = True  # Required for vLLM runner validation
    supports_pp = False  # Pipeline parallelism not supported yet
    supports_multimodal = False

    def __init__(
        self,
        *,
        model_cls: type[ModelProtocol],
        model_args: BaseModelArgs,
        state_dict_adapter: type[BaseStateDictAdapter],
        parallelize_fn: ParallelizeFunction,
        vllm_config: VllmConfig,
        prefix: str = "",
    ):
        super().__init__()

        assert vllm_config is not None, "vllm_config is required"

        # Store components
        self.model_cls = model_cls
        self.state_dict_adapter = state_dict_adapter
        self.parallelize_fn = parallelize_fn

        # Use TorchTitan model args directly (no HF config mapping)
        self.config = model_args
        logger.info(f"Creating {self.model_cls.__name__} with config: {model_args}")
        self.model = self.model_cls(model_args)

        # Setup RoPE cache extension function if provided
        self.rope_cache_extension_fn = partial(
            precompute_rope_cache,
            dim=self.config.head_dim,
            base=self.config.rope_theta,
        )
        # Replace attention with vLLM paged attention
        self._replace_with_vllm_attention(model_args)

        # Create ParallelDims from vLLM config and apply parallelization
        # NOTE: We need to apply parallelize within model.__init__ because w
        parallel_dims = create_parallel_dims_from_vllm_config(vllm_config)
        if parallel_dims.tp_enabled:
            self.world_mesh = parallel_dims.world_mesh
            tp_mesh = self.world_mesh["tp"]
            parallelize_fn(
                model=self.model,
                tp_mesh=tp_mesh,
                loss_parallel=False,
                enable_float8_tensorwise_tp=False,
                enable_async_tp=False,
            )
            logger.info(
                f"Successfully initialized model with with TP={parallel_dims.tp}"
            )
        else:
            logger.info("Single GPU mode - no parallelization needed")

    def _replace_with_vllm_attention(self, model_args):
        """
        Replace TorchTitan attention with vLLM paged attention.

        Assumes model has .layers dict with .attention.inner_attention structure.
        Override in subclass if different structure.
        """
        assert hasattr(
            self.model, "layers"
        ), f"Model {type(self.model).__name__} must have .layers attribute"

        for layer_name, layer in self.model.layers.items():
            assert hasattr(
                layer, "attention"
            ), f"Layer {layer_name} must have .attention attribute"

            vllm_attn = VLLMAttention(
                hidden_size=model_args.dim,
                num_heads=model_args.n_heads,
                num_kv_heads=model_args.n_heads,  # Use n_heads (already replicated)
                head_dim=model_args.head_dim,
                layer_name=layer_name,
                scale=model_args.head_dim**-0.5,
            )

            # Replace inner attention
            layer.attention.inner_attention = vllm_attn

        logger.info(
            f"Successfully replaced TorchTitan attention with VLLMAttention "
            f"({len(self.model.layers)} layers)"
        )

    def _extend_rope_cache_if_needed(
        self, rope_cache: torch.Tensor, max_position: int
    ) -> torch.Tensor:
        """
        Extend RoPE cache if needed during vLLM profiling stage.

        Args:
            rope_cache: Current RoPE cache tensor
            max_position: Maximum position index needed

        Returns:
            Extended RoPE cache if needed, otherwise original cache
        """
        from torch.distributed._tensor import DTensor, Replicate

        required_len = max_position + 1

        # No extension needed
        if required_len <= rope_cache.shape[0]:
            return rope_cache

        # If no extension function provided, return original cache
        if self.rope_cache_extension_fn is None:
            logger.warning(
                f"RoPE cache extension needed (required_len={required_len}, "
                f"current_len={rope_cache.shape[0]}) but no rope_cache_extension_fn provided. "
                "Returning original cache."
            )
            return rope_cache

        # Handle DTensor case
        is_dtensor = isinstance(rope_cache, DTensor)
        if is_dtensor:
            device_mesh = rope_cache.device_mesh
            local_rope_cache = rope_cache.to_local()
            device = local_rope_cache.device
            dtype = local_rope_cache.dtype
        else:
            device = rope_cache.device
            dtype = rope_cache.dtype

        # Use provided extension function
        try:
            extended_cache = self.rope_cache_extension_fn(self.config, required_len)
            extended_cache = extended_cache.to(device=device, dtype=dtype)
        except Exception as e:
            logger.warning(
                f"Failed to extend RoPE cache using rope_cache_extension_fn: {e}. "
                "Returning original cache."
            )
            return rope_cache

        # Convert back to DTensor if needed
        if is_dtensor:
            rope_cache = DTensor.from_local(
                extended_cache,
                device_mesh=device_mesh,
                placements=[Replicate()],
            )
        else:
            rope_cache = extended_cache

        return rope_cache

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Convert input token IDs to embeddings."""
        return self.model.tok_embeddings(input_ids)

    def get_input_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Convert input token IDs to embeddings (deprecated vLLM interface)."""
        return self.embed_input_ids(input_ids)

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
            input_ids: Token IDs [total_tokens] (1D varlen format)
            positions: Position indices [total_tokens] (1D varlen format)
            inputs_embeds: Pre-computed embeddings (optional)
            **kwargs: Additional vLLM kwargs

        Returns:
            hidden_states: Final hidden states [total_tokens, hidden_size]
        """
        if inputs_embeds is not None:
            raise NotImplementedError("inputs_embeds not yet supported")

        if input_ids is None:
            raise ValueError("Either input_ids or inputs_embeds must be provided")

        # Convert vLLM interface to TorchTitan interface
        # vLLM: [total_tokens] → TorchTitan: [batch_size, seq_len]
        tokens_2d = input_ids.unsqueeze(0)

        # Get embeddings
        h = self.model.tok_embeddings(tokens_2d)

        # Get RoPE cache (handle model-specific attribute names)
        # Use hasattr to avoid ambiguous boolean value error with tensors
        if hasattr(self.model, "rope_cache"):
            rope_attr = self.model.rope_cache
        elif hasattr(self.model, "freqs_cis"):
            rope_attr = self.model.freqs_cis
        else:
            rope_attr = None

        # Extend RoPE cache if needed (vLLM profiling may use 2x max_seq_len)
        if positions is not None:
            max_position = positions.max().item()
        else:
            max_position = 0

        rope_cache = self._extend_rope_cache_if_needed(rope_attr, max_position)
        positions = positions.unsqueeze(0)

        # Pass through transformer layers
        for layer in self.model.layers.values():
            h = layer(h, rope_cache, attention_masks=None, positions=positions)

        # Convert to vLLM format: [total_tokens, hidden_size]
        if h.dim() == 3:
            batch_size, seq_len, hidden_size = h.shape
            h = h.view(batch_size * seq_len, hidden_size)

        return h

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
        sampling_metadata=None,
    ) -> torch.Tensor | None:
        """Compute logits from hidden states."""
        h = self.model.norm(hidden_states)
        logits = self.model.output(h)

        return logits

    def load_weights(self, weights_iter):
        """
        Load weights from HF checkpoint using the provided state dict adapter.
        vLLM engine would call this function to load model weights.

        Args:
            weights_iter: Iterator of (name, tensor) pairs from HF checkpoint

        Returns:
            Set of loaded parameter names
        """
        # Collect weights from iterator
        hf_state_dict = {}
        for name, tensor in weights_iter:
            hf_state_dict[name] = tensor

        # Use adapter to convert HF → TorchTitan format
        adapter = self.state_dict_adapter(
            model_args=self.config,
            hf_assets_path=None,
        )

        torchtitan_state_dict = adapter.from_hf(hf_state_dict)
        model_state_dict = {k: v for k, v in self.model.state_dict().items()}

        # Convert to DTensor if target is DTensor
        for name, tensor in torchtitan_state_dict.items():
            if name in model_state_dict and isinstance(model_state_dict[name], DTensor):
                target_dtensor = model_state_dict[name]
                device_mesh = target_dtensor.device_mesh
                torchtitan_state_dict[name] = DTensor.from_local(
                    tensor.to(device_mesh.device_type),
                    device_mesh=device_mesh,
                    placements=[Replicate()],
                )

        # Load state dict
        set_model_state_dict(
            model=self.model,
            model_state_dict=torchtitan_state_dict,
            options=StateDictOptions(strict=False),
        )

        loaded_params = {f"model.{name}" for name in torchtitan_state_dict.keys()}

        return loaded_params
