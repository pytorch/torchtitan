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
from torch.distributed._tensor import DTensor, Replicate, Shard
from torch.distributed.checkpoint.state_dict import (
    set_model_state_dict,
    StateDictOptions,
)

from vllm.config import VllmConfig
from vllm.logger import init_logger

from torchtitan.experiments.deterministic_vllm_rl.models.attention import (
    VLLMPagedFlashAttention,
)
from torchtitan.experiments.deterministic_vllm_rl.models.utils import (
    create_job_config_from_vllm_config,
    create_parallel_dims,
)
from torchtitan.models.qwen3.model.model import precompute_rope_cache
from torchtitan.protocols.model import BaseModelArgs, ModelProtocol
from torchtitan.protocols.state_dict_adapter import BaseStateDictAdapter

logger = init_logger(__name__)

ParallelizeFunction: TypeAlias = Callable[..., nn.Module]


class TorchTitanVLLMModel(nn.Module):
    """
    Generic vLLM-compatible model wrapper for TorchTitan models.

    The wrapper handles:
    - HF config to TorchTitan model args mapping
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
        model_cls: type[ModelProtocol],  # passing types that is not instantiated
        model_args_cls: type[BaseModelArgs],
        state_dict_adapter: type[BaseStateDictAdapter],
        parallelize_fn: ParallelizeFunction,
        vllm_config: VllmConfig,
        prefix: str = "",
    ):
        super().__init__()

        assert vllm_config is not None, "vllm_config is required"

        # Store components
        self.model_cls = model_cls
        self.model_args_cls = model_args_cls
        self.state_dict_adapter = state_dict_adapter
        self.parallelize_fn = parallelize_fn

        # Map HF config to TorchTitan ModelArgs
        hf_config = vllm_config.model_config.hf_config
        logger.info(f"Mapping HF config to {self.model_args_cls.__name__}")
        model_args = self._map_hf_config_to_model_args(hf_config, self.model_args_cls)

        # Create TorchTitan model
        logger.info(f"Creating {self.model_cls.__name__} with config: {model_args}")
        self.model = self.model_cls(model_args)
        self.config = model_args

        # Setup RoPE cache extension function if provided
        self.rope_cache_extension_fn = partial(
            precompute_rope_cache,
            dim=self.config.head_dim,
            base=self.config.rope_theta,
        )
        # Replace attention with vLLM paged attention
        self._replace_with_vllm_paged_attention(model_args)

        # Create JobConfig and ParallelDims from vLLM config
        job_config = create_job_config_from_vllm_config(vllm_config)
        self.parallel_dims = parallel_dims = create_parallel_dims(job_config)

        # Apply parallelization using parallelize_qwen3
        if parallel_dims.tp_enabled:
            parallelize_fn(
                self.model,
                parallel_dims,
                job_config,
            )
            logger.info(f"Successfully initialized model with TP={parallel_dims.tp}")
        else:
            logger.info("Single GPU mode - no parallelization needed")

    def _map_hf_config_to_model_args(self, hf_config, model_args_cls):
        """
        Map HuggingFace config to TorchTitan ModelArgs.

        Default implementation that handles common model args fields.
        Override in subclass if custom mapping is needed.
        """
        # Maps TorchTitan parameter name to HF config attribute name
        mapping = {
            "vocab_size": "vocab_size",
            "dim": "hidden_size",
            "n_layers": "num_hidden_layers",
            "n_heads": "num_attention_heads",
            "n_kv_heads": "num_key_value_heads",
            "head_dim": "head_dim",
            "hidden_dim": "intermediate_size",
            "norm_eps": "rms_norm_eps",
            "max_seq_len": "max_position_embeddings",
            "rope_theta": "rope_theta",
            "qk_norm": "qk_norm",
        }

        # Build kwargs for model args from mapping
        kwargs = {}
        for torchtitan_param, hf_attr in mapping.items():
            # Try to get value from HF config
            if hasattr(hf_config, hf_attr):
                kwargs[torchtitan_param] = getattr(hf_config, hf_attr)

        return model_args_cls(**kwargs)

    def _replace_with_vllm_paged_attention(self, model_args):
        """
        Replace TorchTitan attention with vLLM paged attention.

        Assumes model has .layers dict with .attention.inner_attention structure.
        Override in subclass if different structure.
        """
        if not hasattr(self.model, "layers"):
            raise AttributeError(
                f"Model {type(self.model).__name__} must have .layers attribute"
            )

        for layer_name, layer in self.model.layers.items():
            if not hasattr(layer, "attention"):
                raise ValueError(f"Layer {layer_name} must have .attention attribute")

            # Create vLLM paged attention
            vllm_attn = VLLMPagedFlashAttention(
                hidden_size=model_args.dim,
                num_heads=model_args.n_heads,
                num_kv_heads=model_args.n_heads,  # Use n_heads (already replicated)
                head_dim=model_args.head_dim,
                causal=True,
            )

            # Replace inner attention
            layer.attention.inner_attention = vllm_attn

        logger.info(
            "Successfully replaced TorchTitan attention with vLLM PagedFlashAttention"
        )

    def _extend_rope_cache_if_needed(
        self, rope_cache: torch.Tensor, max_position: int
    ) -> torch.Tensor:
        """
        Extend RoPE cache if needed during vLLM profiling.

        Args:
            rope_cache: Current RoPE cache tensor
            max_position: Maximum position index needed

        Returns:
            Extended RoPE cache if needed, otherwise original cache
        """

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

        # Handle DTensor case when TP is applied
        if self.parallel_dims.tp_enabled:
            assert isinstance(rope_cache, DTensor)
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

        # Convert back to DTensor when TP is applied
        if self.parallel_dims.tp_enabled:
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
        inputs_embeddings = self.model.tok_embeddings(input_ids)
        # NOTE: When TP is applied, the inputs_embedding will be row-wise sharded (Shard(1))
        # and will be a plain tensor, return directly
        return inputs_embeddings

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
        print(
            f"before calling tok_embeddings, tokens_2d is_dtensor {isinstance(tokens_2d, DTensor)}"
        )
        # Get embeddings
        h = self.model.tok_embeddings(tokens_2d)  # h will be shard(1) plain tensor

        print(f"after calling tok_embeddings, h is_dtensor {isinstance(h, DTensor)}")
        if isinstance(h, DTensor):
            print(f"h is DTensor with placements: {h.placements}, shape: {h.shape}")

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

        # Pass through transformer layers
        for layer in self.model.layers.values():
            h = layer(h, rope_cache[positions], attention_masks=None)

        # To make it work with vLLM Engine, turn it into local tensor
        if isinstance(h, DTensor):
            h = h.full_tensor()
        else:
            # Here h is returned as a shard(1) plain tensor because rowwise has use_local_output=True
            h = DTensor.from_local(
                h,
                device_mesh=self.parallel_dims.world_mesh,
                placements=[
                    Shard(1),
                ],
            )
            h = h.full_tensor()

        # Convert to vLLM format: [total_tokens, hidden_size]
        if h.dim() == 3:
            batch_size, seq_len, hidden_size = h.shape
            # When TP is applied, transformer layer's output is Shard(1) because we applied sequence parallel
            # so after flatten, h is Shard(0). And this conversion might be wrong
            h = h.view(batch_size * seq_len, hidden_size)

        return h

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
        sampling_metadata=None,
    ) -> torch.Tensor | None:
        """Compute logits from hidden states."""

        # Here h is returned as a shard(1) plain tensor because rowwise has use_local_output=True
        hidden_states = DTensor.from_local(
            hidden_states,
            device_mesh=self.parallel_dims.world_mesh,
            placements=[
                Replicate(),
            ],
        )
        # When TP is applied, we didn't use SP because DTensor can not handle uneven sharding on sequence-length dimension
        h = self.model.norm(hidden_states)
        logits = self.model.output(h)

        return logits

    def load_weights(self, weights_iter):
        """
        Load weights from HF checkpoint using the provided state dict adapter.

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

        print(f"Loaded weights successfully, loaded {len(loaded_params)} parameters")

        return loaded_params
