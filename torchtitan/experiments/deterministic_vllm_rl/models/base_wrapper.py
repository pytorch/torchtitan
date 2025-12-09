# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Generic vLLM wrapper for TorchTitan models.

This module provides TorchTitanVLLMWrapper, a generic base class that makes
any TorchTitan model compatible with vLLM by:
1. Accepting 4 pluggable model-specific components
2. Replacing attention with vLLM's paged attention
3. Setting up parallelization (TP/EP)
4. Loading weights from HuggingFace checkpoints
"""

import functools
from typing import Callable

import torch
import torch.nn as nn
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import DTensor

from vllm.config import VllmConfig
from vllm.logger import init_logger

from torchtitan.experiments.deterministic_vllm_rl.models.attention import (
    VLLMPagedFlashAttention,
)
from torchtitan.protocols.model import BaseModelArgs
from torchtitan.protocols.state_dict_adapter import BaseStateDictAdapter
from torchtitan.tools.utils import device_type


logger = init_logger(__name__)


class TorchTitanVLLMWrapper(nn.Module):
    """
    Generic vLLM-compatible wrapper for TorchTitan models.

    This base class integrates any TorchTitan model with vLLM by accepting
    4 pluggable model-specific components:

    1. model_cls: The TorchTitan model class (e.g., Qwen3Model)
    2. model_args_cls: The model args class (e.g., Qwen3ModelArgs)
    3. state_dict_adapter: State dict adapter for loading HF weights
    4. parallelize_fn: Function to apply tensor parallelism

    The wrapper handles:
    - HF config → TorchTitan model args mapping
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
        model_cls: type,
        model_args_cls: BaseModelArgs,
        state_dict_adapter: BaseStateDictAdapter,
        parallelize_fn: Callable,
        rope_cache_compute_fn: Callable | None = None,
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
        logger.info(f"Mapping HF config to {model_args_cls.__name__}")
        model_args = self._map_hf_config_to_model_args(hf_config, model_args_cls)

        # Create TorchTitan model
        logger.info(f"Creating {model_cls.__name__} with config: {model_args}")
        self.model = model_cls(model_args)
        self.config = model_args

        # NOTE: Here's assumptions of rope_cache_compute_fn function signature
        self.rope_cache_extension_fn = functools.partial(
            rope_cache_compute_fn, dim=self.config.head_dim, base=self.config.rope_theta
        )

        # Replace attention with vLLM paged attention
        self._replace_with_vllm_paged_attention(model_args)

        # Setup parallelization
        (
            dp_size,
            mp_size,
            cp_size,
            pp_size,
            ep_size,
            etp_size,
        ) = self._process_parallelism_settings(vllm_config)

        if mp_size > 1 or ep_size > 1:
            self._build_device_mesh_and_parallelize(
                dp_size, mp_size, cp_size, pp_size, ep_size, etp_size
            )

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

    def _process_parallelism_settings(self, vllm_config: VllmConfig):
        """Parse parallel config from vLLM config."""
        world_size = (
            vllm_config.parallel_config.data_parallel_size
            * vllm_config.parallel_config.tensor_parallel_size
        )
        ep_size = (
            world_size if vllm_config.parallel_config.enable_expert_parallel else 1
        )
        etp_size = (
            1 if vllm_config.parallel_config.enable_expert_parallel else world_size
        )
        dp_size = vllm_config.parallel_config.data_parallel_size
        mp_size = vllm_config.parallel_config.tensor_parallel_size
        cp_size = vllm_config.parallel_config.decode_context_parallel_size
        pp_size = vllm_config.parallel_config.pipeline_parallel_size
        self.pp_size = pp_size

        return dp_size, mp_size, cp_size, pp_size, ep_size, etp_size

    def _build_device_mesh_and_parallelize(
        self,
        dp_size: int,
        mp_size: int,
        cp_size: int,
        pp_size: int,
        ep_size: int,
        etp_size: int,
    ):
        """
        Build device mesh and apply parallelization using the provided parallelize_fn.
        """
        import torch.distributed as dist

        world_size = dist.get_world_size()
        dp_replicate = dp_size
        dp_shard = 1

        assert dp_replicate * dp_shard * cp_size * mp_size * pp_size == world_size, (
            f"Invalid parallel dims: dp_replicate({dp_replicate}) * dp_shard({dp_shard}) * "
            f"cp({cp_size}) * tp({mp_size}) * pp({pp_size}) != WORLD_SIZE({world_size})"
        )

        # Build device mesh
        dims = []
        names = []
        for d, name in zip(
            [pp_size, dp_replicate, dp_shard, cp_size, mp_size],
            ["pp", "dp_replicate", "dp_shard", "cp", "tp"],
        ):
            if d > 1:
                dims.append(d)
                names.append(name)

        logger.info(f"Building {len(dims)}-D device mesh with {names}, {dims}")
        world_mesh = init_device_mesh(device_type, dims, mesh_dim_names=names)

        # Apply tensor parallelism using provided function
        if mp_size > 1:
            tp_mesh = world_mesh["tp"]
            self.parallelize_fn(
                model=self.model,
                tp_mesh=tp_mesh,
                loss_parallel=False,
                enable_float8_tensorwise_tp=False,
                enable_async_tp=False,
            )
            logger.info(f"Applied Tensor Parallelism with TP={mp_size}")

        self.world_mesh = world_mesh

    def _extend_rope_cache_if_needed(
        self, rope_cache: torch.Tensor, max_position: int
    ) -> torch.Tensor:
        """
        Extend RoPE cache if needed during vLLM profiling.

        Uses the rope_cache_extension_fn provided during initialization if available.

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
        rope_cache = rope_cache[positions]

        # Pass through transformer layers
        for layer in self.model.layers.values():
            h = layer(h, rope_cache, attention_masks=None)

        # Convert to vLLM format: [total_tokens, hidden_size]
        if h.dim() == 3:
            batch_size, seq_len, hidden_size = h.shape
            h = h.view(batch_size * seq_len, hidden_size)

        # Convert DTensor to regular tensor
        if isinstance(h, DTensor):
            h = h.full_tensor()

        return h

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
        sampling_metadata=None,
    ) -> torch.Tensor | None:
        """Compute logits from hidden states."""
        h = self.model.norm(hidden_states)
        logits = self.model.output(h)

        if isinstance(logits, DTensor):
            logits = logits.full_tensor()

        return logits

    def load_weights(self, weights_iter):
        """
        Load weights from HF checkpoint using the provided state dict adapter.

        Args:
            weights_iter: Iterator of (name, tensor) pairs from HF checkpoint

        Returns:
            Set of loaded parameter names
        """
        from torch.distributed._tensor import DTensor, Replicate
        from torch.distributed.checkpoint.state_dict import (
            set_model_state_dict,
            StateDictOptions,
        )

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
