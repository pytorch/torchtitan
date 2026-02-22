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

import torch
import torch.nn as nn
from torch.distributed._tensor import DTensor, Replicate
from torch.distributed.checkpoint.state_dict import (
    set_model_state_dict,
    StateDictOptions,
)

from torchtitan.experiments.rl.unified.infra.parallelism_utils import (
    create_job_config_from_vllm_config,
    create_parallel_dims_from_vllm_config,
)

from torchtitan.experiments.rl.unified.models.utils import replace_with_vllm_attention
from torchtitan.models.qwen3.model import precompute_rope_cache
from torchtitan.protocols.model import BaseModel
from torchtitan.protocols.model_spec import ParallelizeFunction
from torchtitan.protocols.state_dict_adapter import BaseStateDictAdapter

from vllm.config import VllmConfig
from vllm.logger import init_logger


logger = init_logger(__name__)


class TorchTitanVLLMModelWrapper(nn.Module):
    """
    Generic vLLM-compatible model wrapper for TorchTitan models. Implemented
    required interface required by vLLM Engine.
    Doc: https://docs.vllm.ai/en/latest/contributing/model/basic/
    Reference: https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/models/llama.py

    The wrapper handles:
    - Direct usage of TorchTitan model args (no HF config mapping needed)
    - Attention replacement with vLLM paged attention
    - Parallelism setup and DTensor conversion between torchtitan and vLLM
    - Weight loading from HF checkpoints
    - vLLM forward/compute_logits interface
    """

    is_text_generation_model = True  # Required for vLLM runner validation
    supports_pp = False  # Pipeline parallelism not supported yet
    supports_multimodal = False

    def __init__(
        self,
        *,
        model_config: BaseModel.Config,
        state_dict_adapter: type[BaseStateDictAdapter],
        parallelize_fn: ParallelizeFunction,
        vllm_config: VllmConfig,
        prefix: str = "",
    ):
        super().__init__()

        assert vllm_config is not None, "vllm_config is required"

        # Store components
        self.state_dict_adapter = state_dict_adapter
        self.parallelize_fn = parallelize_fn

        # Use TorchTitan model config directly (no HF config mapping)
        self.config = model_config
        logger.info(f"Creating model with config: {model_config}")
        self.model = model_config.build()

        # Setup RoPE cache extension function if provided
        self.rope_cache_extension_fn = partial(
            precompute_rope_cache,
            dim=self.config.head_dim,
            base=self.config.rope_theta,
        )

        # Create ParallelDims and JobConfig from vLLM config at runtime
        # vLLM config contains the tensor_parallel_size from command-line args
        # and this will be consistent across all worker processes
        self.parallel_dims = create_parallel_dims_from_vllm_config(vllm_config)
        self.parallel_config = create_job_config_from_vllm_config(
            vllm_config=vllm_config,
        )
        # Replace attention with vLLM paged attention
        tp_size = self.parallel_dims.tp
        if tp_size > 1:
            assert (
                self.config.layer.attention.n_heads % tp_size == 0
            ), "Only support when n_heads can be divided by tp_size"

        replace_with_vllm_attention(self.model, tp_degree=tp_size)

        # NOTE: We need to apply parallelize within model.__init__ because vllm
        # doesn't separate model creation and parallelism application and instead
        # requires parallelization to be done inside model constructor.
        self.model = parallelize_fn(
            model=self.model,
            parallel_dims=self.parallel_dims,
            job_config=self.parallel_config,
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

        h = self.model.norm(h)
        # When parallelism is applied, get full tensor before return to vLLM Engine
        # The original placement is Shard(1) (shard on sequence dimension, as it will prepare for sequence parallel in `self.norm`).
        # vLLM's engine expects plain, non-distributed tensors to slice the last token for each request.
        if isinstance(h, DTensor):
            h = h.full_tensor()

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
        # When TP is applied, we return the full tensor (plain tensor) to vLLM engine
        # at the end of TorchTitanVLLMModelWrapper.forward().
        # We need to wrap the input from vLLM engine back to DTensor with Replicate() placement.
        if self.parallel_dims.tp_enabled:
            hidden_states = DTensor.from_local(
                hidden_states,
                device_mesh=self.parallel_dims.get_mesh("tp"),
                placements=[
                    Replicate(),
                ],
            )

        logits = self.model.output(hidden_states)

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
            model_config=self.config,
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
