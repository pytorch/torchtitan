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
from torchtitan.models.qwen3.model.model import precompute_rope_cache
from torchtitan.protocols.model import BaseModelArgs, ModelProtocol
from torchtitan.protocols.state_dict_adapter import BaseStateDictAdapter
from torchtitan.protocols.train_spec import ParallelizeFunction
from vllm.config import VllmConfig
from vllm.logger import init_logger


logger = init_logger(__name__)

import vllm.utils.torch_utils as _vllm_torch_utils

# ---------------------------------------------------------------------------
# vLLM weak_ref_tensor + DTensor compatibility patches
#
# Piecewise CUDA-graph capture calls weak_ref_tensor() on every subgraph
# output (see vllm/compilation/cuda_graph.py). When TP is active some of
# those outputs are DTensors with Shard(1) placement.  Three things are
# needed to make that work:
#
# 1. A FakeTensor ("Meta") kernel so torch.compile tracing can infer the
#    output shape/dtype.
# 2. A DTensor sharding strategy so DTensor dispatch knows how to propagate
#    the placement.  pointwise (identity) is correct because weak_ref_tensor
#    is semantically a no-op.
# 3. A Python-level guard that returns the DTensor as-is, because the C++
#    _C.weak_ref_tensor op can fail on DTensors in spawned workers
#    ("The specified pointer resides on host memory").
# ---------------------------------------------------------------------------
from torch.distributed.tensor._ops._pointwise_ops import pointwise_strategy
from torch.distributed.tensor._ops.utils import register_op_strategy

torch.library.register_fake(
    "_C::weak_ref_tensor",
    lambda tensor: torch.empty_like(tensor),
)
register_op_strategy(torch.ops._C.weak_ref_tensor.default)(pointwise_strategy)

_orig_weak_ref_tensor = _vllm_torch_utils.weak_ref_tensor


def _patched_weak_ref_tensor(tensor):
    if isinstance(tensor, DTensor):
        return tensor
    return _orig_weak_ref_tensor(tensor)


_vllm_torch_utils.weak_ref_tensor = _patched_weak_ref_tensor


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
                model_args.n_heads % tp_size == 0
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

        # Store compile-friendly TP flag (bool attribute — no graph break)
        self._tp_enabled = self.parallel_dims.tp > 1

        # Pre-extend RoPE cache to vLLM's max model length for compile-friendly forward
        max_model_len = vllm_config.model_config.max_model_len
        if (
            hasattr(self.model, "rope_cache")
            and self.model.rope_cache.shape[0] < max_model_len
        ):
            extended_cache = precompute_rope_cache(
                dim=self.config.head_dim,
                max_seq_len=max_model_len,
                base=self.config.rope_theta,
            )
            self.model.rope_cache = extended_cache.to(
                device=self.model.rope_cache.device,
                dtype=self.model.rope_cache.dtype,
            )
        elif (
            hasattr(self.model, "freqs_cis")
            and self.model.freqs_cis.shape[0] < max_model_len
        ):
            extended_cache = precompute_rope_cache(
                dim=self.config.head_dim,
                max_seq_len=max_model_len,
                base=self.config.rope_theta,
            )
            self.model.freqs_cis = extended_cache.to(
                device=self.model.freqs_cis.device,
                dtype=self.model.freqs_cis.dtype,
            )

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

        # Convert vLLM 1D format to TorchTitan 2D format
        tokens_2d = input_ids.unsqueeze(0)
        positions_2d = positions.unsqueeze(0)

        # Get embeddings
        h = self.model.tok_embeddings(tokens_2d)

        # RoPE cache: always self.model.rope_cache, pre-extended in __init__
        rope_cache = None
        if hasattr(self.model, "rope_cache"):
            rope_cache = self.model.rope_cache
        elif hasattr(self.model, "freqs_cis"):
            rope_cache = self.model.freqs_cis

        # Pass through transformer layers
        for layer in self.model.layers.values():
            h = layer(h, rope_cache, attention_masks=None, positions=positions_2d)

        h = self.model.norm(h)

        # When TP enabled, norm output is DTensor Shard(1); gather for vLLM engine
        if self._tp_enabled:
            h = h.full_tensor()

        # Flatten to vLLM format: [total_tokens, hidden_size]
        h = h.view(-1, h.shape[-1])

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
        if self._tp_enabled:
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
