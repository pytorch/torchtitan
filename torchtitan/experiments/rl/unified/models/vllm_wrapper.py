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

import dataclasses
import types

import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
import torch.nn as nn
import torch.nn.functional as F
from torch.distributed._tensor import DTensor, Replicate, Shard
from torch.distributed.checkpoint.state_dict import (
    set_model_state_dict,
    StateDictOptions,
)

from torchtitan.config import ParallelismConfig
from torchtitan.distributed.parallel_dims import ParallelDims
from torchtitan.experiments.rl.unified.models.attention import (
    replace_with_vllm_attention,
)
from torchtitan.protocols.model_spec import ModelSpec

from vllm.config import VllmConfig
from vllm.logger import init_logger


logger = init_logger(__name__)


def create_torchtitan_config_from_vllm_config(
    vllm_config: VllmConfig,
) -> tuple[ParallelDims, ParallelismConfig]:
    """
    Create ParallelDims and ParallelismConfig from vLLM configuration.

    Maps vLLM parallelism settings to TorchTitan's config objects so that
    TorchTitan's parallelize functions can be called with the correct kwargs.

    This is needed because vLLM doesn't separate model creation and parallelism
    application — it requires parallelization inside the model constructor
    (TorchTitanVLLMModelWrapper.__init__).

    Args:
        vllm_config: vLLM configuration object

    Returns:
        Tuple of (ParallelDims, ParallelismConfig) mapped from vLLM config

    Note:
        vLLM doesn't use FSDP sharding (dp_shard=1) or expert parallelism (ep=1, etp=1)
        in inference. These are set to default values.
    """
    world_size = dist.get_world_size()
    parallel_config = vllm_config.parallel_config

    parallel_dims = ParallelDims(
        dp_replicate=parallel_config.data_parallel_size,
        dp_shard=1,
        cp=parallel_config.decode_context_parallel_size,
        tp=parallel_config.tensor_parallel_size,
        pp=parallel_config.pipeline_parallel_size,
        ep=1,
        etp=1,
        world_size=world_size,
    )

    parallelism = ParallelismConfig(
        data_parallel_replicate_degree=parallel_config.data_parallel_size,
        data_parallel_shard_degree=1,
        context_parallel_degree=parallel_config.decode_context_parallel_size,
        tensor_parallel_degree=parallel_config.tensor_parallel_size,
        pipeline_parallel_degree=parallel_config.pipeline_parallel_size,
        expert_parallel_degree=1,
        expert_tensor_parallel_degree=1,
    )

    logger.info(
        f"Created TorchTitan config from vLLM: "
        f"DP={parallel_dims.dp_replicate}, TP={parallel_dims.tp}, "
        f"CP={parallel_dims.cp}, PP={parallel_dims.pp}"
    )

    return parallel_dims, parallelism


def _to_local(t):
    """Extract local tensor from DTensor, or return plain tensor as-is."""
    return t.to_local() if isinstance(t, DTensor) else t


def _fuse_weights_and_optimize_for_inference(model):
    """Fuse QKV and gate+up weights, pre-cast RoPE cache for inference.

    Must be called AFTER weight loading and parallelization.
    Reduces matmuls per layer from 7 to 4 (matching native vLLM) and
    eliminates per-layer f32→bf16 RoPE cache cast.

    Fused weights are stored as plain local tensors (not DTensors) because
    torch.cat on Shard(0) DTensors produces incorrect global-to-local mapping
    (local data is interleaved [wq_local, wk_local, wv_local] but Shard(0)
    assumes contiguous global rows). The fused forward operates on local
    tensors and wraps back to DTensor only for RowwiseParallel modules.
    """
    from torchtitan.experiments.rl.unified.models.attention import (
        _torchtitan_to_vllm_cos_sin_cache,
    )
    from torchtitan.models.common.attention import (
        apply_rotary_emb_complex,
        AttentionMasksType,
        GQAttention,
    )
    from torchtitan.models.common.feed_forward import FeedForward

    # Pre-cast RoPE cache to bf16 so the per-layer f32→bf16 conversion
    # becomes a no-op inside the compiled graph.
    model.freqs_cis = model.freqs_cis.to(dtype=torch.bfloat16)
    head_dim = model.config.layer.attention.head_dim

    for layer in model.layers.values():
        attn = layer.attention

        # --- Fuse QKV weights as LOCAL tensors ---
        wq_local = _to_local(attn.wq.weight)
        wk_local = _to_local(attn.wk.weight)
        wv_local = _to_local(attn.wv.weight)
        attn._fused_qkv_weight = torch.cat(
            [wq_local, wk_local, wv_local], dim=0
        )
        attn._qkv_split_sizes = [
            wq_local.shape[0],
            wk_local.shape[0],
            wv_local.shape[0],
        ]
        if attn.wq.bias is not None:
            attn._fused_qkv_bias = torch.cat(
                [_to_local(attn.wq.bias), _to_local(attn.wk.bias),
                 _to_local(attn.wv.bias)], dim=0
            )
        else:
            attn._fused_qkv_bias = None

        # Unwrap q_norm/k_norm weights to plain tensors so they work
        # with the plain-tensor activations in the fused forward.
        if attn.q_norm is not None:
            attn.q_norm.weight = nn.Parameter(
                _to_local(attn.q_norm.weight.data), requires_grad=False
            )
        if attn.k_norm is not None:
            attn.k_norm.weight = nn.Parameter(
                _to_local(attn.k_norm.weight.data), requires_grad=False
            )

        # --- Fuse gate+up weights (w1 + w3) as LOCAL tensors ---
        ffn = layer.feed_forward
        w1_local = _to_local(ffn.w1.weight)
        w3_local = _to_local(ffn.w3.weight)
        ffn._fused_w13_weight = torch.cat([w1_local, w3_local], dim=0)
        ffn._w13_split_size = w1_local.shape[0]
        if ffn.w1.bias is not None:
            ffn._fused_w13_bias = torch.cat(
                [_to_local(ffn.w1.bias), _to_local(ffn.w3.bias)], dim=0
            )
        else:
            ffn._fused_w13_bias = None

    # --- Re-patch attention forward to use fused QKV + pre-cast RoPE ---
    # Operates on plain local tensors throughout, wraps output as DTensor
    # Shard(-1) only for the wo RowwiseParallel linear.
    def _fused_attn_forward(
        self,
        x: torch.Tensor,
        rope_cache: torch.Tensor,
        attention_masks: AttentionMasksType | None,
        positions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        bs, seqlen, _ = x.shape

        # Extract device_mesh for DTensor re-wrap, then go local
        device_mesh = x.device_mesh if isinstance(x, DTensor) else None
        x_local = _to_local(x)

        # Fused QKV: 1 matmul instead of 3 (plain local tensors)
        xqkv = F.linear(x_local, self._fused_qkv_weight, self._fused_qkv_bias)
        xq, xk, xv = xqkv.split(self._qkv_split_sizes, dim=-1)

        xq = xq.view(bs, seqlen, -1, self.head_dim)
        xk = xk.view(bs, seqlen, -1, self.head_dim)
        xv = xv.view(bs, seqlen, -1, self.head_dim)

        if self.q_norm is not None:
            xq = self.q_norm(xq)
        if self.k_norm is not None:
            xk = self.k_norm(xk)

        if self.use_rope:
            if self.rope_backend == "cos_sin":
                assert positions is not None, (
                    "vLLM RoPE kernel requires explicit positions"
                )
                rope_local = _to_local(rope_cache)
                cos_sin_cache = _torchtitan_to_vllm_cos_sin_cache(
                    rope_local, self.head_dim
                )

                num_tokens = bs * seqlen
                flat_q = xq.reshape(num_tokens, -1)
                flat_k = xk.reshape(num_tokens, -1)
                flat_pos = _to_local(positions).reshape(num_tokens)

                flat_q, flat_k = torch.ops.vllm.rotary_embedding_return_tensors(
                    flat_pos, flat_q, flat_k,
                    self.head_dim, cos_sin_cache, True,
                )

                xq = flat_q.view(bs, seqlen, -1, self.head_dim)
                xk = flat_k.view(bs, seqlen, -1, self.head_dim)
            else:
                xq, xk = apply_rotary_emb_complex(
                    xq, xk, freqs_cis=_to_local(rope_cache),
                    positions=_to_local(positions),
                )

        # VLLMAttention: plain tensors → skips DTensor unwrap,
        # uses local shapes directly, returns plain tensor
        output = self.inner_attention(xq, xk, xv)

        # Wrap as Shard(-1) DTensor for wo (RowwiseParallel)
        if device_mesh is not None:
            output = DTensor.from_local(
                output, device_mesh=device_mesh, placements=[Shard(-1)]
            )

        return self.wo(output)

    for layer in model.layers.values():
        if hasattr(layer, "attention") and isinstance(layer.attention, GQAttention):
            layer.attention.forward = types.MethodType(
                _fused_attn_forward, layer.attention
            )

    # --- Patch FFN forward to use fused gate+up ---
    # Same approach: local tensors, wrap for w2 RowwiseParallel.
    def _fused_ffn_forward(self, x: torch.Tensor) -> torch.Tensor:
        device_mesh = x.device_mesh if isinstance(x, DTensor) else None
        x_local = _to_local(x)

        w13_out = F.linear(x_local, self._fused_w13_weight, self._fused_w13_bias)
        w1_out, w3_out = w13_out.split(
            [self._w13_split_size, self._w13_split_size], dim=-1
        )
        intermediate = F.silu(w1_out) * w3_out

        # Wrap as Shard(-1) DTensor for w2 (RowwiseParallel)
        if device_mesh is not None:
            intermediate = DTensor.from_local(
                intermediate, device_mesh=device_mesh, placements=[Shard(-1)]
            )

        return self.w2(intermediate)

    for layer in model.layers.values():
        ffn = layer.feed_forward
        if isinstance(ffn, FeedForward):
            ffn.forward = types.MethodType(_fused_ffn_forward, ffn)

    logger.info(
        "Fused QKV and gate+up weights (7→4 matmuls/layer), "
        "pre-cast RoPE cache to bf16"
    )


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
        model_spec: ModelSpec,
        vllm_config: VllmConfig,
        prefix: str = "",
    ):
        super().__init__()

        assert vllm_config is not None, "vllm_config is required"

        # Store components from model_spec
        self.state_dict_adapter = model_spec.state_dict_adapter
        self.parallelize_fn = model_spec.parallelize_fn

        # Use TorchTitan model config directly (no HF config mapping)
        self.config = model_spec.model
        logger.debug(f"Creating model with config: {self.config}")

        # TODO: Check if it's possible to apply meta init
        self.model = self.config.build()

        # RoPE config from model for cache extension
        self.rope_config = self.config.rope

        # Create ParallelDims and configs from vLLM config at runtime.
        # vLLM config contains the tensor_parallel_size from command-line args
        # and this will be consistent across all worker processes.
        self.parallel_dims, parallelism = create_torchtitan_config_from_vllm_config(
            vllm_config
        )

        # Replace attention with vLLM compatible flash attention
        # TODO: Use config system to replace with vllm Attention
        replace_with_vllm_attention(self.model, tp_degree=self.parallel_dims.tp)

        # NOTE: We need to apply parallelize within model.__init__ because vllm
        # doesn't separate model creation and parallelism application and instead
        # requires parallelization to be done inside model constructor.
        self.model = self.parallelize_fn(
            model=self.model,
            parallel_dims=self.parallel_dims,
            parallelism=parallelism,
            has_position_id=True,  # vLLM always passes positions explicitly
        )

        # Initial load model weights from HuggingFace checkpoint path
        self._initial_load_weights(checkpoint_path=vllm_config.model_config.model)

        # Pre-extend RoPE cache to max_model_len so we never need to extend
        # dynamically (which requires .item() and breaks the compiled graph).
        max_model_len = vllm_config.model_config.max_model_len
        self.model.freqs_cis = self._extend_rope_cache_if_needed(
            self.model.freqs_cis, max_model_len
        )

        # Fuse QKV/gate+up weights and pre-cast RoPE cache for inference perf
        _fuse_weights_and_optimize_for_inference(self.model)

        # Pre-compile functions once (avoids re-wrapping each forward call)
        self._compiled_forward_body = torch.compile(
            self._forward_body, backend="inductor", fullgraph=True
        )
        self._compiled_compute_logits = torch.compile(
            self._compute_logits, backend="inductor", fullgraph=True
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

        # Build a new RoPE module with extended max_seq_len
        extended_rope_config = dataclasses.replace(
            self.rope_config, max_seq_len=required_len
        )
        extended_rope = extended_rope_config.build()
        extended_cache = extended_rope.cache.to(device=device, dtype=dtype)

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
        """vLLM required API.
        Convert input token IDs to embeddings."""
        return self.model.tok_embeddings(input_ids)

    def get_input_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        """vLLM required API.
        Convert input token IDs to embeddings (deprecated vLLM interface)."""
        return self.embed_input_ids(input_ids)

    def _forward_body(self, tokens, rope_cache, positions):
        """Embed + all transformer layers + final norm in one compiled graph."""
        torch._check(tokens.shape[1] >= 2)
        h = self.model.tok_embeddings(tokens)
        for layer in self.model.layers.values():
            h = layer(h, rope_cache, attention_masks=None, positions=positions)
        h = self.model.norm(h)
        return h

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        positions: torch.Tensor | None = None,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor:
        """
        vLLM required API.
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
        positions = positions.unsqueeze(0)
        torch._dynamo.decorators.mark_unbacked(tokens_2d, 1)
        torch._dynamo.decorators.mark_unbacked(positions, 1)

        # RoPE cache is pre-extended in __init__ to max_model_len
        rope_cache = self.model.freqs_cis

        # Single compiled region: embed + layers + norm
        h = self._compiled_forward_body(tokens_2d, rope_cache, positions)

        # When parallelism is applied, get full tensor before return to vLLM Engine
        if isinstance(h, DTensor):
            h = h.full_tensor()

        # Convert to vLLM format: [total_tokens, hidden_size]
        if h.dim() == 3:
            batch_size, seq_len, hidden_size = h.shape
            h = h.view(batch_size * seq_len, hidden_size)

        return h

    def _compute_logits(self, hidden_states):
        torch._check(hidden_states.shape[0] >= 2)
        h = self.model.norm(hidden_states)
        return self.model.output(h)

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
        sampling_metadata=None,
    ) -> torch.Tensor | None:
        """vLLM required API.
        Compute logits from hidden states."""

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

        torch._dynamo.decorators.mark_unbacked(hidden_states, 0)
        logits = self._compiled_compute_logits(hidden_states)
        # Unwrap DTensor to plain tensor before returning to vLLM
        if isinstance(logits, DTensor):
            logits = logits.full_tensor()
        return logits

    def load_weights_from_state_dict(self, state_dict):
        """
        Load model weights from a state dict.

        Expects DTensor-wrapped tensors matching the model's placements.
        The caller is responsible for reconstructing DTensors from plain
        local tensors before calling this method.
        """
        set_model_state_dict(
            model=self.model,
            model_state_dict=state_dict,
            options=StateDictOptions(strict=False),
        )

        return state_dict.keys()

    def _initial_load_weights(self, checkpoint_path):
        """
        Helper function to load torchtitan model weights from HF checkpoint when initialize this model.

        Args:
            checkpoint_path: Path to the HuggingFace checkpoint directory or HF hub model ID
        """
        # Resolve HF hub model IDs (e.g. "Qwen/Qwen3-1.7B") to local cache paths.
        # HuggingFaceStorageReader requires a local filesystem path.
        import os

        if not os.path.exists(checkpoint_path):
            from huggingface_hub import snapshot_download

            checkpoint_path = snapshot_download(
                checkpoint_path, local_files_only=True
            )

        # Create adapter instance
        adapter = self.state_dict_adapter(
            model_config=self.config,
            hf_assets_path=None,
        )

        # Get HF storage reader from adapter
        storage_reader = adapter.get_hf_storage_reader(checkpoint_path)

        # Load HF state dict using DCP
        hf_state_dict = adapter.to_hf(self.model.state_dict())
        dcp.load(hf_state_dict, storage_reader=storage_reader)

        # Convert HF state dict to TorchTitan format
        torchtitan_state_dict = adapter.from_hf(hf_state_dict)

        model_state_dict = {k: v for k, v in self.model.state_dict().items()}

        # Convert to DTensor if target is DTensor (when the target model is sharded)
        # This only happens when initial loading from HF full state dict
        for name, tensor in torchtitan_state_dict.items():
            if name in model_state_dict and isinstance(model_state_dict[name], DTensor):
                if isinstance(tensor, DTensor):
                    continue
                target_dtensor = model_state_dict[name]
                device_mesh = target_dtensor.device_mesh
                torchtitan_state_dict[name] = DTensor.from_local(
                    tensor.to(device_mesh.device_type),
                    device_mesh=device_mesh,
                    placements=[Replicate()],
                )

        return self.load_weights_from_state_dict(torchtitan_state_dict)

    def load_weights(self, weights_iter):
        """
        vLLM required API.

        This is a no-op method since model weights are already loaded during initialization.
        Returns the names of all parameters that have been loaded so vLLM's safety check passes.

        Args:
            weights_iter: Iterator of (name, tensor) pairs from HF checkpoint

        Returns:
            Set of loaded parameter names
        """

        loaded_param_names = set()
        for name, _ in self.model.named_parameters():
            loaded_param_names.add("model." + name)

        logger.info(
            f"Weights already loaded during model initialization. \
            Returning {len(loaded_param_names)} loaded parameter names to satisfy vLLM safety check."
        )

        # Return the names of all loaded parameters so vLLM knows they were handled
        return loaded_param_names
