# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
vLLM-compatible Flash Attention implementation for deterministic RL training.
"""

import itertools

import torch
from torch.distributed.tensor._api import DTensor

from vllm.attention.layer import Attention
from vllm.attention.utils.fa_utils import flash_attn_varlen_func


class VLLMCompatibleFlashAttention(torch.nn.Module):
    """Wrapper around FlashAttention as used by VLLM"""

    def __init__(self) -> None:
        super().__init__()
        self.flash_attn_varlen_func = flash_attn_varlen_func
        from vllm.attention.utils.fa_utils import get_flash_attn_version
        from vllm.model_executor.layers.batch_invariant import vllm_is_batch_invariant

        self.vllm_is_batch_invariant = vllm_is_batch_invariant
        self.fa_version = get_flash_attn_version()

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        *,
        scale: float | None = None,
    ) -> torch.Tensor:
        # Flash Attention varlen expects: (batch, seqlen, nheads, headdim)
        # The input from TorchTitan is always (batch, num_heads, seq_len, head_dim)
        # We need to transpose to (batch, seq_len, num_heads, head_dim)

        # Input is (batch, num_heads, seq_len, head_dim) - need to transpose
        q = q.transpose(1, 2)  # -> (batch, seq_len, num_heads, head_dim)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # Get dimensions
        batch_size, seq_len, num_heads, head_dim = q.shape

        # Convert to varlen format: flatten batch and sequence dimensions
        # (batch, seqlen, nheads, headdim) -> (total_tokens, nheads, headdim)
        q_varlen = q.reshape(-1, num_heads, head_dim)
        k_varlen = k.reshape(-1, k.shape[2], head_dim)
        v_varlen = v.reshape(-1, v.shape[2], head_dim)

        # Create cumulative sequence lengths
        # cu_seqlens: [0, seq_len, 2*seq_len, ..., batch_size*seq_len]
        cu_seqlens = torch.arange(
            0, (batch_size + 1) * seq_len, seq_len, dtype=torch.int32, device=q.device
        )

        # Wrap Flash Attention with manual backward pass
        class FlashAttnWithBackward(torch.autograd.Function):
            @staticmethod
            def forward(
                ctx,
                q,
                k,
                v,
                cu_seqlens,
                seq_len,
                scale,
                num_splits,
                flash_fn,
                fa_version,
            ):
                # Call flash attention for forward (fast)
                output = flash_fn(
                    q,
                    k,
                    v,
                    cu_seqlens_q=cu_seqlens,
                    cu_seqlens_k=cu_seqlens,
                    max_seqlen_q=seq_len,
                    max_seqlen_k=seq_len,
                    softmax_scale=scale,
                    causal=True,
                    num_splits=num_splits,
                    fa_version=fa_version,
                )
                # Save for backward
                ctx.save_for_backward(q, k, v, output)
                ctx.scale = scale
                ctx.seq_len = seq_len
                return output

            @staticmethod
            def backward(ctx, grad_output):
                q, k, v, output = ctx.saved_tensors
                scale = ctx.scale
                seq_len = ctx.seq_len

                # Reshape from varlen back to batch format for attention computation
                # Assume uniform sequence lengths (batch_size = total_tokens / seq_len)
                total_tokens = q.shape[0]
                num_heads = q.shape[1]
                head_dim = q.shape[2]
                batch_size = total_tokens // seq_len

                q_batch = q.reshape(batch_size, seq_len, num_heads, head_dim)
                k_batch = k.reshape(batch_size, seq_len, num_heads, head_dim)
                v_batch = v.reshape(batch_size, seq_len, num_heads, head_dim)
                out_batch = output.reshape(batch_size, seq_len, num_heads, head_dim)
                grad_out_batch = grad_output.reshape(
                    batch_size, seq_len, num_heads, head_dim
                )

                # Transpose to (batch, num_heads, seq_len, head_dim)
                q_t = q_batch.transpose(1, 2)
                k_t = k_batch.transpose(1, 2)
                v_t = v_batch.transpose(1, 2)
                out_t = out_batch.transpose(1, 2)
                grad_out_t = grad_out_batch.transpose(1, 2)

                # Compute attention scores: QK^T
                # q_t: (B, H, N, D), k_t: (B, H, N, D) -> scores: (B, H, N, N)
                scores = torch.matmul(q_t, k_t.transpose(-2, -1)) * scale

                # Apply causal mask
                causal_mask = torch.triu(
                    torch.ones(seq_len, seq_len, device=q.device, dtype=torch.bool),
                    diagonal=1,
                )
                scores = scores.masked_fill(causal_mask, float("-inf"))

                # Softmax
                attn_weights = torch.nn.functional.softmax(
                    scores, dim=-1
                )  # (B, H, N, N)

                # Backward through attention
                # out = attn_weights @ v
                # grad_v = attn_weights^T @ grad_out
                grad_v_t = torch.matmul(attn_weights.transpose(-2, -1), grad_out_t)

                # grad_attn_weights = grad_out @ v^T
                grad_attn_weights = torch.matmul(grad_out_t, v_t.transpose(-2, -1))

                # Backward through softmax
                # d_softmax = attn_weights * (grad_attn_weights - sum(grad_attn_weights * attn_weights))
                sum_term = (grad_attn_weights * attn_weights).sum(dim=-1, keepdim=True)
                grad_scores = attn_weights * (grad_attn_weights - sum_term)

                # Apply causal mask to gradients
                grad_scores = grad_scores.masked_fill(causal_mask, 0.0)

                # Backward through QK^T and scale
                grad_scores = grad_scores * scale

                # grad_q = grad_scores @ K
                grad_q_t = torch.matmul(grad_scores, k_t)

                # grad_k = grad_scores^T @ Q
                grad_k_t = torch.matmul(grad_scores.transpose(-2, -1), q_t)

                # Transpose back and reshape to varlen format
                grad_q = grad_q_t.transpose(1, 2).reshape(
                    total_tokens, num_heads, head_dim
                )
                grad_k = grad_k_t.transpose(1, 2).reshape(
                    total_tokens, num_heads, head_dim
                )
                grad_v = grad_v_t.transpose(1, 2).reshape(
                    total_tokens, num_heads, head_dim
                )

                return grad_q, grad_k, grad_v, None, None, None, None, None, None

        # Call Flash Attention varlen with custom backward
        output_varlen = FlashAttnWithBackward.apply(
            q_varlen,
            k_varlen,
            v_varlen,
            cu_seqlens,
            seq_len,
            scale,
            1 if self.vllm_is_batch_invariant() else 0,
            self.flash_attn_varlen_func,
            self.fa_version,
        )

        # Convert output back to DTensor with original placement
        # from_local is a class method of DTensor, not an instance method
        output = DTensor.from_local(
            output, device_mesh=dtensor_device_mesh, placements=dtensor_placement
        )

        return output
        output = output_varlen.reshape(batch_size, seq_len, num_heads, head_dim)

        # Transpose back to TorchTitan format: (batch, num_heads, seq_len, head_dim)
        output = output.transpose(1, 2)

        return output


class VLLMPagedFlashAttention(torch.nn.Module):
    """
    Wrapper around vLLM's Attention with custom backward pass.

    Forward: Uses vLLM's optimized Attention layer (flash attention kernels)
    Backward: Custom implementation for deterministic gradients

    TODO: This class need to be future refined.
    """

    # Class variable for auto-generating unique layer names (thread-safe)
    _layer_counter = itertools.count()

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
        scale: float | None = None,
        causal: bool = True,
    ) -> None:
        super().__init__()

        self.hidden_size = hidden_size

        # Handle tensor parallelism: adjust num_heads and num_kv_heads for TP
        # NOTE(jianiw): As we use local tensor for this region, we need to manually
        try:
            from vllm.config import get_current_vllm_config
            from vllm.logger import init_logger

            logger = init_logger(__name__)
            vllm_config = get_current_vllm_config()
            self.tp_size = tp_size = vllm_config.parallel_config.tensor_parallel_size

            if tp_size > 1:
                if num_kv_heads % tp_size != 0:
                    # Pad num_kv_heads and num_heads to be divisible by tp_size
                    assert num_heads % num_kv_heads == 0
                    padded_size = tp_size - num_kv_heads % tp_size
                    padded_num_kv_heads = num_kv_heads + padded_size
                    padded_num_heads = (
                        num_heads + padded_size * num_heads // num_kv_heads
                    )
                    assert padded_num_heads % tp_size == 0
                    assert padded_num_kv_heads % tp_size == 0

                    logger.info(
                        f"Padding attention heads for tensor parallelism: "
                        f"{num_heads=}, {padded_num_heads=}, "
                        f"{num_kv_heads=}, {padded_num_kv_heads=}"
                    )

                    num_heads = padded_num_heads // tp_size
                    num_kv_heads = padded_num_kv_heads // tp_size
                else:
                    num_heads //= tp_size
                    num_kv_heads //= tp_size
        except (ImportError, RuntimeError, AttributeError):
            # Not in vLLM context - use original values
            pass

        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.causal = causal

        if scale is None:
            self.scale = head_dim**-0.5
        else:
            self.scale = scale

        # Create vLLM Attention layer
        try:
            from vllm.config import get_current_vllm_config

            config = get_current_vllm_config()
            cache_config = (
                config.cache_config if hasattr(config, "cache_config") else None
            )

            # Generate unique prefix for this attention layer
            # vLLM expects format "layers.X" for layer index extraction
            layer_idx = next(VLLMPagedFlashAttention._layer_counter)
            prefix = f"layers.{layer_idx}"

            self.vllm_attn = Attention(
                num_heads=num_heads,
                head_size=head_dim,
                scale=self.scale,
                num_kv_heads=num_kv_heads,
                cache_config=cache_config,
                quant_config=None,
                prefix=prefix,
            )

        except (ImportError, RuntimeError, AttributeError):
            # Not in vLLM context - will need to set up manually
            self.vllm_attn = None

        # KV cache - will be populated by vLLM during model loading
        self.kv_cache: list[torch.Tensor] | None = None

        # Auto-register for vLLM KV cache if in vLLM context
        self._auto_register_for_kv_cache()

    def _auto_register_for_kv_cache(self):
        """Automatically register this layer for vLLM KV cache allocation.

        This is called during __init__ and will register the layer if we're in
        a vLLM context. If not in vLLM context (e.g., pure PyTorch training),
        this silently does nothing.
        """
        # Initialize layer_name attribute
        self.layer_name: str | None = None

        try:
            from vllm.config import get_current_vllm_config

            config = get_current_vllm_config()
            compilation_config = config.compilation_config

            # Generate unique layer name using class counter
            # Format: "layers.{index}" for compatibility with extract_layer_index()
            layer_name = f"layers.{next(VLLMPagedFlashAttention._layer_counter)}"

            # Register this layer in static forward context
            if layer_name in compilation_config.static_forward_context:
                raise ValueError(f"Duplicate layer name: {layer_name}")
            compilation_config.static_forward_context[layer_name] = self
            self.layer_name = layer_name

        except (ImportError, RuntimeError, AttributeError):
            # Not in vLLM context - this is fine!
            # Layer will work normally for training/inference without vLLM
            pass

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        *,
        scale: float | None = None,
    ) -> torch.Tensor:
        """
        Forward pass using vLLM's Attention layer for inference.

        Args:
            q: Query tensor [batch, num_heads, seq_len, head_dim]
            k: Key tensor [batch, num_kv_heads, seq_len, head_dim]
            v: Value tensor [batch, num_kv_heads, seq_len, head_dim]
            scale: Optional attention scale override (unused, vLLM uses internal scale)

        Returns:
            output: [batch, num_heads, seq_len, head_dim]
        """

        # When TP is applied, q/k/v are DTensors which Shard(1) placement - Colwise shard
        is_dtensor = isinstance(q, DTensor)
        if self.tp_size > 1 and is_dtensor:
            dtensor_placement = q.placements
            dtensor_device_mesh = q.device_mesh
            q = q.to_local()
            k = k.to_local()
            v = v.to_local()

        # Input is (batch, num_heads, seq_len, head_dim)
        # When TP is applied, num_heads are num_local_heads
        batch_size, num_heads, seq_len, head_dim = q.shape

        if self.vllm_attn is None:
            raise RuntimeError(
                "vLLM attention not initialized. This module requires vLLM context."
            )

        # Transpose to (batch, seq_len, num_heads, head_dim) for vLLM
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # NOTE(jianiw): vllm_attention can take input as shape (batch, seq_len, num_heads, head_dim)
        # and handle the reshaping internally
        # Use vLLM's Attention layer for paged attention and
        output_varlen = self.vllm_attn(q, k, v)

        # Reshape back to batch format
        output = output_varlen.view(batch_size, seq_len, num_heads, head_dim)

        # Transpose back to TorchTitan format: (batch, num_heads, seq_len, head_dim)
        output = output.transpose(1, 2)

        # When TP is applied, we need to pack plain tensor back to DTensor
        if self.tp_size > 1 and is_dtensor:
            output = DTensor.from_local(
                output, device_mesh=dtensor_device_mesh, placements=dtensor_placement
            )

        return output
