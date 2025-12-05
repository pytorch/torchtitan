# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
vLLM-compatible Flash Attention implementation for deterministic RL training.

Uses vLLM's flash_attn_varlen_func for forward pass (fast) with custom backward pass.
"""

import itertools

import torch

from vllm.attention.layer import Attention
from vllm.attention.utils.fa_utils import flash_attn_varlen_func, get_flash_attn_version
from vllm.model_executor.layers.batch_invariant import vllm_is_batch_invariant


class VLLMPagedFlashAttention(torch.nn.Module):
    """
    Wrapper around vLLM's Attention with custom backward pass.

    Forward: Uses vLLM's optimized Attention layer (flash attention kernels)
    Backward: Custom implementation for deterministic gradients
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
            tp_size = vllm_config.parallel_config.tensor_parallel_size

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
        Forward with dual-mode behavior:
        - Inference (model.training=False): Use vLLM's Attention layer (KV cache, etc.)
        - Training (model.training=True): Use flash_attn_varlen_func with custom backward
        - vLLM's Attention used flash_attn_varlen_func kernel by default.

        Args:
            q: Query tensor [batch, num_heads, seq_len, head_dim]
            k: Key tensor [batch, num_kv_heads, seq_len, head_dim]
            v: Value tensor [batch, num_kv_heads, seq_len, head_dim]
            scale: Optional attention scale override

        Returns:
            output: [batch, num_heads, seq_len, head_dim]
        """
        # Input is (batch, num_heads, seq_len, head_dim)
        batch_size, num_heads, seq_len, head_dim = q.shape

        # INFERENCE MODE: Use vLLM's Attention layer
        if not self.training and self.vllm_attn is not None:
            # Transpose to (batch, seq_len, num_heads, head_dim) for vLLM
            q = q.transpose(1, 2)
            k = k.transpose(1, 2)
            v = v.transpose(1, 2)

            # # Flatten to (total_tokens, num_heads, head_dim)
            # NOTE(jianiw): vllm_attention can also take input as shape (batch, seq_len, num_heads, head_dim) and do internally

            # q_varlen = q.reshape(-1, num_heads, head_dim)
            # k_varlen = k.reshape(-1, k.shape[2], head_dim)  #  k.shape[2] = num_kv_head
            # v_varlen = v.reshape(-1, v.shape[2], head_dim)

            try:
                # Use vLLM's Attention layer (requires forward context)
                output_varlen = self.vllm_attn(q, k, v)

                # print(f"[jianiw] vllm_attn output is: {output_varlen}")
                # Reshape back to batch format
                output = output_varlen.view(batch_size, seq_len, num_heads, head_dim)

                # Transpose back to TorchTitan format
                output = output.transpose(1, 2)

                return output
            except (AssertionError, RuntimeError) as e:
                # Forward context not available, fall through to training mode
                print(f"Error when calling self.vllm_attn during Inference, {str(e)}")
                raise

        # TRAINING MODE: Use flash_attn_varlen_func with custom backward
        # Transpose to (batch, seq_len, num_heads, head_dim) for vLLM
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # After to_local(), use actual tensor shapes (TP may have sharded heads)
        # Shape: (batch, seq_len, num_heads_local, head_dim)
        _, _, num_heads_local, _ = q.shape
        _, _, num_kv_heads_local, _ = k.shape

        # Convert to varlen format for vLLM: flatten batch and sequence
        # (batch, seq_len, num_heads, head_dim) -> (total_tokens, num_heads, head_dim)
        q_varlen = q.reshape(-1, num_heads_local, head_dim)
        k_varlen = k.reshape(-1, num_kv_heads_local, head_dim)
        v_varlen = v.reshape(-1, num_kv_heads_local, head_dim)

        # Use custom autograd function with flash_attn_varlen_func forward and manual backward
        class VLLMForwardCustomBackward(torch.autograd.Function):
            @staticmethod
            def forward(ctx, q, k, v, scale, batch_size, seq_len, causal, fa_version):
                # Flash Attention only supports fp16 and bf16
                # Store original dtype for conversion back
                original_dtype = q.dtype

                # Convert to bf16 if not already fp16/bf16
                if original_dtype not in [torch.float16, torch.bfloat16]:
                    target_dtype = (
                        torch.bfloat16
                        if torch.cuda.is_bf16_supported()
                        else torch.float16
                    )
                    q = q.to(target_dtype)
                    k = k.to(target_dtype)
                    v = v.to(target_dtype)
                else:
                    target_dtype = original_dtype

                # Use flash_attn_varlen_func directly for fast forward pass
                # This is the SAME kernel vLLM uses internally!
                # TODO(jianiw): Need to double-check
                cu_seqlens_q = torch.arange(
                    0,
                    (batch_size + 1) * seq_len,
                    seq_len,
                    dtype=torch.int32,
                    device=q.device,
                )

                output = flash_attn_varlen_func(
                    q=q,
                    k=k,
                    v=v,
                    cu_seqlens_q=cu_seqlens_q,
                    cu_seqlens_k=cu_seqlens_q,
                    max_seqlen_q=seq_len,
                    max_seqlen_k=seq_len,
                    softmax_scale=scale,
                    causal=causal,
                    num_splits=1 if vllm_is_batch_invariant() else 0,
                    fa_version=fa_version,
                )

                # Convert output back to original dtype if needed
                if original_dtype not in [torch.float16, torch.bfloat16]:
                    output = output.to(original_dtype)

                # Save for backward
                ctx.save_for_backward(q, k, v, output)
                ctx.scale = scale
                ctx.seq_len = seq_len
                ctx.batch_size = batch_size
                ctx.causal = causal
                ctx.original_dtype = original_dtype

                return output

            @staticmethod
            def backward(ctx, grad_output):
                q, k, v, output = ctx.saved_tensors
                scale = ctx.scale
                seq_len = ctx.seq_len
                batch_size = ctx.batch_size
                causal = ctx.causal
                original_dtype = ctx.original_dtype

                # Convert grad_output to match saved tensor dtype
                if grad_output.dtype != q.dtype:
                    grad_output = grad_output.to(q.dtype)

                # Reshape from varlen to batch format
                total_tokens = q.shape[0]
                num_heads = q.shape[1]
                head_dim = q.shape[2]

                q_batch = q.reshape(batch_size, seq_len, num_heads, head_dim)
                k_batch = k.reshape(batch_size, seq_len, num_heads, head_dim)
                v_batch = v.reshape(batch_size, seq_len, num_heads, head_dim)
                grad_out_batch = grad_output.reshape(
                    batch_size, seq_len, num_heads, head_dim
                )

                # Transpose to (batch, num_heads, seq_len, head_dim)
                q_t = q_batch.transpose(1, 2)
                k_t = k_batch.transpose(1, 2)
                v_t = v_batch.transpose(1, 2)
                grad_out_t = grad_out_batch.transpose(1, 2)

                # Compute attention scores: QK^T
                scores = torch.matmul(q_t, k_t.transpose(-2, -1)) * scale

                # Apply causal mask if needed
                if causal:
                    causal_mask = torch.triu(
                        torch.ones(seq_len, seq_len, device=q.device, dtype=torch.bool),
                        diagonal=1,
                    )
                    scores = scores.masked_fill(causal_mask, float("-inf"))

                # Softmax
                attn_weights = torch.nn.functional.softmax(scores, dim=-1)

                # Backward through attention
                # grad_v = attn_weights^T @ grad_out
                grad_v_t = torch.matmul(attn_weights.transpose(-2, -1), grad_out_t)

                # grad_attn_weights = grad_out @ v^T
                grad_attn_weights = torch.matmul(grad_out_t, v_t.transpose(-2, -1))

                # Backward through softmax
                sum_term = (grad_attn_weights * attn_weights).sum(dim=-1, keepdim=True)
                grad_scores = attn_weights * (grad_attn_weights - sum_term)

                # Apply causal mask to gradients
                if causal:
                    grad_scores = grad_scores.masked_fill(causal_mask, 0.0)

                # Backward through scale
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

                # Convert gradients back to original dtype if needed
                if original_dtype not in [torch.float16, torch.bfloat16]:
                    grad_q = grad_q.to(original_dtype)
                    grad_k = grad_k.to(original_dtype)
                    grad_v = grad_v.to(original_dtype)

                return grad_q, grad_k, grad_v, None, None, None, None, None

        # Get flash attention version
        fa_version = get_flash_attn_version()

        # Apply custom autograd function
        output_varlen = VLLMForwardCustomBackward.apply(
            q_varlen,
            k_varlen,
            v_varlen,
            scale or self.scale,
            batch_size,
            seq_len,
            self.causal,
            fa_version,
        )

        # Convert back to batch format
        # (total_tokens, num_heads, head_dim) -> (batch, seq_len, num_heads, head_dim)
        output = output_varlen.reshape(batch_size, seq_len, num_heads, head_dim)

        # Transpose back to TorchTitan format: (batch, num_heads, seq_len, head_dim)
        output = output.transpose(1, 2)

        return output
