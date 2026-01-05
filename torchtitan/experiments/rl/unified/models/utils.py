# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


import functools
import logging
import types
from enum import Enum

import torch
from safetensors.torch import load_file

from torchtitan.experiments.rl.unified.models.attention import VLLMAttention
from torchtitan.experiments.rl.vllm_compat.models.attention import (
    VLLMCompatibleFlashAttention,
)

from torchtitan.experiments.rl.vllm_compat.weights_vllm_compat import (
    torchtitan_to_vllm_compat,
)
from torchtitan.models.qwen3.model.args import Qwen3ModelArgs
from transformers import AutoConfig
from vllm.attention.utils.fa_utils import flash_attn_varlen_func
from vllm.v1.attention.backends.flash_attn import (
    AttentionImpl,
    FlashAttentionImpl,
    FlashAttentionMetadata,
)


logger = logging.getLogger(__name__)


class ModelMode(str, Enum):
    """
    Enum defining which TorchTitan model to use.

    Attributes:
        UNIFIED: Standard TorchTitan model replaced with vLLM attention for unified
            training and inference.
        VLLM_COMPAT: vLLM-compatible TorchTitan model using vLLM's batch invariant kernels,
            ensuring bitwise determinism between training and inference.
        STANDARD: Plain TorchTitan model without any modifications.
    """

    UNIFIED = "unified"
    VLLM_COMPAT = "vllm_compat"
    STANDARD = "standard"


class DifferentiableFlashAttnImpl(FlashAttentionImpl):
    def forward(
        self,
        layer: torch.nn.Module,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: FlashAttentionMetadata,
        output: torch.Tensor | None = None,
        output_scale: torch.Tensor | None = None,
        output_block_scale: torch.Tensor | None = None,
    ):
        class FlashAttnWithBackward(torch.autograd.Function):
            @staticmethod
            def forward(
                ctx,
                q,
                k,
                v,
                attn_metadata,
                scale,
                fa_version,
                alibi_slopes,
                window_size,
                softcap,
                s_aux,
            ):
                assert attn_metadata.num_actual_tokens == q.size(0)
                # TODO Check if this is different from the state returned from FA fwd.
                rng_state = torch.cuda.get_rng_state(device=q.device)

                output, softmax_lse = flash_attn_varlen_func(
                    q=q,
                    k=k,
                    v=v,
                    cu_seqlens_q=attn_metadata.query_start_loc,
                    cu_seqlens_k=attn_metadata.query_start_loc,
                    max_seqlen_q=attn_metadata.max_query_len,
                    max_seqlen_k=attn_metadata.max_seq_len,
                    softmax_scale=scale,
                    causal=attn_metadata.causal,
                    alibi_slopes=alibi_slopes,
                    window_size=window_size,
                    block_table=None,  # TODO Revisit if PagedAttention is really needed.
                    softcap=softcap,
                    fa_version=fa_version,
                    num_splits=attn_metadata.max_num_splits,
                    s_aux=s_aux,
                    return_softmax_lse=True,
                )
                ctx.save_for_backward(q, k, v, output, softmax_lse)
                ctx.scale = scale
                ctx.seq_len = attn_metadata.max_seq_len
                ctx.seq_len_q = attn_metadata.max_query_len
                ctx.cu_seqlens = attn_metadata.query_start_loc

                # Extract the last 16 bytes
                seed_bytes = rng_state[-16:-8]
                offset_bytes = rng_state[-8:]

                # Convert bytes to integers (Little Endian is standard for CUDA/x86)
                ctx.seed = int.from_bytes(seed_bytes.tolist(), byteorder="little")
                ctx.offset = int.from_bytes(offset_bytes.tolist(), byteorder="little")
                return output

            @staticmethod
            def backward(ctx, grad_output):
                q, k, v, output, softmax_lse = ctx.saved_tensors

                # TODO RuntimeError: Input tensor must have contiguous last dimension
                # dq, dk, dv = torch.ops.aten._scaled_dot_product_flash_attention_backward.default(
                #     grad_output,
                #     q,
                #     k,
                #     v,
                #     output,
                #     softmax_lse,
                #     ctx.cu_seqlens,
                #     ctx.cu_seqlens,
                #     ctx.seq_len_q,
                #     ctx.seq_len,
                #     dropout_p=0,
                #     is_causal=True,
                #     philox_seed=torch.tensor(ctx.seed, dtype=torch.int64, device=q.device),
                #     philox_offset=torch.tensor(ctx.offset, dtype=torch.int64, device=q.device),
                #     scale=ctx.scale,
                # )
                # return dq, dk, dv
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

                return grad_q, grad_k, grad_v, None, None, None, None, None, None, None

        assert self.vllm_flash_attn_version == 3, "Only FA3 is supported for now."
        assert output is None
        batch_size, seq_len, num_heads, head_dim = query.shape

        q = query.view(-1, num_heads, head_dim)
        k = key.view(-1, key.shape[2], head_dim)
        v = value.view(-1, value.shape[2], head_dim)

        output_varlen = FlashAttnWithBackward.apply(
            q,
            k,
            v,
            attn_metadata,
            self.scale,
            self.vllm_flash_attn_version,
            self.alibi_slopes,
            self.sliding_window,
            self.logits_soft_cap,
            self.sinks,
        )
        return output_varlen


def replace_with_vllm_attention(model, tp_degree=1, differentiable=False):
    """
    Replace TorchTitan attention with vLLM's Attention.

    Assumes model has .layers dict with .attention.inner_attention structure.
    """
    if not hasattr(model, "layers"):
        raise AttributeError(
            f"Model {type(model).__name__} must have .layers attribute"
        )

    model_args = model.model_args
    for layer_name, layer in model.layers.items():
        if not hasattr(layer, "attention"):
            raise ValueError(f"Layer {layer_name} must have .attention attribute")

        vllm_attn = VLLMAttention(
            hidden_size=model_args.dim,
            num_heads=model_args.n_heads // tp_degree,
            num_kv_heads=model_args.n_heads
            // tp_degree,  # Use n_heads (already replicated)
            head_dim=model_args.head_dim,
            layer_name=layer_name,
            scale=model_args.head_dim**-0.5,
        )

        layer.attention.inner_attention = vllm_attn

    if differentiable:
        from vllm.config import CompilationConfig, VllmConfig
        from vllm.forward_context import set_forward_context
        from vllm.model_executor.layers.batch_invariant import vllm_is_batch_invariant

        static_forward_context = {}
        for layer_name, layer in model.layers.items():
            if not hasattr(layer, "attention"):
                raise ValueError(f"Layer {layer_name} must have .attention attribute")

            vllm_attn = layer.attention.inner_attention.vllm_attn

            attn_impl = vllm_attn.impl
            assert isinstance(attn_impl, AttentionImpl)
            vllm_attn.impl = DifferentiableFlashAttnImpl(
                attn_impl.num_heads,
                attn_impl.head_size,
                attn_impl.scale,
                attn_impl.num_kv_heads,
                attn_impl.alibi_slopes,
                None,  # sliding_window needs to be assigned separately.
                attn_impl.kv_cache_dtype,
                attn_impl.logits_soft_cap,
                attn_impl.attn_type,
                attn_impl.kv_sharing_target_layer_name,
                attn_impl.sinks,
            )
            vllm_attn.impl.sliding_window = attn_impl.sliding_window
            static_forward_context[
                f"model.layers.{layer_name}.attention.inner_attention"
            ] = vllm_attn
            # TODO Investigate why inplace output will break backprop.
            vllm_attn.use_output = False

        original_forward = model.forward

        @functools.wraps(original_forward)
        def forward_with_vllm_context(self, tokens: torch.Tensor, *args, **kwargs):
            batch_size, seq_len = tokens.shape
            cu_seqlens = torch.arange(
                0,
                (batch_size + 1) * seq_len,
                seq_len,
                dtype=torch.int32,
                device=tokens.device,
            )

            seq_lens = torch.full(
                (batch_size,), seq_len, dtype=torch.int32, device=tokens.device
            )

            dummy_metadata = FlashAttentionMetadata(
                num_actual_tokens=batch_size * seq_len,
                query_start_loc=cu_seqlens,
                max_query_len=seq_len,
                seq_lens=seq_lens,
                max_seq_len=seq_len,
                causal=True,
                max_num_splits=1 if vllm_is_batch_invariant() else 0,
                block_table=None,
                slot_mapping=None,
                use_cascade=False,
                common_prefix_len=0,
                cu_prefix_query_lens=None,
                prefix_kv_lens=None,
                suffix_kv_lens=None,
            )
            compilation_config = CompilationConfig()
            compilation_config.static_forward_context = static_forward_context
            with set_forward_context(
                attn_metadata={key: dummy_metadata for key in static_forward_context},
                vllm_config=VllmConfig(
                    compilation_config=compilation_config
                ),  # Dummy vllm config assuming it's not used in trainer.
            ):
                return original_forward(tokens, *args, **kwargs)

        model.forward = types.MethodType(forward_with_vllm_context, model)

    logger.info(
        f"Successfully replaced TorchTitan attention with VLLMAttention "
        f"({len(model.layers)} layers)"
    )


def replace_with_vllm_compatible_flash_attention(model):
    """
    Replace TorchTitan attention with vLLM compatible flash attention.

    Assumes model has .layers dict with .attention.inner_attention structure.
    """
    if not hasattr(model, "layers"):
        raise AttributeError(
            f"Model {type(model).__name__} must have .layers attribute"
        )

    model_args = model.model_args
    for layer_name, layer in model.layers.items():
        if not hasattr(layer, "attention"):
            raise ValueError(f"Layer {layer_name} must have .attention attribute")

        vllm_attn = VLLMCompatibleFlashAttention()

        layer.attention.inner_attention = vllm_attn

    logger.info(
        f"Successfully replaced TorchTitan attention with VLLMCompatibleFlashAttention "
        f"({len(model.layers)} layers)"
    )


def load_model(
    checkpoint_path: str, model_path: str, model_mode: str = ModelMode.VLLM_COMPAT
):
    """
    Load TorchTitan model from checkpoint for trainer.

    Args:
        checkpoint_path: Path to TorchTitan checkpoint
        model_path: Path to HuggingFace model (for config)
        model_mode: Indicates which model to use. Train inferece unified model, batch invariant Torchtitan model,
            or plain Torchtitan model

    Returns:
        model: Loaded TorchTitan model for trainer.
    """
    # Load HuggingFace config
    # TODO: do not depend on transformers.AutoConfig, use qwen_args directly
    hf_config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)

    # Create model args
    model_args = Qwen3ModelArgs(
        dim=hf_config.hidden_size,
        n_layers=hf_config.num_hidden_layers,
        n_heads=hf_config.num_attention_heads,
        n_kv_heads=hf_config.num_key_value_heads,
        vocab_size=hf_config.vocab_size,
        head_dim=getattr(
            hf_config,
            "head_dim",
            hf_config.hidden_size // hf_config.num_attention_heads,
        ),
        hidden_dim=hf_config.intermediate_size,
        norm_eps=hf_config.rms_norm_eps,
        rope_theta=hf_config.rope_theta,
        max_seq_len=getattr(hf_config, "max_position_embeddings", 32768),
        qk_norm=True,
        depth_init=True,
        eos_id=getattr(hf_config, "eos_token_id", 151645),
    )

    # state_dict is in standard TorchTitan format (w1, w2, w3)
    state_dict = load_file(checkpoint_path)

    if model_mode == ModelMode.UNIFIED:
        from torchtitan.models.qwen3 import Qwen3Model

        model = Qwen3Model(model_args)
        # Set global default dtype to bfloat16. This is needed because vLLM's Attention
        # layer uses torch.get_default_dtype() and it doesn't support float32
        torch.set_default_dtype(torch.bfloat16)
        # NOTE: Override attention to vllm compatible attention for backward capability.
        # Only patch to vllm compatible attention for training.
        # replace_with_vllm_compatible_flash_attention(model)
        replace_with_vllm_attention(model, differentiable=True)

        # Load standard TorchTitan format directly
        model.load_state_dict(state_dict, strict=True)
    elif model_mode == ModelMode.VLLM_COMPAT:
        # Create and load model that has bitwise determinism between training and inference
        from torchtitan.experiments.rl.vllm_compat.models.qwen3 import (
            Qwen3VLLMCompatModel,
        )

        model = Qwen3VLLMCompatModel(model_args)
        # Convert to vLLM-compat format (merged gate_up_proj, down_proj)
        vllm_compat_state = torchtitan_to_vllm_compat(state_dict)
        model.load_state_dict(vllm_compat_state, strict=False)
    else:
        # Use standard TorchTitan model
        from torchtitan.models.qwen3 import Qwen3Model

        model = Qwen3Model(model_args)
        # Load standard TorchTitan format directly
        model.load_state_dict(state_dict, strict=False)

    model.to(torch.bfloat16)

    return model
