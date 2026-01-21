# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import math

import torch
from torch import nn
from torch.nn.attention.flex_attention import and_masks, BlockMask

from torchtitan.components.peft.lora import lora_or_linear, per_layer_config
from torchtitan.components.tokenizer import BaseTokenizer
from torchtitan.config.job_config import PEFT
from torchtitan.models.attention import (
    create_attention_mask,
    FlexAttentionWrapper,
    get_causal_mask_mod,
    get_document_mask_mod,
    get_block_causal_mask_mod_by_seq_lens,
    ScaledDotProductAttentionWrapper,
)
from torchtitan.models.moe import build_moe, fast_init_trunc_normal_, fast_init_normal_, FeedForward
from torchtitan.protocols.model import AttentionMasksType
from torchtitan.protocols.train_spec import ModelProtocol

from .args import DeepSeekV3ModelArgs


# Adapted from https://github.com/DeepSeek-ai/DeepSeek-V3/blob/main/inference/model.py#L294
def precompute_freqs_cis(args: DeepSeekV3ModelArgs) -> torch.Tensor:
    """
    Precomputes frequency-based complex exponential values for rotary positional embeddings.

    Args:
        args (DeepSeekV3ModelArgs): Model arguments containing positional embedding parameters.

    Returns:
        torch.Tensor: Precomputed complex exponential values for positional embeddings.
    """
    dim = args.qk_rope_head_dim
    seqlen = args.max_seq_len
    beta_fast = args.beta_fast
    beta_slow = args.beta_slow
    base = args.rope_theta
    factor = args.rope_factor

    def find_correction_dim(
        num_rotations: float, dim: int, base: float, max_seq_len: int
    ) -> float:
        """
        Computes the correction dimension for a given number of rotations in the rotary positional embedding.

        Args:
            num_rotations (float): Number of rotations to compute the correction for.
            dim (int): Dimensionality of the embedding space.
            base (float): Base value for the exponential computation.
            max_seq_len (int): Maximum sequence length.

        Returns:
            float: The correction dimension based on the input parameters.
        """
        return (
            dim
            * math.log(max_seq_len / (num_rotations * 2 * math.pi))
            / (2 * math.log(base))
        )

    def find_correction_range(
        low_rot: float, high_rot: float, dim: int, base: float, max_seq_len: int
    ) -> tuple[int, int]:
        """
        Computes the range of correction dimensions for rotary positional embeddings.

        Args:
            low_rot (float): Lower bound for the number of rotations.
            high_rot (float): Upper bound for the number of rotations.
            dim (int): Dimensionality of the embedding space.
            base (float): Base value for the exponential computation.
            max_seq_len (int): Maximum sequence length.

        Returns:
            tuple[int, int]: The range of correction dimensions (low, high), clamped to valid indices.
        """
        low = math.floor(find_correction_dim(low_rot, dim, base, max_seq_len))
        high = math.ceil(find_correction_dim(high_rot, dim, base, max_seq_len))
        return max(low, 0), min(high, dim - 1)

    def linear_ramp_factor(min: float, max: float, dim: int) -> torch.Tensor:
        """
        Computes a linear ramp function used to smooth values between a minimum and maximum range.

        Args:
            min (float): Minimum value for the ramp function.
            max (float): Maximum value for the ramp function.
            dim (int): Dimensionality of the ramp tensor.

        Returns:
            torch.Tensor: A tensor of shape (dim,) with values linearly interpolated between 0 and 1,
                clamped to the range [0, 1].
        """
        if min == max:
            max += 0.001
        linear_func = (torch.arange(dim, dtype=torch.float32) - min) / (max - min)
        ramp_func = torch.clamp(linear_func, 0, 1)
        return ramp_func

    # Basic RoPE frequency calculation
    freqs = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))

    # YaRN scaling for extended context
    if factor != 1.0:
        low, high = find_correction_range(
            beta_fast, beta_slow, dim, base, args.original_seq_len
        )
        smooth = 1 - linear_ramp_factor(low, high, dim // 2)
        freqs = freqs / factor * (1 - smooth) + freqs * smooth

    # Create position indices
    t = torch.arange(seqlen)

    # Outer product: [positions] × [frequencies]
    freqs = torch.outer(t, freqs)

    # Convert to complex exponentials: e^(i*freq*pos)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
    return freqs_cis


def reshape_for_broadcast(
    freqs_cis: torch.Tensor, x: torch.Tensor, positions: torch.Tensor | None = None
) -> torch.Tensor:
    """
    Reshape frequency tensor for broadcasting it with another tensor.

    This function reshapes the frequency tensor to have the same shape as the target tensor 'x'
    for the purpose of broadcasting the frequency tensor during element-wise operations.

    The input freqs_cis tensor is assumed to be of shape (max_seqlen, dim // 2),
    and the first seqlen elements will be sliced, but dim must match x.

    Args:
        freqs_cis (torch.Tensor): Frequency tensor to be reshaped.
        x (torch.Tensor): Target tensor for broadcasting compatibility.
        positions (torch.Tensor | None): Position indices used to access/shuffle RoPE cache.
            Shape is (1, seqlen) or (bz, seqlen). Defaults to None.

    Returns:
        torch.Tensor: Reshaped frequency tensor.
    """
    ndim = x.ndim
    assert ndim > 1
    seqlen = x.shape[1]
    if positions is None:
        freqs_cis = freqs_cis[0:seqlen]
        assert freqs_cis.shape == (seqlen, x.shape[-1])
        shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
        return freqs_cis.view(*shape)
    elif positions.size(0) == 1:
        assert positions.shape == (1, seqlen)
        freqs_cis = freqs_cis[positions.squeeze(0)]
        assert freqs_cis.shape == (seqlen, x.shape[-1])
        shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
        return freqs_cis.view(*shape)
    else:
        assert positions.shape == (x.shape[0], seqlen)
        freqs_cis_expanded = freqs_cis[None, :, None, :].expand(x.shape[0], -1, -1, -1)
        freqs_cis = torch.gather(
            freqs_cis_expanded,
            dim=1,
            index=positions.view(x.shape[0], seqlen, 1, 1).expand(
                x.shape[0], seqlen, 1, freqs_cis_expanded.shape[-1]
            ),
        )
        return freqs_cis


def apply_rotary_emb(
    x: torch.Tensor, freqs_cis: torch.Tensor, positions: torch.Tensor | None = None
) -> torch.Tensor:
    """
    Applies rotary positional embeddings to the input tensor.

    Args:
        x (torch.Tensor): Input tensor with positional embeddings to be applied.
        freqs_cis (torch.Tensor): Precomputed complex exponential values for positional embeddings.
        positions (torch.Tensor | None): Position indices used to access/shuffle RoPE cache. Defaults to None.

    Returns:
        torch.Tensor: Tensor with rotary embeddings applied.
    """
    dtype = x.dtype
    x = torch.view_as_complex(x.float().view(*x.shape[:-1], -1, 2))
    freqs_cis = reshape_for_broadcast(freqs_cis, x, positions)
    y = torch.view_as_real(x * freqs_cis).flatten(3)
    return y.to(dtype)


class Attention(nn.Module):
    """
    Multi-head attention (MLA) module.

    Note: DeepSeek-V3 uses Multi-head Latent Attention (MLA) which already employs
    low-rank approximation for Q/KV projections. When using LoRA fine-tuning:
    - The built-in low-rank projections (wq_a/wq_b, wkv_a/wkv_b) are NOT wrapped with
      additional LoRA adapters since they already use efficient low-rank representation.
    - LoRA is only applied to the output projection (wo) to avoid redundancy.
    """

    def __init__(self, model_args: DeepSeekV3ModelArgs, peft_config: PEFT):
        super().__init__()
        self.dim = model_args.dim
        self.n_heads = model_args.n_heads
        self.q_lora_rank = model_args.q_lora_rank
        self.kv_lora_rank = model_args.kv_lora_rank
        self.qk_nope_head_dim = model_args.qk_nope_head_dim
        self.qk_rope_head_dim = model_args.qk_rope_head_dim
        self.qk_head_dim = model_args.qk_nope_head_dim + model_args.qk_rope_head_dim
        self.v_head_dim = model_args.v_head_dim

        # NOTE: DeepSeek's built-in low-rank projections are kept as nn.Linear
        # (not wrapped with LoRA) since they already serve as efficient low-rank
        # representations. Adding LoRA on top would be redundant.
        if self.q_lora_rank == 0:
            # When q_lora_rank == 0, use standard projection - apply LoRA here
            self.wq = lora_or_linear(
                self.dim,
                self.n_heads * self.qk_head_dim,
                bias=False,
                peft_config=peft_config,
            )
        else:
            # Built-in low-rank path, don't add LoRA adapters
            self.wq_a = nn.Linear(self.dim, self.q_lora_rank, bias=False)
            self.q_norm = nn.RMSNorm(self.q_lora_rank, eps=model_args.norm_eps)
            self.wq_b = nn.Linear(
                self.q_lora_rank, self.n_heads * self.qk_head_dim, bias=False
            )
            if peft_config.enable_peft and not peft_config.lora_train_norm:
                self.q_norm.weight.requires_grad = False
        self.wkv_a = nn.Linear(
            self.dim, self.kv_lora_rank + self.qk_rope_head_dim, bias=False
        )
        self.kv_norm = nn.RMSNorm(self.kv_lora_rank, eps=model_args.norm_eps)
        self.wkv_b = nn.Linear(
            self.kv_lora_rank,
            self.n_heads * (self.qk_nope_head_dim + self.v_head_dim),
            bias=False,
        )
        if peft_config.enable_peft:
            if not peft_config.lora_train_norm:
                self.kv_norm.weight.requires_grad = False

        # Output projection - apply LoRA here
        self.wo = lora_or_linear(
            self.n_heads * self.v_head_dim,
            self.dim,
            bias=False,
            peft_config=peft_config,
        )
        self.softmax_scale = self.qk_head_dim**-0.5

        # Apply mscale for YaRN using the ratio formula from HuggingFace:
        # effective_mscale = yarn_get_mscale(factor, mscale) / yarn_get_mscale(factor, mscale_all_dim)
        # When mscale == mscale_all_dim (e.g., both 1.0 for Kimi K2), they cancel out → effective_mscale = 1.0
        if model_args.rope_factor != 1.0:

            def yarn_get_mscale(scale: float, mscale: float) -> float:
                return 0.1 * mscale * math.log(scale) + 1.0

            mscale_numerator = yarn_get_mscale(
                model_args.rope_factor, model_args.mscale
            )
            mscale_denominator = yarn_get_mscale(
                model_args.rope_factor, model_args.mscale_all_dim
            )
            effective_mscale = mscale_numerator / mscale_denominator
            self.softmax_scale = (
                self.softmax_scale * effective_mscale * effective_mscale
            )

        self.attn_type = model_args.attn_type
        match self.attn_type:
            case "flex":
                self.inner_attention = FlexAttentionWrapper()
            case "sdpa":
                # pyrefly: ignore [bad-assignment]
                self.inner_attention = ScaledDotProductAttentionWrapper()
            case "varlen":
                raise ValueError("Varlen attention is not supported with Deepseek V3.")
            case _:
                raise ValueError(f"Unknown attention type: {self.attn_type}")

    def forward(
        self,
        x: torch.Tensor,
        freqs_cis: torch.Tensor,
        attention_masks: AttentionMasksType | None,
        positions: torch.Tensor | None = None,
    ):
        """
        Forward pass for the Multi-Head Latent Attention (MLA) Layer.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, dim).
            freqs_cis (torch.Tensor): Precomputed complex exponential values for rotary embeddings.
            attention_masks (AttentionMasksType | None): Masks used when calculating attention scores.
            positions (torch.Tensor | None): Position indices used to access/shuffle RoPE cache. Defaults to None.

        Returns:
            torch.Tensor: Output tensor with the same shape as the input.
        """
        bsz, seqlen, _ = x.size()

        # Query projection
        if self.q_lora_rank == 0:
            q = self.wq(x)  # (bsz, seqlen, n_heads * qk_head_dim)
        else:
            q = self.wq_a(x)
            q = self.wq_b(self.q_norm(q))
        # Use -1 instead of `n_heads` (or `n_kv_heads`) to infer the actual
        # local heads from sizes of q and kv as TP may have sharded them after
        # the above linear ops.
        q = q.view(bsz, seqlen, -1, self.qk_head_dim)
        q_nope, q_pe = torch.split(
            q, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1
        )
        q_pe = apply_rotary_emb(q_pe, freqs_cis, positions)
        q = torch.cat([q_nope, q_pe], dim=-1)  # (bsz, seqlen, n_heads, qk_head_dim)

        # Key-value projection
        kv = self.wkv_a(x)  # (bsz, seqlen, kv_lora_rank + qk_rope_head_dim)
        kv, k_pe = torch.split(kv, [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)

        k_pe = apply_rotary_emb(
            k_pe.unsqueeze(2), freqs_cis, positions
        )  # (bsz, seqlen, 1, qk_rope_head_dim)

        kv = self.wkv_b(
            self.kv_norm(kv)
        )  # (bsz, seqlen, n_heads * (qk_nope_head_dim + v_head_dim))
        kv = kv.view(bsz, seqlen, -1, self.qk_nope_head_dim + self.v_head_dim)
        k_nope, v = torch.split(kv, [self.qk_nope_head_dim, self.v_head_dim], dim=-1)
        # TODO: self.n_heads is global count; with TP this should this be k_nope.size(2) for local heads?
        k = torch.cat(
            [k_nope, k_pe.expand(-1, -1, self.n_heads, -1)], dim=-1
        )  # (bsz, seqlen, n_heads, qk_head_dim)

        q = q.transpose(1, 2)  # (bsz, n_heads, seqlen, qk_head_dim)
        k = k.transpose(1, 2)  # (bsz, n_heads, seqlen, qk_head_dim)
        v = v.transpose(1, 2)  # (bsz, n_heads, seqlen, v_head_dim)

        match self.attn_type:
            case "flex":
                assert isinstance(attention_masks, BlockMask)
                output = self.inner_attention(
                    q, k, v, block_mask=attention_masks, scale=self.softmax_scale
                )
            case _:
                assert attention_masks is None
                output = self.inner_attention(q, k, v, scale=self.softmax_scale)

        # Reshape and project output
        output = output.transpose(
            1, 2
        ).contiguous()  # (bsz, seqlen, n_heads, v_head_dim)
        output = output.view(bsz, seqlen, -1)  # (bsz, seqlen, n_heads * v_head_dim)
        return self.wo(output)  # (bsz, seqlen, dim)

    def init_weights(self, init_std: float):
        linear_list = [
            self.wkv_a,
            self.wkv_b,
        ]
        if self.q_lora_rank > 0:
            linear_list.extend([self.wq_a, self.wq_b])
        else:
            linear_list.append(self.wq)

        for linear in linear_list:
            fast_init_trunc_normal_(linear.weight, mean=0.0, std=0.02)
        fast_init_trunc_normal_(self.wo.weight, mean=0.0, std=init_std)

        self.kv_norm.reset_parameters()
        if self.q_lora_rank > 0:
            self.q_norm.reset_parameters()


class TransformerBlock(nn.Module):
    """
    Transformer block with attention and feed-forward layers.
    """

    def __init__(
        self, layer_id: int, model_args: DeepSeekV3ModelArgs, peft_config: PEFT
    ):

        super().__init__()
        self.attention = Attention(model_args, peft_config)
        self.attention_norm = nn.RMSNorm(model_args.dim, eps=model_args.norm_eps)
        self.ffn_norm = nn.RMSNorm(model_args.dim, eps=model_args.norm_eps)

        if peft_config.enable_peft and not peft_config.lora_train_norm:
            self.attention_norm.weight.requires_grad = False
            self.ffn_norm.weight.requires_grad = False

        self.moe_enabled = layer_id >= model_args.n_dense_layers
        if self.moe_enabled:
            self.moe = build_moe(
                args=model_args.moe_args,
                dim=model_args.dim,
                hidden_dim=model_args.moe_inter_dim,
                moe_impl=model_args.moe_impl,
                peft_config=peft_config,
            )
        else:
            self.feed_forward = FeedForward(
                model_args.dim, model_args.inter_dim, peft_config=peft_config
            )

        self.weight_init_std = 0.02 / (2 * (layer_id + 1)) ** 0.5
        self.layer_id = layer_id
        self.n_layers = model_args.n_layers

    def forward(
        self,
        x: torch.Tensor,
        freqs_cis: torch.Tensor,
        attention_masks: AttentionMasksType | None,
        positions: torch.Tensor | None = None,
    ):
        """
        Forward pass for the Transformer block.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, dim).
            freqs_cis (torch.Tensor): Precomputed complex exponential values for rotary embeddings.
            attention_masks (AttentionMasksType | None): Masks used when calculating attention scores.
            positions (torch.Tensor | None): Position indices used to access/shuffle RoPE cache. Defaults to None.

        Returns:
            torch.Tensor: Output tensor with the same shape as the input.
        """
        x = x + self.attention(
            self.attention_norm(x), freqs_cis, attention_masks, positions
        )
        if self.moe_enabled:
            x = x + self.moe(self.ffn_norm(x))
        else:
            x = x + self.feed_forward(self.ffn_norm(x))
        return x

    def init_weights(self, buffer_device: torch.device):
        for norm in (self.attention_norm, self.ffn_norm):
            norm.reset_parameters()
        self.attention.init_weights(self.weight_init_std)
        if self.moe_enabled:
            self.moe.init_weights(self.weight_init_std, buffer_device, self.n_layers)
        else:
            self.feed_forward.init_weights(self.weight_init_std)


class DeepSeekV3Model(ModelProtocol):
    """
    DeepSeek-V3 Transformer model with attention and feed-forward layers.
    """

    def __init__(self, model_args: DeepSeekV3ModelArgs, peft_config: PEFT):
        super().__init__(model_args)
        self.max_seq_len = model_args.max_seq_len
        self.tok_embeddings = nn.Embedding(model_args.vocab_size, model_args.dim)
        self.register_buffer(
            "freqs_cis", precompute_freqs_cis(model_args), persistent=False
        )

        self.layers = torch.nn.ModuleDict()
        for layer_id in range(model_args.n_layers):
            peft_config_layer = per_layer_config(peft_config, layer_id)
            self.layers[str(layer_id)] = TransformerBlock(
                layer_id, model_args, peft_config_layer
            )
            if (
                peft_config.enable_peft
                and (peft_config.layers_to_train is not None)
                and (layer_id not in peft_config.layers_to_train)
            ):
                # We have layers that are not in the layers_to_train list, so we need to freeze them.
                for param in self.layers[str(layer_id)].parameters():
                    param.requires_grad = False

        self.norm = nn.RMSNorm(model_args.dim)
        self.output = nn.Linear(
            model_args.dim,
            model_args.vocab_size,
            dtype=torch.get_default_dtype(),
            bias=False,
        )
        self.model_args = model_args
        if peft_config.enable_peft:
            if not peft_config.train_embeddings:
                self.tok_embeddings.weight.requires_grad = False
            if not peft_config.train_output_layer:
                self.output.weight.requires_grad = False
                if not peft_config.lora_train_norm:
                    self.norm.weight.requires_grad = False

    def init_weights(self, buffer_device: torch.device | None = None) -> None:
        buffer_device = buffer_device or self.freqs_cis.device
        with torch.device(buffer_device):
            self.freqs_cis = precompute_freqs_cis(self.model_args)
        if self.tok_embeddings is not None:
            fast_init_normal_(self.tok_embeddings.weight)
        for layer in self.layers.values():
            if layer is not None:
                # pyrefly: ignore [not-callable]
                layer.init_weights(buffer_device=buffer_device)
        if self.norm is not None:
            self.norm.reset_parameters()
        final_out_std = self.model_args.dim**-0.5
        cutoff_factor = 3
        if self.output is not None:
            fast_init_trunc_normal_(
                self.output.weight,
                mean=0.0,
                std=final_out_std,
                a=-cutoff_factor * final_out_std,
                b=cutoff_factor * final_out_std,
            )

    def get_attention_masks(
        self,
        input_batch: torch.Tensor,
        tokenizer: BaseTokenizer,
        extra_inputs: dict[str, torch.Tensor] | None = None,
    ) -> AttentionMasksType:
        mask_mods = [get_causal_mask_mod()]
        match self.model_args.attn_mask_type:
            case "causal":
                B = 1
            case "block_causal":
                B = input_batch.shape[0]
                mask_mods.append(get_document_mask_mod(input_batch, tokenizer.eos_id))
            case "block_causal_by_sequence_lengths":
                sequence_lengths = extra_inputs.pop("sequence_lengths", None)
                if sequence_lengths is None:
                    raise RuntimeError(
                        "`sequence_lengths` required for `block_causal_by_sequence_lengths`"
                    )
                B = input_batch.shape[0]
                mask_mods.append(
                    get_block_causal_mask_mod_by_seq_lens(sequence_lengths)
                )
            case _:
                raise ValueError(
                    f"Unknown attention mask type: {self.model_args.attn_mask_type}"
                )
        return create_attention_mask(
            and_masks(*mask_mods), B, None, input_batch.shape[1], input_batch.shape[1]
        )

    def forward(
        self,
        tokens: torch.Tensor,
        attention_masks: AttentionMasksType | None = None,
        positions: torch.Tensor | None = None,
    ):
        """
        Forward pass for the Transformer model.

        Args:
            tokens (torch.Tensor): Input token indices if pipeline parallelism is not enabled.
                If pipeline parallelism is enabled, this will be the input token indices
                for the ranks on the first pipeline stage. This will be the activation of the
                previous pipeline stage if the current rank is not on the first stage.
            attention_masks (AttentionMasksType | None): Masks used when calculating attention scores.
            positions (torch.Tensor | None): Position indices used to access/shuffle RoPE cache. Defaults to None.

        Returns:
            torch.Tensor: Logits tensor of shape (batch_size, vocab_size).
        """

        h = self.tok_embeddings(tokens) if self.tok_embeddings is not None else tokens

        for layer in self.layers.values():
            h = layer(h, self.freqs_cis, attention_masks, positions)
        h = self.norm(h) if self.norm is not None else h
        output = self.output(h) if self.output is not None else h
        return output
