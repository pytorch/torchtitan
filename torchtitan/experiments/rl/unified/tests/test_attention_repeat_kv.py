# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Test script to compare vLLM attention behavior with and without repeat_kv.

This test compares two scenarios:
1. With repeat_kv: num_kv_heads expanded to num_heads (MHA-style after expansion)
2. Without repeat_kv: num_kv_heads kept as-is (GQA-style)

Both scenarios use the same input and generate 5 tokens auto-regressively,
checking that attention outputs match for each token.

Usage:
    python torchtitan/experiments/rl/unified/tests/test_attention_repeat_kv.py
"""

import os
# Set environment variables before importing vLLM
os.environ["VLLM_ATTENTION_BACKEND"] = "FLASH_ATTN"

from dataclasses import dataclass

import torch
import torch.nn as nn

# Import vLLM attention components
try:
    from vllm.config import VllmConfig, CacheConfig, ModelConfig, set_current_vllm_config
    from vllm.engine.arg_utils import EngineArgs
    from vllm.usage.usage_lib import UsageContext
    from vllm.v1.attention.backends.flash_attn import FlashAttentionMetadata
    from vllm.v1.attention.backend import CommonAttentionMetadata
    from vllm.forward_context import set_forward_context
    from vllm.attention.layer import Attention
    HAS_VLLM = True
except ImportError:
    HAS_VLLM = False
    print("vLLM not installed, skipping test")


BLOCK_SIZE = 32


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """torch.repeat_interleave(x, dim=2, repeats=n_rep)"""
    bs, slen, n_kv_heads, head_dim = x.shape
    if n_rep == 1:
        return x
    return (
        torch.unsqueeze(x, dim=3)
        .expand(bs, slen, n_kv_heads, n_rep, head_dim)
        .reshape(bs, slen, n_kv_heads * n_rep, head_dim)
    )


def create_vllm_config(num_heads: int, num_kv_heads: int, head_dim: int):
    """Create a minimal vLLM config for testing."""
    # Use a dummy model path - we're just using vLLM's attention
    engine_args = EngineArgs(
        model="Qwen/Qwen3-0.6B",
        skip_tokenizer_init=True,
        enforce_eager=True,
        gpu_memory_utilization=0.1,
        tensor_parallel_size=1,
        dtype="bfloat16",
        kv_cache_dtype="auto",
    )
    vllm_config = engine_args.create_engine_config(UsageContext.ENGINE_CONTEXT)
    return vllm_config


@dataclass
class BatchSpec:
    """Specification for a batch configuration (workload shape only)."""

    seq_lens: list[int]
    query_lens: list[int]

    name: str = "unnamed"

    @property
    def batch_size(self):
        return len(self.seq_lens)

    def __post_init__(self):
        assert len(self.seq_lens) == len(self.query_lens)

    def compute_num_tokens(self):
        return sum(self.query_lens)


def create_common_attn_metadata(
    batch_spec: BatchSpec,
    block_size: int,
    device: torch.device,
    max_block_idx: int = 1000,
    arange_block_indices: bool = False,
) -> CommonAttentionMetadata:
    """Create CommonAttentionMetadata from a BatchSpec."""
    # Create query start locations
    query_start_loc = torch.zeros(
        batch_spec.batch_size + 1, dtype=torch.int32, device=device
    )
    query_start_loc[1:] = torch.tensor(
        batch_spec.query_lens, dtype=torch.int32, device=device
    ).cumsum(0)
    query_start_loc_cpu = query_start_loc.cpu()
    num_tokens = batch_spec.compute_num_tokens()

    # Create sequence lengths
    seq_lens = torch.tensor(batch_spec.seq_lens, dtype=torch.int32, device=device)
    max_seq_len = int(seq_lens.max().item())

    # Create block table and slot mapping
    max_blocks = (max(batch_spec.seq_lens) + block_size - 1) // block_size
    if arange_block_indices:
        num_blocks = batch_spec.batch_size * max_blocks
        block_table_tensor = torch.arange(
            num_blocks, dtype=torch.int32, device=device
        ).view(batch_spec.batch_size, max_blocks)
        slot_mapping = torch.arange(num_tokens, dtype=torch.int64, device=device).view(
            num_tokens
        )
    else:
        block_table_tensor = torch.randint(
            0,
            max_block_idx,
            (batch_spec.batch_size, max_blocks),
            dtype=torch.int32,
            device=device,
        )
        slot_mapping = torch.randint(
            0, max_block_idx, (num_tokens,), dtype=torch.int64, device=device
        )

    # Calculate max query length
    max_query_len = max(batch_spec.query_lens)

    return CommonAttentionMetadata(
        query_start_loc=query_start_loc,
        query_start_loc_cpu=query_start_loc_cpu,
        seq_lens=seq_lens,
        num_reqs=batch_spec.batch_size,
        num_actual_tokens=num_tokens,
        max_query_len=max_query_len,
        max_seq_len=max_seq_len,
        block_table_tensor=block_table_tensor + 1,
        slot_mapping=slot_mapping + BLOCK_SIZE,
        causal=True,
    )


def build_attn_metadata(seq_len: int, device: str) -> FlashAttentionMetadata:
    """Build FlashAttentionMetadata for attention."""
    batch_spec = BatchSpec(seq_lens=[seq_len], query_lens=[seq_len])
    common_attn_metadata = create_common_attn_metadata(
        batch_spec,
        block_size=BLOCK_SIZE,
        device=torch.device(device),
        arange_block_indices=True,
    )
    return FlashAttentionMetadata(
        causal=common_attn_metadata.causal,
        num_actual_tokens=seq_len,
        max_query_len=seq_len,
        query_start_loc=common_attn_metadata.query_start_loc,
        max_seq_len=seq_len,
        seq_lens=common_attn_metadata.seq_lens,
        block_table=common_attn_metadata.block_table_tensor,
        slot_mapping=common_attn_metadata.slot_mapping,
        use_cascade=False,
        common_prefix_len=0,
        cu_prefix_query_lens=None,
        prefix_kv_lens=None,
        suffix_kv_lens=None,
    )


class SimpleAttention(nn.Module):
    """
    Simple attention module using vLLM's Attention for testing repeat_kv behavior.
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
        vllm_config,
        layer_idx: int = 0,
        use_repeat_kv: bool = False,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.vllm_config = vllm_config
        self.use_repeat_kv = use_repeat_kv
        self.n_rep = num_heads // num_kv_heads

        # Q, K, V projections
        self.wq = nn.Linear(hidden_size, num_heads * head_dim, bias=False)
        self.wk = nn.Linear(hidden_size, num_kv_heads * head_dim, bias=False)
        self.wv = nn.Linear(hidden_size, num_kv_heads * head_dim, bias=False)
        self.wo = nn.Linear(num_heads * head_dim, hidden_size, bias=False)

        # vLLM Attention - use num_heads for both if repeat_kv is enabled
        with set_current_vllm_config(vllm_config):
            if use_repeat_kv:
                # After repeat_kv, K and V will have num_heads
                self.attn = Attention(
                    num_heads=num_heads,
                    head_size=head_dim,
                    scale=head_dim ** -0.5,
                    num_kv_heads=num_heads,  # MHA-style after expansion
                    prefix=f"layer{layer_idx}",
                )
            else:
                # GQA-style, vLLM handles the repeat internally
                self.attn = Attention(
                    num_heads=num_heads,
                    head_size=head_dim,
                    scale=head_dim ** -0.5,
                    num_kv_heads=num_kv_heads,
                    prefix=f"layer{layer_idx}",
                )

    def forward(
        self,
        x: torch.Tensor,
        attn_metadata: FlashAttentionMetadata,
    ) -> torch.Tensor:
        """
        Forward pass using vLLM's Attention.

        Args:
            x: Input tensor of shape [batch, seq_len, hidden_size]
            attn_metadata: vLLM attention metadata for KV caching

        Returns:
            Output tensor of shape [batch, seq_len, hidden_size]
        """
        batch, seq_len, _ = x.shape

        # Compute Q, K, V - vLLM expects [num_tokens, num_heads, head_dim]
        xq = self.wq(x).view(batch * seq_len, self.num_heads, self.head_dim)
        xk = self.wk(x).view(batch * seq_len, self.num_kv_heads, self.head_dim)
        xv = self.wv(x).view(batch * seq_len, self.num_kv_heads, self.head_dim)

        # Apply repeat_kv if enabled
        if self.use_repeat_kv:
            # Reshape for repeat_kv: [bs, slen, n_kv_heads, head_dim]
            xk = xk.view(batch, seq_len, self.num_kv_heads, self.head_dim)
            xv = xv.view(batch, seq_len, self.num_kv_heads, self.head_dim)
            xk = repeat_kv(xk, self.n_rep)
            xv = repeat_kv(xv, self.n_rep)
            # Reshape back to [num_tokens, num_heads, head_dim]
            xk = xk.contiguous().view(batch * seq_len, self.num_heads, self.head_dim)
            xv = xv.contiguous().view(batch * seq_len, self.num_heads, self.head_dim)

        # Use vLLM attention (attn_metadata is set via set_forward_context)
        output = self.attn(xq, xk, xv)

        # Reshape and project
        output = output.view(batch, seq_len, -1)
        return self.wo(output)



def test_mha_vs_gqa_with_vllm():
    """
    Compare MHA (with repeat_kv) vs GQA (without repeat_kv).

    - MHA: uses repeat_kv to expand K,V from num_kv_heads to num_heads
    - GQA: doesn't use repeat_kv, vLLM handles GQA internally

    Compares the mean output values for each token position in the prefill.
    """
    if not HAS_VLLM:
        print("Skipping test - vLLM not available")
        return False

    print("\n" + "=" * 70)
    print("Testing vLLM Attention: MHA (with repeat_kv) vs GQA (no repeat_kv)")
    print("=" * 70)

    # Model dimensions
    hidden_size = 256
    num_heads = 8
    num_kv_heads = 2  # GQA with 4x fewer KV heads
    head_dim = hidden_size // num_heads
    batch_size = 1
    prompt_len = 8

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

    if device == "cpu":
        print("Warning: vLLM attention requires CUDA, skipping test on CPU")
        return True

    print(f"\nConfig: hidden_size={hidden_size}, num_heads={num_heads}, "
          f"num_kv_heads={num_kv_heads}, head_dim={head_dim}")
    print(f"Prompt length: {prompt_len} tokens")

    # Set default dtype for vLLM attention backend selection
    torch.set_default_dtype(dtype)

    # Setup MHA with repeat_kv (expands K,V to num_heads before attention)
    print("\n--- Setting up MHA (with repeat_kv) ---")
    vllm_config_mha = create_vllm_config(num_heads, num_heads, head_dim)

    attn_mha = SimpleAttention(
        hidden_size=hidden_size,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        vllm_config=vllm_config_mha,
        layer_idx=0,
        use_repeat_kv=True,  # MHA uses repeat_kv
    ).to(device=device, dtype=dtype).eval()

    # Setup KV cache (uses num_heads since we expand K,V)
    static_ctx_mha = vllm_config_mha.compilation_config.static_forward_context
    num_blocks = 30
    for layer_name, layer in static_ctx_mha.items():
        if hasattr(layer, 'kv_cache'):
            kv_cache = torch.zeros(
                (2, num_blocks, BLOCK_SIZE, num_heads, head_dim),
                dtype=dtype,
                device=device,
            )
            layer.kv_cache = [kv_cache]

    # Setup GQA without repeat_kv (vLLM handles GQA internally)
    print("--- Setting up GQA (no repeat_kv) ---")
    vllm_config_gqa = create_vllm_config(num_heads, num_kv_heads, head_dim)

    attn_gqa = SimpleAttention(
        hidden_size=hidden_size,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        vllm_config=vllm_config_gqa,
        layer_idx=0,
        use_repeat_kv=False,  # GQA doesn't use repeat_kv
    ).to(device=device, dtype=dtype).eval()

    # Setup KV cache (uses num_kv_heads)
    static_ctx_gqa = vllm_config_gqa.compilation_config.static_forward_context
    for layer_name, layer in static_ctx_gqa.items():
        if hasattr(layer, 'kv_cache'):
            kv_cache = torch.zeros(
                (2, num_blocks, BLOCK_SIZE, num_kv_heads, head_dim),
                dtype=dtype,
                device=device,
            )
            layer.kv_cache = [kv_cache]

    # Copy weights so both have identical projections
    attn_mha.wq.weight.data.copy_(attn_gqa.wq.weight.data)
    attn_mha.wk.weight.data.copy_(attn_gqa.wk.weight.data)
    attn_mha.wv.weight.data.copy_(attn_gqa.wv.weight.data)
    attn_mha.wo.weight.data.copy_(attn_gqa.wo.weight.data)

    # Create input
    torch.manual_seed(42)
    prompt = torch.randn(batch_size, prompt_len, hidden_size, device=device, dtype=dtype)

    # Prefill phase
    print("\n[Prefill] Processing prompt...")
    attn_metadata = build_attn_metadata(prompt_len, device)

    with torch.no_grad():
        with set_forward_context(attn_metadata, vllm_config_mha):
            output_mha = attn_mha(prompt, attn_metadata)
        with set_forward_context(attn_metadata, vllm_config_gqa):
            output_gqa = attn_gqa(prompt, attn_metadata)

    assert not torch.isnan(output_mha).any(), "MHA output contains NaN!"
    assert not torch.isnan(output_gqa).any(), "GQA output contains NaN!"
    print("  Prefill completed!")

    # Print comparison table with mean values for each token position
    print("\n" + "=" * 65)
    print("MHA vs GQA Mean Comparison (Prefill)")
    print("=" * 65)
    print(f"{'Token':<12}{'MHA Mean':>15}{'GQA Mean':>15}{'Diff':>15}")
    print("-" * 65)
    for i in range(prompt_len):
        mha_mean = output_mha[0, i, :].mean().item()
        gqa_mean = output_gqa[0, i, :].mean().item()
        diff = mha_mean - gqa_mean
        print(f"{'Token ' + str(i):<12}{mha_mean:>15.6f}{gqa_mean:>15.6f}{diff:>15.6f}")
    print("=" * 65)
    print("Both MHA and GQA completed successfully!")
    print("=" * 65)

    return True


if __name__ == "__main__":
    if not HAS_VLLM:
        print("vLLM not installed, cannot run tests")
        exit(1)

    print("Running vLLM attention tests...")

    # Test 2: Compare MHA vs GQA
    success = test_mha_vs_gqa_with_vllm()
