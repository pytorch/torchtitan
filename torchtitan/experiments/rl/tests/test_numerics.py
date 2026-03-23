# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Numerics parity test: vLLM native Qwen3 vs vLLM + TorchTitan wrapper.

Single-GPU (TP=1) only. Verifies that TorchTitanVLLMModelWrapper produces
the same outputs as vLLM's native Qwen3 model when loaded from the same
checkpoint.

The vLLM native model uses vLLM's default FLASH_ATTN backend (bundled
flash-attn kernel). The TorchTitan model uses the CUSTOM backend
(PyTorchFlashAttentionImpl), which routes attention through PyTorch's
native FlashAttention varlen API.

All tests exercise the **prefill** forward path only (single pass, no
autoregressive decode). KV cache is written during prefill but never
read back — the decode path is not covered here.

Test cases and tolerance standards:
  - test_weights_match:          max_diff <= 1e-5 (exact weight loading)
  - test_attention_module:       atol=1e-5 (FLASH_ATTN vs CUSTOM backend)
  - test_end_to_end_logits:      atol=1e-3

test_attention_module compares vLLM's bundled FlashAttention (FLASH_ATTN)
against our PyTorchFlashAttentionImpl (CUSTOM backend) to verify parity
between the two attention implementations.

Usage:
    pytest torchtitan/experiments/rl/tests/test_numerics.py -v -s

    MODEL_CHECKPOINT_PATH=Qwen/Qwen3-0.6B pytest \
        torchtitan/experiments/rl/tests/test_numerics.py -v -s
"""

import os

import pytest
import torch

vllm = pytest.importorskip("vllm")

from vllm.config import set_current_vllm_config
from vllm.distributed import (
    init_distributed_environment,
    initialize_model_parallel,
    model_parallel_is_initialized,
)
from vllm.engine.arg_utils import EngineArgs
from vllm.forward_context import set_forward_context
from vllm.model_executor.model_loader import get_model_loader
from vllm.usage.usage_lib import UsageContext
from vllm.v1.attention.backends.flash_attn import FlashAttentionMetadata

# Import to register the CUSTOM attention backend with vLLM's registry
from torchtitan.experiments.rl.models.attention import (  # noqa: F401
    PyTorchFlashAttentionBackend,
)
from torchtitan.experiments.rl.models.parallelize import parallelize_qwen3
from torchtitan.experiments.rl.models.vllm_wrapper import TorchTitanVLLMModelWrapper
from torchtitan.models.qwen3 import model_registry

DEFAULT_MODEL_CHECKPOINT = "Qwen/Qwen3-0.6B"

# FA2 requires page_size (block_size) to be a multiple of 256.
# FA3 (SM 9.0+) supports any block_size.
_SM_90_AVAILABLE = (
    torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 9
)
BLOCK_SIZE = 32 if _SM_90_AVAILABLE else 256
NUM_BLOCKS = 30


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _get_model_flavor(checkpoint_path: str) -> str:
    """'Qwen/Qwen3-0.6B' -> '0.6B'."""
    basename = os.path.basename(checkpoint_path.rstrip("/"))
    if "Qwen3" in basename:
        parts = basename.split("-", 1)
        if len(parts) == 2:
            return parts[1]
    return "0.6B"


def _build_attn_metadata(num_tokens: int, device) -> FlashAttentionMetadata:
    """Prefill attention metadata for a single sequence."""
    query_start_loc = torch.zeros(2, dtype=torch.int32, device=device)
    query_start_loc[1] = num_tokens
    max_blocks = (num_tokens + BLOCK_SIZE - 1) // BLOCK_SIZE
    return FlashAttentionMetadata(
        causal=True,
        num_actual_tokens=num_tokens,
        max_query_len=num_tokens,
        query_start_loc=query_start_loc,
        max_seq_len=num_tokens,
        seq_lens=torch.tensor([num_tokens], dtype=torch.int32, device=device),
        block_table=torch.arange(
            max_blocks, dtype=torch.int32, device=device
        ).unsqueeze(0)
        + 1,
        slot_mapping=torch.arange(num_tokens, dtype=torch.int64, device=device)
        + BLOCK_SIZE,
        use_cascade=False,
        common_prefix_len=0,
        cu_prefix_query_lens=None,
        prefix_kv_lens=None,
        suffix_kv_lens=None,
    )


def _reset_kv_caches(vllm_config, num_kv_heads, head_dim, device):
    """Zero all KV caches so each forward starts from a clean slate."""
    for layer in vllm_config.compilation_config.static_forward_context.values():
        if hasattr(layer, "kv_cache"):
            layer.kv_cache = [
                torch.zeros(
                    (2, NUM_BLOCKS, BLOCK_SIZE, num_kv_heads, head_dim),
                    dtype=torch.bfloat16,
                    device=device,
                )
            ]


def _replace_vllm_rmsnorm_with_torch(model: torch.nn.Module) -> None:
    """Replace vLLM's fused RMSNorm with torch.nn.RMSNorm.

    vLLM's RMSNorm uses a fused CUDA kernel that differs numerically from
    PyTorch's implementation. We swap it out so both models use the same
    norm arithmetic for an apples-to-apples comparison.
    """
    from vllm.model_executor.layers.layernorm import RMSNorm as VLLMRMSNorm

    for name, child in list(model.named_children()):
        if isinstance(child, VLLMRMSNorm):
            wrapper = _TorchRMSNormWrapper(
                child.weight.shape[0], eps=child.variance_epsilon
            )
            wrapper.norm.weight.data.copy_(child.weight.data)
            wrapper = wrapper.to(device=child.weight.device, dtype=child.weight.dtype)
            setattr(model, name, wrapper)
        else:
            _replace_vllm_rmsnorm_with_torch(child)


class _TorchRMSNormWrapper(torch.nn.Module):
    """Adapts torch.nn.RMSNorm to vLLM's (hidden, residual) interface."""

    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.norm = torch.nn.RMSNorm(hidden_size, eps=eps)
        self.variance_epsilon = eps

    @property
    def weight(self):
        return self.norm.weight

    def forward(self, x, residual=None):
        if residual is not None:
            x = x + residual
            residual = x
        out = self.norm(x)
        return (out, residual) if residual is not None else out


def _make_engine_args(checkpoint: str, attention_backend: str | None = None):
    """Create EngineArgs with optional attention backend override."""
    kwargs = dict(
        model=checkpoint,
        skip_tokenizer_init=True,
        enforce_eager=True,
        gpu_memory_utilization=0.9,
        tensor_parallel_size=1,
        dtype="bfloat16",
        block_size=BLOCK_SIZE,
    )
    if attention_backend is not None:
        kwargs["attention_backend"] = attention_backend
    return EngineArgs(**kwargs)


# ---------------------------------------------------------------------------
# Fixture: build both models once per test class
# ---------------------------------------------------------------------------


@pytest.fixture(scope="class")
def ctx():
    """Load vLLM native (FLASH_ATTN attention backend) and TorchTitan (CUSTOM attention backend) models from the same checkpoint."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    torch.set_default_dtype(torch.bfloat16)
    device = torch.device("cuda:0")

    checkpoint = os.environ.get("MODEL_CHECKPOINT_PATH", DEFAULT_MODEL_CHECKPOINT)
    if not os.path.isdir(checkpoint):
        from huggingface_hub import snapshot_download

        checkpoint = snapshot_download(checkpoint)

    # Two separate configs: one per attention backend
    native_vllm_config = _make_engine_args(checkpoint).create_engine_config(
        UsageContext.ENGINE_CONTEXT
    )
    custom_vllm_config = _make_engine_args(
        checkpoint, attention_backend="CUSTOM"
    ).create_engine_config(UsageContext.ENGINE_CONTEXT)

    hf_cfg = native_vllm_config.model_config.hf_config

    # vLLM requires distributed init even for TP=1
    with set_current_vllm_config(native_vllm_config):
        if not model_parallel_is_initialized():
            os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
            init_distributed_environment(
                world_size=1,
                rank=0,
                distributed_init_method="env://",
                local_rank=0,
                backend="nccl",
            )
            initialize_model_parallel(1, 1)

    # vLLM native model (FLASH_ATTN — vLLM's bundled flash-attn)
    with set_current_vllm_config(native_vllm_config):
        loader = get_model_loader(native_vllm_config.load_config)
        vllm_native_model = loader.load_model(
            native_vllm_config, native_vllm_config.model_config
        )
        _replace_vllm_rmsnorm_with_torch(vllm_native_model)

    # TorchTitan model (CUSTOM — PyTorchFlashAttentionImpl)
    with set_current_vllm_config(custom_vllm_config):
        model_spec = model_registry(_get_model_flavor(checkpoint))
        model_spec.parallelize_fn = parallelize_qwen3
        torchtitan_model = TorchTitanVLLMModelWrapper(
            model_spec=model_spec, vllm_config=custom_vllm_config
        )

    torchtitan_model.to(device=device, dtype=torch.bfloat16).eval()

    # Shared test inputs
    num_kv_heads = hf_cfg.num_key_value_heads
    head_dim = getattr(
        hf_cfg, "head_dim", hf_cfg.hidden_size // hf_cfg.num_attention_heads
    )
    seq_len = 8

    torch.manual_seed(42)
    tokens = torch.randint(0, hf_cfg.vocab_size, (seq_len,), device=device)
    positions = torch.arange(seq_len, device=device)
    attn_meta = _build_attn_metadata(seq_len, device)

    # Reset KV caches for both configs
    _reset_kv_caches(native_vllm_config, num_kv_heads, head_dim, device)
    _reset_kv_caches(custom_vllm_config, num_kv_heads, head_dim, device)

    return {
        "vllm_native_model": vllm_native_model,
        "torchtitan_model": torchtitan_model,
        "native_vllm_config": native_vllm_config,
        "custom_vllm_config": custom_vllm_config,
        "tokens": tokens,
        "positions": positions,
        "prefill_attn_metadata": attn_meta,
        "dim": hf_cfg.hidden_size,
        "num_kv_heads": num_kv_heads,
        "head_dim": head_dim,
        "seq_len": seq_len,
        "device": device,
    }


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestVLLMTorchTitanNumerics:
    """Compare TorchTitanVLLMModelWrapper vs vLLM native Qwen3 (TP=1)."""

    @pytest.fixture(autouse=True)
    def _skip_no_cuda(self):
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        torch.set_default_dtype(torch.bfloat16)

    @torch.inference_mode()
    def test_end_to_end_logits(self, ctx):
        """Full forward pass: logits must match within 1e-3."""
        device = ctx["device"]

        _reset_kv_caches(
            ctx["native_vllm_config"], ctx["num_kv_heads"], ctx["head_dim"], device
        )
        with set_forward_context(
            ctx["prefill_attn_metadata"], ctx["native_vllm_config"]
        ):
            native_hidden = ctx["vllm_native_model"](
                input_ids=ctx["tokens"], positions=ctx["positions"]
            )
            native_logits = ctx["vllm_native_model"].compute_logits(native_hidden)

        _reset_kv_caches(
            ctx["custom_vllm_config"], ctx["num_kv_heads"], ctx["head_dim"], device
        )
        with set_forward_context(
            ctx["prefill_attn_metadata"], ctx["custom_vllm_config"]
        ):
            tt_hidden = ctx["torchtitan_model"](
                input_ids=ctx["tokens"], positions=ctx["positions"]
            )
            tt_logits = ctx["torchtitan_model"].compute_logits(tt_hidden)

        torch.testing.assert_close(
            native_logits.float(),
            tt_logits.float(),
            rtol=1e-3,
            atol=1e-3,
            msg="End-to-end logits mismatch",
        )

    @torch.inference_mode()
    def test_attention_module(self, ctx):
        """Attention backend parity: FLASH_ATTN vs CUSTOM (PyTorchFlashAttentionImpl).

        Feeds identical random hidden states through vLLM native's self_attn
        (using vLLM's bundled FlashAttention) and TorchTitan's attention module
        (using PyTorchFlashAttentionImpl). Verifies the two attention backends
        produce matching outputs.
        """
        device = ctx["device"]

        vllm_attn = ctx["vllm_native_model"].model.layers[0].self_attn
        tt_attn = ctx["torchtitan_model"].model.layers["0"].attention

        torch.manual_seed(123)
        hidden = torch.randn(
            ctx["seq_len"], ctx["dim"], dtype=torch.bfloat16, device=device
        )
        positions = torch.arange(ctx["seq_len"], device=device)

        # Native model forward with FLASH_ATTN backend
        _reset_kv_caches(
            ctx["native_vllm_config"], ctx["num_kv_heads"], ctx["head_dim"], device
        )
        with set_forward_context(
            ctx["prefill_attn_metadata"], ctx["native_vllm_config"]
        ):
            native_out = vllm_attn(positions=positions, hidden_states=hidden.clone())

        # TorchTitan model forward with CUSTOM backend (PyTorchFlashAttentionImpl)
        _reset_kv_caches(
            ctx["custom_vllm_config"], ctx["num_kv_heads"], ctx["head_dim"], device
        )
        with set_forward_context(
            ctx["prefill_attn_metadata"], ctx["custom_vllm_config"]
        ):
            freqs_cis = ctx["torchtitan_model"].model.freqs_cis
            tt_out = tt_attn(
                hidden.unsqueeze(0).clone(), freqs_cis, None, positions.unsqueeze(0)
            )
            if isinstance(tt_out, tuple):
                tt_out = tt_out[0]
            tt_out = tt_out.squeeze(0)

        torch.testing.assert_close(
            native_out.float(),
            tt_out.float(),
            rtol=1e-5,
            atol=1e-5,
            msg="Attention module output mismatch",
        )

    @torch.inference_mode()
    def test_weights_match(self, ctx):
        """Spot-check that key weight tensors loaded identically."""
        native_sd = dict(ctx["vllm_native_model"].named_parameters())
        tt_sd = dict(ctx["torchtitan_model"].model.named_parameters())

        pairs = [
            ("model.embed_tokens.weight", "tok_embeddings.weight"),
            ("model.layers.0.self_attn.o_proj.weight", "layers.0.attention.wo.weight"),
            ("model.layers.0.mlp.down_proj.weight", "layers.0.feed_forward.w2.weight"),
            ("model.norm.norm.weight", "norm.weight"),
        ]

        for vllm_name, tt_name in pairs:
            if vllm_name not in native_sd or tt_name not in tt_sd:
                continue
            vllm_w = native_sd[vllm_name]
            tt_w = tt_sd[tt_name]
            assert vllm_w.shape == tt_w.shape, f"Shape mismatch: {vllm_name}"
            max_diff = (vllm_w.float() - tt_w.float()).abs().max().item()
            assert max_diff <= 1e-5, f"Weight mismatch for {vllm_name}: {max_diff}"

        # Fused QKV
        qkv = native_sd.get("model.layers.0.self_attn.qkv_proj.weight")
        wq = tt_sd.get("layers.0.attention.wq.weight")
        wk = tt_sd.get("layers.0.attention.wk.weight")
        wv = tt_sd.get("layers.0.attention.wv.weight")
        if qkv is not None and wq is not None:
            fused = torch.cat([wq, wk, wv], dim=0)
            max_diff = (qkv.float() - fused.float()).abs().max().item()
            assert max_diff <= 1e-5, f"QKV weight mismatch: {max_diff}"

        # Fused gate_up
        gate_up = native_sd.get("model.layers.0.mlp.gate_up_proj.weight")
        w1 = tt_sd.get("layers.0.feed_forward.w1.weight")
        w3 = tt_sd.get("layers.0.feed_forward.w3.weight")
        if gate_up is not None and w1 is not None:
            fused = torch.cat([w1, w3], dim=0)
            max_diff = (gate_up.float() - fused.float()).abs().max().item()
            assert max_diff <= 1e-5, f"gate_up weight mismatch: {max_diff}"
