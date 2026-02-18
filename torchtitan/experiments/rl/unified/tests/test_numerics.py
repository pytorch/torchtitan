# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Tests for vLLM integration with TorchTitan models.

Compares vLLM running TorchTitan model vs vLLM running native Qwen3 model.
Both models use vLLM's attention and forward context.

Usage:
    torchrun --nproc-per-node=1 -m pytest \
        torchtitan/experiments/rl/unified/tests/test_numerics.py -v -s

    torchrun --nproc-per-node=2 -m pytest \
        torchtitan/experiments/rl/unified/tests/test_numerics.py -v -s

    MODEL_CHECKPOINT_PATH=Qwen/Qwen3-1.7B torchrun --nproc-per-node=2 \
        -m pytest torchtitan/experiments/rl/unified/tests/test_numerics.py -v -s
"""

import glob
import os
from dataclasses import dataclass

import pytest
import torch
from safetensors.torch import load_file as load_safetensors

vllm = pytest.importorskip("vllm")

from torch.distributed._tensor import DTensor
from torchtitan.experiments.rl.unified.infra.parallelize import parallelize_qwen3
from torchtitan.experiments.rl.unified.models.vllm_wrapper import (
    TorchTitanVLLMModelWrapper,
)
from torchtitan.models.qwen3 import Qwen3Model, Qwen3ModelArgs
from torchtitan.models.qwen3.model.state_dict_adapter import Qwen3StateDictAdapter
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

BLOCK_SIZE = 32
NUM_BLOCKS = 30
DEFAULT_MODEL_CHECKPOINT = "Qwen/Qwen3-0.6B"

_distributed_initialized = False


def get_distributed_info():
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    rank = int(os.environ.get("RANK", 0))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    return world_size, rank, local_rank


def ensure_distributed_initialized():
    global _distributed_initialized
    world_size, rank, local_rank = get_distributed_info()

    if model_parallel_is_initialized() or _distributed_initialized:
        return

    os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
    init_distributed_environment(
        world_size=world_size,
        rank=rank,
        distributed_init_method="env://",
        local_rank=local_rank,
        backend="nccl",
    )
    initialize_model_parallel(world_size, 1)
    _distributed_initialized = True

    if rank == 0:
        print(f"Initialized distributed environment: TP={world_size}")


def reset_kv_caches(vllm_config, num_kv_heads_local, head_dim, device):
    """Zero all KV caches in the static forward context."""
    static_ctx = vllm_config.compilation_config.static_forward_context
    for layer_name, layer in static_ctx.items():
        if hasattr(layer, "kv_cache"):
            layer.kv_cache = [
                torch.zeros(
                    (2, NUM_BLOCKS, BLOCK_SIZE, num_kv_heads_local, head_dim),
                    dtype=torch.bfloat16,
                    device=device,
                )
            ]


class TorchRMSNormWrapper(torch.nn.Module):
    """Wrapper around torch.nn.RMSNorm that mimics vLLM's RMSNorm interface.

    vLLM's RMSNorm accepts (hidden_states, residual) and returns (output, residual),
    but torch.nn.RMSNorm only accepts (x) and returns a single tensor.
    """

    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.norm = torch.nn.RMSNorm(hidden_size, eps=eps)
        self.variance_epsilon = eps

    @property
    def weight(self):
        return self.norm.weight

    def forward(self, x: torch.Tensor, residual: torch.Tensor = None):
        if residual is not None:
            x = x + residual
            residual = x
        output = self.norm(x)
        if residual is not None:
            return output, residual
        return output


def replace_vllm_rmsnorm_with_torch(model: torch.nn.Module, rank: int = 0) -> None:
    """Replace vLLM's RMSNorm with TorchRMSNormWrapper for numerical equivalence."""
    from vllm.model_executor.layers.layernorm import RMSNorm as VLLMRMSNorm

    replaced_count = 0

    def replace_in_module(parent_module):
        nonlocal replaced_count
        for name, child in list(parent_module.named_children()):
            if isinstance(child, VLLMRMSNorm):
                torch_norm = TorchRMSNormWrapper(
                    child.weight.shape[0], eps=child.variance_epsilon
                )
                torch_norm.norm.weight.data.copy_(child.weight.data)
                torch_norm = torch_norm.to(
                    device=child.weight.device, dtype=child.weight.dtype
                )
                setattr(parent_module, name, torch_norm)
                replaced_count += 1
            else:
                replace_in_module(child)

    replace_in_module(model)
    if rank == 0:
        print(f"Replaced {replaced_count} vLLM RMSNorm layers with TorchRMSNormWrapper")


def build_attn_metadata(num_tokens: int, local_rank: int = 0) -> FlashAttentionMetadata:
    """Build FlashAttentionMetadata for a prefill of num_tokens."""
    device = torch.device(f"cuda:{local_rank}")

    query_start_loc = torch.zeros(2, dtype=torch.int32, device=device)
    query_start_loc[1] = num_tokens
    seq_lens = torch.tensor([num_tokens], dtype=torch.int32, device=device)

    max_blocks = (num_tokens + BLOCK_SIZE - 1) // BLOCK_SIZE
    block_table = torch.arange(max_blocks, dtype=torch.int32, device=device).unsqueeze(
        0
    )
    slot_mapping = torch.arange(num_tokens, dtype=torch.int64, device=device)

    return FlashAttentionMetadata(
        causal=True,
        num_actual_tokens=num_tokens,
        max_query_len=num_tokens,
        query_start_loc=query_start_loc,
        max_seq_len=num_tokens,
        seq_lens=seq_lens,
        block_table=block_table + 1,
        slot_mapping=slot_mapping + BLOCK_SIZE,
        use_cascade=False,
        common_prefix_len=0,
        cu_prefix_query_lens=None,
        prefix_kv_lens=None,
        suffix_kv_lens=None,
    )


@dataclass
class SetupContext:
    """Context object holding setup results for tests."""

    vllm_native_model: torch.nn.Module
    torchtitan_vllm_model: torch.nn.Module
    vllm_config: object
    tokens: torch.Tensor
    positions: torch.Tensor
    attn_metadata: FlashAttentionMetadata
    model_dim: int
    num_kv_heads_local: int
    head_dim: int
    seq_len: int
    world_size: int


def setup_models_and_inputs(
    model_checkpoint_path: str = DEFAULT_MODEL_CHECKPOINT,
    seq_len: int = 8,
) -> SetupContext:
    """Set up vLLM native and TorchTitan models with weights from the same checkpoint."""
    ensure_distributed_initialized()
    world_size, rank, local_rank = get_distributed_info()
    device = f"cuda:{local_rank}"

    # Create vLLM config
    engine_args = EngineArgs(
        model=model_checkpoint_path,
        skip_tokenizer_init=True,
        enforce_eager=True,
        gpu_memory_utilization=0.1,
        tensor_parallel_size=world_size,
        dtype="bfloat16",
    )
    vllm_config = engine_args.create_engine_config(UsageContext.ENGINE_CONTEXT)

    # Extract model dimensions
    hf_config = vllm_config.model_config.hf_config
    model_dim = hf_config.hidden_size
    num_heads = hf_config.num_attention_heads
    num_kv_heads = hf_config.num_key_value_heads
    num_layers = hf_config.num_hidden_layers
    head_dim = getattr(hf_config, "head_dim", model_dim // num_heads)
    hidden_dim = hf_config.intermediate_size
    vocab_size = hf_config.vocab_size

    if rank == 0:
        print(
            f"\nModel: {model_checkpoint_path}, dim={model_dim}, layers={num_layers}, "
            f"heads={num_heads}, kv_heads={num_kv_heads}, head_dim={head_dim}, TP={world_size}"
        )

    model_args = Qwen3ModelArgs(
        dim=model_dim,
        n_layers=num_layers,
        n_heads=num_heads,
        n_kv_heads=num_kv_heads,
        vocab_size=vocab_size,
        head_dim=head_dim,
        hidden_dim=hidden_dim,
        norm_eps=hf_config.rms_norm_eps,
        rope_theta=hf_config.rope_theta,
        max_seq_len=seq_len * 2,
        qk_norm=getattr(hf_config, "qk_norm", True),
        depth_init=False,
        attn_type="sdpa",
        attn_mask_type="causal",
    )

    # Load vLLM native model
    loader = get_model_loader(vllm_config.load_config)
    vllm_native_model = loader.load_model(vllm_config, vllm_config.model_config)

    # Create TorchTitan model with vLLM wrapper
    with set_current_vllm_config(vllm_config):
        torchtitan_vllm_model = TorchTitanVLLMModelWrapper(
            model_cls=Qwen3Model,
            model_args=model_args,
            state_dict_adapter=Qwen3StateDictAdapter,
            parallelize_fn=parallelize_qwen3,
            vllm_config=vllm_config,
        )
    torchtitan_vllm_model.to(device=device, dtype=torch.bfloat16)
    torchtitan_vllm_model.eval()

    # Replace vLLM's RMSNorm with torch.nn.RMSNorm for numerical equivalence
    replace_vllm_rmsnorm_with_torch(vllm_native_model, rank)

    # Load weights into TorchTitan model from the same checkpoint
    def load_hf_weights_iter(checkpoint_path: str):
        if os.path.isdir(checkpoint_path):
            model_path = checkpoint_path
        else:
            from huggingface_hub import snapshot_download

            model_path = snapshot_download(checkpoint_path)

        safetensor_files = glob.glob(os.path.join(model_path, "*.safetensors"))
        if not safetensor_files:
            raise ValueError(f"No safetensors files found in {model_path}")

        for st_file in sorted(safetensor_files):
            weights = load_safetensors(st_file)
            yield from weights.items()

    loaded_params = torchtitan_vllm_model.load_weights(
        load_hf_weights_iter(model_checkpoint_path)
    )
    if rank == 0:
        print(f"Loaded {len(loaded_params)} parameters into TorchTitan model")

    # Create inputs
    torch.manual_seed(42)
    tokens = torch.randint(0, vocab_size, (seq_len,), device="cpu").to(device)
    positions = torch.arange(seq_len, device=device)

    attn_metadata = build_attn_metadata(seq_len, local_rank)

    # Set up KV caches
    num_kv_heads_local = num_kv_heads // world_size
    reset_kv_caches(vllm_config, num_kv_heads_local, head_dim, device)

    if rank == 0:
        print("Setup complete\n")

    return SetupContext(
        vllm_native_model=vllm_native_model,
        torchtitan_vllm_model=torchtitan_vllm_model,
        vllm_config=vllm_config,
        tokens=tokens,
        positions=positions,
        attn_metadata=attn_metadata,
        model_dim=model_dim,
        num_kv_heads_local=num_kv_heads_local,
        head_dim=head_dim,
        seq_len=seq_len,
        world_size=world_size,
    )


class TestVLLMTorchTitan:
    """Compare TorchTitanVLLMModelWrapper vs vLLM native Qwen3 model."""

    @pytest.fixture(autouse=True)
    def setup(self):
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        torch.set_default_dtype(torch.bfloat16)

    @pytest.fixture(scope="class")
    def ctx(self):
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        torch.set_default_dtype(torch.bfloat16)
        checkpoint = os.environ.get("MODEL_CHECKPOINT_PATH", DEFAULT_MODEL_CHECKPOINT)
        return setup_models_and_inputs(model_checkpoint_path=checkpoint)

    @torch.inference_mode()
    def test_weights_load(self, ctx: SetupContext):
        """Verify key weights match between vLLM native and TorchTitan models.

        vLLM fuses certain weight matrices:
        - q_proj + k_proj + v_proj -> qkv_proj
        - gate_proj + up_proj -> gate_up_proj
        - RMSNorm replaced by TorchRMSNormWrapper (adds .norm. in path)
        """
        _, rank, _ = get_distributed_info()
        vllm_sd = dict(ctx.vllm_native_model.named_parameters())
        tt_sd = dict(ctx.torchtitan_vllm_model.model.named_parameters())

        # Compare local shards: vLLM stores plain (already-sharded) tensors,
        # TorchTitan uses DTensor.
        comparisons = []

        direct_pairs = [
            ("model.embed_tokens.weight", "tok_embeddings.weight"),
            ("model.layers.0.self_attn.o_proj.weight", "layers.0.attention.wo.weight"),
            ("model.layers.0.mlp.down_proj.weight", "layers.0.feed_forward.w2.weight"),
            (
                "model.layers.0.input_layernorm.norm.weight",
                "layers.0.attention_norm.weight",
            ),
            (
                "model.layers.0.post_attention_layernorm.norm.weight",
                "layers.0.ffn_norm.weight",
            ),
            ("model.norm.norm.weight", "norm.weight"),
            (
                "model.layers.0.self_attn.q_norm.norm.weight",
                "layers.0.attention.q_norm.weight",
            ),
            (
                "model.layers.0.self_attn.k_norm.norm.weight",
                "layers.0.attention.k_norm.weight",
            ),
        ]

        for vllm_name, tt_name in direct_pairs:
            if vllm_name not in vllm_sd or tt_name not in tt_sd:
                continue
            tt_tensor = tt_sd[tt_name]
            if isinstance(tt_tensor, DTensor):
                tt_tensor = tt_tensor.to_local()
            comparisons.append((vllm_name, vllm_sd[vllm_name], tt_tensor))

        # Fused QKV: vLLM's qkv_proj = concat(q, k, v)
        qkv_name = "model.layers.0.self_attn.qkv_proj.weight"
        if qkv_name in vllm_sd:
            wq = tt_sd.get("layers.0.attention.wq.weight")
            wk = tt_sd.get("layers.0.attention.wk.weight")
            wv = tt_sd.get("layers.0.attention.wv.weight")
            if wq is not None and wk is not None and wv is not None:
                parts = [
                    p.to_local() if isinstance(p, DTensor) else p for p in [wq, wk, wv]
                ]
                comparisons.append(
                    (qkv_name, vllm_sd[qkv_name], torch.cat(parts, dim=0))
                )

        # Fused gate_up: vLLM's gate_up_proj = concat(gate, up)
        gate_up_name = "model.layers.0.mlp.gate_up_proj.weight"
        if gate_up_name in vllm_sd:
            w1 = tt_sd.get("layers.0.feed_forward.w1.weight")
            w3 = tt_sd.get("layers.0.feed_forward.w3.weight")
            if w1 is not None and w3 is not None:
                parts = [
                    p.to_local() if isinstance(p, DTensor) else p for p in [w1, w3]
                ]
                comparisons.append(
                    (gate_up_name, vllm_sd[gate_up_name], torch.cat(parts, dim=0))
                )

        for name, vllm_tensor, tt_tensor in comparisons:
            assert (
                vllm_tensor.shape == tt_tensor.shape
            ), f"Shape mismatch for {name}: {vllm_tensor.shape} vs {tt_tensor.shape}"
            max_diff = (vllm_tensor.float() - tt_tensor.float()).abs().max().item()
            if rank == 0:
                print(f"  {name}: max_diff={max_diff:.2e}")
            assert max_diff <= 1e-5, f"Weight mismatch for {name}: max_diff={max_diff}"

    @torch.inference_mode()
    def test_forward_attention(self, ctx: SetupContext):
        """Test attention module outputs match (TP=1 only)."""
        world_size, rank, local_rank = get_distributed_info()

        if world_size > 1:
            pytest.skip("Skipped for TP > 1")

        device = f"cuda:{local_rank}"
        vllm_attn = ctx.vllm_native_model.model.layers[0].self_attn
        tt_attn = ctx.torchtitan_vllm_model.model.layers["0"].attention

        torch.manual_seed(123)
        attn_input = torch.randn(
            ctx.seq_len, ctx.model_dim, dtype=torch.bfloat16, device=device
        )
        attn_positions = torch.arange(ctx.seq_len, device=device)

        # vLLM attention forward
        reset_kv_caches(ctx.vllm_config, ctx.num_kv_heads_local, ctx.head_dim, device)
        with set_forward_context(ctx.attn_metadata, ctx.vllm_config):
            vllm_out = vllm_attn(
                positions=attn_positions, hidden_states=attn_input.clone()
            )

        # TorchTitan attention forward
        reset_kv_caches(ctx.vllm_config, ctx.num_kv_heads_local, ctx.head_dim, device)
        with set_forward_context(ctx.attn_metadata, ctx.vllm_config):
            rope_cache = ctx.torchtitan_vllm_model.model.rope_cache
            if isinstance(rope_cache, DTensor):
                rope_cache = rope_cache.to_local()
            tt_out = tt_attn(
                attn_input.unsqueeze(0).clone(),
                rope_cache,
                None,
                attn_positions.unsqueeze(0),
            )
            tt_out = (
                tt_out.full_tensor().squeeze(0)
                if isinstance(tt_out, DTensor)
                else tt_out.squeeze(0)
            )

        torch.testing.assert_close(
            vllm_out.float(),
            tt_out.float(),
            rtol=1e-3,
            atol=1e-3,
            msg="Attention output mismatch",
        )

    @torch.inference_mode()
    def test_forward_embedding_and_transformer_layer(self, ctx: SetupContext):
        """Test embedding + first transformer layer outputs match."""
        world_size, rank, local_rank = get_distributed_info()
        device = f"cuda:{local_rank}"

        # Run vLLM native embedding + first layer
        reset_kv_caches(ctx.vllm_config, ctx.num_kv_heads_local, ctx.head_dim, device)
        with set_forward_context(ctx.attn_metadata, ctx.vllm_config):
            vllm_embeddings = ctx.vllm_native_model.model.embed_tokens(ctx.tokens)
            hidden_states, residual = ctx.vllm_native_model.model.layers[0](
                positions=ctx.positions,
                hidden_states=vllm_embeddings.clone(),
                residual=None,
            )

        # Run TorchTitan embedding + first layer
        reset_kv_caches(ctx.vllm_config, ctx.num_kv_heads_local, ctx.head_dim, device)
        with set_forward_context(ctx.attn_metadata, ctx.vllm_config):
            tokens_2d = ctx.tokens.unsqueeze(0)
            positions_2d = ctx.positions.unsqueeze(0)
            tt_embeddings = ctx.torchtitan_vllm_model.model.tok_embeddings(tokens_2d)
            tt_embeddings_full = (
                tt_embeddings.full_tensor()
                if isinstance(tt_embeddings, DTensor)
                else tt_embeddings
            )

            rope_cache = ctx.torchtitan_vllm_model.model.rope_cache
            if isinstance(rope_cache, DTensor):
                rope_cache = rope_cache.to_local()

            h = ctx.torchtitan_vllm_model.model.layers["0"](
                tt_embeddings.clone(), rope_cache, None, positions_2d
            )
            tt_layer0 = h.full_tensor() if isinstance(h, DTensor) else h

        # Compare embeddings
        torch.testing.assert_close(
            vllm_embeddings.view(-1, ctx.model_dim).float(),
            tt_embeddings_full.view(-1, ctx.model_dim).float(),
            rtol=1e-2,
            atol=1e-2,
            msg="Embedding mismatch",
        )

        # Compare layer 0: vLLM returns (hidden, residual), TorchTitan returns combined
        vllm_combined = (hidden_states + residual).view(-1, ctx.model_dim)
        tt_flat = tt_layer0.view(-1, ctx.model_dim)

        if rank == 0:
            diff = (vllm_combined.float() - tt_flat.float()).abs()
            print(
                f"Layer 0 max abs diff: {diff.max().item():.6f}, mean: {diff.mean().item():.6f}"
            )

        # TP > 1 has larger diffs due to fused-vs-separate matmuls + Sequence Parallel
        atol = 1e-3 if world_size == 1 else 1e-2
        torch.testing.assert_close(
            vllm_combined.float(),
            tt_flat.float(),
            rtol=atol,
            atol=atol,
            msg="Layer 0 output mismatch",
        )

    @torch.inference_mode()
    def test_forward_end_to_end(self, ctx: SetupContext):
        """Test full forward pass + logits match."""
        world_size, rank, local_rank = get_distributed_info()
        device = f"cuda:{local_rank}"

        # Run vLLM native forward
        reset_kv_caches(ctx.vllm_config, ctx.num_kv_heads_local, ctx.head_dim, device)
        with set_forward_context(ctx.attn_metadata, ctx.vllm_config):
            vllm_hidden = ctx.vllm_native_model(
                input_ids=ctx.tokens, positions=ctx.positions
            )
            vllm_logits = ctx.vllm_native_model.compute_logits(vllm_hidden)

        # Run TorchTitan forward
        reset_kv_caches(ctx.vllm_config, ctx.num_kv_heads_local, ctx.head_dim, device)
        with set_forward_context(ctx.attn_metadata, ctx.vllm_config):
            tt_hidden = ctx.torchtitan_vllm_model(
                input_ids=ctx.tokens, positions=ctx.positions
            )
            tt_logits = ctx.torchtitan_vllm_model.compute_logits(tt_hidden)

        if world_size == 1:
            # TP=1: no collective communication difference, expect exact match
            torch.testing.assert_close(
                vllm_logits.float(),
                tt_logits.float(),
                rtol=1e-3,
                atol=1e-3,
                msg="End-to-end logits mismatch (TP=1)",
            )
        else:
            # TP > 1: bf16 diffs compound across layers, use cosine similarity
            cos_sim = torch.nn.functional.cosine_similarity(
                vllm_logits.float(), tt_logits.float(), dim=-1
            )
            top1_match = (
                (vllm_logits.argmax(-1) == tt_logits.argmax(-1)).float().mean().item()
            )

            if rank == 0:
                print(
                    f"Logits cosine sim: min={cos_sim.min().item():.4f}, mean={cos_sim.mean().item():.4f}, "
                    f"top-1 agreement: {top1_match:.1%}"
                )

            assert (
                cos_sim.min().item() > 0.90
            ), f"Cosine similarity too low: {cos_sim.min().item()}"
            assert (
                cos_sim.mean().item() > 0.95
            ), f"Mean cosine similarity too low: {cos_sim.mean().item()}"
