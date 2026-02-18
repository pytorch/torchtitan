# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Tests for vLLM integration with TorchTitan models.

This script tests vLLM running TorchTitan model vs vLLM running native Qwen3 model.
Both models use vLLM's attention and forward context, and we compare their outputs.

Usage:
    # Run with TP=1 on a single GPU (default checkpoint: Qwen/Qwen3-0.6B)
    torchrun --nproc_per_node=1 -m pytest \
        torchtitan/experiments/rl/unified/tests/test_numerics.py -v -s

    # Run with TP=2 on 2 GPUs
    torchrun --nproc_per_node=2 -m pytest \
        torchtitan/experiments/rl/unified/tests/test_numerics.py -v -s

    # Specify a custom model checkpoint using environment variable
    MODEL_CHECKPOINT_PATH=/path/to/checkpoint torchrun --nproc_per_node=1 \
        -m pytest torchtitan/experiments/rl/unified/tests/test_numerics.py -v -s

    # Use a different HuggingFace model
    MODEL_CHECKPOINT_PATH=Qwen/Qwen3-1.7B torchrun --nproc_per_node=2 \
        -m pytest torchtitan/experiments/rl/unified/tests/test_numerics.py -v -s
"""

import glob
import os
from dataclasses import dataclass

import pytest
import torch
from safetensors.torch import load_file as load_safetensors

# Import vLLM (skip tests if not available)
vllm = pytest.importorskip("vllm")

from torch.distributed._tensor import DTensor
from torchtitan.experiments.rl.unified.infra.parallelize import parallelize_qwen3
from torchtitan.experiments.rl.unified.models.vllm_wrapper import (
    TorchTitanVLLMModelWrapper,
)

# TorchTitan imports
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
from vllm.v1.attention.backend import CommonAttentionMetadata
from vllm.v1.attention.backends.flash_attn import FlashAttentionMetadata

BLOCK_SIZE = 32

# Track if distributed environment has been initialized
_distributed_initialized = False


def get_distributed_info():
    """Get distributed environment info from environment variables."""
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    rank = int(os.environ.get("RANK", 0))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    return world_size, rank, local_rank


def ensure_distributed_initialized():
    """Initialize distributed environment if not already done."""
    global _distributed_initialized

    world_size, rank, local_rank = get_distributed_info()

    # Check if already initialized
    if model_parallel_is_initialized() or _distributed_initialized:
        return

    # Use env:// method which is compatible with torchrun
    # torchrun sets MASTER_ADDR and MASTER_PORT
    os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"

    init_distributed_environment(
        world_size=world_size,
        rank=rank,
        distributed_init_method="env://",
        local_rank=local_rank,
        backend="nccl",
    )
    initialize_model_parallel(world_size, 1)  # TP=world_size, PP=1
    _distributed_initialized = True

    if rank == 0:
        print(f"Initialized distributed environment: TP={world_size}")


@dataclass
class TestContext:
    """Context object holding setup results for tests."""

    vllm_native_model: torch.nn.Module
    torchtitan_vllm_model: torch.nn.Module
    vllm_config: object
    tokens: torch.Tensor
    positions: torch.Tensor
    attn_metadata: FlashAttentionMetadata
    model_dim: int
    vocab_size: int
    num_heads: int
    num_kv_heads: int
    num_layers: int
    head_dim: int
    hidden_dim: int
    seq_len: int
    q_size: int
    kv_size: int
    # Local (sharded) sizes for TP > 1
    num_kv_heads_local: int
    world_size: int


def setup_models_and_inputs(
    model_checkpoint_path: str = "Qwen/Qwen3-0.6B",
    seq_len: int = 8,
) -> TestContext:
    """
    Set up vLLM native and TorchTitan models with weights loaded from the same checkpoint.

    This function:
    1. Initializes distributed environment
    2. Creates vLLM config and loads native Qwen3 model with weights from checkpoint
    3. Creates TorchTitan model with vLLM wrapper and loads weights from same checkpoint
    4. Creates input tokens, positions, and attention metadata
    5. Sets up KV caches

    Args:
        model_checkpoint_path: Path to HuggingFace checkpoint (local path or HF hub model name)
        seq_len: Sequence length for test inputs

    Returns:
        TestContext containing both models, inputs, and configuration.
    """
    # Setup distributed environment
    ensure_distributed_initialized()

    # Get world size for tensor parallel
    world_size, rank, local_rank = get_distributed_info()
    device = f"cuda:{local_rank}"

    # Create vLLM EngineArgs for native Qwen3 model
    engine_args = EngineArgs(
        model=model_checkpoint_path,
        skip_tokenizer_init=True,
        enforce_eager=True,
        gpu_memory_utilization=0.1,
        tensor_parallel_size=world_size,
        dtype="bfloat16",
    )

    vllm_config = engine_args.create_engine_config(UsageContext.ENGINE_CONTEXT)

    # Extract model dimensions from vLLM model config
    hf_config = vllm_config.model_config.hf_config
    model_dim = hf_config.hidden_size
    vocab_size = hf_config.vocab_size
    num_heads = hf_config.num_attention_heads
    num_kv_heads = hf_config.num_key_value_heads
    num_layers = hf_config.num_hidden_layers
    head_dim = getattr(hf_config, "head_dim", model_dim // num_heads)
    hidden_dim = hf_config.intermediate_size

    if rank == 0:
        print(f"\n=== Model dimensions from {model_checkpoint_path} ===")
        print(
            f"model_dim={model_dim}, vocab_size={vocab_size}, num_heads={num_heads}, "
            f"num_kv_heads={num_kv_heads}, num_layers={num_layers}, hidden_dim={hidden_dim}, "
            f"head_dim={head_dim}, TP={world_size}"
        )

    # Create TorchTitan model args matching vLLM model config
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

    # Load vLLM native Qwen3 model
    if rank == 0:
        print("\n=== Loading vLLM native Qwen3 model ===")
    loader = get_model_loader(vllm_config.load_config)
    vllm_native_model = loader.load_model(vllm_config, vllm_config.model_config)
    if rank == 0:
        print(f"vLLM native model: {type(vllm_native_model)}")

    # Create TorchTitan model with vLLM wrapper
    if rank == 0:
        print("\n=== Creating TorchTitan model with vLLM wrapper ===")
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
    # We do this on the vLLM native model since it uses regular tensors (not DTensor)
    replace_vllm_rmsnorm_with_torch(vllm_native_model, rank)

    # Load weights into TorchTitan model from the same checkpoint
    if rank == 0:
        print("\n=== Loading weights into TorchTitan model from checkpoint ===")

    # Load HF weights from checkpoint and create iterator for load_weights
    def load_hf_weights_iter(checkpoint_path: str):
        """Load HF checkpoint weights and yield (name, tensor) pairs."""
        if os.path.isdir(checkpoint_path):
            model_path = checkpoint_path
        else:
            # Download from HuggingFace Hub
            from huggingface_hub import snapshot_download

            model_path = snapshot_download(checkpoint_path)

        # Find all safetensors files
        safetensor_files = glob.glob(os.path.join(model_path, "*.safetensors"))
        if not safetensor_files:
            raise ValueError(f"No safetensors files found in {model_path}")

        if rank == 0:
            print(f"Found {len(safetensor_files)} safetensors files in {model_path}")

        # Load and yield weights from each file
        for st_file in sorted(safetensor_files):
            weights = load_safetensors(st_file)
            for name, tensor in weights.items():
                yield name, tensor

    # Load weights using TorchTitanVLLMModelWrapper's load_weights method
    weights_iter = load_hf_weights_iter(model_checkpoint_path)
    loaded_params = torchtitan_vllm_model.load_weights(weights_iter)
    if rank == 0:
        print(f"Loaded {len(loaded_params)} parameters into TorchTitan model")

    # Calculate local sizes for TP (used in TestContext)
    q_size = num_heads * head_dim // world_size
    kv_size = num_kv_heads * head_dim // world_size

    # Create input tokens - generate on CPU first to ensure same tokens across all ranks
    torch.manual_seed(42)
    tokens = torch.randint(0, vocab_size, (seq_len,), device="cpu").to(device)
    positions = torch.arange(seq_len, device=device)
    if rank == 0:
        print(f"\nInput tokens shape: {tokens.shape}")

    # Build attention metadata
    attn_metadata = build_attn_metadata(seq_len, local_rank)

    # Bind KV caches for all attention layers in static_forward_context
    # When TP > 1, KV cache uses sharded num_kv_heads (each rank stores its portion)
    num_blocks = 30
    num_kv_heads_local = num_kv_heads // world_size
    static_ctx = vllm_config.compilation_config.static_forward_context
    if rank == 0:
        print(
            f"\n=== Setting up KV caches (num_kv_heads_local={num_kv_heads_local}) ==="
        )

    for layer_name, layer in static_ctx.items():
        if hasattr(layer, "kv_cache"):
            kv_cache = torch.zeros(
                (2, num_blocks, BLOCK_SIZE, num_kv_heads_local, head_dim),
                dtype=torch.bfloat16,
                device=device,
            )
            layer.kv_cache = [kv_cache]

    if rank == 0:
        print("=== Setup complete ===\n")

    return TestContext(
        vllm_native_model=vllm_native_model,
        torchtitan_vllm_model=torchtitan_vllm_model,
        vllm_config=vllm_config,
        tokens=tokens,
        positions=positions,
        attn_metadata=attn_metadata,
        model_dim=model_dim,
        vocab_size=vocab_size,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        num_layers=num_layers,
        head_dim=head_dim,
        hidden_dim=hidden_dim,
        seq_len=seq_len,
        q_size=q_size,
        kv_size=kv_size,
        num_kv_heads_local=num_kv_heads_local,
        world_size=world_size,
    )


class TorchRMSNormWrapper(torch.nn.Module):
    """
    Wrapper around torch.nn.RMSNorm that mimics vLLM's RMSNorm interface.

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
    """
    Replace vLLM's RMSNorm in the vLLM native model with TorchRMSNormWrapper.

    This ensures numerical equivalence between TorchTitan and vLLM models
    by making both use the same RMSNorm implementation.

    We replace vLLM's RMSNorm (which uses custom CUDA kernels) with a wrapper
    around torch.nn.RMSNorm that mimics vLLM's interface.
    """
    from vllm.model_executor.layers.layernorm import RMSNorm as VLLMRMSNorm

    replaced_count = 0

    def replace_in_module(parent_module, parent_name=""):
        nonlocal replaced_count
        for name, child in list(parent_module.named_children()):
            full_name = f"{parent_name}.{name}" if parent_name else name

            # Check if this is vLLM's RMSNorm
            if isinstance(child, VLLMRMSNorm):
                # Get the original parameters
                hidden_size = child.weight.shape[0]
                eps = child.variance_epsilon
                weight = child.weight.data.clone()

                # Create TorchRMSNormWrapper
                torch_norm = TorchRMSNormWrapper(hidden_size, eps=eps)
                torch_norm.norm.weight.data.copy_(weight)
                torch_norm = torch_norm.to(
                    device=child.weight.device, dtype=child.weight.dtype
                )

                # Replace the module
                setattr(parent_module, name, torch_norm)
                replaced_count += 1

            else:
                # Recurse into child modules
                replace_in_module(child, full_name)

    replace_in_module(model)

    if rank == 0:
        print(f"Replaced {replaced_count} vLLM RMSNorm layers with TorchRMSNormWrapper")


# Reference: https://github.com/vllm-project/vllm/blob/58a05b0ca184048e11ea1a54759732ad634ce34a/tests/v1/attention/utils.py#L31
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


def build_attn_metadata(num_tokens: int, local_rank: int = 0) -> FlashAttentionMetadata:
    """Build FlashAttentionMetadata for a given number of tokens."""
    batch_spec = BatchSpec(seq_lens=[num_tokens], query_lens=[num_tokens])
    common_attn_metadata = create_common_attn_metadata(
        batch_spec,
        block_size=BLOCK_SIZE,
        device=torch.device(f"cuda:{local_rank}"),
        arange_block_indices=True,
    )
    return FlashAttentionMetadata(
        causal=common_attn_metadata.causal,
        num_actual_tokens=num_tokens,
        max_query_len=num_tokens,
        query_start_loc=common_attn_metadata.query_start_loc,
        max_seq_len=num_tokens,
        seq_lens=common_attn_metadata.seq_lens,
        block_table=common_attn_metadata.block_table_tensor,
        slot_mapping=common_attn_metadata.slot_mapping,
        use_cascade=False,
        common_prefix_len=0,
        cu_prefix_query_lens=None,
        prefix_kv_lens=None,
        suffix_kv_lens=None,
    )


def create_debug_model_args(
    model_dim: int = 64,
    vocab_size: int = 128,
    num_heads: int = 4,
    num_kv_heads: int = 4,
    num_layers: int = 2,
    max_seq_len: int = 32,
) -> Qwen3ModelArgs:
    """Create a small debug model args for testing."""
    head_dim = model_dim // num_heads
    hidden_dim = model_dim * 4  # Standard 4x multiplier for FFN

    return Qwen3ModelArgs(
        dim=model_dim,
        n_layers=num_layers,
        n_heads=num_heads,
        n_kv_heads=num_kv_heads,
        vocab_size=vocab_size,
        head_dim=head_dim,
        hidden_dim=hidden_dim,
        norm_eps=1e-6,
        rope_theta=10000.0,
        max_seq_len=max_seq_len,
        qk_norm=True,
        depth_init=False,
        attn_type="sdpa",
        attn_mask_type="causal",
    )


# Default model checkpoint path
DEFAULT_MODEL_CHECKPOINT = "Qwen/Qwen3-0.6B"


def get_model_checkpoint_path():
    """
    Get the model checkpoint path from environment variable or use default.
    """
    env_value = os.environ.get("MODEL_CHECKPOINT_PATH")
    if env_value:
        return env_value
    return DEFAULT_MODEL_CHECKPOINT


class TestVLLMTorchTitan:
    """Test class for vLLM integration with TorchTitan models.
    Comparing TorchTitanVLLMModelWrapper v.s. vllm native Qwen3 model"""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Setup for each test."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        # Set default dtype to bfloat16 for vLLM compatibility
        torch.set_default_dtype(torch.bfloat16)

    @pytest.fixture(scope="class")
    def ctx(self):
        """Shared test context fixture that sets up models once per class."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        torch.set_default_dtype(torch.bfloat16)
        model_checkpoint_path = get_model_checkpoint_path()
        return setup_models_and_inputs(model_checkpoint_path=model_checkpoint_path)

    @torch.inference_mode()
    def test_forward_embedding_and_transformer_layer(self, ctx: TestContext):
        """
        Test forward pass of embedding layer + first transformer layer.

        Compares hidden states after embedding and first transformer layer.
        """
        world_size, rank, local_rank = get_distributed_info()

        if rank == 0:
            print("\n" + "=" * 70)
            print(
                f"=== TEST: Embedding + Transformer Layer Forward Pass (TP={world_size}) ==="
            )
            print("=" * 70)

        # Run vLLM native embedding + first layer
        with set_forward_context(ctx.attn_metadata, ctx.vllm_config):
            vllm_embeddings = ctx.vllm_native_model.model.embed_tokens(ctx.tokens)

            residual = None
            hidden_states = vllm_embeddings.clone()
            hidden_states, residual = ctx.vllm_native_model.model.layers[0](
                positions=ctx.positions,
                hidden_states=hidden_states,
                residual=residual,
            )
            vllm_layer0_output_hidden = hidden_states
            vllm_layer0_output_residual = residual

        # Run TorchTitan embedding + first layer
        with set_forward_context(ctx.attn_metadata, ctx.vllm_config):
            tokens_2d = ctx.tokens.unsqueeze(0)
            positions_2d = ctx.positions.unsqueeze(0)
            tt_embeddings = ctx.torchtitan_vllm_model.model.tok_embeddings(tokens_2d)
            if isinstance(tt_embeddings, DTensor):
                tt_embeddings_local = tt_embeddings.full_tensor()
            else:
                tt_embeddings_local = tt_embeddings

            rope_cache = ctx.torchtitan_vllm_model.model.rope_cache
            if isinstance(rope_cache, DTensor):
                rope_cache = rope_cache.to_local()

            h = tt_embeddings.clone()
            # Must pass all args positionally to match PrepareModuleInput layouts
            h = ctx.torchtitan_vllm_model.model.layers["0"](
                h, rope_cache, None, positions_2d
            )
            if isinstance(h, DTensor):
                tt_layer0_output = h.full_tensor()
            else:
                tt_layer0_output = h

        # Compare outputs
        vllm_embed_flat = vllm_embeddings.view(-1, ctx.model_dim)
        tt_embed_flat = tt_embeddings_local.view(-1, ctx.model_dim)
        torch.testing.assert_close(
            vllm_embed_flat.float(),
            tt_embed_flat.float(),
            rtol=1e-2,
            atol=1e-2,
            msg="Embedding outputs: vLLM vs TorchTitan mismatch",
        )

        # vLLM uses pre-RMSNorm residual pattern
        vllm_combined = vllm_layer0_output_hidden + vllm_layer0_output_residual
        vllm_combined_flat = vllm_combined.view(-1, ctx.model_dim)
        tt_layer0_flat = tt_layer0_output.view(-1, ctx.model_dim)

        torch.testing.assert_close(
            vllm_combined_flat.float(),
            tt_layer0_flat.float(),
            rtol=1e-3,
            atol=1e-3,
            msg="Layer 0 outputs (vLLM hidden+residual vs TorchTitan) mismatch",
        )

    @torch.inference_mode()
    def test_forward_attention(self, ctx: TestContext):
        """
        Test forward pass of the attention model only.

        NOTE: This test is skipped for TP > 1 because testing individual parallelized
        components directly is complex - the PrepareModuleInput hooks interpret local
        tensors as shards and transform them unexpectedly.
        """
        world_size, rank, local_rank = get_distributed_info()

        # Skip for TP > 1 - testing individual parallelized attention is complex
        if world_size > 1:
            if rank == 0:
                print("Skipping test_forward_attention for TP > 1")
            return

        if rank == 0:
            print("\n" + "=" * 70)
            print(f"=== TEST: Attention Model Forward Pass (TP={world_size}) ===")
            print("=" * 70)

        # Get attention modules
        vllm_attn = ctx.vllm_native_model.model.layers[0].self_attn
        tt_attn = ctx.torchtitan_vllm_model.model.layers["0"].attention

        # Create a fixed input for attention comparison
        torch.manual_seed(123)
        device = f"cuda:{local_rank}"
        attn_input = torch.randn(
            ctx.seq_len, ctx.model_dim, dtype=torch.bfloat16, device=device
        )
        attn_positions = torch.arange(ctx.seq_len, device=device)

        # Get static forward context for KV cache
        static_ctx = ctx.vllm_config.compilation_config.static_forward_context

        # Clear KV cache before vLLM forward
        num_blocks = 30
        for layer_name, layer in static_ctx.items():
            if hasattr(layer, "kv_cache"):
                kv_cache = torch.zeros(
                    (2, num_blocks, BLOCK_SIZE, ctx.num_kv_heads_local, ctx.head_dim),
                    dtype=torch.bfloat16,
                    device=device,
                )
                layer.kv_cache = [kv_cache]

        # vLLM Attention Forward
        with set_forward_context(ctx.attn_metadata, ctx.vllm_config):
            # Full attention forward
            vllm_attn_output = vllm_attn(
                positions=attn_positions,
                hidden_states=attn_input.clone(),
            )

        # Clear KV cache before TorchTitan forward
        for layer_name, layer in static_ctx.items():
            if hasattr(layer, "kv_cache"):
                kv_cache = torch.zeros(
                    (2, num_blocks, BLOCK_SIZE, ctx.num_kv_heads_local, ctx.head_dim),
                    dtype=torch.bfloat16,
                    device=device,
                )
                layer.kv_cache = [kv_cache]

        # TorchTitan expects [batch, seq, hidden] format
        tt_attn_input = attn_input.unsqueeze(0)
        tt_attn_positions = attn_positions.unsqueeze(0)

        # TorchTitan Attention Forward
        with set_forward_context(ctx.attn_metadata, ctx.vllm_config):
            # Get rope cache
            rope_cache = ctx.torchtitan_vllm_model.model.rope_cache
            if isinstance(rope_cache, DTensor):
                rope_cache = rope_cache.to_local()

            # Full attention forward. Qwen3.models.model.Attention
            # Must pass all 4 args positionally to match PrepareModuleInput layouts
            tt_attn_output = tt_attn(
                tt_attn_input.clone(),
                rope_cache,
                None,  # attention_masks
                tt_attn_positions,  # positions
            )

            if isinstance(tt_attn_output, DTensor):
                tt_attn_output_local = tt_attn_output.full_tensor().squeeze(0)
            else:
                tt_attn_output_local = tt_attn_output.squeeze(0)

        # Print comparison table with mean of diff for each token position
        if rank == 0:
            print("\n" + "=" * 50)
            print("vLLM vs TorchTitan Attention Diff (per token)")
            print("=" * 50)
            print(f"{'Token':<12}{'Mean(Diff)':>18}{'Max(|Diff|)':>18}")
            print("-" * 50)
            for i in range(ctx.seq_len):
                token_diff = (
                    vllm_attn_output[i, :] - tt_attn_output_local[i, :]
                ).float()
                mean_diff = token_diff.mean().item()
                max_abs_diff = token_diff.abs().max().item()
                print(f"{'Token ' + str(i):<12}{mean_diff:>18.9f}{max_abs_diff:>18.9f}")
            print("=" * 50)

        torch.testing.assert_close(
            vllm_attn_output.float(),
            tt_attn_output_local.float(),
            rtol=1e-3,
            atol=1e-3,
            msg="Attention output: vLLM vs TorchTitan mismatch",
        )

    @torch.inference_mode()
    def test_forward_end_to_end(self, ctx: TestContext):
        """
        Test end-to-end forward pass comparing vLLM native vs TorchTitan.

        This runs the full model forward pass on both models and compares
        the final logits output.
        """
        world_size, rank, local_rank = get_distributed_info()

        if rank == 0:
            print("\n" + "=" * 70)
            print(f"=== TEST: End-to-End Forward Pass (TP={world_size}) ===")
            print("=" * 70)

        # Run vLLM native forward
        if rank == 0:
            print("\n=== Running vLLM native forward ===")
        with set_forward_context(ctx.attn_metadata, ctx.vllm_config):
            vllm_native_hidden = ctx.vllm_native_model(ctx.tokens, ctx.positions)
            vllm_native_logits = ctx.vllm_native_model.compute_logits(
                vllm_native_hidden
            )
        if rank == 0:
            print(f"vLLM native hidden shape: {vllm_native_hidden.shape}")
            print(f"vLLM native logits shape: {vllm_native_logits.shape}")

        # Run TorchTitan vLLM wrapper forward
        if rank == 0:
            print("\n=== Running TorchTitan vLLM wrapper forward ===")
        with set_forward_context(ctx.attn_metadata, ctx.vllm_config):
            torchtitan_hidden = ctx.torchtitan_vllm_model(ctx.tokens, ctx.positions)
            torchtitan_logits = ctx.torchtitan_vllm_model.compute_logits(
                torchtitan_hidden
            )

        # Handle DTensor output
        if isinstance(torchtitan_hidden, DTensor):
            torchtitan_hidden = torchtitan_hidden.full_tensor()
        if isinstance(torchtitan_logits, DTensor):
            torchtitan_logits = torchtitan_logits.full_tensor()

        if rank == 0:
            print(f"TorchTitan hidden shape: {torchtitan_hidden.shape}")
            print(f"TorchTitan logits shape: {torchtitan_logits.shape}")

        # Compare outputs
        if rank == 0:
            print("\n=== Comparing outputs ===")
            print(
                f"vLLM native hidden stats: min={vllm_native_hidden.min():.4f}, "
                f"max={vllm_native_hidden.max():.4f}, mean={vllm_native_hidden.float().mean():.4f}"
            )
            print(
                f"TorchTitan hidden stats: min={torchtitan_hidden.min():.4f}, "
                f"max={torchtitan_hidden.max():.4f}, mean={torchtitan_hidden.float().mean():.4f}"
            )
            print(
                f"vLLM native logits stats: min={vllm_native_logits.min():.4f}, "
                f"max={vllm_native_logits.max():.4f}, mean={vllm_native_logits.float().mean():.4f}"
            )
            print(
                f"TorchTitan logits stats: min={torchtitan_logits.min():.4f}, "
                f"max={torchtitan_logits.max():.4f}, mean={torchtitan_logits.float().mean():.4f}"
            )

        # Verify outputs are close
        torch.testing.assert_close(
            vllm_native_logits,
            torchtitan_logits,
            rtol=1e-3,
            atol=1e-3,
        )
        if rank == 0:
            print(
                f"\n=== Test passed! vLLM TorchTitan and vLLM native produce matching outputs (TP={world_size}) ==="
            )
