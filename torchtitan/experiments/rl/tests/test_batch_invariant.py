# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Unit tests for batch-invariant mode.

Tests verify that the same sequence produces bit-identical results regardless
of what other sequences are in the batch — mimicking the batcher effect where
the generator produces sequences individually but the trainer processes them
as a batch.

Single-GPU test: run with pytest.
TP=2 test: torchrun --nproc_per_node=2 -m pytest <file>::test_batch_invariance_tp2 -v
TP=4 test: torchrun --nproc_per_node=4 -m pytest <file>::test_batch_invariance_tp4 -v
"""

import os
import unittest

import torch

from torchtitan.experiments.rl.batch_invariant import (
    disable_batch_invariant_mode,
    enable_batch_invariant_mode,
    log_softmax,
    matmul_persistent,
    set_batch_invariant_mode,
)


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class TestBatchInvariantEndToEnd(unittest.TestCase):
    """End-to-end test mimicking the generator->batcher->trainer pipeline."""

    def setUp(self):
        disable_batch_invariant_mode()

    def tearDown(self):
        disable_batch_invariant_mode()

    def test_transformer_layer_batch_invariance(self):
        """A simplified transformer layer produces identical outputs for the
        same sequence regardless of batch composition.

        This mimics the full pipeline:
        1. Generator produces 5 sequences independently
        2. Controller batches them together
        3. Trainer's forward pass (linear -> log_softmax -> linear) processes the batch
        4. Each sequence's output must be identical to processing it alone
        """
        torch.manual_seed(42)
        D = 256
        seq_len = 32

        # Simulated model weights (shared across all forward passes)
        W1 = torch.randn(D, D, device="cuda", dtype=torch.bfloat16)
        W2 = torch.randn(D, D, device="cuda", dtype=torch.bfloat16)

        # 5 sequences from the generator
        sequences = [
            torch.randn(seq_len, D, device="cuda", dtype=torch.bfloat16)
            for _ in range(5)
        ]

        def forward(x: torch.Tensor) -> torch.Tensor:
            """Simplified transformer: linear -> log_softmax -> linear."""
            h = matmul_persistent(x, W1)
            h = log_softmax(h, dim=-1)
            return matmul_persistent(h, W2)

        with set_batch_invariant_mode():
            # Process each sequence alone
            outputs_alone = [forward(seq) for seq in sequences]

            # Process all 5 as a batch
            batched = torch.cat(sequences, dim=0)
            output_batched = forward(batched)

            # Extract each sequence's output from the batch
            for i, out_alone in enumerate(outputs_alone):
                start = i * seq_len
                end = start + seq_len
                out_from_batch = output_batched[start:end]
                assert torch.equal(out_alone, out_from_batch), (
                    f"Sequence {i} max diff: "
                    f"{(out_alone - out_from_batch).abs().max():.6e}"
                )


def _run_batch_invariance_tp(tp_degree: int) -> None:
    """Test batch invariance with TP on a real Qwen3 debug model.

    Verifies that the same sequences produce bit-wise identical outputs
    regardless of what other sequences are in the batch, under the given
    TP degree with NCCL deterministic settings and batch-invariant kernels.

    Args:
        tp_degree: Tensor parallel degree (must match nproc_per_node).
    """
    from unittest.mock import patch

    import torch.distributed as dist

    from torchtitan.config import ParallelismConfig
    from torchtitan.distributed import ParallelDims
    from torchtitan.experiments.rl.models.parallelize import parallelize_qwen3
    from torchtitan.models.qwen3 import model_registry

    # Set NCCL deterministic env vars (same as PolicyTrainer)
    os.environ["NCCL_ALGO"] = "Ring"
    os.environ["NCCL_MIN_NCHANNELS"] = "1"
    os.environ["NCCL_MAX_NCHANNELS"] = "1"
    os.environ["NCCL_PROTO"] = "Simple"
    os.environ["NCCL_COLLNET_ENABLE"] = "0"
    os.environ["NCCL_NVLS_ENABLE"] = "0"

    # Initialize distributed
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl")

    rank = dist.get_rank()
    world_size = dist.get_world_size()
    assert world_size == tp_degree, (
        f"This test requires exactly {tp_degree} GPUs, got {world_size}"
    )

    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)

    # Enable deterministic mode
    enable_batch_invariant_mode()
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    # Build model
    model_spec = model_registry("debugmodel")
    old_dtype = torch.get_default_dtype()
    torch.set_default_dtype(torch.bfloat16)
    with torch.device("meta"):
        model = model_spec.model.build()
    torch.set_default_dtype(old_dtype)

    with patch("torchtitan.distributed.parallel_dims.device_type", "cuda"):
        parallel_dims = ParallelDims(
            dp_replicate=1,
            dp_shard=1,
            cp=1,
            tp=tp_degree,
            pp=1,
            ep=1,
            etp=1,
            world_size=tp_degree,
        )

    parallelize_qwen3(
        model,
        parallel_dims=parallel_dims,
        parallelism=ParallelismConfig(tensor_parallel_degree=tp_degree),
    )

    model.to_empty(device="cuda")
    with torch.no_grad():
        model.init_weights(buffer_device=None)
    model.eval()

    # Create 5 sequences
    vocab_size = 2048  # debugmodel vocab_size
    seq_len = 32
    torch.manual_seed(123)
    sequences = [
        torch.randint(0, vocab_size, (1, seq_len), device="cuda") for _ in range(5)
    ]

    with torch.no_grad():
        # Forward on the first 3 sequences
        batch_3 = torch.cat(sequences[:3], dim=0)
        out_3 = model(batch_3)

        # Forward on all 5 sequences (containing the same first 3)
        batch_5 = torch.cat(sequences, dim=0)
        out_5 = model(batch_5)

    # The first 3 outputs must be bit-wise identical
    for i in range(3):
        assert torch.equal(out_3[i], out_5[i]), (
            f"[Rank {rank}] Sequence {i} NOT bit-wise identical between "
            f"batch-of-3 and batch-of-5. "
            f"Max diff: {(out_3[i] - out_5[i]).abs().max():.6e}"
        )

    if rank == 0:
        print(
            f"PASSED (TP={tp_degree}): "
            "All 3 sequences are bit-wise identical across batches"
        )

    # Cleanup
    disable_batch_invariant_mode()
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    dist.destroy_process_group()


def test_batch_invariance_tp2():
    """Test batch invariance with TP=2 on a real Qwen3 debug model.

    Run with: torchrun --nproc_per_node=2 -m pytest
        torchtitan/experiments/rl/tests/test_batch_invariant.py::test_batch_invariance_tp2 -v
    """
    _run_batch_invariance_tp(tp_degree=2)


def test_batch_invariance_tp4():
    """Test batch invariance with TP=4 on a real Qwen3 debug model.

    Exercises NCCL all-reduce across 4 ranks with deterministic settings
    (Ring algorithm, 1 channel, Simple protocol).

    Run with: torchrun --nproc_per_node=4 -m pytest
        torchtitan/experiments/rl/tests/test_batch_invariant.py::test_batch_invariance_tp4 -v
    """
    _run_batch_invariance_tp(tp_degree=4)


if __name__ == "__main__":
    unittest.main()
