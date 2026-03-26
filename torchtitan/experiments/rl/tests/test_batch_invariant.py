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
TP test: torchrun --nproc_per_node=<N> -m pytest <file>::test_batch_invariance_tp<N> -v
"""

import os
import unittest

import torch

from torchtitan.experiments.rl.batch_invariant import (
    disable_batch_invariant_mode,
    enable_batch_invariant_mode,
)

_SEQ_LEN = 32


def _build_debug_model(device="cuda"):
    """Build a Qwen3 debugmodel with random weights on the given device."""
    from torchtitan.models.qwen3 import model_registry

    model_spec = model_registry("debugmodel")
    old_dtype = torch.get_default_dtype()
    torch.set_default_dtype(torch.bfloat16)
    with torch.device("meta"):
        model = model_spec.model.build()
    torch.set_default_dtype(old_dtype)

    model.to_empty(device=device)
    torch.manual_seed(42)
    with torch.no_grad():
        model.init_weights(buffer_device=None)
    return model, model_spec.model.vocab_size


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class TestBatchInvariant(unittest.TestCase):
    """Test batch invariance on Qwen3 debugmodel."""

    def setUp(self):
        disable_batch_invariant_mode()

    def tearDown(self):
        disable_batch_invariant_mode()

    def test_forward_invariance(self):
        """The same sequence produces bit-identical logits regardless of
        what other sequences are in the batch.

        Runs the full Qwen3 debugmodel forward on 5 sequences individually,
        then as a batch of 5, and checks that outputs match bitwise.
        """
        enable_batch_invariant_mode()

        model, vocab_size = _build_debug_model()
        model.eval()

        torch.manual_seed(123)
        sequences = [
            torch.randint(0, vocab_size, (1, _SEQ_LEN), device="cuda") for _ in range(5)
        ]

        with torch.no_grad():
            outputs_alone = [model(seq) for seq in sequences]

            batched = torch.cat(sequences, dim=0)
            output_batched = model(batched)

        for i, out_alone in enumerate(outputs_alone):
            out_from_batch = output_batched[i : i + 1]
            assert torch.equal(out_alone, out_from_batch), (
                f"Sequence {i} max diff: "
                f"{(out_alone - out_from_batch).abs().max():.6e}"
            )


def _run_batch_invariance_tp(tp_degree: int) -> None:
    """Test batch invariance with TP on a real Qwen3 debug model.

    Verifies that the same sequences produce bit-wise identical outputs
    regardless of what other sequences are in the batch, under the given
    TP degree.

    Args:
        tp_degree: Tensor parallel degree (must match nproc_per_node).
    """
    from unittest.mock import patch

    import torch.distributed as dist

    from torchtitan.config import ParallelismConfig
    from torchtitan.distributed import ParallelDims
    from torchtitan.experiments.rl.models.parallelize import parallelize_qwen3

    if not dist.is_initialized():
        dist.init_process_group(backend="nccl")

    rank = dist.get_rank()
    world_size = dist.get_world_size()
    assert (
        world_size == tp_degree
    ), f"This test requires exactly {tp_degree} GPUs, got {world_size}"

    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)

    enable_batch_invariant_mode()

    model, vocab_size = _build_debug_model()

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

    torch.manual_seed(123)
    sequences = [
        torch.randint(0, vocab_size, (1, _SEQ_LEN), device="cuda") for _ in range(5)
    ]

    with torch.no_grad():
        batch_3 = torch.cat(sequences[:3], dim=0)
        out_3 = model(batch_3)

        batch_5 = torch.cat(sequences, dim=0)
        out_5 = model(batch_5)

    for i in range(3):
        assert torch.equal(out_3[i], out_5[i]), (
            f"[Rank {rank}] Sequence {i} NOT bit-wise identical between "
            f"batch-of-3 and batch-of-5. "
            f"Max diff: {(out_3[i] - out_5[i]).abs().max():.6e}"
        )

    disable_batch_invariant_mode()
    dist.destroy_process_group()


def test_batch_invariance_tp2():
    """Run with: torchrun --nproc_per_node=2 -m pytest <file>::test_batch_invariance_tp2 -v"""
    _run_batch_invariance_tp(tp_degree=2)


def test_batch_invariance_tp4():
    """Run with: torchrun --nproc_per_node=4 -m pytest <file>::test_batch_invariance_tp4 -v"""
    _run_batch_invariance_tp(tp_degree=4)


if __name__ == "__main__":
    unittest.main()
