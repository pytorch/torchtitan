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
TP test: torchrun --nproc_per_node=<N> -m pytest <file>::TestBatchInvariant::test_forward_invariance_tp -v
"""

import os
import unittest
from unittest.mock import patch

import torch
import torch.distributed as dist

from torchtitan.config import TORCH_DTYPE_MAP
from torchtitan.config.configs import DebugConfig
from torchtitan.experiments.rl.actors.utils import build_varlen_metadata
from torchtitan.experiments.rl.batch_invariant import enable_batch_invariant_mode
from torchtitan.tools.utils import set_default_dtype

_SEQ_LEN = 256


def _build_debug_model(
    debug: DebugConfig,
    *,
    dtype: str = "bfloat16",
    device: str = "cuda",
):
    """Build a Qwen3 debugmodel with random weights on the given device.

    Mirrors PolicyTrainer._build_model() — uses DebugConfig to drive
    batch_invariant and training dtype from config instead of hardcoding.
    """
    from torchtitan.models.common.attention import VarlenAttention
    from torchtitan.models.qwen3 import model_registry

    model_spec = model_registry("debugmodel", attn_backend_override="varlen")

    if debug.batch_invariant_mode:
        assert isinstance(
            model_spec.model.layer.attention.inner_attention,
            VarlenAttention.Config,
        ), "Only varlen attention backend is allowed."
        model_spec.model.layer.attention.inner_attention.batch_invariant = True

    with torch.device("meta"):
        with set_default_dtype(TORCH_DTYPE_MAP[dtype]):
            model = model_spec.model.build()

    model.to_empty(device=device)
    torch.manual_seed(42)
    with torch.no_grad():
        model.init_weights(buffer_device=None)
    return model, model_spec.model.vocab_size


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class TestBatchInvariant(unittest.TestCase):
    """Test batch invariance on Qwen3 debugmodel."""

    def setUp(self):
        enable_batch_invariant_mode()
        self.debug = DebugConfig(batch_invariant_mode=True, deterministic=True)

    def test_forward_invariance(self):
        """The same sequence produces bit-identical logits regardless of
        what other sequences are in the batch.

        Runs the full Qwen3 debugmodel forward on 5 sequences individually,
        then as a batch of 5, and checks that outputs match bitwise.
        """
        enable_batch_invariant_mode()

        model, vocab_size = _build_debug_model(self.debug)
        model.eval()

        torch.manual_seed(123)
        sequences = [
            torch.randint(0, vocab_size, (1, _SEQ_LEN), device="cuda") for _ in range(5)
        ]

        with torch.no_grad():
            outputs_alone = [
                model(
                    seq,
                    attention_masks=build_varlen_metadata([(seq[0], 0, 0)], seq.device),
                )
                for seq in sequences
            ]

            batched = torch.cat(sequences, dim=0)
            batch_meta = build_varlen_metadata(
                [(seq[0], 0, 0) for seq in sequences], batched.device
            )
            output_batched = model(batched, attention_masks=batch_meta)

        for i, out_alone in enumerate(outputs_alone):
            out_from_batch = output_batched[i : i + 1]
            assert torch.equal(out_alone, out_from_batch), (
                f"Sequence {i} max diff: "
                f"{(out_alone - out_from_batch).abs().max():.6e}"
            )

    @unittest.skipUnless(
        dist.is_initialized() or "RANK" in os.environ,
        "requires torchrun launcher",
    )
    def test_forward_invariance_tp(self):
        """Test batch invariance with tensor parallelism.

        Run with: torchrun --nproc_per_node=<N> -m pytest <file>::TestBatchInvariant::test_forward_invariance_tp -v
        """
        from torchtitan.config import ParallelismConfig
        from torchtitan.distributed import ParallelDims
        from torchtitan.experiments.rl.models.parallelize import parallelize_qwen3

        if not dist.is_initialized():
            dist.init_process_group(backend="nccl")

        rank = dist.get_rank()
        world_size = dist.get_world_size()

        local_rank = int(os.environ.get("LOCAL_RANK", rank))
        device = torch.device(f"cuda:{local_rank}")
        torch.cuda.set_device(device)

        debug = DebugConfig(batch_invariant_mode=True, deterministic=True)
        enable_batch_invariant_mode()

        model, vocab_size = _build_debug_model(debug)

        # tp_degree == world_size (all GPUs used for TP)
        with patch("torchtitan.distributed.parallel_dims.device_type", "cuda"):
            parallel_dims = ParallelDims(
                dp_replicate=1,
                dp_shard=1,
                cp=1,
                tp=world_size,
                pp=1,
                ep=1,
                etp=1,
                world_size=world_size,
            )

        parallelize_qwen3(
            model,
            parallel_dims=parallel_dims,
            parallelism=ParallelismConfig(tensor_parallel_degree=world_size),
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
            meta_3 = build_varlen_metadata(
                [(seq[0], 0, 0) for seq in sequences[:3]], batch_3.device
            )
            out_3 = model(batch_3, attention_masks=meta_3)

            batch_5 = torch.cat(sequences, dim=0)
            meta_5 = build_varlen_metadata(
                [(seq[0], 0, 0) for seq in sequences], batch_5.device
            )
            out_5 = model(batch_5, attention_masks=meta_5)

        for i in range(3):
            assert torch.equal(out_3[i], out_5[i]), (
                f"[Rank {rank}] Sequence {i} NOT bit-wise identical between "
                f"batch-of-3 and batch-of-5. "
                f"Max diff: {(out_3[i] - out_5[i]).abs().max():.6e}"
            )

        dist.destroy_process_group()


if __name__ == "__main__":
    unittest.main()
