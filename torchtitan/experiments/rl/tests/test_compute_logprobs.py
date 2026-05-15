# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Unit tests for compute_logprobs chunked vs non-chunked equivalence.

Run:
    pytest torchtitan/experiments/rl/tests/test_compute_logprobs.py -v
"""

import pytest
import torch

from torchtitan.experiments.rl.actors.utils import compute_logprobs


@pytest.mark.parametrize("chunk_size", [1, 16, 32, 64, 128])
def test_chunked_bitwise_identical(chunk_size):
    """Chunked compute_logprobs produces bit-identical results to non-chunked."""
    torch.manual_seed(42)
    logits = torch.randn(2, 200, 1000)
    token_ids = torch.randint(0, 1000, (2, 200))

    result_full = compute_logprobs(logits, token_ids)
    result_chunked = compute_logprobs(logits, token_ids, chunk_size=chunk_size)

    assert torch.equal(result_full, result_chunked), (
        f"chunk_size={chunk_size}: max delta="
        f"{(result_full - result_chunked).abs().max().item():.2e}"
    )


def test_chunk_size_larger_than_seq():
    """When chunk_size > seq_len, behaves identically to non-chunked."""
    torch.manual_seed(0)
    logits = torch.randn(1, 50, 500)
    token_ids = torch.randint(0, 500, (1, 50))

    result_full = compute_logprobs(logits, token_ids)
    result_chunked = compute_logprobs(logits, token_ids, chunk_size=1024)

    assert torch.equal(result_full, result_chunked)


def test_chunk_size_one():
    """Edge case: chunk_size=1 processes one token at a time."""
    torch.manual_seed(7)
    logits = torch.randn(3, 20, 100)
    token_ids = torch.randint(0, 100, (3, 20))

    result_full = compute_logprobs(logits, token_ids)
    result_chunked = compute_logprobs(logits, token_ids, chunk_size=1)

    assert torch.equal(result_full, result_chunked)


def test_output_shape():
    """Output shape is [batch, seq_len - 1]."""
    logits = torch.randn(4, 100, 256)
    token_ids = torch.randint(0, 256, (4, 100))

    result = compute_logprobs(logits, token_ids)
    assert result.shape == (4, 99)

    result_chunked = compute_logprobs(logits, token_ids, chunk_size=16)
    assert result_chunked.shape == (4, 99)


def test_gradient_flows_through_chunked():
    """Gradients flow correctly through the chunked path."""
    logits = torch.randn(2, 50, 100, requires_grad=True)
    token_ids = torch.randint(0, 100, (2, 50))

    result = compute_logprobs(logits, token_ids, chunk_size=8)
    loss = result.sum()
    loss.backward()

    assert logits.grad is not None
    assert logits.grad.shape == logits.shape
    assert not torch.all(logits.grad == 0)


@pytest.mark.parametrize("bad_chunk_size", [0, -1, -100])
def test_invalid_chunk_size_raises(bad_chunk_size):
    """Non-positive chunk_size raises ValueError."""
    logits = torch.randn(1, 10, 50)
    token_ids = torch.randint(0, 50, (1, 10))

    with pytest.raises(ValueError, match="chunk_size must be positive"):
        compute_logprobs(logits, token_ids, chunk_size=bad_chunk_size)


def test_seq_len_one_chunked():
    """Chunked path handles seq_len=1 (shift produces empty sequence) with grad_fn."""
    logits = torch.randn(2, 1, 50, requires_grad=True)
    token_ids = torch.randint(0, 50, (2, 1))

    result_full = compute_logprobs(logits, token_ids)
    result_chunked = compute_logprobs(logits, token_ids, chunk_size=4)

    assert result_full.shape == (2, 0)
    assert result_chunked.shape == (2, 0)
    assert result_chunked.dtype == result_full.dtype == torch.float32
    assert torch.equal(result_full, result_chunked)
    # Both paths must preserve grad_fn for autograd parity
    assert result_full.grad_fn is not None
    assert result_chunked.grad_fn is not None
    result_chunked.sum().backward()
