# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Regression tests for the bincount-based token-count step in MoE
routing and dispatch.

Background: ``TokenChoiceTopKRouter`` (in ``models/common/moe.py``)
and ``LocalTokenDispatcher`` (in ``models/common/token_dispatcher.py``)
both compute ``num_tokens_per_expert`` from the integer expert
indices returned by ``torch.topk``. The implementation uses
``torch.bincount`` rather than ``torch.histc`` so that runs under
``torch.use_deterministic_algorithms(True)`` work on backends that
lack a deterministic ``_histc`` kernel (e.g. Intel XPU).

These tests guard the equivalence with ``histc`` (which was the
previous implementation) so a future refactor doesn't silently
regress to the non-deterministic path.
"""

from __future__ import annotations

import unittest

import torch


class TestMoeRoutingCountingEquivalence(unittest.TestCase):
    """``bincount`` is contractually equivalent to ``histc`` for the
    router/dispatcher input shape (integer indices in ``[0, N)``)."""

    def _make_indices(self, num_tokens: int, top_k: int, num_experts: int):
        torch.manual_seed(0)
        return torch.randint(0, num_experts, (num_tokens, top_k))

    def _check(self, num_tokens: int, top_k: int, num_experts: int):
        idx = self._make_indices(num_tokens, top_k, num_experts).view(-1)
        # histc requires floating dtype; bincount requires integer.
        histc_counts = torch.histc(
            idx.float(), bins=num_experts, min=0, max=num_experts
        ).long()
        bincount_counts = torch.bincount(idx, minlength=num_experts)
        self.assertEqual(histc_counts.shape, bincount_counts.shape)
        self.assertTrue(torch.equal(histc_counts, bincount_counts))
        # Total must equal num_tokens * top_k.
        self.assertEqual(
            bincount_counts.sum().item(), num_tokens * top_k
        )

    def test_typical_router_shape(self):
        # 8K tokens × top_2 over 32 experts — DeepSeek-V3-ish.
        self._check(num_tokens=8192, top_k=2, num_experts=32)

    def test_small_routing(self):
        # Small case to exercise edge bins.
        self._check(num_tokens=16, top_k=1, num_experts=4)

    def test_large_expert_count(self):
        self._check(num_tokens=2048, top_k=4, num_experts=128)


class TestMoeRoutingUnderDeterministicAlgorithms(unittest.TestCase):
    """``torch.bincount`` (no weights) must not raise under
    ``torch.use_deterministic_algorithms(True)`` on CPU or XPU —
    that's the motivating reason for using it over ``torch.histc``.

    Note on CUDA: ``torch.bincount(..., weights=tensor)`` does raise
    under deterministic algorithms on CUDA, but the router/dispatcher
    call site here uses no weights, which is exempt. We don't assert
    no-raise on CUDA here because (a) the swap is motivated by XPU,
    not CUDA, where ``histc`` was already deterministic, and (b)
    upstream may tighten the bincount determinism rules in the
    future. CPU + XPU coverage is sufficient.
    """

    def _sync_device(self, device: torch.device) -> None:
        # Avoid ``torch.accelerator.synchronize`` (newer API not present
        # in all supported PyTorch versions); dispatch to the
        # device-specific synchronize that has shipped for years.
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        elif device.type == "xpu":
            torch.xpu.synchronize(device)

    def _run_no_raise_check(self, device: torch.device) -> None:
        torch.manual_seed(0)
        idx = torch.randint(0, 32, (8192,), device=device, dtype=torch.int64)
        prev = torch.are_deterministic_algorithms_enabled()
        torch.use_deterministic_algorithms(True)
        try:
            counts = torch.bincount(idx, minlength=32)
            self._sync_device(device)
        finally:
            torch.use_deterministic_algorithms(prev)
        self.assertEqual(counts.shape, (32,))
        self.assertEqual(counts.sum().item(), 8192)

    def test_bincount_does_not_raise_on_cpu(self):
        self._run_no_raise_check(torch.device("cpu"))

    @unittest.skipUnless(
        hasattr(torch, "xpu") and torch.xpu.is_available(),
        "requires Intel XPU",
    )
    def test_bincount_does_not_raise_on_xpu(self):
        self._run_no_raise_check(torch.device("xpu:0"))


if __name__ == "__main__":
    unittest.main()
