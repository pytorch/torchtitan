# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Quantile Balancing -- K3 tech report sec 2.3.3, Eqs. 13-14.

The defining property, and the reason QB exists: it SOLVES for the bias that
gives each expert its target load q = m*k/n, instead of nudging by a step size
whose gamma trades adaptation speed against oscillation. So the tests check the
property, not just the formula.
"""

import unittest

import torch

from torchtitan.components.quantile_balance import (
    expert_loads,
    margin_histogram,
    quantile_balance_bias,
    quantile_balance_bias_histogram,
    topk_with_cutoff,
)


def _skewed_scores(T=512, E=8, seed=0):
    """Router scores with a deliberately imbalanced preference."""
    torch.manual_seed(seed)
    logits = torch.randn(T, E) + torch.linspace(2.0, -2.0, E)  # expert 0 hot
    return torch.sigmoid(logits)


class TestTopKWithCutoff(unittest.TestCase):
    def test_cutoff_is_the_k_plus_1_th_biased_score(self):
        s = _skewed_scores(T=16, E=6)
        b = torch.zeros(6)
        ids, cut = topk_with_cutoff(s, b, top_k=2)
        self.assertEqual(ids.shape, (16, 2))
        srt = (s + b).sort(dim=-1, descending=True).values
        torch.testing.assert_close(cut, srt[:, 2])

    def test_bias_shifts_selection_only(self):
        s = _skewed_scores(T=32, E=6)
        hot = torch.zeros(6)
        hot[0] = -10.0  # suppress the hot expert
        ids_a, _ = topk_with_cutoff(s, torch.zeros(6), top_k=2)
        ids_b, _ = topk_with_cutoff(s, hot, top_k=2)
        self.assertGreater((ids_a == 0).sum().item(), (ids_b == 0).sum().item())

    def test_rejects_k_plus_1_over_num_experts(self):
        with self.assertRaises(ValueError):
            topk_with_cutoff(_skewed_scores(T=4, E=3), torch.zeros(3), top_k=3)


class TestQuantileBalanceProperty(unittest.TestCase):
    def test_bias_drives_loads_to_the_target(self):
        T, E, k = 1024, 8, 2
        s = _skewed_scores(T, E)
        b0 = torch.zeros(E)

        before = expert_loads(s, b0, k).float()
        _, cutoff = topk_with_cutoff(s, b0, k)
        b1 = quantile_balance_bias(s, cutoff, k)
        after = expert_loads(s, b1, k).float()

        target = T * k / E
        # imbalance must shrink a lot in ONE step -- that is the whole point
        self.assertLess(
            (after - target).abs().max().item(),
            0.35 * (before - target).abs().max().item(),
        )

    def test_bias_is_zero_mean(self):
        s = _skewed_scores(T=256, E=8)
        _, cutoff = topk_with_cutoff(s, torch.zeros(8), top_k=2)
        b = quantile_balance_bias(s, cutoff, top_k=2)
        self.assertAlmostEqual(b.mean().item(), 0.0, places=5)

    def test_zero_mean_offset_does_not_change_selection(self):
        # Eq. 14's second line: a common offset leaves Top-k unchanged
        s = _skewed_scores(T=64, E=8)
        b = quantile_balance_bias(s, topk_with_cutoff(s, torch.zeros(8), 2)[1], 2)
        ids_a, _ = topk_with_cutoff(s, b, top_k=2)
        ids_b, _ = topk_with_cutoff(s, b + 3.7, top_k=2)
        torch.testing.assert_close(ids_a, ids_b)

    def test_already_balanced_scores_get_a_near_zero_bias(self):
        torch.manual_seed(1)
        s = torch.sigmoid(torch.randn(2048, 8))  # no expert preference
        _, cutoff = topk_with_cutoff(s, torch.zeros(8), top_k=2)
        b = quantile_balance_bias(s, cutoff, top_k=2)
        self.assertLess(b.abs().max().item(), 0.05)


class TestHistogramEstimator(unittest.TestCase):
    def test_counts_are_additive_across_shards(self):
        # the property that makes one all-reduce equal the global batch
        s = _skewed_scores(T=512, E=8)
        _, cutoff = topk_with_cutoff(s, torch.zeros(8), top_k=2)
        whole = margin_histogram(s, cutoff)
        a = margin_histogram(s[:200], cutoff[:200])
        b = margin_histogram(s[200:], cutoff[200:])
        torch.testing.assert_close(whole, a + b)
        self.assertEqual(whole.sum().item(), 512 * 8)

    def test_histogram_bias_approximates_the_exact_bias(self):
        s = _skewed_scores(T=2048, E=8)
        _, cutoff = topk_with_cutoff(s, torch.zeros(8), top_k=2)
        exact = quantile_balance_bias(s, cutoff, top_k=2)
        approx = quantile_balance_bias_histogram(
            margin_histogram(s, cutoff, num_bins=512), top_k=2
        )
        # exact up to the bin width (2.0 range / 512 bins ~= 0.004), with a
        # little slack for where the quantile lands inside a bin
        self.assertLess((exact - approx).abs().max().item(), 0.02)

    def test_histogram_bias_also_balances(self):
        T, E, k = 2048, 8, 2
        s = _skewed_scores(T, E)
        _, cutoff = topk_with_cutoff(s, torch.zeros(E), k)
        b = quantile_balance_bias_histogram(margin_histogram(s, cutoff), k)
        before = expert_loads(s, torch.zeros(E), k).float()
        after = expert_loads(s, b, k).float()
        target = T * k / E
        self.assertLess(
            (after - target).abs().max().item(),
            0.35 * (before - target).abs().max().item(),
        )


if __name__ == "__main__":
    unittest.main()


class TestQuantileBalancerRuntime(unittest.TestCase):
    """The runtime driver, tested on the property QB is defined by: after the
    bias is installed, the per-expert loads must sit at the target m*k/n."""

    def _fake_moe(self, num_experts=16, top_k=2):
        """A stand-in with the two attributes the balancer touches: a router
        that returns core's 3-tuple, and an expert_bias_E buffer."""
        import torch.nn as nn

        from torchtitan.models.common.moe import MoE

        class Router(nn.Module):
            def __init__(self, e, k):
                super().__init__()
                self.top_k = k
                self.gate = nn.Linear(8, e, bias=False)

            def forward(self, x_BLD, expert_bias_E=None):
                scores = torch.sigmoid(self.gate(x_BLD))
                biased = scores if expert_bias_E is None else scores + expert_bias_E
                _, ids = torch.topk(biased, self.top_k, dim=-1)
                return scores.gather(-1, ids), ids, scores

        moe = MoE.__new__(MoE)  # skip MoE.__init__, which needs a full config
        nn.Module.__init__(moe)
        moe.router = Router(num_experts, top_k)
        moe.register_buffer("expert_bias_E", torch.zeros(num_experts))
        return moe

    def test_installed_bias_moves_loads_toward_the_target(self):
        from torchtitan.components.quantile_balance import (
            expert_loads,
            QuantileBalancer,
        )

        torch.manual_seed(0)
        E, K, T = 16, 2, 4096
        moe = self._fake_moe(E, K)
        # skew the gate so routing starts badly imbalanced
        with torch.no_grad():
            moe.router.gate.weight.normal_(std=1.0)
            moe.router.gate.weight[:4] += 2.0

        class Part(torch.nn.Module):
            def __init__(self, moe):
                super().__init__()
                self.moe = moe

        balancer = QuantileBalancer([Part(moe)], num_bins=1024)
        x = torch.randn(1, T, 8)

        with torch.no_grad():
            scores = torch.sigmoid(moe.router.gate(x)).reshape(-1, E)

        def cv():
            loads = expert_loads(scores, moe.expert_bias_E, K).float()
            return (loads.std() / loads.mean()).item(), loads

        # QB solves for the bias from margins measured at the CURRENT bias, so
        # applying it shifts every token's cutoff and one shot cannot land on
        # the fixed point. In training the update runs every step; this mirrors
        # that and checks it converges rather than oscillating.
        history = [cv()[0]]
        for _ in range(30):
            moe.router(x, moe.expert_bias_E)  # the hook fills the histogram
            balancer.step()
            history.append(cv()[0])

        # The histogram estimator reaches a resolution-limited fixed point
        # (module docstring has the bins-vs-plateau table); assert it gets a
        # large fraction of the way there and then stays put, which is the
        # behaviour that distinguishes it from the sign rule's oscillation.
        self.assertLess(
            history[-1],
            history[0] / 3,
            f"QB did not converge: cv trajectory {[round(c, 3) for c in history]}",
        )
        self.assertLess(abs(history[-1] - history[-2]), 1e-6, "not at a fixed point")
        _, loads = cv()
        self.assertLess(abs(loads.mean().item() - T * K / E), 1e-6)
        balancer.remove()

    def test_bias_is_overwritten_not_accumulated(self):
        from torchtitan.components.quantile_balance import QuantileBalancer

        torch.manual_seed(0)
        moe = self._fake_moe()

        class Part(torch.nn.Module):
            def __init__(self, moe):
                super().__init__()
                self.moe = moe

        balancer = QuantileBalancer([Part(moe)], num_bins=256)
        x = torch.randn(1, 512, 8)

        moe.router(x, moe.expert_bias_E)
        balancer.step()
        first = moe.expert_bias_E.clone()
        moe.router(x, moe.expert_bias_E)
        balancer.step()
        second = moe.expert_bias_E.clone()

        # QB solves for the bias, so repeating an identical batch must not
        # drift it the way an accumulating sign rule would.
        self.assertLess((second - first).abs().max().item(), 0.05)
        balancer.remove()

    def test_step_without_a_forward_is_a_noop(self):
        from torchtitan.components.quantile_balance import QuantileBalancer

        moe = self._fake_moe()

        class Part(torch.nn.Module):
            def __init__(self, moe):
                super().__init__()
                self.moe = moe

        balancer = QuantileBalancer([Part(moe)], num_bins=64)
        before = moe.expert_bias_E.clone()
        balancer.step()
        self.assertTrue(torch.equal(before, moe.expert_bias_E))
        balancer.remove()

    def test_missing_expert_bias_buffer_is_rejected(self):
        from torchtitan.components.quantile_balance import QuantileBalancer

        moe = self._fake_moe()
        moe.expert_bias_E = None

        class Part(torch.nn.Module):
            def __init__(self, moe):
                super().__init__()
                self.moe = moe

        with self.assertRaisesRegex(ValueError, "expert_bias_E"):
            QuantileBalancer([Part(moe)])
