# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Unit tests for compute_batch_token_log_probs.

Verifies that the batched implementation produces identical results to the
per-episode compute_token_log_probs across various sequence length combinations,
including models with causal self-attention where padding could leak information.
"""

import importlib.util
import math
import os
import unittest

import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleModel(nn.Module):
    """Minimal LM with no attention: embedding -> linear -> logits.

    Each position is processed independently, so padding can't affect results
    by design. Used as a baseline sanity check.
    """

    def __init__(self, vocab_size: int, dim: int):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, dim)
        self.linear = nn.Linear(dim, vocab_size, bias=False)

    def forward(self, tokens, attention_masks=None, positions=None):
        h = self.embed(tokens)
        return self.linear(h)


class CausalAttentionModel(nn.Module):
    """LM with causal self-attention: embedding -> causal attn -> linear -> logits.

    This model has cross-position interactions through attention, making it
    the critical test for padding correctness. If padding tokens leak into
    earlier positions' attention, log probs will differ between batched and
    single-episode computation.
    """

    def __init__(self, vocab_size: int, dim: int, n_heads: int = 4):
        super().__init__()
        self.dim = dim
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.embed = nn.Embedding(vocab_size, dim)
        self.wq = nn.Linear(dim, dim, bias=False)
        self.wk = nn.Linear(dim, dim, bias=False)
        self.wv = nn.Linear(dim, dim, bias=False)
        self.wo = nn.Linear(dim, dim, bias=False)
        self.output = nn.Linear(dim, vocab_size, bias=False)

    def forward(self, tokens, attention_masks=None, positions=None):
        B, S = tokens.shape
        h = self.embed(tokens)  # (B, S, D)

        q = self.wq(h).view(B, S, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.wk(h).view(B, S, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.wv(h).view(B, S, self.n_heads, self.head_dim).transpose(1, 2)

        # Causal attention with SDPA
        out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        out = out.transpose(1, 2).contiguous().view(B, S, self.dim)
        h = h + self.wo(out)

        return self.output(h)


class MultiLayerCausalModel(nn.Module):
    """Multi-layer causal LM: embedding -> N x (causal attn + FFN) -> linear.

    Stacks multiple transformer blocks to amplify any padding leakage.
    """

    def __init__(self, vocab_size: int, dim: int, n_heads: int = 4, n_layers: int = 2):
        super().__init__()
        self.dim = dim
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.embed = nn.Embedding(vocab_size, dim)
        self.layers = nn.ModuleList(
            [self._make_layer(dim, n_heads) for _ in range(n_layers)]
        )
        self.output = nn.Linear(dim, vocab_size, bias=False)

    def _make_layer(self, dim, n_heads):
        return nn.ModuleDict(
            {
                "wq": nn.Linear(dim, dim, bias=False),
                "wk": nn.Linear(dim, dim, bias=False),
                "wv": nn.Linear(dim, dim, bias=False),
                "wo": nn.Linear(dim, dim, bias=False),
                "ffn": nn.Sequential(
                    nn.Linear(dim, dim * 4, bias=False),
                    nn.GELU(),
                    nn.Linear(dim * 4, dim, bias=False),
                ),
                "norm1": nn.LayerNorm(dim),
                "norm2": nn.LayerNorm(dim),
            }
        )

    def forward(self, tokens, attention_masks=None, positions=None):
        B, S = tokens.shape
        h = self.embed(tokens)

        for layer in self.layers:
            residual = h
            h_norm = layer["norm1"](h)
            q = (
                layer["wq"](h_norm)
                .view(B, S, self.n_heads, self.head_dim)
                .transpose(1, 2)
            )
            k = (
                layer["wk"](h_norm)
                .view(B, S, self.n_heads, self.head_dim)
                .transpose(1, 2)
            )
            v = (
                layer["wv"](h_norm)
                .view(B, S, self.n_heads, self.head_dim)
                .transpose(1, 2)
            )
            out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
            out = out.transpose(1, 2).contiguous().view(B, S, self.dim)
            h = residual + layer["wo"](out)
            h = h + layer["ffn"](layer["norm2"](h))

        return self.output(h)


# Load utils.py directly by file path to avoid triggering rl/__init__.py
# which imports vLLM/triton that aren't available on all platforms.
_utils_path = os.path.join(
    os.path.dirname(__file__),
    "..",
    "..",
    "torchtitan",
    "experiments",
    "rl",
    "actors",
    "utils.py",
)
_spec = importlib.util.spec_from_file_location("rl_actors_utils", _utils_path)
_utils = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_utils)

compute_batch_token_log_probs = _utils.compute_batch_token_log_probs
compute_token_log_probs = _utils.compute_token_log_probs
compute_policy_gradient_loss = _utils.compute_policy_gradient_loss


class TestBatchTokenLogProbs(unittest.TestCase):
    """Tests using a simple model (no attention) as a baseline."""

    def setUp(self):
        torch.manual_seed(42)
        self.vocab_size = 64
        self.dim = 32
        self.device = torch.device("cpu")
        self.model = SimpleModel(self.vocab_size, self.dim).to(self.device)
        self.model.eval()

    def _run_comparison(self, prompt_token_ids, gen_token_ids, atol=1e-6, rtol=1e-5):
        """Run both single and batched paths, assert results match within tolerance.

        float32 log_softmax can produce tiny differences (~1e-7) between
        batched and per-episode computation due to different tensor layouts.
        """
        # Per-episode (reference)
        single_results = []
        for prompt_toks, gen_toks in zip(prompt_token_ids, gen_token_ids):
            lps = compute_token_log_probs(
                self.model, prompt_toks, gen_toks, self.device
            )
            single_results.append(lps)

        # Batched
        batch_results = compute_batch_token_log_probs(
            self.model, prompt_token_ids, gen_token_ids, self.device
        )

        self.assertEqual(len(single_results), len(batch_results))
        for i, (single, batched) in enumerate(zip(single_results, batch_results)):
            self.assertEqual(
                single.shape,
                batched.shape,
                f"Episode {i}: shape mismatch {single.shape} vs {batched.shape}",
            )
            torch.testing.assert_close(
                single,
                batched,
                atol=atol,
                rtol=rtol,
                msg=f"Episode {i}: values differ",
            )

    def test_uniform_lengths(self):
        """All episodes have the same prompt and gen length."""
        prompts = [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]]
        gens = [[13, 14, 15], [16, 17, 18], [19, 20, 21]]
        self._run_comparison(prompts, gens)

    def test_variable_prompt_lengths(self):
        """Different prompt lengths, same gen length."""
        prompts = [[1, 2], [3, 4, 5, 6, 7], [8, 9, 10]]
        gens = [[11, 12, 13], [14, 15, 16], [17, 18, 19]]
        self._run_comparison(prompts, gens)

    def test_variable_gen_lengths(self):
        """Same prompt length, different gen lengths."""
        prompts = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        gens = [[10], [11, 12, 13, 14, 15], [16, 17]]
        self._run_comparison(prompts, gens)

    def test_variable_both(self):
        """Both prompt and gen lengths vary."""
        prompts = [[1], [2, 3, 4, 5, 6, 7], [8, 9]]
        gens = [[10, 11, 12, 13, 14], [15], [16, 17, 18]]
        self._run_comparison(prompts, gens)

    def test_single_episode(self):
        """Batch of 1 should match single-episode function."""
        prompts = [[1, 2, 3, 4, 5]]
        gens = [[6, 7, 8]]
        self._run_comparison(prompts, gens)

    def test_single_gen_token(self):
        """Each episode generates only 1 token."""
        prompts = [[1, 2, 3], [4, 5], [6, 7, 8, 9]]
        gens = [[10], [11], [12]]
        self._run_comparison(prompts, gens)

    def test_long_sequences(self):
        """Longer sequences to stress-test padding."""
        prompts = [
            list(range(1, 51)),  # 50 tokens
            list(range(1, 11)),  # 10 tokens
            list(range(1, 31)),  # 30 tokens
        ]
        gens = [
            list(range(51, 56)),  # 5 tokens
            list(range(11, 41)),  # 30 tokens
            list(range(31, 46)),  # 15 tokens
        ]
        self._run_comparison(prompts, gens)

    def test_large_batch(self):
        """Larger batch size with random variable lengths."""
        torch.manual_seed(123)
        import random

        random.seed(123)
        prompts = [list(range(1, random.randint(3, 20))) for _ in range(16)]
        gens = [list(range(30, 30 + random.randint(1, 15))) for _ in range(16)]
        self._run_comparison(prompts, gens)

    def test_gradients_flow(self):
        """Verify gradients propagate through the batched path."""
        self.model.train()
        prompts = [[1, 2, 3], [4, 5, 6, 7]]
        gens = [[8, 9], [10, 11, 12]]

        results = compute_batch_token_log_probs(self.model, prompts, gens, self.device)

        loss = sum(r.sum() for r in results)
        loss.backward()

        has_grad = any(
            p.grad is not None and p.grad.abs().sum() > 0
            for p in self.model.parameters()
        )
        self.assertTrue(has_grad, "No gradients flowed through batched path")

    def test_gradients_match(self):
        """Verify batched and per-episode paths produce identical gradients."""
        prompts = [[1, 2], [3, 4, 5, 6]]
        gens = [[7, 8, 9], [10, 11]]

        # Per-episode gradients
        self.model.zero_grad()
        single_results = []
        for p, g in zip(prompts, gens):
            lps = compute_token_log_probs(self.model, p, g, self.device)
            single_results.append(lps)
        single_loss = sum(r.sum() for r in single_results)
        single_loss.backward()
        single_grads = {
            n: p.grad.clone()
            for n, p in self.model.named_parameters()
            if p.grad is not None
        }

        # Batched gradients
        self.model.zero_grad()
        batch_results = compute_batch_token_log_probs(
            self.model, prompts, gens, self.device
        )
        batch_loss = sum(r.sum() for r in batch_results)
        batch_loss.backward()
        batch_grads = {
            n: p.grad.clone()
            for n, p in self.model.named_parameters()
            if p.grad is not None
        }

        self.assertEqual(set(single_grads.keys()), set(batch_grads.keys()))
        for name in single_grads:
            torch.testing.assert_close(
                single_grads[name],
                batch_grads[name],
                atol=1e-6,
                rtol=1e-5,
                msg=f"Gradient mismatch for {name}",
            )


class TestBatchTokenLogProbsCausalAttention(unittest.TestCase):
    """Tests using a model with causal self-attention.

    This is the critical test class: causal attention has cross-position
    interactions, so if padding tokens leak into earlier positions through
    the attention mechanism, log probs WILL differ between batched and
    single-episode computation.
    """

    def setUp(self):
        torch.manual_seed(42)
        self.vocab_size = 64
        self.dim = 32
        self.device = torch.device("cpu")
        self.model = CausalAttentionModel(self.vocab_size, self.dim, n_heads=4).to(
            self.device
        )
        self.model.eval()

    def _run_comparison(self, prompt_token_ids, gen_token_ids, atol=1e-5, rtol=1e-4):
        """Run both single and batched paths, assert results match."""
        single_results = []
        for prompt_toks, gen_toks in zip(prompt_token_ids, gen_token_ids):
            lps = compute_token_log_probs(
                self.model, prompt_toks, gen_toks, self.device
            )
            single_results.append(lps)

        batch_results = compute_batch_token_log_probs(
            self.model, prompt_token_ids, gen_token_ids, self.device
        )

        self.assertEqual(len(single_results), len(batch_results))
        for i, (single, batched) in enumerate(zip(single_results, batch_results)):
            self.assertEqual(
                single.shape, batched.shape, f"Episode {i}: shape mismatch"
            )
            torch.testing.assert_close(
                single,
                batched,
                atol=atol,
                rtol=rtol,
                msg=f"Episode {i}: values differ (padding may be leaking through attention)",
            )

    def test_variable_lengths(self):
        """Variable prompt and gen lengths with attention."""
        prompts = [[1, 2], [3, 4, 5, 6, 7], [8, 9, 10]]
        gens = [[11, 12, 13, 14], [15], [16, 17, 18]]
        self._run_comparison(prompts, gens)

    def test_extreme_padding_disparity(self):
        """One short and one long sequence — maximal padding on short one."""
        prompts = [[1, 2], [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]]
        gens = [[16], [17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30]]
        self._run_comparison(prompts, gens)

    def test_single_vs_batch_identical(self):
        """Batch of 1 with attention should be exact match."""
        prompts = [[1, 2, 3, 4, 5]]
        gens = [[6, 7, 8]]
        single = compute_token_log_probs(self.model, prompts[0], gens[0], self.device)
        batch = compute_batch_token_log_probs(self.model, prompts, gens, self.device)
        torch.testing.assert_close(single, batch[0], atol=0, rtol=0)

    def test_padding_does_not_leak(self):
        """Directly verify that adding a padded sequence doesn't change results.

        Runs the same episode as batch-of-1 and then as part of a batch-of-2
        where the second episode is much longer (causing heavy padding on the
        first). Results for the first episode must match.
        """
        prompt = [1, 2, 3]
        gen = [10, 11, 12]

        # Alone
        alone = compute_batch_token_log_probs(self.model, [prompt], [gen], self.device)[
            0
        ]

        # Paired with a much longer sequence
        long_prompt = list(range(1, 20))
        long_gen = list(range(30, 50))
        paired = compute_batch_token_log_probs(
            self.model, [prompt, long_prompt], [gen, long_gen], self.device
        )[0]

        torch.testing.assert_close(
            alone,
            paired,
            atol=1e-5,
            rtol=1e-4,
            msg="Padding from longer sequence leaked into shorter sequence's results",
        )

    def test_gradients_with_attention(self):
        """Verify gradients match between batched and per-episode with attention."""
        self.model.train()
        prompts = [[1, 2, 3], [4, 5, 6, 7, 8]]
        gens = [[9, 10], [11, 12, 13, 14]]

        # Per-episode
        self.model.zero_grad()
        single_loss = sum(
            compute_token_log_probs(self.model, p, g, self.device).sum()
            for p, g in zip(prompts, gens)
        )
        single_loss.backward()
        single_grads = {
            n: p.grad.clone()
            for n, p in self.model.named_parameters()
            if p.grad is not None
        }

        # Batched
        self.model.zero_grad()
        batch_loss = sum(
            r.sum()
            for r in compute_batch_token_log_probs(
                self.model, prompts, gens, self.device
            )
        )
        batch_loss.backward()

        for n, p in self.model.named_parameters():
            if p.grad is not None:
                torch.testing.assert_close(
                    single_grads[n],
                    p.grad,
                    atol=1e-5,
                    rtol=1e-4,
                    msg=f"Gradient mismatch for {n} (attention model)",
                )

    def test_large_batch_with_attention(self):
        """Larger batch with attention to stress-test padding correctness."""
        import random

        random.seed(42)
        prompts = [list(range(1, random.randint(3, 15))) for _ in range(8)]
        gens = [list(range(30, 30 + random.randint(1, 10))) for _ in range(8)]
        self._run_comparison(prompts, gens)


class TestMultiLayerCausalModel(unittest.TestCase):
    """Tests using a multi-layer transformer to amplify any leakage."""

    def setUp(self):
        torch.manual_seed(42)
        self.vocab_size = 64
        self.dim = 32
        self.device = torch.device("cpu")
        self.model = MultiLayerCausalModel(
            self.vocab_size, self.dim, n_heads=4, n_layers=3
        ).to(self.device)
        self.model.eval()

    def test_padding_does_not_leak_multi_layer(self):
        """Multi-layer model amplifies any leakage — must still match."""
        prompt = [1, 2, 3, 4]
        gen = [10, 11, 12]

        alone = compute_batch_token_log_probs(self.model, [prompt], [gen], self.device)[
            0
        ]

        long_prompt = list(range(1, 25))
        long_gen = list(range(30, 55))
        paired = compute_batch_token_log_probs(
            self.model, [prompt, long_prompt], [gen, long_gen], self.device
        )[0]

        torch.testing.assert_close(
            alone,
            paired,
            atol=1e-5,
            rtol=1e-4,
            msg="Padding leaked through multi-layer transformer",
        )

    def test_variable_lengths_multi_layer(self):
        """Variable lengths through multiple layers."""
        prompts = [[1], [2, 3, 4, 5, 6, 7, 8], [9, 10]]
        gens = [[11, 12, 13, 14, 15], [16], [17, 18, 19]]

        single_results = []
        for p, g in zip(prompts, gens):
            single_results.append(
                compute_token_log_probs(self.model, p, g, self.device)
            )
        batch_results = compute_batch_token_log_probs(
            self.model, prompts, gens, self.device
        )

        for i, (s, b) in enumerate(zip(single_results, batch_results)):
            torch.testing.assert_close(
                s, b, atol=1e-5, rtol=1e-4, msg=f"Episode {i} differs (multi-layer)"
            )

    def test_gradients_multi_layer(self):
        """Gradients through multi-layer model must match."""
        self.model.train()
        prompts = [[1, 2], [3, 4, 5, 6]]
        gens = [[7, 8, 9], [10, 11]]

        self.model.zero_grad()
        single_loss = sum(
            compute_token_log_probs(self.model, p, g, self.device).sum()
            for p, g in zip(prompts, gens)
        )
        single_loss.backward()
        single_grads = {
            n: p.grad.clone()
            for n, p in self.model.named_parameters()
            if p.grad is not None
        }

        self.model.zero_grad()
        batch_loss = sum(
            r.sum()
            for r in compute_batch_token_log_probs(
                self.model, prompts, gens, self.device
            )
        )
        batch_loss.backward()

        for n, p in self.model.named_parameters():
            if p.grad is not None:
                torch.testing.assert_close(
                    single_grads[n],
                    p.grad,
                    atol=1e-4,
                    rtol=1e-3,
                    msg=f"Gradient mismatch for {n} (multi-layer)",
                )


class TestPolicyGradientLossIntegration(unittest.TestCase):
    """End-to-end test of compute_policy_gradient_loss with batched internals."""

    def setUp(self):
        torch.manual_seed(42)
        self.vocab_size = 64
        self.dim = 32
        self.device = torch.device("cpu")
        self.model = CausalAttentionModel(self.vocab_size, self.dim, n_heads=4).to(
            self.device
        )
        self.ref_model = CausalAttentionModel(self.vocab_size, self.dim, n_heads=4).to(
            self.device
        )
        # Give ref_model same weights
        self.ref_model.load_state_dict(self.model.state_dict())
        self.model.train()
        self.ref_model.eval()

    def test_loss_computation_runs(self):
        """Full loss computation with batched log probs doesn't crash."""
        prompts = [[1, 2, 3], [4, 5, 6, 7], [8, 9]]
        gens = [[10, 11], [12, 13, 14], [15, 16, 17, 18]]
        advantages = torch.tensor([1.0, -0.5, 0.3])

        # Compute ref log probs (batched)
        with torch.no_grad():
            ref_log_probs = compute_batch_token_log_probs(
                self.ref_model, prompts, gens, self.device
            )

        # Compute policy gradient loss (uses batched internally)
        loss, metrics, batch_log_probs = compute_policy_gradient_loss(
            self.model, gens, prompts, advantages, ref_log_probs
        )

        self.assertTrue(torch.isfinite(loss), f"Loss is not finite: {loss}")
        self.assertIn("pg_loss", metrics)
        self.assertIn("entropy", metrics)
        self.assertIn("kl_div", metrics)
        self.assertEqual(len(batch_log_probs), 3)

        # Verify backward works
        loss.backward()
        has_grad = any(
            p.grad is not None and p.grad.abs().sum() > 0
            for p in self.model.parameters()
        )
        self.assertTrue(has_grad, "No gradients from policy gradient loss")

    def test_kl_zero_when_same_weights(self):
        """KL divergence should be ~0 when policy == reference model."""
        prompts = [[1, 2, 3], [4, 5, 6, 7]]
        gens = [[10, 11, 12], [13, 14]]
        advantages = torch.tensor([1.0, -1.0])

        with torch.no_grad():
            ref_log_probs = compute_batch_token_log_probs(
                self.ref_model, prompts, gens, self.device
            )

        _, metrics, _ = compute_policy_gradient_loss(
            self.model, gens, prompts, advantages, ref_log_probs
        )

        self.assertAlmostEqual(
            metrics["kl_div"], 0.0, places=5, msg="KL should be ~0 for identical models"
        )
        self.assertAlmostEqual(
            metrics["ratio_mean"],
            1.0,
            places=5,
            msg="Ratio should be ~1 for identical models",
        )

    def test_loss_changes_with_different_advantages(self):
        """Loss should change when advantages change."""
        prompts = [[1, 2, 3], [4, 5, 6]]
        gens = [[10, 11], [12, 13]]

        with torch.no_grad():
            ref_log_probs = compute_batch_token_log_probs(
                self.ref_model, prompts, gens, self.device
            )

        loss1, _, _ = compute_policy_gradient_loss(
            self.model, gens, prompts, torch.tensor([1.0, 1.0]), ref_log_probs
        )
        loss2, _, _ = compute_policy_gradient_loss(
            self.model, gens, prompts, torch.tensor([-1.0, -1.0]), ref_log_probs
        )

        self.assertNotAlmostEqual(
            loss1.item(),
            loss2.item(),
            places=3,
            msg="Loss should differ with different advantages",
        )


if __name__ == "__main__":
    unittest.main()
