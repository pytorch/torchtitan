# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest
import torch
import torch.nn.functional as F
from torchtitan.components.optim import AdamW, OptimizersContainer

from torchtitan.config.parallelism import ParallelismConfig
from torchtitan.models.common.activation import SwiGLU
from torchtitan.models.common.aux_loss import AuxLoss
from torchtitan.models.deepseek_v4.mhc import HcPre
from torchtitan.models.deepseek_v4_1 import build_model_config
from torchtitan.models.deepseek_v4_1.attention import hierarchical_select
from torchtitan.models.deepseek_v4_1.mhc import HcCoefficients, mix_branches
from torchtitan.models.deepseek_v4_1.moe import Float32LimitedSwiGLU
from torchtitan.models.deepseek_v4_1.quantization import (
    fp4_activation,
    fp4_values,
    fp8_activation,
)


@pytest.fixture(autouse=True)
def single_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def test_fp4_rounding_and_straight_through():
    values = torch.tensor([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0])
    expected = torch.tensor([0.0, 1.0, 1.0, 2.0, 2.0, 4.0, 4.0])
    torch.testing.assert_close(fp4_values(values), expected, rtol=0, atol=0)
    torch.testing.assert_close(fp4_values(-values), -expected, rtol=0, atol=0)
    for quantizer in (
        fp4_activation,
        lambda x: fp4_activation(x, compressed_kv=True),
        fp8_activation,
    ):
        x = torch.randn(5, 64, requires_grad=True)
        y = quantizer(x)
        assert torch.isfinite(quantizer(torch.zeros_like(x))).all()
        y.sum().backward()
        torch.testing.assert_close(x.grad, torch.ones_like(x), rtol=0, atol=0)


def test_quantization_block_scales_against_scalar_oracle():
    x = torch.linspace(-3.0, 3.0, 64).reshape(2, 32)
    levels = np.array([-6, -4, -3, -2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2, 3, 4, 6])
    expected = np.empty_like(x.numpy())
    for row, block in enumerate(x.numpy()):
        scale = 2.0 ** np.ceil(np.log2(max(np.abs(block).max(), 6 * 2**-126) / 6))
        for col, value in enumerate(block):
            expected[row, col] = levels[np.abs(levels - value / scale).argmin()] * scale
    torch.testing.assert_close(
        fp4_activation(x), torch.from_numpy(expected), rtol=0, atol=0
    )


def test_single_pass_mhc_uses_incoming_coefficients():
    cfg = HcCoefficients.Config(dim=8, hc_mult=4, sinkhorn_iters=3, norm_eps=1e-20)
    module = cfg.build()
    with torch.no_grad():
        module.hc_fn.normal_(std=0.1)
        module.hc_base.normal_(std=0.1)
        module.hc_scale.fill_(0.2)
    x = torch.randn(7, 4, 8, requires_grad=True)
    pre, post, comb = module(x)
    legacy = HcPre.Config(dim=8, hc_mult=4, sinkhorn_iters=3, norm_eps=1e-20).build()
    legacy.load_state_dict(module.state_dict())
    mixed, old_post, old_comb = legacy(x)
    torch.testing.assert_close(mix_branches(x, pre), mixed, rtol=0, atol=0)
    torch.testing.assert_close(post, old_post, rtol=0, atol=0)
    logits = F.linear(x.flatten(-2).float(), module.hc_fn)
    logits = logits * torch.rsqrt(
        x.flatten(-2).square().mean(-1, keepdim=True) + cfg.norm_eps
    )
    expected = (logits[:, 8:] * module.hc_scale[2] + module.hc_base[8:]).view(7, 4, 4)
    expected = expected.softmax(-1) + cfg.eps
    expected = expected / (expected.sum(-2, keepdim=True) + cfg.eps)
    for _ in range(cfg.sinkhorn_iters - 1):
        expected = expected / (expected.sum(-1, keepdim=True) + cfg.eps)
        expected = expected / (expected.sum(-2, keepdim=True) + cfg.eps)
    torch.testing.assert_close(comb, expected, rtol=0, atol=0)
    incoming = torch.zeros(7, 4)
    incoming[:, 2] = 1
    torch.testing.assert_close(mix_branches(x, incoming), x[:, 2], rtol=0, atol=0)
    assert not torch.allclose(mix_branches(x, incoming), mixed)
    (pre.sum() + post.sum() + comb.square().sum()).backward()
    assert x.grad.norm() > 0 and module.hc_fn.grad.norm() > 0


def test_bounded_swiglu_preserves_ordinary_swiglu():
    gate = torch.tensor([-20.0, -1.0, 2.0, 20.0])
    up = torch.tensor([20.0, 2.0, -20.0, -3.0])
    plain = SwiGLU.Config().build()
    bounded = Float32LimitedSwiGLU.Config(limit=10).build()
    torch.testing.assert_close(
        plain._activation_fn(gate, up), F.silu(gate) * up, rtol=0, atol=0
    )
    torch.testing.assert_close(
        bounded._activation_fn(gate, up),
        F.silu(gate.clamp(max=10)) * up.clamp(-10, 10),
        rtol=0,
        atol=0,
    )
    gate = torch.linspace(-12, 12, 128, dtype=torch.bfloat16).requires_grad_()
    up = torch.linspace(12, -12, 128, dtype=torch.bfloat16).requires_grad_()
    expected_gate = gate.detach().clone().requires_grad_()
    expected_up = up.detach().clone().requires_grad_()
    expected = (
        F.silu(expected_gate.float().clamp(max=10)) * expected_up.float().clamp(-10, 10)
    ).to(torch.bfloat16)
    actual = bounded._activation_fn(gate, up)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    actual.sum().backward()
    expected.sum().backward()
    torch.testing.assert_close(gate.grad, expected_gate.grad, rtol=0, atol=0)
    torch.testing.assert_close(up.grad, expected_up.grad, rtol=0, atol=0)


def test_engram_hash_boundaries_and_reference_rng():
    config = build_model_config(
        "debugmodel",
        parallelism=ParallelismConfig(),
        seq_len=16,
        attention_backend="reference",
    )
    hasher = config.engram_hash.build()
    hasher.init_states()
    tokens = torch.tensor([8, 9, 10, 11, 12, 13, 14])
    positions = torch.tensor([0, 1, 2, 0, 1, 2, 3])
    alive = torch.tensor([True, True, False, True, True, True, True])
    hashes = hasher(tokens, positions, token_mask_T=alive)
    expected = torch.empty_like(hashes)
    for layer_index, layer_id in enumerate(config.engram_layer_ids):
        bound = (np.iinfo(np.int64).max // config.vocab_size) // 2
        multipliers = (
            np.random.default_rng(10007 * layer_id).integers(
                0, bound, 4, dtype=np.int64
            )
            * 2
            + 1
        )
        primes = config.engram_hash.primes[layer_index]
        for token_index in range(tokens.numel()):
            rolling, blocked, offset, columns = 0, False, 0, []
            for lookback in range(4):
                blocked |= (
                    positions[token_index] < lookback
                    or not alive[max(token_index - lookback, 0)]
                )
                source = 2 if blocked else int(tokens[token_index - lookback])
                rolling ^= source * int(multipliers[lookback])
                if lookback:
                    for prime in primes[lookback - 1]:
                        columns.append(rolling % prime + offset)
                        offset += prime
            expected[token_index, layer_index] = torch.tensor(columns)
    torch.testing.assert_close(hashes, expected, rtol=0, atol=0)
    isolated = hasher(tokens[3:], torch.arange(4), token_mask_T=alive[3:])
    torch.testing.assert_close(hashes[3:], isolated, rtol=0, atol=0)


def test_candidate_hierarchy_matches_dense_scores_and_pins_recent_block():
    torch.manual_seed(3)
    q, keys, weights = torch.randn(11, 2, 8), torch.randn(11, 8), torch.rand(11, 2)
    positions = torch.arange(11)
    indices, candidates = hierarchical_select(
        q,
        keys,
        weights,
        positions,
        topk=3,
        ratio=1,
        num_candidate_blocks=2,
        block_size=2,
        max_seqlen=11,
        chunk_size=3,
    )
    scores = (torch.einsum("thd,kd->thk", q, keys).relu() * weights[:, :, None]).sum(1)
    for token in range(11):
        valid_indices = indices[token][indices[token] >= 0].long()
        expected_scores = (
            scores[token, : token + 1].topk(min(3, token + 1)).values.sort().values
        )
        torch.testing.assert_close(
            scores[token, valid_indices].sort().values, expected_scores
        )
        valid_candidates = candidates[token][candidates[token] >= 0]
        assert token in valid_candidates and (valid_candidates <= token).all()
        assert valid_candidates.unique().numel() == valid_candidates.numel()
    reindex_q = torch.randn_like(q)
    bounded, _ = hierarchical_select(
        reindex_q,
        keys,
        weights,
        positions,
        topk=3,
        ratio=1,
        candidates_TC=candidates,
        max_seqlen=11,
        chunk_size=4,
    )
    bounded_scores = (
        torch.einsum("thd,kd->thk", reindex_q, keys).relu() * weights[:, :, None]
    ).sum(1)
    for token in range(11):
        pool = candidates[token][candidates[token] >= 0].long()
        picked = bounded[token][bounded[token] >= 0].long()
        assert torch.isin(picked, pool).all()
        torch.testing.assert_close(
            bounded_scores[token, picked].sort().values,
            bounded_scores[token, pool].topk(min(3, pool.numel())).values.sort().values,
        )


def test_full_index_selection_matches_dense_topk_for_tied_scores():
    query, keys = torch.zeros(32, 2, 8), torch.zeros(32, 8)
    positions = torch.arange(32)
    indices, _ = hierarchical_select(
        query,
        keys,
        torch.ones(32, 2),
        positions,
        topk=4,
        ratio=1,
        max_seqlen=32,
    )
    scores = torch.zeros(32, 32).masked_fill(
        torch.arange(32)[None] > positions[:, None],
        -torch.inf,
    )
    expected = scores.topk(4, sorted=False).indices.sort(-1).values
    expected = expected.masked_fill(expected > positions[:, None], -1).int()
    torch.testing.assert_close(indices, expected, rtol=0, atol=0)


def test_modality_router_weights_counts_and_image_bias_updates(monkeypatch):
    config = build_model_config(
        "debugmodel",
        parallelism=ParallelismConfig(),
        seq_len=8,
        attention_backend="reference",
        vision_enabled=True,
        num_draft_layers=3,
    )
    with torch.device("meta"):
        model = config.build()
    model.to_empty(device="cpu")
    model.init_states()
    moe = model.layers["0"].feed_forward
    router = moe.router
    with torch.no_grad():
        router.gate.weight.zero_()
        moe.expert_bias_E.zero_()
        moe.expert_bias_E[:2] = 3
        moe.expert_bias_vl_E.zero_()
        moe.expert_bias_vl_E[2:4] = 3
    x = torch.randn(6, config.dim)
    image = torch.tensor([False, False, True, True, True, False])
    padding = torch.tensor([False, False, False, False, True, True])
    scores, indices, _ = router(
        x,
        moe.expert_bias_E,
        image_mask_T=image,
        expert_bias_vl_E=moe.expert_bias_vl_E,
        padding_mask_T=padding,
    )
    assert torch.isin(indices[~image], torch.tensor([0, 1])).all()
    assert torch.isin(indices[image], torch.tensor([2, 3])).all()
    torch.testing.assert_close(scores, torch.full_like(scores, 0.75), rtol=0, atol=0)
    torch.testing.assert_close(
        router.tokens_per_expert_E, torch.tensor([2, 2, 0, 0, 0, 0, 0, 0]).float()
    )
    torch.testing.assert_close(
        router.image_tokens_per_expert_E, torch.tensor([0, 0, 2, 2, 0, 0, 0, 0]).float()
    )
    router.eval()
    router(
        x, moe.expert_bias_E, image_mask_T=image, expert_bias_vl_E=moe.expert_bias_vl_E
    )
    assert router.image_tokens_per_expert_E.sum() == 4
    router.train()
    with patch(
        "torchtitan.models.deepseek_v4_1.moe.remat.is_recomputing", return_value=True
    ):
        router(
            x,
            moe.expert_bias_E,
            image_mask_T=image,
            expert_bias_vl_E=moe.expert_bias_vl_E,
        )
    assert router.image_tokens_per_expert_E.sum() == 4
    previous = moe.expert_bias_vl_E.clone()
    previous_text = moe.expert_bias_E.clone()
    context = SimpleNamespace(get_optional_mesh=lambda name: None, ep_enabled=False)
    optimizers = OptimizersContainer.Config(
        optimizers=[AdamW.Config(pattern=r".*", fused=False, lr=0.0)]
    ).build(model_parts=[model])
    monkeypatch.setattr(AuxLoss, "group_acc", {})
    auxiliary = next(
        module for module in model.modules() if isinstance(module, AuxLoss)
    )
    auxiliary.instance_acc.fill_(2.5)
    type(model)._register_optimizer_hooks(optimizers, [model], context)
    optimizers.step()
    delta = torch.tensor([1, 1, -1, -1, 1, 1, 1, 1]).float() * moe.load_balance_coeff
    torch.testing.assert_close(moe.expert_bias_vl_E, previous + delta - delta.mean())
    text_delta = (
        torch.tensor([-1, -1, 1, 1, 1, 1, 1, 1]).float() * moe.load_balance_coeff
    )
    torch.testing.assert_close(
        moe.expert_bias_E, previous_text + text_delta - text_delta.mean()
    )
    assert router.image_tokens_per_expert_E.sum() == 0
    assert router.tokens_per_expert_E.sum() == 0
    draft_moe = model.drafter.layers["0"].feed_forward
    with torch.no_grad():
        draft_moe.router.gate.weight.zero_()
        draft_moe.expert_bias_E.zero_()
        draft_moe.expert_bias_E[:2] = 3
    draft_moe.router(x, draft_moe.expert_bias_E)
    previous_draft = draft_moe.expert_bias_E.clone()
    draft_optimizers = OptimizersContainer.Config(
        optimizers=[AdamW.Config(pattern=r".*", fused=False, lr=0.0)]
    ).build(model_parts=[model.drafter])
    type(model.drafter)._register_optimizer_hooks(
        draft_optimizers, [model.drafter], context
    )
    draft_optimizers.step()
    draft_delta = torch.tensor([-1, -1, 1, 1]).float() * draft_moe.load_balance_coeff
    torch.testing.assert_close(draft_moe.expert_bias_E, previous_draft + draft_delta)
    assert draft_moe.router.tokens_per_expert_E.sum() == 0
    assert auxiliary.instance_acc == 0
    assert AuxLoss.group_acc[(auxiliary.reduce_mesh, auxiliary.metric_name)] == 2.5
