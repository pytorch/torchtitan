# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Kimi K3's torch_remat regions: RegionAC saves or recomputes them by name."""

import dataclasses
import unittest
from copy import deepcopy
from unittest.mock import patch

import torch
import torch_remat as remat

from torchtitan.distributed.activation_checkpoint import RegionAC
from torchtitan.models.kimi_k3 import model as k3_model, model_registry
from torchtitan.protocols.module import Module, ModuleDict

_TOKENS = 8
# The attention kernels are CUDA-only; these CPU stand-ins keep the call signatures.


class _CountingOp(Module):
    def __init__(self, operation):
        super().__init__()
        self.operation = operation
        self.num_forwards = 0

    def forward(self, *args, **kwargs):
        self.num_forwards += 1
        return self.operation(*args, **kwargs)


def _mla_inner_attention(q_THK, k_THK, v_THV, *, attention_masks=None, scale=None):
    del attention_masks
    return torch.nn.functional.scaled_dot_product_attention(
        q_THK.transpose(0, 1),
        k_THK.transpose(0, 1),
        v_THV.transpose(0, 1),
        is_causal=True,
        scale=scale,
    ).transpose(0, 1)


def _kda_inner(
    query_TC, key_TC, value_TC, raw_gate_THK, raw_beta_TH, *params, cu_seqlens=None
):
    del cu_seqlens
    num_tokens, num_heads = raw_beta_TH.shape
    q_THK, k_THK, v_THV = (
        t.view(num_tokens, num_heads, -1) for t in (query_TC, key_TC, value_TC)
    )
    gate_THK = torch.sigmoid(raw_gate_THK) * torch.sigmoid(raw_beta_TH).unsqueeze(-1)
    scale = 1 + sum(p.float().mean() for p in params)
    return (q_THK * k_THK).sum(-1, keepdim=True).tanh() * v_THV * gate_THK * scale


class _TwoBlocks(Module):
    """A KDA block that opens the attention-residual block, then an MLA block."""

    def __init__(self):
        super().__init__()
        layers = model_registry("debugmodel").model.layers
        kda_config = layers[0]
        assert kda_config.delta_attention is not None and kda_config.moe is None
        mla_config = dataclasses.replace(
            next(layer for layer in layers if layer.attention is not None),
            layer_id=1,
            moe=None,
            feed_forward=kda_config.feed_forward,
        )
        torch.manual_seed(0)
        blocks = [kda_config.build(), mla_config.build()]
        for block in blocks:
            block.init_states()
            block.feed_forward.w13 = _CountingOp(block.feed_forward.w13)
        blocks[0].delta_attention.inner_kda = _CountingOp(_kda_inner)
        blocks[1].attention.inner_attention = _CountingOp(_mla_inner_attention)
        self.layers = ModuleDict({str(i): block for i, block in enumerate(blocks)})

    def forward(self, x_TD: torch.Tensor) -> torch.Tensor:
        block_residual_TND = x_TD.unsqueeze(1)[:, :0]
        for block in self.layers.values():
            x_TD, block_residual_TND = block(x_TD, block_residual_TND)
        return x_TD.pow(2).mean()


def _run_forward_backward(model: Module, x_TD: torch.Tensor):
    model.zero_grad(set_to_none=True)
    input_TD = x_TD.detach().clone().requires_grad_(True)
    output = model(input_TD)
    output.backward()
    grads = {n: p.grad for n, p in model.named_parameters() if p.grad is not None}
    return output.detach(), input_TD.grad, grads


def _counts(model: _TwoBlocks, residual_calls: int) -> tuple[int, int, int, int]:
    kda, mla = model.layers["0"], model.layers["1"]
    return (
        kda.delta_attention.inner_kda.num_forwards,
        mla.attention.inner_attention.num_forwards,
        residual_calls,
        kda.feed_forward.w13.num_forwards + mla.feed_forward.w13.num_forwards,
    )


def _unwrapped_residual():
    """The residual math without its checkpoint: the reference and the control."""
    return patch.object(k3_model.remat, "checkpoint", lambda **_: (lambda fn: fn))


def _saved_shapes(model: Module, x_TD: torch.Tensor) -> list[tuple[int, ...]]:
    shapes = []

    def pack(tensor):
        shapes.append(tuple(tensor.shape))
        return tensor

    with torch.autograd.graph.saved_tensors_hooks(pack, lambda tensor: tensor):
        model(x_TD.detach().clone().requires_grad_(True))
    return shapes


class TestKimiK3AttentionResidualRecompute(unittest.TestCase):
    def test_residual_math_reruns_in_backward_bitwise(self):
        model = _TwoBlocks()
        reference = deepcopy(model)
        x_TD = torch.randn(_TOKENS, 1024)
        with _unwrapped_residual():
            expected = _run_forward_backward(reference, x_TD)
        calls = []
        original = k3_model._apply_attention_residual

        def counting(*args):
            calls.append(None)
            return original(*args)

        with patch.object(k3_model, "_apply_attention_residual", counting):
            actual = _run_forward_backward(model, x_TD)
        torch.testing.assert_close(actual[0], expected[0], rtol=0, atol=0)
        torch.testing.assert_close(actual[1], expected[1], rtol=0, atol=0)
        self.assertEqual(actual[2].keys(), expected[2].keys())
        for name, grad in expected[2].items():
            torch.testing.assert_close(actual[2][name], grad, rtol=0, atol=0, msg=name)
        # Three residual computations in forward, each run again in backward.
        self.assertEqual(len(calls), 6)

    def test_covered_block_runs_the_residual_once(self):
        model = _TwoBlocks()
        for block in model.layers.values():
            block.checkpoint_residual = False
        reference = deepcopy(model)
        x_TD = torch.randn(_TOKENS, 1024)
        with _unwrapped_residual():
            expected = _run_forward_backward(reference, x_TD)
        calls = []
        original = k3_model._apply_attention_residual

        def counting(*args):
            calls.append(None)
            return original(*args)

        with patch.object(k3_model, "_apply_attention_residual", counting):
            actual = _run_forward_backward(model, x_TD)
        torch.testing.assert_close(actual[0], expected[0], rtol=0, atol=0)
        for name, grad in expected[2].items():
            torch.testing.assert_close(actual[2][name], grad, rtol=0, atol=0, msg=name)
        # The block's own activation checkpoint is the recompute; no second one here.
        self.assertEqual(len(calls), 3)

    def test_no_stack_shaped_activation_is_saved(self):
        model = _TwoBlocks()
        x_TD = torch.randn(_TOKENS, 1024)
        # Every residual here reads a one-entry stack plus the prefix sum.
        stack_shape = (_TOKENS, 2, 1024)
        with _unwrapped_residual():
            self.assertIn(stack_shape, _saved_shapes(model, x_TD))
        self.assertNotIn(stack_shape, _saved_shapes(model, x_TD))


class TestKimiK3RematRegions(unittest.TestCase):
    def test_regions_are_declared_relative_to_the_block(self):
        model = _TwoBlocks()
        RegionAC.Config(save_regions=[]).build().apply(model)
        with remat.collect_trace() as trace:
            model(torch.randn(_TOKENS, 1024))
        kda_names = [
            f"delta_attention.{n}"
            for n in (
                "forget",
                "beta",
                "qkv",
                "inner_kda",
                "output_gate",
                "output_norm",
                "output_proj",
            )
        ]
        mla_names = [
            f"attention.{n}" for n in ("q", "kv", "inner_attention", "gate", "wo")
        ]
        ffn_names = ["ffn_res", "feed_forward.w13", "feed_forward.w2"]
        self.assertEqual(
            [entry.name for entry in trace.entries],
            kda_names + ffn_names + ["attention_res"] + mla_names + ffn_names,
        )

    def test_save_regions_control_recomputation_bitwise(self):
        # (kda kernel, mla kernel, residual math calls, ffn w13 calls) after one step.
        for save_regions, expected in (
            ([], (2, 2, 6, 4)),
            (["attention.*", "delta_attention.*"], (1, 1, 6, 4)),
            (["*attention.*"], (1, 1, 6, 4)),
            (["attention_res", "ffn_res"], (2, 2, 3, 4)),
            (["*"], (1, 1, 3, 2)),
        ):
            with self.subTest(save_regions=save_regions):
                baseline = _TwoBlocks()
                region_model = deepcopy(baseline)
                RegionAC.Config(save_regions=save_regions).build().apply(region_model)
                x_TD = torch.randn(_TOKENS, 1024)
                expected_out = _run_forward_backward(baseline, x_TD)
                calls = []
                original = k3_model._apply_attention_residual

                def counting(*args):
                    calls.append(None)
                    return original(*args)

                with patch.object(k3_model, "_apply_attention_residual", counting):
                    actual_out = _run_forward_backward(region_model, x_TD)
                torch.testing.assert_close(
                    actual_out[0], expected_out[0], rtol=0, atol=0
                )
                torch.testing.assert_close(
                    actual_out[1], expected_out[1], rtol=0, atol=0
                )
                self.assertEqual(actual_out[2].keys(), expected_out[2].keys())
                for name, grad in expected_out[2].items():
                    torch.testing.assert_close(
                        actual_out[2][name], grad, rtol=0, atol=0, msg=name
                    )
                self.assertEqual(_counts(region_model, len(calls)), expected)

    def test_region_ac_keeps_the_state_dict(self):
        with torch.device("meta"):
            model = model_registry("debugmodel").model.build()
        state_keys = list(model.state_dict())
        RegionAC.Config(
            save_regions=["attention.*", "delta_attention.*"]
        ).build().apply(model)
        self.assertEqual(list(model.state_dict()), state_keys)


if __name__ == "__main__":
    unittest.main()
