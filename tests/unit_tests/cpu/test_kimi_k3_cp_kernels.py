# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Kimi K3's context-parallel kernels and the recipe transforms, on CPU."""

import unittest
from dataclasses import replace
from types import SimpleNamespace
from unittest import mock

import spmd_types as spmd

import torch

from torchtitan.distributed.context_parallel import validate_context_parallel
from torchtitan.models.common.attention import FlexAttention
from torchtitan.models.kimi_k3.config_registry import kimi_k3_debugmodel
from torchtitan.models.kimi_k3.context_parallel import (
    ContextParallelInnerKDA,
    MLAAllGatherCPFlexAttention,
    MLAUlyssesCPFlexAttention,
)
from torchtitan.models.kimi_k3.kda import InnerKDA
from torchtitan.models.kimi_k3.model import KimiK3Model
from torchtitan_recipes.kimi_k3 import kimi_k3_context_parallel

T, H, NOPE, ROPE, V = 8, 4, 8, 4, 6


class _FakeMesh:
    mesh_dim_names = ("dp", "cp", "tp")

    def __init__(self, cp_size: int):
        self._cp_size = cp_size

    def get_group(self, axis):
        assert axis == "cp"
        return SimpleNamespace(size=lambda: self._cp_size)


def _in_mesh(cp_size: int):
    return mock.patch(
        "torchtitan.models.common.cp_attention.current_spmd_mesh",
        return_value=_FakeMesh(cp_size),
    )


def _mla_qkv(dtype=torch.float32):
    """q, k, v the way MLA hands them over: one rope vector expanded onto every head."""
    q_THK = torch.randn(T, H, NOPE + ROPE, dtype=dtype)
    k_nope_THN = torch.randn(T, H, NOPE, dtype=dtype)
    k_rope_TR = torch.randn(T, ROPE, dtype=dtype)
    k_THK = torch.cat((k_nope_THN, k_rope_TR.unsqueeze(1).expand(-1, H, -1)), dim=-1)
    v_THV = torch.randn(T, H, V, dtype=dtype)
    return q_THK, k_THK, v_THV


def _run(kernel, q, k, v):
    """Run ``kernel`` with pass-through collectives; return the calls and what
    reached FlexAttention."""
    calls, seen = [], {}

    def record(x, group, *, src, dst, backward_options=None):
        calls.append((x, group, src, dst, backward_options))
        return x

    def capture(self, q, k, v, **kwargs):
        seen.update(q=q, k=k, v=v)
        return q

    with _in_mesh(2), mock.patch.object(
        spmd, "redistribute", record
    ), mock.patch.object(FlexAttention, "forward", capture):
        out = kernel.forward(q, k, v)
    return calls, seen, out


class TestPackedMLAKernels(unittest.TestCase):
    def test_ulysses_moves_one_packed_tensor_and_the_rope_slice_once(self):
        q, k, v = _mla_qkv()
        kernel = MLAUlyssesCPFlexAttention(
            MLAUlyssesCPFlexAttention.Config(rope_head_dim=ROPE)
        )
        calls, seen, out = _run(kernel, q, k, v)

        self.assertEqual(3, len(calls))
        packed, _, src, dst, _ = calls[0]
        self.assertEqual((T, H, (NOPE + ROPE) + NOPE + V), tuple(packed.shape))
        self.assertEqual((spmd.S(0), spmd.S(1)), (src, dst))
        rope, _, src, dst, backward_options = calls[1]
        self.assertEqual((T, ROPE), tuple(rope.shape))
        self.assertEqual((spmd.S(0), spmd.R), (src, dst))
        self.assertEqual({"op_dtype": torch.float32}, backward_options)
        _, _, src, dst, _ = calls[2]
        self.assertEqual((spmd.S(1), spmd.S(0)), (src, dst))
        # With pass-through collectives the kernel must hand FlexAttention
        # exactly what MLA produced: the split, the pack and the expansion
        # are each other's inverse.
        for name, original in (("q", q), ("k", k), ("v", v)):
            self.assertTrue(torch.equal(seen[name], original), name)
        self.assertTrue(torch.equal(out, q))

    def test_all_gather_moves_the_packed_kv_and_the_rope_slice(self):
        q, k, v = _mla_qkv(torch.bfloat16)
        kernel = MLAAllGatherCPFlexAttention(
            MLAAllGatherCPFlexAttention.Config(rope_head_dim=ROPE)
        )
        calls, seen, _ = _run(kernel, q, k, v)

        self.assertEqual(2, len(calls))
        packed, _, src, dst, backward_options = calls[0]
        self.assertEqual((T, H, NOPE + V), tuple(packed.shape))
        self.assertEqual((spmd.S(0), spmd.R), (src, dst))
        self.assertEqual({"op_dtype": torch.bfloat16}, backward_options)
        rope, _, _, _, _ = calls[1]
        self.assertEqual((T, ROPE), tuple(rope.shape))
        for name, original in (("q", q), ("k", k), ("v", v)):
            self.assertTrue(torch.equal(seen[name], original), name)

    def test_all_gather_reduce_dtype_reaches_both_gathers(self):
        q, k, v = _mla_qkv(torch.bfloat16)
        kernel = MLAAllGatherCPFlexAttention(
            MLAAllGatherCPFlexAttention.Config(
                rope_head_dim=ROPE, reduce_dtype="float32"
            )
        )
        calls, _, _ = _run(kernel, q, k, v)
        self.assertEqual(
            [torch.float32, torch.float32], [c[4]["op_dtype"] for c in calls]
        )

    def test_kernels_keep_the_generic_kernels_flags(self):
        self.assertFalse(MLAUlyssesCPFlexAttention.Config.shard_attention_mask)
        self.assertTrue(MLAUlyssesCPFlexAttention.Config.shard_attention_heads)
        self.assertTrue(
            getattr(MLAAllGatherCPFlexAttention.Config, "shard_attention_mask", True)
        )
        self.assertFalse(
            getattr(MLAAllGatherCPFlexAttention.Config, "shard_attention_heads", False)
        )


class TestRecipeTransforms(unittest.TestCase):
    @staticmethod
    def _layers(config):
        model = config.model_spec.model
        assert isinstance(model, KimiK3Model.Config)
        mla = [l.attention for l in model.layers if l.attention is not None]
        kda = [l.delta_attention for l in model.layers if l.delta_attention is not None]
        return model, mla, kda

    def test_every_layer_gets_its_kernel(self):
        config = kimi_k3_debugmodel()
        _, mla, kda = self._layers(config)
        self.assertTrue(mla and kda)
        # A non-default field must ride through the retype.
        flex = mla[0].inner_attention
        assert isinstance(flex, FlexAttention.Config)
        mla[0].inner_attention = replace(flex, block_size=64)

        config = kimi_k3_context_parallel(config, cp_degree=2)

        model, mla, kda = self._layers(config)
        for attention in mla:
            inner = attention.inner_attention
            self.assertIsInstance(inner, MLAUlyssesCPFlexAttention.Config)
            assert isinstance(inner, MLAUlyssesCPFlexAttention.Config)
            self.assertEqual(attention.qk_rope_head_dim, inner.rope_head_dim)
        first = mla[0].inner_attention
        assert isinstance(first, MLAUlyssesCPFlexAttention.Config)
        self.assertEqual(64, first.block_size)
        for delta_attention in kda:
            self.assertIsInstance(
                delta_attention.inner_kda, ContextParallelInnerKDA.Config
            )
        self.assertEqual(2, config.parallelism.context_parallel_degree)
        self.assertIsNone(config.parallelism.context_parallel_load_balancer)
        validate_context_parallel(model, config.parallelism)

    def test_all_gather_is_a_choice(self):
        config = kimi_k3_context_parallel(
            kimi_k3_debugmodel(), cp_degree=2, mla_kernel=MLAAllGatherCPFlexAttention
        )
        _, mla, kda = self._layers(config)
        for attention in mla:
            self.assertIsInstance(
                attention.inner_attention, MLAAllGatherCPFlexAttention.Config
            )
        for delta_attention in kda:
            self.assertIsInstance(
                delta_attention.inner_kda, ContextParallelInnerKDA.Config
            )

    def test_kda_without_its_kernel_is_rejected(self):
        config = kimi_k3_context_parallel(kimi_k3_debugmodel(), cp_degree=2)
        model, _, kda = self._layers(config)
        inner = kda[0].inner_kda
        assert isinstance(inner, InnerKDA.Config)
        kda[0].inner_kda = InnerKDA.Config(head_dim=inner.head_dim, kernel=inner.kernel)
        with self.assertRaisesRegex(ValueError, "KCP kernel on every KDA layer"):
            model.update_from_config(config=config)


if __name__ == "__main__":
    unittest.main()
