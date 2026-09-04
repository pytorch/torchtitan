# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Context-parallel attention kernel selection and mesh lookup."""

import unittest
from types import SimpleNamespace
from unittest import mock

import spmd_types as spmd

import torch
import torch.distributed as dist

from torchtitan.models.common.attention import FlexAttention
from torchtitan.models.common.config_utils import get_attention_config
from torchtitan.models.common.cp_attention import (
    AllGatherCPFlexAttention,
    ContextParallelKernel,
    UlyssesCPFlexAttention,
)


class TestKernelSelection(unittest.TestCase):
    def test_cp_kernel_is_a_flex_kernel(self):
        self.assertIsInstance(AllGatherCPFlexAttention.Config(), FlexAttention.Config)

    def test_cp_kernel_inherits_flex_fields(self):
        config = AllGatherCPFlexAttention.Config(block_size=256)
        self.assertEqual(config.block_size, 256)

    def test_cp_kernel_is_not_an_attention_backend(self):
        with self.assertRaisesRegex(ValueError, "Unknown backend"):
            get_attention_config("allgather_cp_flex")

    def test_plain_flex_is_not_a_cp_kernel(self):
        kernel = get_attention_config("flex")._owner
        assert kernel is not None
        self.assertFalse(issubclass(kernel, ContextParallelKernel))


class _FakeMesh:
    def __init__(self, cp_size: int | None):
        self.mesh_dim_names = ("dp", "tp") if cp_size is None else ("dp", "cp", "tp")
        self._cp_size = cp_size

    def get_group(self, axis):
        assert axis == "cp"
        return SimpleNamespace(size=lambda: self._cp_size)


def _in_mesh(cp_size):
    return mock.patch(
        "torchtitan.models.common.cp_attention.current_spmd_mesh",
        return_value=_FakeMesh(cp_size),
    )


class TestCpGroup(unittest.TestCase):
    """CP kernels require a multi-rank CP group."""

    @staticmethod
    def _kernel():
        return AllGatherCPFlexAttention(AllGatherCPFlexAttention.Config())

    def test_cp_axis_above_one_yields_its_group(self):
        with _in_mesh(8):
            self.assertEqual(self._kernel().cp_group.size(), 8)

    def test_no_mesh_context_is_an_error(self):
        with self.assertRaisesRegex(RuntimeError, "requires an active SPMD mesh"):
            self._kernel().cp_group

    def test_degree_one_is_an_error(self):
        with _in_mesh(1), self.assertRaisesRegex(
            RuntimeError, "requires an active CP mesh"
        ):
            self._kernel().cp_group

    def test_mesh_without_a_cp_axis_is_an_error(self):
        with _in_mesh(None), self.assertRaisesRegex(
            RuntimeError, "requires an active CP mesh"
        ):
            self._kernel().cp_group

    def test_forward_without_a_cp_group_is_an_error(self):
        num_tokens, heads, head_dim = 8, 2, 16
        q, k, v = (torch.randn(num_tokens, heads, head_dim) for _ in range(3))
        with _in_mesh(1), self.assertRaisesRegex(
            RuntimeError, "requires an active CP mesh"
        ):
            self._kernel().forward(q, k, v)

    def test_the_kernel_holds_no_mesh_state(self):
        self.assertNotIn("parallelize", ContextParallelKernel.__dict__)


class TestAllGather(unittest.TestCase):
    def test_gathers_k_and_v_over_the_cp_group(self):
        num_tokens, heads, head_dim = 8, 2, 16
        q, k, v = (torch.randn(num_tokens, heads, head_dim) for _ in range(3))
        calls = []

        def record(x, group, *, src, dst, backward_options):
            calls.append((x, group, src, dst, backward_options))
            return x

        with _in_mesh(8), mock.patch.object(
            spmd, "redistribute", record
        ), mock.patch.object(FlexAttention, "forward", lambda self, q, *a, **kw: q):
            AllGatherCPFlexAttention(AllGatherCPFlexAttention.Config()).forward(q, k, v)

        self.assertEqual(2, len(calls))
        self.assertIs(k, calls[0][0])
        self.assertIs(v, calls[1][0])
        for _, group, src, dst, backward_options in calls:
            self.assertEqual(8, group.size())
            self.assertEqual(spmd.S(0), src)
            self.assertEqual(spmd.R, dst)
            self.assertEqual({"op_dtype": torch.float32}, backward_options)

    @staticmethod
    def _reduce_dtypes(config):
        """Reduction dtype the kernel asks for, once per gathered tensor."""
        seen = []

        def record(x, group, *, src, dst, backward_options):
            seen.append(backward_options["op_dtype"])
            return x

        q, k, v = (torch.randn(8, 2, 16, dtype=torch.bfloat16) for _ in range(3))
        with _in_mesh(8), mock.patch.object(
            spmd, "redistribute", record
        ), mock.patch.object(FlexAttention, "forward", lambda self, q, *a, **kw: q):
            AllGatherCPFlexAttention(config).forward(q, k, v)
        return seen

    def test_reduces_in_the_input_dtype_by_default(self):
        config = AllGatherCPFlexAttention.Config()
        self.assertEqual([torch.bfloat16] * 2, self._reduce_dtypes(config))

    def test_reduce_dtype_overrides_the_input_dtype(self):
        config = AllGatherCPFlexAttention.Config(reduce_dtype="float32")
        self.assertEqual([torch.float32] * 2, self._reduce_dtypes(config))


class TestAllGatherCollective(unittest.TestCase):
    """Exercise the real collective and its backward.

    A single-rank group is enough: the reduction dtype is validated first.
    """

    @classmethod
    def setUpClass(cls):
        cls._owns_pg = not dist.is_initialized()
        if cls._owns_pg:
            dist.init_process_group(
                backend="gloo",
                init_method="tcp://localhost:12362",
                world_size=1,
                rank=0,
            )

    @classmethod
    def tearDownClass(cls):
        if cls._owns_pg and dist.is_initialized():
            dist.destroy_process_group()

    def _gather_and_backward(self, dtype):
        """Pair each of K and V with the gradient the gather returns to it."""
        kernel = AllGatherCPFlexAttention(AllGatherCPFlexAttention.Config())
        q, k, v = (
            torch.randn(4, 2, 8, dtype=dtype, requires_grad=True) for _ in range(3)
        )
        with mock.patch.object(
            AllGatherCPFlexAttention,
            "cp_group",
            new_callable=mock.PropertyMock,
            return_value=dist.group.WORLD,
        ), mock.patch.object(
            FlexAttention, "forward", lambda self, q, k, v, **kw: k + v
        ):
            kernel.forward(q, k, v).float().sum().backward()
        k_grad, v_grad = k.grad, v.grad
        assert k_grad is not None and v_grad is not None
        return ((k, k_grad), (v, v_grad))

    def test_bfloat16_kv_reach_the_reducing_backward(self):
        for tensor, grad in self._gather_and_backward(torch.bfloat16):
            self.assertEqual(torch.bfloat16, grad.dtype)
            self.assertEqual(tensor.shape, grad.shape)

    def test_float32_kv_reach_the_reducing_backward(self):
        for _, grad in self._gather_and_backward(torch.float32):
            self.assertEqual(torch.float32, grad.dtype)


class TestUlysses(unittest.TestCase):
    def test_is_still_a_flex_kernel(self):
        self.assertIsInstance(UlyssesCPFlexAttention.Config(), FlexAttention.Config)

    def test_is_not_an_attention_backend(self):
        with self.assertRaisesRegex(ValueError, "Unknown backend"):
            get_attention_config("ulysses_cp_flex")

    def test_keeps_its_mask_global(self):
        self.assertFalse(UlyssesCPFlexAttention.Config().shard_attention_mask)

    def test_shards_attention_heads(self):
        self.assertTrue(UlyssesCPFlexAttention.Config().shard_attention_heads)

    def test_reshards_sequence_to_heads_and_back(self):
        q, k, v = (torch.randn(8, 4, 16) for _ in range(3))
        calls = []

        def record(x, group, *, src, dst):
            calls.append((x, group, src, dst))
            return x

        kernel = UlyssesCPFlexAttention(UlyssesCPFlexAttention.Config())
        with _in_mesh(2), mock.patch.object(
            spmd, "redistribute", record
        ), mock.patch.object(FlexAttention, "forward", lambda self, q, *a, **kw: q):
            kernel.forward(q, k, v)

        self.assertEqual(4, len(calls))
        for _, group, src, dst in calls[:3]:
            self.assertEqual(2, group.size())
            self.assertEqual(spmd.S(0), src)
            self.assertEqual(spmd.S(1), dst)
        _, group, src, dst = calls[3]
        self.assertEqual(2, group.size())
        self.assertEqual(spmd.S(1), src)
        self.assertEqual(spmd.S(0), dst)


if __name__ == "__main__":
    unittest.main()
