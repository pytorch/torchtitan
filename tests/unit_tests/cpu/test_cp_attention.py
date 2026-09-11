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

from torchtitan.distributed.parallel_dims import MeshAxisName
from torchtitan.distributed.spmd_types import spmd_mesh_group
from torchtitan.models.common.attention import FlexInnerAttention
from torchtitan.models.common.config_utils import get_attention_config
from torchtitan.models.common.cp_attention import (
    CPInnerAttention,
    KVAllGatherCPFlexInnerAttention,
    UlyssesCPFlexInnerAttention,
)


class TestKernelSelection(unittest.TestCase):
    def test_cp_kernel_is_a_flex_kernel(self):
        self.assertIsInstance(
            KVAllGatherCPFlexInnerAttention.Config(), FlexInnerAttention.Config
        )

    def test_cp_kernel_inherits_flex_fields(self):
        config = KVAllGatherCPFlexInnerAttention.Config(block_size=256)
        self.assertEqual(config.block_size, 256)

    def test_cp_kernel_is_not_an_attention_backend(self):
        with self.assertRaisesRegex(ValueError, "Unknown backend"):
            get_attention_config("allgather_cp_flex")

    def test_plain_flex_is_not_a_cp_kernel(self):
        kernel = get_attention_config("flex")._owner
        assert kernel is not None
        self.assertFalse(issubclass(kernel, CPInnerAttention))

    def test_cp_inner_attention_owns_input_sharding(self):
        batch = {"input": torch.arange(8)}
        sharded = {"input": torch.arange(4)}
        mesh = object()
        with mock.patch(
            "torchtitan.distributed.context_parallel.api."
            "prepare_context_parallel_input",
            return_value=sharded,
        ) as prepare:
            result = KVAllGatherCPFlexInnerAttention.cp_shard(
                batch, None, mesh, "headtail", None
            )

        self.assertIs(result, sharded)
        prepare.assert_called_once_with(batch, None, mesh, "headtail", None)


class _FakeMesh:
    def __init__(self, cp_size: int | None):
        self.mesh_dim_names = ("dp", "tp") if cp_size is None else ("dp", "cp", "tp")
        self._cp_size = cp_size

    def get_group(self, axis):
        assert axis == "cp"
        return SimpleNamespace(size=lambda: self._cp_size)


def _in_mesh(cp_size):
    return mock.patch(
        "torchtitan.distributed.spmd_types.current_spmd_mesh",
        return_value=_FakeMesh(cp_size),
    )


class TestCpGroup(unittest.TestCase):
    """CP inner attention requires a multi-rank CP group."""

    @staticmethod
    def _kernel():
        return KVAllGatherCPFlexInnerAttention(KVAllGatherCPFlexInnerAttention.Config())

    def test_cp_axis_above_one_yields_its_group(self):
        with _in_mesh(8):
            group = spmd_mesh_group(MeshAxisName.CP)
            assert group is not None
            self.assertEqual(group.size(), 8)

    def test_no_mesh_context_returns_none(self):
        with mock.patch(
            "torchtitan.distributed.spmd_types.current_spmd_mesh", return_value=None
        ):
            self.assertIsNone(spmd_mesh_group(MeshAxisName.CP))

    def test_degree_one_returns_none(self):
        with _in_mesh(1):
            self.assertIsNone(spmd_mesh_group(MeshAxisName.CP))

    def test_mesh_without_a_cp_axis_returns_none(self):
        with _in_mesh(None):
            self.assertIsNone(spmd_mesh_group(MeshAxisName.CP))

    def test_forward_without_a_cp_group_is_an_error(self):
        num_tokens, heads, head_dim = 8, 2, 16
        q, k, v = (torch.randn(num_tokens, heads, head_dim) for _ in range(3))
        with _in_mesh(1), self.assertRaisesRegex(
            RuntimeError, "active multi-rank CP mesh axis"
        ):
            self._kernel().forward(q, k, v)

    def test_cp_inner_attention_holds_no_mesh_state(self):
        self.assertNotIn("cp_group", CPInnerAttention.__dict__)


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
        ), mock.patch.object(
            FlexInnerAttention, "forward", lambda self, q, *a, **kw: q
        ):
            KVAllGatherCPFlexInnerAttention(
                KVAllGatherCPFlexInnerAttention.Config()
            ).forward(q, k, v)

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
        ), mock.patch.object(
            FlexInnerAttention, "forward", lambda self, q, *a, **kw: q
        ):
            KVAllGatherCPFlexInnerAttention(config).forward(q, k, v)
        return seen

    def test_reduces_in_float32_by_default(self):
        config = KVAllGatherCPFlexInnerAttention.Config()
        self.assertEqual([torch.float32] * 2, self._reduce_dtypes(config))

    def test_reduce_dtype_can_use_bfloat16(self):
        config = KVAllGatherCPFlexInnerAttention.Config(reduce_dtype="bfloat16")
        self.assertEqual([torch.bfloat16] * 2, self._reduce_dtypes(config))


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
        kernel = KVAllGatherCPFlexInnerAttention(
            KVAllGatherCPFlexInnerAttention.Config()
        )
        q, k, v = (
            torch.randn(4, 2, 8, dtype=dtype, requires_grad=True) for _ in range(3)
        )
        with mock.patch(
            "torchtitan.models.common.cp_attention." "spmd_mesh_group",
            return_value=dist.group.WORLD,
        ), mock.patch.object(
            FlexInnerAttention, "forward", lambda self, q, k, v, **kw: k + v
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
        self.assertIsInstance(
            UlyssesCPFlexInnerAttention.Config(), FlexInnerAttention.Config
        )

    def test_is_not_an_attention_backend(self):
        with self.assertRaisesRegex(ValueError, "Unknown backend"):
            get_attention_config("ulysses_cp_flex")

    def test_keeps_its_mask_global(self):
        mask = object()
        batch = {"input": torch.arange(8), "attention_masks": mask}

        def shard(inputs, *args):
            self.assertNotIn("attention_masks", inputs)
            inputs["input"] = inputs["input"][:4]
            return inputs

        with mock.patch(
            "torchtitan.distributed.context_parallel.api."
            "prepare_context_parallel_input",
            side_effect=shard,
        ):
            result = UlyssesCPFlexInnerAttention.cp_shard(
                batch, None, object(), None, None
            )

        self.assertIs(result["attention_masks"], mask)
        self.assertEqual(result["input"].shape, (4,))

    def test_reshards_sequence_to_heads_and_back(self):
        q, k, v = (torch.randn(8, 4, 16) for _ in range(3))
        calls = []

        def record(x, group, *, src, dst):
            calls.append((x, group, src, dst))
            return x

        kernel = UlyssesCPFlexInnerAttention(UlyssesCPFlexInnerAttention.Config())
        with _in_mesh(2), mock.patch.object(
            spmd, "redistribute", record
        ), mock.patch.object(
            FlexInnerAttention, "forward", lambda self, q, *a, **kw: q
        ):
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
