# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Context-parallel attention kernel selection and mesh lookup."""

import unittest
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, cast
from unittest import mock

import spmd_types as spmd

import torch
import torch.distributed as dist
from torch.distributed.device_mesh import DeviceMesh
from torch.nn.attention.flex_attention import BlockMask

from torchtitan.config import ParallelismConfig
from torchtitan.distributed.parallel_dims import MeshAxisName, ParallelDims
from torchtitan.distributed.spmd_types import spmd_mesh_group
from torchtitan.models.common.attention import FlexInnerAttention
from torchtitan.models.common.config_utils import get_attention_config
from torchtitan.models.common.cp_attention import (
    CPInnerAttention,
    KVAllGatherCPFlexInnerAttention,
    UlyssesCPFlexInnerAttention,
)
from torchtitan.protocols.module import Module


class _MetadataCpBackend(CPInnerAttention, Module):
    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        pass

    @classmethod
    def cp_shard(cls, input_dict, cp_mesh, load_balancer):
        return input_dict


class _MixedCpModel(Module):
    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        backends: list[Module.Config]


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

    def test_all_gather_shards_only_block_mask_metadata(self):
        batch: dict[str, Any] = {"input": torch.arange(8)}
        block_mask = object.__new__(BlockMask)
        linear_attention_metadata = object()
        batch["attention_masks"] = {
            "quadratic_attention": block_mask,
            "linear_attention": linear_attention_metadata,
        }
        sharded_block_mask = object.__new__(BlockMask)
        mesh = cast(DeviceMesh, object())
        with mock.patch(
            "torchtitan.models.common.cp_attention."
            "shard_context_parallel_attention_masks",
            return_value={"quadratic_attention": sharded_block_mask},
        ) as shard_masks:
            result = KVAllGatherCPFlexInnerAttention.cp_shard(batch, mesh, None)

        self.assertIs(result, batch)
        self.assertIs(result["input"], batch["input"])
        self.assertIs(
            result["attention_masks"]["quadratic_attention"], sharded_block_mask
        )
        self.assertIs(
            result["attention_masks"]["linear_attention"],
            linear_attention_metadata,
        )
        shard_masks.assert_called_once_with(
            mesh, {"quadratic_attention": block_mask}, None
        )


class TestDecoderCpSharding(unittest.TestCase):
    def test_common_input_sharding_does_not_modify_metadata(self):
        from torchtitan.models.common.decoder import prepare_context_parallel_input

        input_T = torch.arange(8)
        labels_T = torch.arange(8)
        attention_metadata = object()
        batch = {
            "input": input_T,
            "labels": labels_T,
            "attention_masks": attention_metadata,
        }
        sharded_input_T = input_T[:4]
        sharded_labels_T = labels_T[:4]
        cp_mesh = cast(DeviceMesh, object())
        load_balancer = object()

        with mock.patch(
            "torchtitan.models.common.decoder.create_context_parallel_load_balancer",
            return_value=load_balancer,
        ) as create_load_balancer, mock.patch(
            "torchtitan.models.common.decoder.shard_context_parallel_inputs",
            return_value=(sharded_input_T, sharded_labels_T),
        ) as shard_inputs:
            result, result_load_balancer = prepare_context_parallel_input(
                batch, None, cp_mesh, "headtail", None
            )

        self.assertIs(result, batch)
        self.assertIs(result_load_balancer, load_balancer)
        self.assertIs(result["input"], sharded_input_T)
        self.assertIs(result["labels"], sharded_labels_T)
        self.assertIs(result["attention_masks"], attention_metadata)
        create_load_balancer.assert_called_once_with(
            cp_mesh, 8, attention_metadata, "headtail", None
        )
        args = shard_inputs.call_args.args
        self.assertIs(args[0], cp_mesh)
        self.assertEqual(args[2], (0, 0))
        self.assertIs(args[3], load_balancer)

    def test_shards_inputs_once_and_metadata_once_per_backend(self):
        from torchtitan.models.common.decoder import Decoder

        model_config = _MixedCpModel.Config(
            backends=[
                KVAllGatherCPFlexInnerAttention.Config(),
                KVAllGatherCPFlexInnerAttention.Config(),
                _MetadataCpBackend.Config(),
            ]
        )

        model = cast(Decoder, SimpleNamespace(config=model_config))
        batch = {"input": torch.arange(8)}
        input_shardings = {}
        cp_mesh = cast(DeviceMesh, object())
        load_balancer = object()
        parallel_dims = cast(
            ParallelDims, SimpleNamespace(get_mesh=lambda axis: cp_mesh)
        )
        parallelism = cast(
            ParallelismConfig,
            SimpleNamespace(
                context_parallel_load_balancer=None,
                context_parallel_ptrr_mask_key=None,
            ),
        )

        with mock.patch(
            "torchtitan.models.common.decoder.prepare_context_parallel_input",
            return_value=(batch, load_balancer),
        ) as shard_inputs, mock.patch.object(
            KVAllGatherCPFlexInnerAttention,
            "cp_shard",
            side_effect=lambda inputs, *_args: inputs,
        ) as shard_all_gather, mock.patch.object(
            _MetadataCpBackend,
            "cp_shard",
            side_effect=lambda inputs, *_args: inputs,
        ) as shard_metadata:
            result = Decoder._cp_shard_inputs(
                model,
                batch,
                input_shardings,
                parallel_dims,
                parallelism,
            )

        self.assertIs(result, batch)
        shard_inputs.assert_called_once_with(
            batch, input_shardings, cp_mesh, None, None
        )
        shard_all_gather.assert_called_once_with(batch, cp_mesh, load_balancer)
        shard_metadata.assert_called_once_with(batch, cp_mesh, load_balancer)


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

        result = UlyssesCPFlexInnerAttention.cp_shard(
            batch, cast(DeviceMesh, object()), None
        )

        self.assertIs(result, batch)
        self.assertIs(result["attention_masks"], mask)
        self.assertEqual(result["input"].shape, (8,))

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
