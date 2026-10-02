# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Context-parallel attention kernel selection and mesh lookup."""

import unittest
from types import SimpleNamespace
from typing import Any, cast
from unittest import mock

import spmd_types as spmd

import torch
import torch.distributed as dist
from torch.nn.attention.flex_attention import BlockMask

from torchtitan.distributed import context_parallel
from torchtitan.distributed.context_parallel import (
    ContextParallelLoadBalancer,
    HeadTailCPLoadBalancer,
    PTRRFlexAttentionCPLoadBalancer,
)
from torchtitan.distributed.parallelism_context import MeshAxisName
from torchtitan.distributed.spmd_types import spmd_mesh_group
from torchtitan.models.common.attention import (
    FlexInnerAttention,
    InnerAttention,
    SlidingWindowFlexInnerAttention,
    VarlenInnerAttention,
)
from torchtitan.models.common.config_utils import get_attention_config
from torchtitan.models.common.cp_attention import (
    CPInnerAttention,
    KVAllGatherCPFlexInnerAttention,
    KVAllGatherCPSlidingWindowFlexInnerAttention,
    UlyssesCPFlexInnerAttention,
    UlyssesCPInnerAttention,
    UlyssesCPVarlenInnerAttention,
)
from torchtitan.models.common.decoder import Decoder
from torchtitan.models.common.decoder_sharding import decoder_input_sharding


class TestKernelSelection(unittest.TestCase):
    def test_cp_kernel_base_is_an_inner_attention(self):
        self.assertTrue(issubclass(CPInnerAttention, InnerAttention))
        self.assertTrue(issubclass(CPInnerAttention.Config, InnerAttention.Config))

    def test_cp_kernel_is_a_flex_kernel(self):
        config = KVAllGatherCPFlexInnerAttention.Config()
        self.assertIsInstance(config, CPInnerAttention.Config)
        self.assertIsInstance(config, FlexInnerAttention.Config)

    def test_cp_kernel_inherits_flex_fields(self):
        config = KVAllGatherCPFlexInnerAttention.Config(block_size=256)
        self.assertEqual(config.block_size, 256)

    def test_sliding_cp_kernel_is_a_sliding_flex_kernel(self):
        config = KVAllGatherCPSlidingWindowFlexInnerAttention.Config(window_size=128)
        self.assertIsInstance(config, CPInnerAttention.Config)
        self.assertIsInstance(config, SlidingWindowFlexInnerAttention.Config)

    def test_cp_kernel_is_not_an_attention_backend(self):
        with self.assertRaisesRegex(ValueError, "Unknown backend"):
            get_attention_config("allgather_cp_flex")

    def test_plain_flex_is_not_a_cp_kernel(self):
        self.assertNotIsInstance(get_attention_config("flex"), CPInnerAttention.Config)

    def test_all_gather_backends_share_metadata_preparation(self):
        self.assertIs(
            KVAllGatherCPSlidingWindowFlexInnerAttention.prepare_cp_metadata,
            KVAllGatherCPFlexInnerAttention.prepare_cp_metadata,
        )


class TestFluxCpSharding(unittest.TestCase):
    def test_flux_input_sharding_uses_sequence_dim_one(self):
        from torchtitan.models.flux.sharding import flux_input_sharding

        input_shardings = flux_input_sharding()

        self.assertEqual(
            set(input_shardings),
            {"img", "img_ids", "txt", "txt_ids", "target"},
        )
        for sharding in input_shardings.values():
            cp = sharding.local_type[MeshAxisName.CP]
            self.assertIsInstance(cp, spmd.Shard)
            self.assertEqual(cp.dim, 1)


class TestDecoderCpSharding(unittest.TestCase):
    def _assert_selected_ptrr_metadata(
        self,
        *,
        attention_metadata,
        expected_metadata,
    ):
        sliding_config = KVAllGatherCPSlidingWindowFlexInnerAttention.Config(
            window_size=128
        )
        model = SimpleNamespace(
            config=SimpleNamespace(
                first_attention=SimpleNamespace(inner_attention=sliding_config),
                traverse=lambda *_args, **_kwargs: (),
            )
        )
        load_balancer = mock.Mock()
        load_balancer.generate_permutation.return_value = None
        load_balancer_config = PTRRFlexAttentionCPLoadBalancer.Config()
        parallelism = SimpleNamespace(
            context_parallel_load_balancer=load_balancer_config
        )
        input_dict = {
            "input": torch.arange(8),
            "attention_metadata": attention_metadata,
        }

        with mock.patch.object(
            PTRRFlexAttentionCPLoadBalancer.Config,
            "build",
            return_value=load_balancer,
        ) as build_load_balancer, mock.patch.object(
            context_parallel,
            "get_cp_input_seq_len",
            return_value=8,
        ), mock.patch.object(
            context_parallel,
            "shard_tensors",
            return_value=input_dict,
        ):
            Decoder._cp_shard(
                cast(Any, model),
                input_dict,
                input_shardings=decoder_input_sharding(),
                parallelism_context=cast(Any, SimpleNamespace()),
                parallelism=cast(Any, parallelism),
            )

        build_load_balancer.assert_called_once_with(
            seq_len=8,
            attention_metadata=expected_metadata,
        )

    def test_ptrr_uses_sliding_metadata_when_full_metadata_is_absent(self):
        sliding_metadata = object.__new__(BlockMask)

        self._assert_selected_ptrr_metadata(
            attention_metadata={
                KVAllGatherCPSlidingWindowFlexInnerAttention: sliding_metadata
            },
            expected_metadata=sliding_metadata,
        )

    def test_ptrr_prefers_full_metadata(self):
        full_metadata = object.__new__(BlockMask)
        sliding_metadata = object.__new__(BlockMask)

        self._assert_selected_ptrr_metadata(
            attention_metadata={
                KVAllGatherCPFlexInnerAttention: full_metadata,
                KVAllGatherCPSlidingWindowFlexInnerAttention: sliding_metadata,
            },
            expected_metadata=full_metadata,
        )

    def test_config_builds_selected_load_balancer(self):
        input_T = torch.arange(8)
        block_mask = object.__new__(BlockMask)
        cp_group = SimpleNamespace(size=lambda: 2)
        spmd_mesh = SimpleNamespace(device_type="cuda")
        cases = (
            (HeadTailCPLoadBalancer, None),
            (
                PTRRFlexAttentionCPLoadBalancer,
                block_mask,
            ),
        )
        with mock.patch(
            "torchtitan.distributed.context_parallel._HeadTailLoadBalancer"
        ), mock.patch(
            "torchtitan.distributed.context_parallel._PTRRLoadBalancer"
        ), mock.patch(
            "torchtitan.distributed.context_parallel.spmd_mesh_group",
            return_value=cp_group,
        ), mock.patch(
            "torchtitan.distributed.context_parallel.current_spmd_mesh",
            return_value=spmd_mesh,
        ):
            for load_balancer_type, attention_metadata in cases:
                with self.subTest(load_balancer_type=load_balancer_type.__name__):
                    load_balancer = load_balancer_type.Config().build(
                        seq_len=input_T.shape[0],
                        attention_metadata=attention_metadata,
                    )
                    self.assertIs(type(load_balancer), load_balancer_type)

    def test_base_load_balancer_config_is_not_buildable(self):
        with self.assertRaises(NotImplementedError):
            ContextParallelLoadBalancer.Config().build()

    def test_no_shardable_inputs_is_a_noop(self):
        batch = {"attention_metadata": object()}
        result = context_parallel.shard_tensors(
            batch,
            input_shardings={},
            permutation=None,
        )

        self.assertIs(result, batch)

    def test_sharded_sequence_length_must_be_divisible_by_cp_size(self):
        cp_group = SimpleNamespace(size=lambda: 2)
        with mock.patch(
            "torchtitan.distributed.context_parallel.spmd_mesh_group",
            return_value=cp_group,
        ), self.assertRaisesRegex(ValueError, r"sequence length \(5\)"):
            context_parallel.shard_tensors(
                {"input": torch.arange(5)},
                input_shardings=decoder_input_sharding(),
                permutation=None,
            )

    def test_common_input_sharding_does_not_modify_metadata(self):
        input_T = torch.arange(8)
        labels_T = torch.arange(8)
        attention_metadata = object.__new__(BlockMask)
        batch = {
            "input": input_T,
            "labels": labels_T,
            "attention_metadata": attention_metadata,
        }
        sharded_input_T = input_T[:4]
        sharded_labels_T = labels_T[:4]
        cp_group = SimpleNamespace(size=lambda: 2)
        spmd_mesh = SimpleNamespace(device_type="cuda")
        permutation = torch.arange(8).unsqueeze(0)
        torch_load_balancer = mock.Mock()
        torch_load_balancer._generate_indices.return_value = permutation

        with mock.patch(
            "torchtitan.distributed.context_parallel._HeadTailLoadBalancer",
            return_value=torch_load_balancer,
        ) as create_load_balancer, mock.patch(
            "torchtitan.distributed.context_parallel.spmd.shard",
            side_effect=(sharded_input_T, sharded_labels_T),
        ) as shard_input, mock.patch(
            "torchtitan.distributed.context_parallel.spmd_mesh_group",
            return_value=cp_group,
        ), mock.patch(
            "torchtitan.distributed.context_parallel.current_spmd_mesh",
            return_value=spmd_mesh,
        ):
            load_balancer = HeadTailCPLoadBalancer.Config().build(
                seq_len=input_T.shape[0],
                attention_metadata=attention_metadata,
            )
            assert isinstance(load_balancer, ContextParallelLoadBalancer)
            permutation = load_balancer.generate_permutation()
            result = context_parallel.shard_tensors(
                batch,
                input_shardings=decoder_input_sharding(),
                permutation=permutation,
            )

        self.assertIs(result, batch)
        self.assertIs(result["input"], sharded_input_T)
        self.assertIs(result["labels"], sharded_labels_T)
        self.assertIs(result["attention_metadata"], attention_metadata)
        create_load_balancer.assert_called_once_with(8, 2, "cuda")
        self.assertEqual(shard_input.call_count, 2)
        self.assertIs(shard_input.call_args_list[0].args[1], cp_group)

    def test_headtail_uses_the_first_named_cp_input(self):
        tokens_BS = torch.arange(8).view(1, 8)
        cp_group = SimpleNamespace(size=lambda: 2)
        spmd_mesh = SimpleNamespace(device_type="cuda")

        with mock.patch(
            "torchtitan.distributed.context_parallel._HeadTailLoadBalancer"
        ) as create_load_balancer, mock.patch(
            "torchtitan.distributed.context_parallel.spmd_mesh_group",
            return_value=cp_group,
        ), mock.patch(
            "torchtitan.distributed.context_parallel.current_spmd_mesh",
            return_value=spmd_mesh,
        ):
            load_balancer = HeadTailCPLoadBalancer.Config().build(
                seq_len=tokens_BS.shape[1],
                attention_metadata=None,
            )
            load_balancer.generate_permutation()

        create_load_balancer.assert_called_once_with(8, 2, "cuda")

    def test_headtail_requires_two_chunks_per_cp_rank(self):
        cp_group = SimpleNamespace(size=lambda: 2)
        spmd_mesh = SimpleNamespace(device_type="cuda")

        with mock.patch(
            "torchtitan.distributed.context_parallel.spmd_mesh_group",
            return_value=cp_group,
        ), mock.patch(
            "torchtitan.distributed.context_parallel.current_spmd_mesh",
            return_value=spmd_mesh,
        ), self.assertRaisesRegex(
            ValueError, r"divisible by 2 \* CP size \(4\)"
        ):
            HeadTailCPLoadBalancer.Config().build(
                seq_len=6,
                attention_metadata=None,
            )

    def test_ptrr_is_rebuilt_from_each_batch_mask(self):
        input_T = torch.arange(8)
        first_mask = object.__new__(BlockMask)
        second_mask = object.__new__(BlockMask)
        cp_group = SimpleNamespace(size=lambda: 2)
        config = PTRRFlexAttentionCPLoadBalancer.Config()

        with mock.patch(
            "torchtitan.distributed.context_parallel._PTRRLoadBalancer"
        ) as create_load_balancer, mock.patch(
            "torchtitan.distributed.context_parallel.spmd_mesh_group",
            return_value=cp_group,
        ):
            first_load_balancer = config.build(
                seq_len=input_T.shape[0],
                attention_metadata=first_mask,
            )
            first_load_balancer.generate_permutation()
            second_load_balancer = config.build(
                seq_len=input_T.shape[0],
                attention_metadata=second_mask,
            )
            second_load_balancer.generate_permutation()

        self.assertEqual(
            create_load_balancer.call_args_list,
            [mock.call(first_mask, 2), mock.call(second_mask, 2)],
        )


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
        config = UlyssesCPFlexInnerAttention.Config()
        self.assertIsInstance(config, UlyssesCPInnerAttention.Config)
        self.assertIsInstance(config, FlexInnerAttention.Config)

    def test_is_not_an_attention_backend(self):
        with self.assertRaisesRegex(ValueError, "Unknown backend"):
            get_attention_config("ulysses_cp_flex")

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


class TestUlyssesVarlen(unittest.TestCase):
    def test_is_still_a_varlen_kernel(self):
        config = UlyssesCPVarlenInnerAttention.Config()
        self.assertIsInstance(config, UlyssesCPInnerAttention.Config)
        self.assertIsInstance(config, VarlenInnerAttention.Config)

    def test_dispatches_to_varlen_inner_attention(self):
        q, k, v = (torch.randn(8, 4, 16) for _ in range(3))
        mask = object()
        kernel = UlyssesCPVarlenInnerAttention(UlyssesCPVarlenInnerAttention.Config())

        with _in_mesh(2), mock.patch.object(
            spmd,
            "redistribute",
            side_effect=lambda x, *_args, **_kwargs: x,
        ), mock.patch.object(
            VarlenInnerAttention,
            "forward",
            autospec=True,
            return_value=q,
        ) as inner_forward:
            result = kernel.forward(q, k, v, attention_metadata=mask)

        inner_forward.assert_called_once_with(kernel, q, k, v, attention_metadata=mask)
        self.assertIs(result, q)


if __name__ == "__main__":
    unittest.main()
