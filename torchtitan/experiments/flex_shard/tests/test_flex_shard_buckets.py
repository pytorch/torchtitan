#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Tests for FlexShard Phase 2b: BucketSpec and bucket validation.

Usage:
    # Single-process tests (no GPU/NCCL required):
    python -m pytest \
      torchtitan/experiments/flex_shard/tests/test_flex_shard_buckets.py \
      -v -k "not Distributed"

    # Distributed correctness tests:
    python -m pytest -q -k Distributed \
      torchtitan/experiments/flex_shard/tests/test_flex_shard_buckets.py
"""

from types import ModuleType, SimpleNamespace
from typing import cast

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed.device_mesh import init_device_mesh
from torch.testing._internal.common_distributed import skip_if_lt_x_gpu
from torch.testing._internal.common_fsdp import (
    FSDPTest,
    FSDPTestMultiThread,
    get_devtype,
)
from torch.testing._internal.common_utils import run_tests, TestCase

from torchtitan.experiments.flex_shard import (
    BucketSpec,
    flex_shard,
    is_flex_shard_param,
    LocalStorageLayout,
    Placement,
)
from torchtitan.experiments.flex_shard.example.owned import Owned
from torchtitan.experiments.flex_shard.example.shard import per_param_placements, Shard
from torchtitan.experiments.flex_shard.flex_shard.bucket_runtime import (
    BucketCommContext,
    BucketRuntime,
)
from torchtitan.experiments.flex_shard.flex_shard.bucket_storage import (
    _assign_params_to_buckets,
    ParamInfo,
    ShardedBucketStorage,
)
from torchtitan.experiments.flex_shard.flex_shard.flex_shard import (
    _materialize_bucket_storages,
    PreparedFlexShardInputs,
)
from torchtitan.experiments.flex_shard.tests.common import (
    make_transformer_model,
    single_rank_cpu_mesh,
    single_rank_cuda_mesh,
    transformer_bucket_specs,
    transformer_inputs,
)


device_type = torch.device(get_devtype())


class _IncompletePlacement(Placement):
    def __eq__(self, other: object) -> bool:
        return isinstance(other, _IncompletePlacement)

    def __hash__(self) -> int:
        return hash(type(self))


class _TestPlacement(Placement):
    def __eq__(self, other: object) -> bool:
        return isinstance(other, _TestPlacement)

    def __hash__(self) -> int:
        return hash(type(self))

    def compute_local_shape(
        self, global_shape: torch.Size, rank: int, world_size: int
    ) -> torch.Size:
        return global_shape

    def extract_local_shard(
        self,
        param: torch.Tensor,
        rank: int,
        world_size: int,
    ) -> torch.Tensor:
        return param.contiguous()


class _PaddedShard(Shard):
    def __init__(self, padding_nbytes: int) -> None:
        super().__init__(0)
        self.padding_nbytes = padding_nbytes

    def local_storage_layout(
        self,
        global_shape: torch.Size,
        dtype: torch.dtype,
        rank: int,
        world_size: int,
    ) -> LocalStorageLayout:
        layout = super().local_storage_layout(global_shape, dtype, rank, world_size)
        return LocalStorageLayout(
            local_shape=layout.local_shape,
            local_numel=layout.local_numel,
            storage_nbytes=layout.storage_nbytes + self.padding_nbytes,
        )


# ---------------------------------------------------------------------------
# Bucket communication scheduling tests (single-process, no NCCL)
# ---------------------------------------------------------------------------


class TestBucketCommScheduling(TestCase):
    @staticmethod
    def _context_with_reshard_flags(flags: list[bool]) -> BucketCommContext:
        context = BucketCommContext(
            device_handle=ModuleType("dummy_device_handle"),
            unshard_stream=cast(torch.Stream, object()),
            reduce_grad_stream=cast(torch.Stream, object()),
        )
        context.buckets = [
            cast(
                BucketRuntime,
                SimpleNamespace(
                    bucket_storage=SimpleNamespace(_reshard_after_forward=flag),
                ),
            )
            for flag in flags
        ]
        return context

    def test_defer_reduce_grad_for_previous_bucket_backward_unshard(self):
        context = self._context_with_reshard_flags([True, True, True])

        self.assertFalse(
            context.should_defer_reduce_grad_for_backward_prefetch(
                context.buckets[0],
            )
        )
        self.assertIs(
            context.next_backward_unshard_bucket(context.buckets[1]),
            context.buckets[0],
        )
        self.assertTrue(
            context.should_defer_reduce_grad_for_backward_prefetch(
                context.buckets[1],
            )
        )
        self.assertIs(
            context.next_backward_unshard_bucket(context.buckets[2]),
            context.buckets[1],
        )
        self.assertTrue(
            context.should_defer_reduce_grad_for_backward_prefetch(
                context.buckets[2],
            )
        )

    def test_reduce_grad_defer_depends_on_previous_bucket_not_current_bucket(self):
        context = self._context_with_reshard_flags([True, False, True])

        self.assertTrue(
            context.should_defer_reduce_grad_for_backward_prefetch(
                context.buckets[1],
            )
        )
        self.assertFalse(
            context.should_defer_reduce_grad_for_backward_prefetch(
                context.buckets[2],
            )
        )


# ---------------------------------------------------------------------------
# Shard mesh tests
# ---------------------------------------------------------------------------


class TestFlexShardMesh(TestCase):
    """Test FlexShard shard mesh validation."""

    def test_accepts_1d_mesh(self):
        from torchtitan.experiments.flex_shard.flex_shard.utils import (
            _validate_flex_shard_mesh,
        )

        with single_rank_cuda_mesh() as mesh:
            _validate_flex_shard_mesh(mesh)

    def test_rejects_cpu_mesh(self):
        from torchtitan.experiments.flex_shard.flex_shard.utils import (
            _validate_flex_shard_mesh,
        )

        with single_rank_cpu_mesh() as mesh:
            with self.assertRaisesRegex(NotImplementedError, "CUDA DeviceMesh"):
                _validate_flex_shard_mesh(mesh)

    def test_rejects_multi_dim_mesh(self):
        from torchtitan.experiments.flex_shard.flex_shard.utils import (
            _validate_flex_shard_mesh,
        )

        with single_rank_cuda_mesh():
            mesh = init_device_mesh(
                "cuda",
                (1, 1),
                mesh_dim_names=("fsdp", "tp"),
            )
            with self.assertRaisesRegex(ValueError, "1D DeviceMesh"):
                _validate_flex_shard_mesh(mesh)


# ---------------------------------------------------------------------------
# Bucket assignment tests (single-process, no NCCL)
# ---------------------------------------------------------------------------


class TestBucketAssignment(TestCase):
    """Test _assign_params_to_buckets."""

    def test_assigns_params_to_correct_bucket(self):
        """Params match the right bucket by fnmatch."""
        from torchtitan.experiments.flex_shard.flex_shard.bucket_storage import (
            _assign_params_to_buckets,
        )

        fqns = ["attn.weight", "attn.bias", "ffn.weight", "ffn.bias"]
        buckets = [
            BucketSpec(
                ["attn.*"],
                placement_fn=per_param_placements,
                reshard_after_forward=False,
            ),
            BucketSpec(
                ["ffn.*"],
                placement_fn=per_param_placements,
                reshard_after_forward=False,
            ),
        ]
        result = _assign_params_to_buckets(fqns, buckets)

        self.assertEqual(result[0], ["attn.weight", "attn.bias"])
        self.assertEqual(result[1], ["ffn.weight", "ffn.bias"])

    def test_rejects_orphan_params(self):
        """Params matching zero buckets raise ValueError."""
        from torchtitan.experiments.flex_shard.flex_shard.bucket_storage import (
            _assign_params_to_buckets,
        )

        fqns = ["attn.weight", "norm.weight"]
        buckets = [
            BucketSpec(
                ["attn.*"],
                placement_fn=per_param_placements,
                reshard_after_forward=False,
            )
        ]
        with self.assertRaises(ValueError, msg="not covered by any bucket"):
            _assign_params_to_buckets(fqns, buckets)

    def test_rejects_overlapping_params(self):
        """Params matching multiple buckets raise ValueError."""
        from torchtitan.experiments.flex_shard.flex_shard.bucket_storage import (
            _assign_params_to_buckets,
        )

        fqns = ["attn.weight"]
        buckets = [
            BucketSpec(
                ["attn.*"],
                placement_fn=per_param_placements,
                reshard_after_forward=False,
            ),
            BucketSpec(
                ["*"],
                placement_fn=per_param_placements,
                reshard_after_forward=False,
            ),
        ]
        with self.assertRaises(ValueError, msg="matched multiple buckets"):
            _assign_params_to_buckets(fqns, buckets)


# ---------------------------------------------------------------------------
# Placement consistency tests (single-process, no NCCL)
# ---------------------------------------------------------------------------


class TestBucketPlacementValidation(TestCase):
    """Test explicit placement and bucket validation."""

    @staticmethod
    def _named_params(
        dtypes: dict[str, torch.dtype] | None = None,
    ) -> list[tuple[str, nn.Parameter]]:
        dtypes = dtypes or {}
        return [
            (
                fqn,
                nn.Parameter(torch.empty(2, 2, dtype=dtypes.get(fqn, torch.float32))),
            )
            for fqn in ("a.weight", "b.weight")
        ]

    def test_rejects_unhashable_placement_equality(self):
        """Placement subclasses that override equality must remain hashable."""
        with self.assertRaisesRegex(TypeError, "__hash__"):

            class _UnhashablePlacement(Placement):
                def __eq__(self, other: object) -> bool:
                    return isinstance(other, _UnhashablePlacement)

    def test_rejects_missing_or_extra_placements(self):
        """Placement validation requires exact managed parameter coverage."""
        from torchtitan.experiments.flex_shard.flex_shard.utils import (
            _validate_placements,
        )

        with single_rank_cpu_mesh() as mesh:
            named_params = self._named_params()
            with self.assertRaisesRegex(ValueError, "missing placements"):
                _validate_placements(
                    {"a.weight": (Shard(0),)},
                    named_params,
                    mesh,
                )

            with self.assertRaisesRegex(ValueError, "unexpected placements"):
                _validate_placements(
                    {
                        "a.weight": (Shard(0),),
                        "b.weight": (Shard(0),),
                        "extra.weight": (Shard(0),),
                    },
                    named_params,
                    mesh,
                )

    def test_rejects_non_placement_object(self):
        """Placement validation requires Placement instances."""
        from torchtitan.experiments.flex_shard.flex_shard.utils import (
            _validate_placements,
        )

        with single_rank_cpu_mesh() as mesh:
            named_params = self._named_params()
            with self.assertRaisesRegex(TypeError, "Placement instances"):
                _validate_placements(
                    {
                        "a.weight": (object(),),
                        "b.weight": (object(),),
                    },
                    named_params,
                    mesh,
                )

    def test_rejects_incomplete_placement_contract(self):
        """Placement subclasses must implement the storage layout contract."""
        from torchtitan.experiments.flex_shard.flex_shard.utils import (
            _validate_placements,
        )

        with single_rank_cpu_mesh() as mesh:
            named_params = self._named_params()
            with self.assertRaisesRegex(TypeError, "storage layout contract"):
                _validate_placements(
                    {
                        "a.weight": (_IncompletePlacement(),),
                        "b.weight": (_IncompletePlacement(),),
                    },
                    named_params,
                    mesh,
                )

    def test_accepts_valid_placement_subclasses(self):
        """Core validation does not import or hard-code the example Shard."""
        from torchtitan.experiments.flex_shard.flex_shard.utils import (
            _validate_placements,
        )

        with single_rank_cpu_mesh() as mesh:
            _validate_placements(
                {
                    "a.weight": (_TestPlacement(),),
                    "b.weight": (_TestPlacement(),),
                },
                self._named_params(),
                mesh,
            )

    def test_copy_param_to_storage_accepts_non_contiguous_shard_view(self):
        """Placement local payloads may be non-contiguous views."""

        class _TransposePlacement(Placement):
            def __eq__(self, other: object) -> bool:
                return isinstance(other, _TransposePlacement)

            def __hash__(self) -> int:
                return hash(type(self))

            def compute_local_shape(
                self, global_shape: torch.Size, rank: int, world_size: int
            ) -> torch.Size:
                if world_size != 1:
                    raise AssertionError("Expected single-rank test placement.")
                return torch.Size([global_shape[1], global_shape[0]])

            def extract_local_shard(
                self,
                param: torch.Tensor,
                rank: int,
                world_size: int,
            ) -> torch.Tensor:
                if world_size != 1:
                    raise AssertionError("Expected single-rank test placement.")
                return param.t()

        placement = _TransposePlacement()
        param = torch.arange(6, dtype=torch.float32).view(2, 3)
        local_shape = placement.compute_local_shape(param.shape, rank=0, world_size=1)
        info = ParamInfo(
            fqn="weight",
            global_shape=param.shape,
            global_stride=param.stride(),
            dtype=param.dtype,
            requires_grad=True,
            placements=(placement,),
            local_shape=local_shape,
            local_numel=param.numel(),
            storage_nbytes=param.numel() * param.element_size(),
            global_numel=param.numel(),
        )
        byte_storage = torch.empty(info.storage_nbytes, dtype=torch.uint8)

        local_payload = placement.extract_local_shard(param, rank=0, world_size=1)
        self.assertFalse(local_payload.is_contiguous())
        placement.copy_param_to_storage(byte_storage, info, param, rank=0, world_size=1)

        copied = byte_storage.view(torch.float32).view(local_shape)
        self.assertEqual(copied, local_payload.contiguous())

    def test_rejects_shard_dim_out_of_range(self):
        """Placement layout validation is front-loaded."""
        from torchtitan.experiments.flex_shard.flex_shard.utils import (
            _validate_placements,
        )

        with single_rank_cpu_mesh() as mesh:
            with self.assertRaisesRegex(ValueError, "invalid for parameter"):
                _validate_placements(
                    {"scalar": (Shard(0),)},
                    [("scalar", nn.Parameter(torch.empty(())))],
                    mesh,
                )

    def test_rejects_owned_owner_rank_out_of_range(self):
        """Owned placement owner_rank must be valid for the mesh."""
        from torchtitan.experiments.flex_shard.flex_shard.utils import (
            _validate_placements,
        )

        with single_rank_cpu_mesh() as mesh:
            with self.assertRaisesRegex(ValueError, "owner_rank"):
                _validate_placements(
                    {"weight": (Owned(1),)},
                    [("weight", nn.Parameter(torch.empty(2, 2)))],
                    mesh,
                )

    def test_owned_reduce_grad_single_rank(self):
        """Owned placement can reduce gradients through its placement contract."""
        with single_rank_cpu_mesh() as mesh:
            placement = Owned(0)
            global_shape = torch.Size([2, 2])
            info = ParamInfo(
                fqn="weight",
                global_shape=global_shape,
                global_stride=(2, 1),
                dtype=torch.float32,
                requires_grad=True,
                placements=(placement,),
                local_shape=global_shape,
                local_numel=4,
                global_numel=4,
            )
            grad = torch.ones(global_shape)

            prepared = placement.prepare_reduce_grad([grad], [info], mesh, None)
            result = placement.reduce_prepared_grad(prepared).sharded_grads[0]

            self.assertEqual(result, grad)

    def test_rejects_mixed_dtypes(self):
        """Parameters in one bucket must share the same storage dtype."""
        from torchtitan.experiments.flex_shard.flex_shard.utils import (
            _validate_bucket_uniform_dtype_and_placement,
        )

        assignments = [["a.weight", "b.weight"]]
        placements = {
            "a.weight": (Shard(0),),
            "b.weight": (Shard(0),),
        }
        buckets = [
            BucketSpec(
                ["*"],
                placement_fn=per_param_placements,
                reshard_after_forward=False,
            )
        ]
        with self.assertRaisesRegex(ValueError, "mixed parameter dtypes"):
            _validate_bucket_uniform_dtype_and_placement(
                assignments,
                placements,
                buckets,
                self._named_params({"b.weight": torch.bfloat16}),
            )

    def test_rejects_mixed_placements_in_one_bucket(self):
        """A bucket collective uses one placement layout for all params."""
        from torchtitan.experiments.flex_shard.flex_shard.utils import (
            _validate_bucket_uniform_dtype_and_placement,
        )

        assignments = [["a.weight", "b.weight"]]
        placements = {
            "a.weight": (Shard(0),),
            "b.weight": (Shard(1),),
        }
        buckets = [
            BucketSpec(
                ["*"],
                placement_fn=per_param_placements,
                reshard_after_forward=False,
            )
        ]
        with self.assertRaisesRegex(ValueError, "mixed placements"):
            _validate_bucket_uniform_dtype_and_placement(
                assignments,
                placements,
                buckets,
                self._named_params(),
            )

    def test_flex_shard_rejects_mixed_placements_before_materializing(self):
        """Invalid bucket placement config should not partially shard the module."""

        class TwoParamModule(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.a = nn.Parameter(torch.empty(2, 2))
                self.b = nn.Parameter(torch.empty(2, 2))

        def mixed_placements(named_params, mesh):
            return {
                "a": (Shard(0),),
                "b": (Shard(1),),
            }

        with single_rank_cuda_mesh() as mesh:
            model = TwoParamModule()
            with self.assertRaisesRegex(ValueError, "mixed placements"):
                flex_shard(
                    model,
                    mesh,
                    buckets=[
                        BucketSpec(
                            ["*"],
                            placement_fn=mixed_placements,
                            reshard_after_forward=False,
                        )
                    ],
                )
            self.assertFalse(hasattr(model, "_sharded_bucket_storages"))


# ---------------------------------------------------------------------------
# Bucket storage layout tests (single-process, no NCCL)
# ---------------------------------------------------------------------------


class TestBucketStorageLayout(FSDPTestMultiThread):
    """Test ParamInfo and ShardedBucketStorage layout for bucket materialization."""

    @property
    def world_size(self) -> int:
        return 2

    def test_create_param_infos_uses_sequential_byte_offsets(self):
        mesh = init_device_mesh("cpu", (self.world_size,), mesh_dim_names=("fsdp",))
        args, model = make_transformer_model()
        named_params = [
            ("tok_embeddings.weight", model.tok_embeddings.weight),
            ("pos_embeddings.weight", model.pos_embeddings.weight),
        ]
        placements = {fqn: (Shard(0),) for fqn, _ in named_params}

        infos, total_bytes = ShardedBucketStorage.create_param_infos(
            named_params, mesh, placements
        )

        tok_embeddings = infos["tok_embeddings.weight"]
        pos_embeddings = infos["pos_embeddings.weight"]
        self.assertEqual(
            tok_embeddings.local_shape,
            Shard(0).compute_local_shape(
                torch.Size([args.vocab_size, args.dim]),
                self.rank,
                self.world_size,
            ),
        )
        self.assertEqual(tok_embeddings.byte_offset, 0)
        self.assertEqual(
            pos_embeddings.local_shape,
            Shard(0).compute_local_shape(
                torch.Size([args.max_seq_len, args.dim]),
                self.rank,
                self.world_size,
            ),
        )
        self.assertEqual(
            pos_embeddings.byte_offset,
            (args.vocab_size // 2) * args.dim * torch.float32.itemsize,
        )
        self.assertEqual(
            total_bytes,
            ((args.vocab_size // 2) + (args.max_seq_len // 2))
            * args.dim
            * torch.float32.itemsize,
        )

    def test_create_param_infos_uses_placement_owned_storage_layout(self):
        mesh = init_device_mesh("cpu", (self.world_size,), mesh_dim_names=("fsdp",))
        args, model = make_transformer_model()
        named_params = [
            ("tok_embeddings.weight", model.tok_embeddings.weight),
            ("pos_embeddings.weight", model.pos_embeddings.weight),
        ]
        padding_nbytes = 16
        placements = {fqn: (_PaddedShard(padding_nbytes),) for fqn, _ in named_params}

        infos, total_bytes = ShardedBucketStorage.create_param_infos(
            named_params, mesh, placements
        )

        tok_embeddings = infos["tok_embeddings.weight"]
        pos_embeddings = infos["pos_embeddings.weight"]
        tok_nbytes = (
            tok_embeddings.local_numel * tok_embeddings.dtype.itemsize + padding_nbytes
        )
        pos_nbytes = (
            pos_embeddings.local_numel * pos_embeddings.dtype.itemsize + padding_nbytes
        )
        self.assertEqual(tok_embeddings.storage_nbytes, tok_nbytes)
        self.assertEqual(pos_embeddings.byte_offset, tok_nbytes)
        self.assertEqual(pos_embeddings.storage_nbytes, pos_nbytes)
        self.assertEqual(total_bytes, tok_nbytes + pos_nbytes)
        self.assertEqual(
            tok_embeddings.local_shape,
            Shard(0).compute_local_shape(
                torch.Size([args.vocab_size, args.dim]),
                self.rank,
                self.world_size,
            ),
        )

    def test_materialized_params_are_views_into_bucket_storage(self):
        mesh = init_device_mesh("cpu", (self.world_size,), mesh_dim_names=("fsdp",))
        args, model = make_transformer_model()
        named_params = list(model.named_parameters())
        placements = {fqn: (Shard(0),) for fqn, _ in named_params}
        buckets = transformer_bucket_specs(
            args.n_layers,
            reshard_after_forward=False,
        )
        assignments = _assign_params_to_buckets(
            [fqn for fqn, _ in named_params],
            buckets,
        )

        inputs = PreparedFlexShardInputs(
            named_params=named_params,
            shard_mesh=mesh,
            device=torch.device("cpu"),
            param_placements=placements,
            bucket_assignments=assignments,
        )
        bucket_storages, fqn_to_bucket_spec = _materialize_bucket_storages(
            model,
            inputs,
            buckets,
        )

        self.assertEqual(len(bucket_storages), len(buckets))
        self.assertIs(fqn_to_bucket_spec["tok_embeddings.weight"], buckets[0])
        self.assertIs(fqn_to_bucket_spec["output.weight"], buckets[-1])

        current_params = dict(model.named_parameters())
        for bucket_storage in bucket_storages:
            storage_ptr = bucket_storage.byte_storage.untyped_storage().data_ptr()
            for fqn, info in bucket_storage.param_infos.items():
                param = current_params[fqn]
                local_view = bucket_storage.get_local_view(fqn)

                self.assertEqual(param.shape, info.local_shape)
                self.assertEqual(
                    param.untyped_storage().data_ptr(),
                    storage_ptr,
                )
                self.assertEqual(
                    local_view.untyped_storage().data_ptr(),
                    storage_ptr,
                )
                self.assertEqual(param, local_view)
                self.assertTrue(is_flex_shard_param(param))

    def test_materialized_params_match_placement_local_shards(self):
        mesh = init_device_mesh("cpu", (self.world_size,), mesh_dim_names=("fsdp",))

        class TinyModule(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.a = nn.Parameter(torch.arange(3, dtype=torch.float32).view(1, 3))
                self.b = nn.Parameter(torch.arange(6, dtype=torch.float32).view(3, 2))

        model = TinyModule()
        named_params = list(model.named_parameters())
        padding_nbytes = 16
        placements = {fqn: (_PaddedShard(padding_nbytes),) for fqn, _ in named_params}
        buckets = [
            BucketSpec(
                ["*"],
                placement_fn=per_param_placements,
                reshard_after_forward=False,
            )
        ]
        assignments = [[fqn for fqn, _ in named_params]]

        inputs = PreparedFlexShardInputs(
            named_params=named_params,
            shard_mesh=mesh,
            device=torch.device("cpu"),
            param_placements=placements,
            bucket_assignments=assignments,
        )
        bucket_storages, _ = _materialize_bucket_storages(
            model,
            inputs,
            buckets,
        )

        bucket_storage = bucket_storages[0]
        current_params = dict(model.named_parameters())
        original_params = dict(named_params)
        for fqn, info in bucket_storage.param_infos.items():
            param = current_params[fqn]
            expected = Shard(0).extract_local_shard(
                original_params[fqn].detach(),
                self.rank,
                self.world_size,
            )
            self.assertEqual(info.local_shape, expected.shape)
            self.assertEqual(
                info.storage_nbytes,
                expected.numel() * expected.dtype.itemsize + padding_nbytes,
            )
            self.assertEqual(param, expected)
            self.assertEqual(param, bucket_storage.get_local_view(fqn))

    def test_owned_materialized_params_match_owner_rank(self):
        mesh = init_device_mesh("cpu", (self.world_size,), mesh_dim_names=("fsdp",))

        class TinyModule(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.weight = nn.Parameter(
                    torch.arange(6, dtype=torch.float32).view(3, 2)
                )

        def owned_rank0(named_params, mesh):
            return {fqn: (Owned(0),) for fqn, _ in named_params}

        model = TinyModule()
        named_params = list(model.named_parameters())
        placements = {fqn: (Owned(0),) for fqn, _ in named_params}
        buckets = [
            BucketSpec(
                ["*"],
                placement_fn=owned_rank0,
                reshard_after_forward=False,
            )
        ]

        inputs = PreparedFlexShardInputs(
            named_params=named_params,
            shard_mesh=mesh,
            device=torch.device("cpu"),
            param_placements=placements,
            bucket_assignments=[[fqn for fqn, _ in named_params]],
        )
        bucket_storages, _ = _materialize_bucket_storages(
            model,
            inputs,
            buckets,
        )

        bucket_storage = bucket_storages[0]
        param = dict(model.named_parameters())["weight"]
        if self.rank == 0:
            self.assertEqual(param.shape, torch.Size([3, 2]))
            self.assertEqual(param, named_params[0][1].detach())
            self.assertEqual(bucket_storage.total_bytes, 6 * torch.float32.itemsize)
        else:
            self.assertEqual(param.shape, torch.Size([0, 0]))
            self.assertEqual(param.numel(), 0)
            self.assertEqual(bucket_storage.total_bytes, 0)

    def test_owned_unshard_broadcasts_from_owner(self):
        mesh = init_device_mesh("cpu", (self.world_size,), mesh_dim_names=("fsdp",))
        placement = Owned(0)
        info = ParamInfo(
            fqn="weight",
            global_shape=torch.Size([2, 2]),
            global_stride=(2, 1),
            dtype=torch.float32,
            requires_grad=True,
            placements=(placement,),
            local_shape=placement.compute_local_shape(
                torch.Size([2, 2]),
                self.rank,
                self.world_size,
            ),
            local_numel=placement.compute_local_numel(
                torch.Size([2, 2]),
                self.rank,
                self.world_size,
            ),
            global_numel=4,
        )
        expected = torch.arange(4, dtype=torch.float32).view(2, 2)
        local = expected.clone() if self.rank == 0 else torch.empty(0)

        prepared = placement.prepare_unshard_bucket([local], [info], mesh, None)
        placement.run_prepared_unshard(prepared)
        result = placement.finish_prepared_unshard(prepared).full_params[0]

        self.assertEqual(result, expected)


# ---------------------------------------------------------------------------
# Distributed per-bucket ShardedBucketStorage tests (torchrun only)
# ---------------------------------------------------------------------------


class TestDistributedBuckets(FSDPTest):
    """Multi-process correctness tests for bucketed FlexShard.

    Run with:
        python -m pytest -q -k Distributed \
          torchtitan/experiments/flex_shard/tests/test_flex_shard_buckets.py
    """

    @property
    def world_size(self) -> int:
        return 2

    def _init_mesh(self):
        return init_device_mesh(
            device_type.type,
            (self.world_size,),
            mesh_dim_names=("fsdp",),
        )

    def _flex_shard(self, model, mesh, **kwargs):
        from torchtitan.experiments.flex_shard import flex_shard
        from torchtitan.experiments.flex_shard.example.shard import per_param_placements

        kwargs.setdefault(
            "buckets",
            [
                BucketSpec(
                    ["*"],
                    placement_fn=per_param_placements,
                    reshard_after_forward=False,
                )
            ],
        )
        return flex_shard(
            model,
            mesh,
            **kwargs,
        )

    @skip_if_lt_x_gpu(2)
    def test_multi_bucket_forward_correct(self):
        """Model with explicit buckets produces correct forward output."""
        mesh = self._init_mesh()
        args, model = make_transformer_model(device=device_type.type)
        for p in model.parameters():
            dist.broadcast(p.data, src=0)

        x = transformer_inputs(args, device=device_type.type)
        dist.broadcast(x, src=0)
        ref_output = model(x).clone()

        self._flex_shard(
            model,
            mesh,
            buckets=transformer_bucket_specs(
                args.n_layers,
                reshard_after_forward=False,
            ),
        )

        self.assertEqual(len(model.sharded_bucket_storages), 5)
        output = model(x)
        self.assertEqual(output, ref_output)

    @skip_if_lt_x_gpu(2)
    def test_multi_bucket_state_dict_sharded(self):
        """state_dict returns sharded params across all buckets."""
        mesh = self._init_mesh()
        args, model = make_transformer_model(device=device_type.type)
        for p in model.parameters():
            dist.broadcast(p.data, src=0)

        self._flex_shard(
            model,
            mesh,
            buckets=transformer_bucket_specs(
                args.n_layers,
                reshard_after_forward=False,
            ),
        )

        sd = model.state_dict()
        self.assertEqual(
            sd["tok_embeddings.weight"].shape,
            (args.vocab_size // self.world_size, args.dim),
        )
        self.assertEqual(
            sd["output.weight"].shape,
            (args.vocab_size // self.world_size, args.dim),
        )


if __name__ == "__main__":
    run_tests()
