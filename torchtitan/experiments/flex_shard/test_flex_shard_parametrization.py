#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Tests for FlexShard parametrization (Phase 2a).

Usage:
    # Single-process tests (no GPU/NCCL required):
    python -m pytest test_flex_shard_parametrization.py -v -k "not Distributed"

    # Distributed correctness tests:
    torchrun --nproc_per_node=2 test_flex_shard_parametrization.py
"""

import unittest
import weakref

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Guard behavior tests (single-process, no NCCL)
# ---------------------------------------------------------------------------


class TestActiveParametrizationGuard(unittest.TestCase):
    """Test _active_parametrization guard and disable_active_parametrization."""

    def test_guard_disabled_returns_raw_shard(self):
        """With guard disabled, ShardParametrization returns input unchanged."""
        from torchtitan.experiments.flex_shard import (
            disable_active_parametrization,
            ShardParametrization,
        )

        param = ShardParametrization(shard_dim=0, group_name="fake", world_size=2)
        shard = torch.randn(4, 8)
        with disable_active_parametrization():
            result = param(shard)
        self.assertIs(result, shard)

    def test_guard_disabled_returns_raw_flat_shard(self):
        """With guard disabled, FlatShardParametrization returns input unchanged."""
        from torchtitan.experiments.flex_shard import (
            disable_active_parametrization,
            FlatShardParametrization,
        )

        param = FlatShardParametrization(
            group_name="fake",
            world_size=2,
            original_shape=torch.Size([4, 8]),
        )
        flat_shard = torch.randn(16)
        with disable_active_parametrization():
            result = param(flat_shard)
        self.assertIs(result, flat_shard)

    def test_guard_restores_after_context(self):
        """Guard restores to True after context manager exits."""
        import importlib

        fs = importlib.import_module("torchtitan.experiments.flex_shard.flex_shard")

        self.assertTrue(fs._active_parametrization)
        with fs.disable_active_parametrization():
            self.assertFalse(fs._active_parametrization)
        self.assertTrue(fs._active_parametrization)

    def test_guard_restores_on_exception(self):
        """Guard restores to True even if exception is raised."""
        import importlib

        fs = importlib.import_module("torchtitan.experiments.flex_shard.flex_shard")

        try:
            with fs.disable_active_parametrization():
                raise RuntimeError("test")
        except RuntimeError:
            pass
        self.assertTrue(fs._active_parametrization)


# ---------------------------------------------------------------------------
# Property registration tests (single-process, no NCCL)
# ---------------------------------------------------------------------------


class TestRegisterParametrization(unittest.TestCase):
    """Test _register_parametrization creates correct property getters."""

    def test_property_created_on_module(self):
        """Property getter is created on the module's dynamic subclass."""
        from torchtitan.experiments.flex_shard.flex_shard import (
            _register_parametrization,
            ShardParametrization,
        )

        module = nn.Linear(8, 4, bias=False)
        param = ShardParametrization(shard_dim=0, group_name="fake", world_size=2)
        _register_parametrization(module, {"weight": param})

        # The module's class should be a dynamic subclass
        self.assertIn("FlexShard", type(module).__name__)
        # Property should exist on the class
        self.assertIsInstance(type(module).__dict__["weight"], property)

    def test_state_dict_bypasses_property(self):
        """state_dict reads _parameters directly, not through property."""
        from torchtitan.experiments.flex_shard.flex_shard import (
            _register_parametrization,
            ShardParametrization,
        )

        module = nn.Linear(8, 4, bias=False)
        original_shape = module.weight.shape

        # Register parametrization with guard disabled so no NCCL needed
        param = ShardParametrization(shard_dim=0, group_name="fake", world_size=2)
        _register_parametrization(module, {"weight": param})

        # state_dict should return the raw parameter (bypasses property)
        sd = module.state_dict()
        self.assertEqual(sd["weight"].shape, original_shape)

    def test_multiple_params_on_same_module(self):
        """Multiple parameters can be parametrized on the same module."""
        from torchtitan.experiments.flex_shard.flex_shard import (
            _register_parametrization,
            ShardParametrization,
        )

        module = nn.Linear(8, 4)  # has weight and bias
        param_w = ShardParametrization(shard_dim=0, group_name="fake", world_size=2)
        param_b = ShardParametrization(shard_dim=0, group_name="fake", world_size=2)
        _register_parametrization(module, {"weight": param_w, "bias": param_b})

        self.assertIsInstance(type(module).__dict__["weight"], property)
        self.assertIsInstance(type(module).__dict__["bias"], property)


# ---------------------------------------------------------------------------
# Distributed correctness tests (torchrun only)
# ---------------------------------------------------------------------------


class TestDistributedParametrization(unittest.TestCase):
    """Multi-process correctness tests for parametrized FlexShard.

    Run with: torchrun --nproc_per_node=2 test_flex_shard_parametrization.py
    """

    @classmethod
    def setUpClass(cls):
        if not torch.distributed.is_initialized():
            torch.distributed.init_process_group(backend="nccl")
        cls.rank = torch.distributed.get_rank()
        cls.world_size = torch.distributed.get_world_size()
        torch.cuda.set_device(cls.rank % torch.cuda.device_count())

    @classmethod
    def tearDownClass(cls):
        if torch.distributed.is_initialized():
            torch.cuda.synchronize()
            torch.distributed.destroy_process_group()

    def _init_mesh(self):
        from torch.distributed.device_mesh import init_device_mesh

        return init_device_mesh("cuda", (self.world_size,), mesh_dim_names=("fsdp",))

    def _flex_shard(self, model, mesh, **kwargs):
        from torch.distributed.fsdp import DataParallelMeshDims

        from torchtitan.experiments.flex_shard import (
            BucketSpec,
            flex_shard,
            lift_params_to_global_spmd_mesh,
            per_param_placements,
        )

        lift_params_to_global_spmd_mesh(model, mesh)
        kwargs.setdefault("shard_placement_fn", per_param_placements)
        kwargs.setdefault("buckets", [BucketSpec(["*"])])
        return flex_shard(
            model,
            mesh,
            DataParallelMeshDims(shard="fsdp"),
            **kwargs,
        )

    def _current_device(self):
        return torch.device("cuda", torch.cuda.current_device())

    def _expected_chunk(self, tensor, chunks, dim, rank):
        result = list(torch.chunk(tensor, chunks, dim=dim))
        empty_shape = list(tensor.shape)
        empty_shape[dim] = 0
        while len(result) < chunks:
            result.append(tensor.new_empty(empty_shape))
        return result[rank].contiguous()

    def test_param_access_triggers_allgather(self):
        """Accessing module.weight returns the full (unsharded) tensor."""
        mesh = self._init_mesh()

        model = nn.Linear(8, 4, bias=False, device="cuda")
        # Broadcast weights so all ranks start with the same full tensor
        torch.distributed.broadcast(model.weight.data, src=0)
        full_ref = model.weight.data.clone()

        self._flex_shard(model, mesh)

        # Accessing model.weight should trigger all-gather via property
        result = model.weight
        torch.testing.assert_close(result, full_ref)

    def test_state_dict_returns_sharded(self):
        """state_dict() returns sharded params, not unsharded."""
        mesh = self._init_mesh()

        model = nn.Linear(8, 4, bias=False, device="cuda")
        torch.distributed.broadcast(model.weight.data, src=0)

        self._flex_shard(model, mesh)

        sd = model.state_dict()
        # state_dict bypasses property, returns local shard
        expected_rows = 4 // self.world_size
        self.assertEqual(sd["weight"].shape, (expected_rows, 8))

    def test_disable_guard_returns_sharded(self):
        """With guard disabled, param access returns raw sharded tensor."""
        from torchtitan.experiments.flex_shard import disable_active_parametrization

        mesh = self._init_mesh()

        model = nn.Linear(8, 4, bias=False, device="cuda")
        torch.distributed.broadcast(model.weight.data, src=0)

        self._flex_shard(model, mesh)

        with disable_active_parametrization():
            result = model.weight
        expected_rows = 4 // self.world_size
        self.assertEqual(result.shape, (expected_rows, 8))

    def test_forward_produces_correct_output(self):
        """Forward pass through parametrized model produces correct results."""
        mesh = self._init_mesh()

        model = nn.Linear(8, 4, bias=False, device="cuda")
        torch.distributed.broadcast(model.weight.data, src=0)
        ref_weight = model.weight.data.clone()

        # Reference output
        x = torch.randn(2, 8, device="cuda")
        torch.distributed.broadcast(x, src=0)
        ref_output = x @ ref_weight.t()

        self._flex_shard(model, mesh)
        output = model(x)

        torch.testing.assert_close(output, ref_output)

    def test_eager_missing_batched_hook_raises(self):
        """Eager Shard buckets must not fall back to per-param functional AG."""
        from torchtitan.experiments.flex_shard import BucketSpec

        class TwoLinears(nn.Module):
            def __init__(self):
                super().__init__()
                self.a = nn.Linear(8, 8, bias=False, device="cuda")
                self.b = nn.Linear(8, 8, bias=False, device="cuda")

            def forward(self, x):
                return self.b(self.a(x))

        mesh = self._init_mesh()
        model = TwoLinears()
        for param in model.parameters():
            torch.distributed.broadcast(param.data, src=0)

        self._flex_shard(
            model,
            mesh,
            buckets=[BucketSpec(["a.*", "b.*"], reshard_after_forward=True)],
        )

        x = torch.randn(2, 8, device="cuda")
        torch.distributed.broadcast(x, src=0)
        with self.assertRaisesRegex(
            RuntimeError,
            "fall back to per-parameter _c10d_functional collectives",
        ):
            model(x)

    def test_eager_buckets_share_root_comm_context(self):
        """All eager Shard buckets share one root-owned AG/RS stream context."""
        from torchtitan.experiments.flex_shard import BucketSpec

        class TwoLinears(nn.Module):
            def __init__(self):
                super().__init__()
                self.a = nn.Linear(8, 8, bias=False, device="cuda")
                self.b = nn.Linear(8, 8, bias=False, device="cuda")

            def forward(self, x):
                return self.b(self.a(x))

        mesh = self._init_mesh()
        model = TwoLinears()
        for param in model.parameters():
            torch.distributed.broadcast(param.data, src=0)

        self._flex_shard(
            model,
            mesh,
            buckets=[
                BucketSpec(["a.*"], reshard_after_forward=True),
                BucketSpec(["b.*"], reshard_after_forward=True),
            ],
        )

        self.assertEqual(len(model.eager_comm_contexts), 1)
        context = next(iter(model.eager_comm_contexts.values()))
        self.assertEqual(len(context.buckets), 2)
        self.assertIs(context.buckets[0].storage._module, model)
        self.assertIs(context.buckets[1].storage._module, model)

        x = torch.randn(2, 8, device="cuda")
        torch.distributed.broadcast(x, src=0)
        model(x).sum().backward()

    def test_eager_reshard_after_forward_uses_autograd_bucket(self):
        """RAF eager buckets return non-leaf full params and autograd RS grads."""
        from torchtitan.experiments.flex_shard import BucketSpec

        class UsesWeight(nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = nn.Parameter(torch.randn(4, 8, device="cuda"))
                self.weight_was_leaf = True
                self.weight_had_grad_fn = False

            def forward(self, x):
                weight = self.weight
                self.weight_was_leaf = weight.is_leaf
                self.weight_had_grad_fn = weight.grad_fn is not None
                return x @ weight.t()

        mesh = self._init_mesh()
        model = UsesWeight()
        torch.distributed.broadcast(model.weight.data, src=0)

        self._flex_shard(
            model,
            mesh,
            buckets=[BucketSpec(["*"], reshard_after_forward=True)],
        )

        x = torch.randn(2, 8, device="cuda")
        torch.distributed.broadcast(x, src=0)
        output = model(x)
        self.assertFalse(model.weight_was_leaf)
        self.assertTrue(model.weight_had_grad_fn)

        output.sum().backward()
        expected_full_grad = torch.ones_like(output).t() @ x
        expected_local_grad = self._expected_chunk(
            expected_full_grad,
            self.world_size,
            0,
            self.rank,
        )
        torch.testing.assert_close(
            model._parameters["weight"].grad,
            expected_local_grad,
        )

    def test_eager_reshard_after_forward_raises_for_shard_dim1(self):
        """RAF eager custom buckets reject unsupported Shard dimensions."""
        from torchtitan.experiments.flex_shard import BucketSpec, Shard

        mesh = self._init_mesh()
        model = nn.Linear(8, 4, bias=False, device="cuda")
        torch.distributed.broadcast(model.weight.data, src=0)

        def shard_dim1(named_params, _mesh):
            return {fqn: (Shard(1),) for fqn, _ in named_params}

        with self.assertRaisesRegex(
            NotImplementedError,
            r"only CUDA Shard\(0\) buckets.*Shard dimension is \[1\]",
        ):
            self._flex_shard(
                model,
                mesh,
                shard_placement_fn=shard_dim1,
                buckets=[BucketSpec(["*"], reshard_after_forward=True)],
            )

    def test_eager_reshard_after_forward_prefetches_next_bucket(self):
        """RAF eager buckets prefetch the next bucket as raw AG state."""
        from torchtitan.experiments.flex_shard import BucketSpec

        class InspectLinear(nn.Module):
            def __init__(self, name):
                super().__init__()
                self.name = name
                self.weight = nn.Parameter(torch.randn(8, 8, device="cuda"))
                self.pending_debug_fqn = None

            def forward(self, x):
                root = self._root_ref()
                context = next(iter(root.eager_comm_contexts.values()))
                pending = context.pending
                self.pending_debug_fqn = (
                    pending.bucket.debug_fqn if pending is not None else None
                )
                return x @ self.weight.t()

        class TwoLinears(nn.Module):
            def __init__(self):
                super().__init__()
                self.a = InspectLinear("a")
                self.b = InspectLinear("b")
                self.a._root_ref = weakref.ref(self)
                self.b._root_ref = weakref.ref(self)

            def forward(self, x):
                return self.b(self.a(x))

        mesh = self._init_mesh()
        model = TwoLinears()
        for param in model.parameters():
            torch.distributed.broadcast(param.data, src=0)

        self._flex_shard(
            model,
            mesh,
            buckets=[
                BucketSpec(["a.*"], reshard_after_forward=True),
                BucketSpec(["b.*"], reshard_after_forward=True),
            ],
        )

        x = torch.randn(2, 8, device="cuda")
        torch.distributed.broadcast(x, src=0)
        output = model(x)
        self.assertEqual(model.a.pending_debug_fqn, "b")
        self.assertIsNone(model.b.pending_debug_fqn)
        context = next(iter(model.eager_comm_contexts.values()))
        self.assertIsNone(context.pending)

        output.sum().backward()

    def test_global_spmd_mesh_param_access(self):
        """Global SPMD mesh path derives the DP mesh and rewraps DTensor params."""
        from torch.distributed.device_mesh import init_device_mesh
        from torch.distributed.fsdp import DataParallelMeshDims
        from torch.distributed.tensor import DTensor, Replicate

        from torchtitan.experiments.flex_shard import (
            BucketSpec,
            flex_shard,
            per_param_placements,
        )

        mesh = init_device_mesh(
            "cuda",
            (self.world_size, 1),
            mesh_dim_names=("fsdp", "tp"),
        )

        model = nn.Linear(8, 4, bias=False, device="cuda")
        torch.distributed.broadcast(model.weight.data, src=0)
        full_ref = model.weight.detach().clone()
        model.weight = nn.Parameter(
            DTensor.from_local(
                model.weight.detach(),
                mesh,
                [Replicate(), Replicate()],
                run_check=False,
                grad_placements=[Replicate(), Replicate()],
            )
        )

        flex_shard(
            model,
            mesh,
            DataParallelMeshDims(shard="fsdp"),
            shard_placement_fn=per_param_placements,
            buckets=[BucketSpec(["*"])],
        )

        result = model.weight
        self.assertIsInstance(result, DTensor)
        self.assertEqual(result.device_mesh.mesh_dim_names, ("tp",))
        torch.testing.assert_close(result.to_local(), full_ref)

    def test_cpu_init_distribute_tensor_to_cuda_storage(self):
        """CPU-built params become CUDA sharded storage after DTensor distribute."""
        from torch.distributed.fsdp import DataParallelMeshDims
        from torch.distributed.tensor import distribute_tensor, Replicate

        from torchtitan.experiments.flex_shard import (
            BucketSpec,
            flex_shard,
            per_param_placements,
        )

        mesh = self._init_mesh()
        device = self._current_device()

        torch.manual_seed(42)
        model = nn.Linear(8, 4, bias=False, device="cpu")
        full_ref = model.weight.detach().clone().to(device)
        model.weight = nn.Parameter(
            distribute_tensor(model.weight, mesh, [Replicate()]),
            requires_grad=model.weight.requires_grad,
        )
        self.assertEqual(model.weight.to_local().device, device)

        flex_shard(
            model,
            mesh,
            DataParallelMeshDims(shard="fsdp"),
            shard_placement_fn=per_param_placements,
            buckets=[BucketSpec(["*"])],
        )

        self.assertEqual(model.dstorage.byte_storage.device, device)
        self.assertEqual(model.state_dict()["weight"].device, device)
        torch.testing.assert_close(model.weight, full_ref)

    def test_cpu_init_global_spmd_2d_storage_on_cuda(self):
        """CPU-built params support full SPMD mesh before FlexShard."""
        from torch.distributed.device_mesh import init_device_mesh
        from torch.distributed.fsdp import DataParallelMeshDims
        from torch.distributed.tensor import distribute_tensor, DTensor, Replicate

        from torchtitan.experiments.flex_shard import (
            BucketSpec,
            flex_shard,
            per_param_placements,
        )

        mesh = init_device_mesh(
            "cuda",
            (self.world_size, 1),
            mesh_dim_names=("fsdp", "tp"),
        )
        device = self._current_device()

        torch.manual_seed(42)
        model = nn.Linear(8, 4, bias=False, device="cpu")
        full_ref = model.weight.detach().clone().to(device)
        model.weight = nn.Parameter(
            distribute_tensor(model.weight, mesh, [Replicate(), Replicate()]),
            requires_grad=model.weight.requires_grad,
        )
        self.assertEqual(model.weight.to_local().device, device)

        flex_shard(
            model,
            mesh,
            DataParallelMeshDims(shard="fsdp"),
            shard_placement_fn=per_param_placements,
            buckets=[BucketSpec(["*"])],
        )

        self.assertEqual(model.dstorage.byte_storage.device, device)
        result = model.weight
        self.assertIsInstance(result, DTensor)
        self.assertEqual(result.device_mesh.mesh_dim_names, ("tp",))
        torch.testing.assert_close(result.to_local(), full_ref)

    def test_parallelize_without_materializing_state_keeps_cpu_until_flex_shard(self):
        """Non-materializing parallelize avoids full CUDA params before FlexShard."""
        from torch.distributed.fsdp import DataParallelMeshDims
        from torch.distributed.tensor import DTensor, Replicate

        from torchtitan.experiments.flex_shard import (
            BucketSpec,
            flex_shard,
            per_param_placements,
        )
        from torchtitan.protocols.module import Module
        from torchtitan.protocols.sharding import ShardingConfig
        from torchtitan.protocols.types import MeshAxisName

        mesh = self._init_mesh()
        device = self._current_device()
        Linear = Module.from_nn_module(nn.Linear)

        torch.manual_seed(42)
        model = Linear(8, 4, bias=False, device="cpu")
        model._sharding_config = ShardingConfig(
            state_shardings={
                "weight": {MeshAxisName.DP_SHARD: Replicate()},
            }
        )
        full_ref = model.weight.detach().clone()

        before = torch.cuda.memory_allocated(device)
        model.parallelize(
            mesh,
            wrap_forward=False,
            distribute_buffers=False,
            materialize_state=False,
        )
        after = torch.cuda.memory_allocated(device)
        self.assertEqual(after, before)
        self.assertEqual(model.weight.device.type, "cpu")
        self.assertNotIsInstance(model.weight, DTensor)

        flex_shard(
            model,
            mesh,
            DataParallelMeshDims(shard="fsdp"),
            shard_placement_fn=per_param_placements,
            buckets=[BucketSpec(["*"])],
        )

        expected = self._expected_chunk(
            full_ref.to(device), self.world_size, dim=0, rank=self.rank
        )
        self.assertEqual(model.dstorage.byte_storage.device, device)
        self.assertEqual(
            model.dstorage.byte_storage.numel(),
            expected.numel() * expected.element_size(),
        )
        torch.testing.assert_close(model.state_dict()["weight"], expected)
        torch.testing.assert_close(model.weight.cpu(), full_ref)

    def test_parallelize_without_materializing_state_extracts_tp_local_param(self):
        """Non-materialized SPMD metadata drives TP-local FlexShard storage."""
        from torch.distributed.device_mesh import init_device_mesh
        from torch.distributed.fsdp import DataParallelMeshDims
        from torch.distributed.tensor import DTensor, Replicate, Shard as DTensorShard

        from torchtitan.experiments.flex_shard import (
            BucketSpec,
            flex_shard,
            per_param_placements,
        )
        from torchtitan.protocols.module import Module
        from torchtitan.protocols.sharding import ShardingConfig
        from torchtitan.protocols.types import MeshAxisName

        mesh = init_device_mesh(
            "cuda",
            (1, self.world_size),
            mesh_dim_names=("fsdp", "tp"),
        )
        device = self._current_device()
        Linear = Module.from_nn_module(nn.Linear)

        torch.manual_seed(42)
        model = Linear(8, 4, bias=False, device="cpu")
        model._sharding_config = ShardingConfig(
            state_shardings={
                "weight": {
                    MeshAxisName.DP_SHARD: Replicate(),
                    MeshAxisName.TP: DTensorShard(0),
                },
            }
        )
        full_ref = model.weight.detach().clone()

        model.parallelize(
            mesh,
            activation_mesh=mesh["tp"],
            wrap_forward=False,
            distribute_buffers=False,
            materialize_state=False,
        )
        self.assertEqual(model.weight.device.type, "cpu")
        self.assertNotIsInstance(model.weight, DTensor)

        flex_shard(
            model,
            mesh,
            DataParallelMeshDims(shard="fsdp"),
            shard_placement_fn=per_param_placements,
            buckets=[BucketSpec(["*"])],
        )

        result = model.weight
        self.assertIsInstance(result, DTensor)
        self.assertEqual(result.device_mesh.mesh_dim_names, ("tp",))
        tp_rank = mesh.get_coordinate()[1]
        expected_local = self._expected_chunk(
            full_ref.to(device), self.world_size, dim=0, rank=tp_rank
        )
        torch.testing.assert_close(result.to_local(), expected_local)
        self.assertEqual(
            model.dstorage.byte_storage.numel(),
            expected_local.numel() * expected_local.element_size(),
        )

    def test_cpu_init_with_cpu_offload_keeps_storage_on_cpu(self):
        """CPU init and CPU offload are distinct: storage stays CPU, compute is CUDA."""
        from torch.distributed.fsdp import DataParallelMeshDims
        from torch.distributed.tensor import distribute_tensor, Replicate

        from torchtitan.experiments.flex_shard import (
            BucketSpec,
            flex_shard,
            OffloadPolicy,
            per_param_placements,
        )

        mesh = self._init_mesh()
        device = self._current_device()

        torch.manual_seed(42)
        model = nn.Linear(8, 4, bias=False, device="cpu")
        full_ref = model.weight.detach().clone().to(device)
        model.weight = nn.Parameter(
            distribute_tensor(model.weight, mesh, [Replicate()]),
            requires_grad=model.weight.requires_grad,
        )

        flex_shard(
            model,
            mesh,
            DataParallelMeshDims(shard="fsdp"),
            shard_placement_fn=per_param_placements,
            buckets=[
                BucketSpec(
                    patterns=["*"],
                    offload_policy=OffloadPolicy(pin_memory=True),
                )
            ],
        )

        self.assertEqual(model.dstorage.byte_storage.device.type, "cpu")
        self.assertTrue(model.dstorage.byte_storage.is_pinned())
        result = model.weight
        self.assertEqual(result.device, device)
        torch.testing.assert_close(result, full_ref)

    def test_plain_cpu_params_rejected_with_global_spmd(self):
        """Global SPMD FlexShard requires full-mesh state metadata."""
        from torch.distributed.fsdp import DataParallelMeshDims

        from torchtitan.experiments.flex_shard import (
            BucketSpec,
            flex_shard,
            per_param_placements,
        )

        mesh = self._init_mesh()
        model = nn.Linear(8, 4, bias=False, device="cpu")

        with self.assertRaisesRegex(ValueError, "DTensors.*annotated"):
            flex_shard(
                model,
                mesh,
                DataParallelMeshDims(shard="fsdp"),
                shard_placement_fn=per_param_placements,
                buckets=[BucketSpec(["*"])],
            )

    def test_checkpoint_roundtrip_with_guard(self):
        """Per-rank save/load roundtrip preserves sharded params and forward correctness.

        FlexShard params are plain tensors (not DTensors), so checkpoint uses
        per-rank torch.save/load. disable_active_parametrization ensures
        state_dict access and load_state_dict don't trigger collectives.
        """
        import os
        import shutil
        import tempfile

        from torchtitan.experiments.flex_shard import disable_active_parametrization

        mesh = self._init_mesh()

        # Create and shard model
        torch.manual_seed(42)
        model = nn.Linear(8, 4, bias=False, device="cuda")
        torch.distributed.broadcast(model.weight.data, src=0)
        full_ref = model.weight.data.clone()

        self._flex_shard(model, mesh)

        # state_dict returns sharded params (bypasses property)
        sd_before = {k: v.clone() for k, v in model.state_dict().items()}
        expected_rows = 4 // self.world_size
        self.assertEqual(sd_before["weight"].shape, (expected_rows, 8))

        # Guard also returns sharded params via param access
        with disable_active_parametrization():
            guarded_weight = model.weight
        self.assertEqual(guarded_weight.shape, (expected_rows, 8))

        # Share tmpdir across ranks
        obj_list = [tempfile.mkdtemp() if self.rank == 0 else ""]
        torch.distributed.broadcast_object_list(obj_list, src=0)
        tmpdir = obj_list[0]

        try:
            # Per-rank save
            torch.save(
                model.state_dict(),
                os.path.join(tmpdir, f"rank_{self.rank}.pt"),
            )
            torch.distributed.barrier()

            # Create fresh model with different weights, shard it
            torch.manual_seed(99)
            model2 = nn.Linear(8, 4, bias=False, device="cuda")
            torch.distributed.broadcast(model2.weight.data, src=0)
            self._flex_shard(model2, mesh)

            # Per-rank load
            sd2 = torch.load(
                os.path.join(tmpdir, f"rank_{self.rank}.pt"),
                weights_only=True,
                map_location=f"cuda:{self.rank}",
            )
            model2.load_state_dict(sd2)
        finally:
            torch.distributed.barrier()
            if self.rank == 0:
                shutil.rmtree(tmpdir, ignore_errors=True)

        # Sharded params should match
        sd_after = model2.state_dict()
        torch.testing.assert_close(sd_after["weight"], sd_before["weight"])

        # Forward should produce the same result as original full weights
        x = torch.randn(2, 8, device="cuda")
        torch.distributed.broadcast(x, src=0)
        ref_output = x @ full_ref.t()
        output = model2(x)
        torch.testing.assert_close(output, ref_output)

    def _to_dtensor_sd(self, model, mesh):
        """Wrap FlexShard state_dict as DTensors for DCP.

        Maps FlexShard placements to DTensor Shard(0):
        - Shard(dim): each rank holds a chunk along dim → DTensor Shard(dim)
        - FlatShard: in parametrization mode, decomposed to per-param
          FlatShard(0, numel, numel), i.e. 1D Shard(0) → DTensor Shard(0)
        """
        from torch.distributed.tensor import DTensor, Shard as DTensorShard

        from torchtitan.experiments.flex_shard import Shard as FlexShard
        from torchtitan.experiments.flex_shard.flex_shard import FlatShard

        sd = {}
        plain_sd = model.state_dict()
        fqn_to_placement = {}
        for ds in model.dstorages:
            for fqn, info in ds.param_infos.items():
                fqn_to_placement[fqn] = info.placements
        for k, v in plain_sd.items():
            placements = fqn_to_placement.get(k)
            if placements is not None:
                dt_placements = []
                for p in placements:
                    if isinstance(p, FlexShard):
                        dt_placements.append(DTensorShard(p.dim))
                    elif isinstance(p, FlatShard):
                        # Per-param FlatShard(0, numel, numel) is 1D Shard(0)
                        dt_placements.append(DTensorShard(0))
                    else:
                        raise ValueError(f"Unsupported placement {p}")
                sd[k] = DTensor.from_local(v, mesh, dt_placements, run_check=False)
            else:
                sd[k] = v
        return sd

    def _dcp_roundtrip(self, model, model2, mesh, full_ref):
        """Save model via DCP, load into model2, verify correctness."""
        import shutil
        import tempfile

        import torch.distributed.checkpoint as dcp
        from torch.distributed.tensor import DTensor

        sd_before_clone = {k: v.clone() for k, v in model.state_dict().items()}

        obj_list = [tempfile.mkdtemp() if self.rank == 0 else ""]
        torch.distributed.broadcast_object_list(obj_list, src=0)
        tmpdir = obj_list[0]

        try:
            dcp.save(self._to_dtensor_sd(model, mesh), checkpoint_id=tmpdir)

            sd2 = self._to_dtensor_sd(model2, mesh)
            dcp.load(sd2, checkpoint_id=tmpdir)

            plain_sd2 = model2.state_dict()
            for k, v in sd2.items():
                local = v.to_local() if isinstance(v, DTensor) else v
                plain_sd2[k].copy_(local)
        finally:
            torch.distributed.barrier()
            if self.rank == 0:
                shutil.rmtree(tmpdir, ignore_errors=True)

        # Sharded params should match original
        sd_after = model2.state_dict()
        for k in sd_before_clone:
            torch.testing.assert_close(
                sd_after[k], sd_before_clone[k], msg=f"{k} mismatch"
            )

        # Forward should produce the same result as original full weights
        x = torch.randn(2, 8, device="cuda")
        torch.distributed.broadcast(x, src=0)
        ref_output = x @ full_ref.t()
        output = model2(x)
        torch.testing.assert_close(output, ref_output)

    def test_dcp_save_load_shard(self):
        """DCP roundtrip with Shard(0) (FSDP2-style)."""
        mesh = self._init_mesh()

        torch.manual_seed(42)
        model = nn.Linear(8, 4, bias=False, device="cuda")
        torch.distributed.broadcast(model.weight.data, src=0)
        full_ref = model.weight.data.clone()
        self._flex_shard(model, mesh)

        torch.manual_seed(99)
        model2 = nn.Linear(8, 4, bias=False, device="cuda")
        torch.distributed.broadcast(model2.weight.data, src=0)
        self._flex_shard(model2, mesh)

        self._dcp_roundtrip(model, model2, mesh, full_ref)

    def test_dcp_save_load_flat_shard(self):
        """DCP roundtrip with FlatShard (FSDP1-style)."""
        from torchtitan.experiments.flex_shard import flat_shard_placements

        mesh = self._init_mesh()

        torch.manual_seed(42)
        model = nn.Linear(8, 4, bias=False, device="cuda")
        torch.distributed.broadcast(model.weight.data, src=0)
        full_ref = model.weight.data.clone()
        self._flex_shard(model, mesh, shard_placement_fn=flat_shard_placements)

        torch.manual_seed(99)
        model2 = nn.Linear(8, 4, bias=False, device="cuda")
        torch.distributed.broadcast(model2.weight.data, src=0)
        self._flex_shard(model2, mesh, shard_placement_fn=flat_shard_placements)

        self._dcp_roundtrip(model, model2, mesh, full_ref)


if __name__ == "__main__":
    unittest.main()
