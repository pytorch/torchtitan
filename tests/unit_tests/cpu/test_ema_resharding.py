# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Multi-rank coverage for EMA checkpoint resharding.

``_materialize_dtensor`` is what lets DCP save and reshard offloaded EMA state:
the pinned copy is only a local shard, so it has to be rewrapped around the
parameter's live sharding before DCP sees it. A single-rank mesh cannot test
this -- a ``Shard(0)`` local shard is bit-identical to the global tensor there,
so the rewrap is unobservable and the GPU suite passes even with the function
deleted. These run on a 2-rank gloo mesh, where dropping the rewrap halves the
saved tensor.

The second class covers the claim the whole design rests on: EMA subclasses
``OptimizersContainer`` so that DCP can save and *reshard* EMA state across a
world-size change with no bespoke format. That is asserted nowhere else.
"""

import tempfile

import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
import torch.nn as nn
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import distribute_tensor, DTensor, Replicate, Shard
from torch.testing._internal.distributed._tensor.common_dtensor import (
    DTensorTestBase,
    with_comms,
)

from torchtitan.components.optimizer import EMA


class TestEMADTensorRewrap(DTensorTestBase):
    @property
    def world_size(self) -> int:
        return 2

    @property
    def backend(self) -> str:
        return "gloo"

    @property
    def device_type(self) -> str:
        return "cpu"

    def _ema(self) -> EMA:
        """A container over a throwaway parameter -- these tests exercise
        _materialize_dtensor, which does not read the tracked tensors."""
        return EMA.Config().build(model_parts=[torch.nn.Linear(2, 2)])

    def _sharded_param(self):
        mesh = self.build_device_mesh()
        global_tensor = torch.arange(4 * 8, dtype=torch.float32).reshape(4, 8)
        return distribute_tensor(global_tensor, mesh, [Shard(0)]), global_tensor

    @with_comms
    def test_rewrap_restores_the_global_shape(self):
        param, global_tensor = self._sharded_param()
        self.assertEqual(param.to_local().shape, (2, 8))  # genuinely sharded

        ema = self._ema()
        local = param.to_local().clone()
        rewrapped = ema._materialize_dtensor(param, local)

        self.assertIsInstance(rewrapped, DTensor)
        self.assertEqual(rewrapped.shape, global_tensor.shape)
        self.assertEqual(rewrapped.placements, param.placements)
        self.assertEqual(rewrapped.device_mesh, param.device_mesh)
        torch.testing.assert_close(rewrapped.to_local(), local)

    @with_comms
    def test_rewrap_is_not_a_no_op_on_a_sharded_param(self):
        """Pins the property that makes the test above meaningful: returning
        the local shard unchanged must be observably different."""
        param, global_tensor = self._sharded_param()
        local = param.to_local().clone()
        ema = self._ema()
        rewrapped = ema._materialize_dtensor(param, local)
        self.assertNotEqual(tuple(rewrapped.shape), tuple(local.shape))
        self.assertEqual(tuple(rewrapped.shape), tuple(global_tensor.shape))

    @with_comms
    def test_plain_tensor_passes_through(self):
        ema = self._ema()
        plain = torch.ones(3)
        self.assertIs(ema._materialize_dtensor(plain, plain), plain)


class TestEMACheckpointResharding(DTensorTestBase):
    """Save EMA state from sharded parameters, then read it back in a
    different layout -- whole global tensors instead of per-rank shards."""

    @property
    def world_size(self) -> int:
        return 2

    @property
    def backend(self) -> str:
        return "gloo"

    @property
    def device_type(self) -> str:
        return "cpu"

    @with_comms
    def test_sharded_ema_state_reloads_as_whole_tensors(self):
        mesh = self.build_device_mesh()
        model = nn.Linear(8, 8, bias=False)
        # shard the parameter so each rank owns half of it
        model.weight = nn.Parameter(
            distribute_tensor(model.weight.detach(), mesh, [Shard(0)])
        )
        ema = EMA.Config().build(model_parts=[model])

        # a value that differs per row, so a mis-assembled reshard is visible
        expected = torch.arange(64, dtype=torch.float32).reshape(8, 8)
        stored = ema.optimizers[0].state[model.weight]["ema_params"]
        with torch.no_grad():
            stored.to_local().copy_(
                distribute_tensor(expected, mesh, [Shard(0)]).to_local()
            )
        self.assertEqual(stored.to_local().shape, (4, 8))  # genuinely sharded

        folder = [tempfile.mkdtemp() if self.rank == 0 else None]
        dist.broadcast_object_list(folder, src=0)
        checkpoint = folder[0]
        dcp.save({"ema": ema}, checkpoint_id=checkpoint)

        # Read the same checkpoint back into plain whole tensors -- a layout
        # nothing wrote. DCP can only do this if the saved shards carry their
        # global coordinates, which is what "resharding-safe" means here.
        target = {"ema.state.weight.ema_params": torch.zeros(8, 8, dtype=torch.float32)}
        dcp.load(target, checkpoint_id=checkpoint)
        torch.testing.assert_close(
            target["ema.state.weight.ema_params"], expected, rtol=0, atol=0
        )

    @with_comms
    def test_buffer_ema_reshards_too(self):
        """Buffer EMA rides in the same "ema" key, so it has to reshard as
        well -- and buffer FQNs never collide with parameter FQNs."""
        mesh = self.build_device_mesh()

        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = nn.Parameter(torch.zeros(8, 4))
                self.register_buffer("expert_bias_E", torch.zeros(8))

        model = Model()
        model.weight = nn.Parameter(
            distribute_tensor(model.weight.detach(), mesh, [Shard(0)])
        )
        model.expert_bias_E = distribute_tensor(model.expert_bias_E, mesh, [Shard(0)])
        ema = EMA.Config(buffer_patterns=[r"expert_bias_E$"]).build(model_parts=[model])
        expected = torch.arange(8, dtype=torch.float32)
        stored = ema._buffer_optimizers[0].state[model.expert_bias_E]["ema_params"]
        with torch.no_grad():
            stored.to_local().copy_(
                distribute_tensor(expected, mesh, [Shard(0)]).to_local()
            )

        folder = [tempfile.mkdtemp() if self.rank == 0 else None]
        dist.broadcast_object_list(folder, src=0)
        checkpoint = folder[0]
        dcp.save({"ema": ema}, checkpoint_id=checkpoint)

        target = {
            "ema.state.expert_bias_E.ema_params": torch.zeros(8, dtype=torch.float32)
        }
        dcp.load(target, checkpoint_id=checkpoint)
        torch.testing.assert_close(
            target["ema.state.expert_bias_E.ema_params"], expected, rtol=0, atol=0
        )


class TestEMAMultiRankLifecycle(DTensorTestBase):
    """The full EMA lifecycle over genuinely sharded parameters: update, DCP
    save, DCP load into a fresh container. A 1-rank mesh cannot show that the
    saved state is world-size independent, because the local shard is then the
    whole tensor.
    """

    @property
    def world_size(self) -> int:
        return 2

    @property
    def backend(self) -> str:
        return "gloo"

    @property
    def device_type(self) -> str:
        return "cpu"

    def _sharded_model(self):
        mesh = self.build_device_mesh()
        model = nn.Linear(8, 4, bias=False)
        with torch.no_grad():
            model.weight.fill_(1.0)
        model.weight = nn.Parameter(
            distribute_tensor(model.weight.detach(), mesh, [Shard(0)])
        )
        return model

    @with_comms
    def test_save_and_load_round_trip_over_sharded_params(self):
        model = self._sharded_model()
        self.assertEqual(model.weight.to_local().shape, (2, 8))  # really sharded
        ema = EMA.Config().build(model_parts=[model])
        with torch.no_grad():
            model.weight.fill_(3.0)
        for step in range(1, 4):
            ema.step(step)
        expected = {
            key: value.full_tensor().clone() for key, value in ema.state_dict().items()
        }

        folder = [tempfile.mkdtemp() if self.rank == 0 else None]
        torch.distributed.broadcast_object_list(folder, src=0)
        dcp.save({"ema": ema}, checkpoint_id=folder[0])

        # the on-disk tensor must be the global one, not a shard
        metadata = dcp.FileSystemReader(folder[0]).read_metadata()
        for key, item in metadata.state_dict_metadata.items():
            if key.endswith(".ema_params"):
                self.assertEqual(tuple(item.size), (4, 8))

        fresh_model = self._sharded_model()
        fresh = EMA.Config().build(model_parts=[fresh_model])
        dcp.load({"ema": fresh}, checkpoint_id=folder[0])
        restored = {
            key: value.full_tensor().clone()
            for key, value in fresh.state_dict().items()
        }
        self.assertEqual(sorted(expected), sorted(restored))
        for key, value in expected.items():
            torch.testing.assert_close(restored[key], value, rtol=0, atol=0)


class TestEMAHsdpPlacements(DTensorTestBase):
    """``_materialize_dtensor`` passes ``run_check=False``, justified in its
    docstring by the update being a per-rank-deterministic function of
    already-consistent data -- which matters precisely because FSDP2 params do
    carry ``Replicate()`` under HSDP. This exercises that 2D case.
    """

    @property
    def world_size(self) -> int:
        return 4

    @property
    def backend(self) -> str:
        return "gloo"

    @property
    def device_type(self) -> str:
        return "cpu"

    def _hsdp_param(self):
        mesh = init_device_mesh(
            "cpu", (2, 2), mesh_dim_names=("dp_replicate", "dp_shard")
        )
        global_tensor = torch.arange(8 * 4, dtype=torch.float32).reshape(8, 4)
        return distribute_tensor(global_tensor, mesh, [Replicate(), Shard(0)])

    @with_comms
    def test_rewrap_preserves_replicate_and_shard(self):
        param = self._hsdp_param()
        self.assertEqual(param.to_local().shape, (4, 4))
        ema = EMA.Config().build(model_parts=[nn.Linear(2, 2)])
        rewrapped = ema._materialize_dtensor(param, param.to_local().clone())
        self.assertIsInstance(rewrapped, DTensor)
        self.assertEqual(rewrapped.placements, param.placements)
        self.assertEqual(rewrapped.shape, param.shape)
        self.assertEqual(rewrapped.device_mesh.ndim, 2)

    @with_comms
    def test_update_is_identical_across_replicate_ranks(self):
        """The invariant that makes run_check=False safe: every replica
        computes the same EMA from the same inputs, with no communication."""
        mesh = init_device_mesh(
            "cpu", (2, 2), mesh_dim_names=("dp_replicate", "dp_shard")
        )
        model = nn.Linear(4, 8, bias=False)
        with torch.no_grad():
            model.weight.fill_(1.0)
        model.weight = nn.Parameter(
            distribute_tensor(model.weight.detach(), mesh, [Replicate(), Shard(0)])
        )
        ema = EMA.Config().build(model_parts=[model])
        with torch.no_grad():
            model.weight.fill_(5.0)
        for step in range(1, 4):
            ema.step(step)
        local = ema.optimizers[0].state[model.weight]["ema_params"].to_local()

        # gather across the replicate axis; every replica must agree exactly
        replicate_group = mesh.get_group("dp_replicate")
        gathered = [torch.empty_like(local) for _ in range(mesh["dp_replicate"].size())]
        torch.distributed.all_gather(
            gathered, local.contiguous(), group=replicate_group
        )
        for other in gathered[1:]:
            torch.testing.assert_close(gathered[0], other, rtol=0, atol=0)


if __name__ == "__main__":
    import unittest

    unittest.main()
