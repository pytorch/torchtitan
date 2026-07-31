# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import os
import shutil
import tempfile
import unittest

import torch
import torch.distributed as dist
import torch.nn as nn

from torchtitan.components.ema import EMAOptimizersContainer


class TestEMADynamicDecay(unittest.TestCase):
    """CPU-only: verifies the dynamic (half_life_fraction-based) decay
    schedule -- in particular that update_every_n_steps != 1 uses the EMA
    firing count (not raw elapsed steps) as the schedule's age, since the
    half-life formula's math is defined in terms of applications of decay.
    """

    def test_default_n1_matches_closed_form(self):
        model = nn.Linear(4, 4)
        ema = EMAOptimizersContainer.Config(enable=True).build(model_parts=[model])
        expected = ema.optimizers[0].state[model.weight]["ema_params"].clone()
        for step in range(1, 6):
            with torch.no_grad():
                model.weight.fill_(float(step))
            ema.step(step)
            beta = 2.0 ** (-1.0 / (0.05 * step))
            expected = expected * beta + model.weight.detach() * (1 - beta)
            actual = ema.optimizers[0].state[model.weight]["ema_params"]
            torch.testing.assert_close(actual, expected, atol=1e-5, rtol=0)

    def test_update_every_n_steps_uses_firing_count_not_raw_steps(self):
        """Regression test: the schedule's age must be the number of EMA
        firings, not the raw step count, or it ages far faster than
        intended whenever update_every_n_steps > 1."""
        model = nn.Linear(4, 4)
        ema = EMAOptimizersContainer.Config(enable=True, update_every_n_steps=2).build(
            model_parts=[model]
        )
        expected = ema.optimizers[0].state[model.weight]["ema_params"].clone()
        fire_count = 0
        for step in range(1, 11):
            with torch.no_grad():
                model.weight.fill_(float(step))
            ema.step(step)
            if step % 2 != 0:
                continue
            fire_count += 1
            beta = 2.0 ** (-1.0 / (0.05 * fire_count))
            expected = expected * beta + model.weight.detach() * (1 - beta)
            actual = ema.optimizers[0].state[model.weight]["ema_params"]
            torch.testing.assert_close(actual, expected, atol=1e-5, rtol=0)
        self.assertEqual(fire_count, 5)

    def test_step_bias_renumbers_aging(self):
        """step_bias lets a deliberate Trainer.step reset keep the EMA aging
        as if training had continued uninterrupted."""
        model = nn.Linear(4, 4)
        ema = EMAOptimizersContainer.Config(enable=True, step_bias=6).build(
            model_parts=[model]
        )
        start = ema.optimizers[0].state[model.weight]["ema_params"].clone()
        with torch.no_grad():
            model.weight.fill_(7.0)
        ema.step(1)  # current_step=1, step_bias=6 -> num_updates = 7
        beta7 = 2.0 ** (-1.0 / (0.05 * 7))
        expected = start * beta7 + model.weight.detach() * (1 - beta7)
        actual = ema.optimizers[0].state[model.weight]["ema_params"]
        torch.testing.assert_close(actual, expected, atol=1e-5, rtol=0)

    def test_fixed_decay_ignores_firing_count(self):
        """A fixed `decay` bypasses the half-life schedule entirely -- same
        value regardless of how many times it's fired."""
        model = nn.Linear(4, 4)
        ema = EMAOptimizersContainer.Config(enable=True, decay=0.9).build(
            model_parts=[model]
        )
        expected = ema.optimizers[0].state[model.weight]["ema_params"].clone()
        for step in range(1, 4):
            with torch.no_grad():
                model.weight.fill_(float(step))
            ema.step(step)
            expected = expected * 0.9 + model.weight.detach() * 0.1
            actual = ema.optimizers[0].state[model.weight]["ema_params"]
            torch.testing.assert_close(actual, expected, atol=1e-6, rtol=0)

    def test_disabled_is_true_noop(self):
        model = nn.Linear(4, 4)
        ema = EMAOptimizersContainer.Config(enable=False).build(model_parts=[model])
        ema.step(1)
        self.assertEqual(ema.state_dict(), {})
        self.assertTrue(all(len(st) == 0 for st in ema.optimizers[0].state.values()))


@unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
class TestEMACpuOffload(unittest.TestCase):
    """GPU-only: exercises offload_to_cpu under real FSDP2 DTensor params --
    the DTensor pin_memory workaround (_pin_local/_materialize_dtensor in
    torchtitan/components/ema.py) can't be tested without real DTensors.
    """

    @classmethod
    def setUpClass(cls):
        cls._owns_pg = not dist.is_initialized()
        if cls._owns_pg:
            os.environ.setdefault("MASTER_ADDR", "localhost")
            os.environ.setdefault("MASTER_PORT", "29602")
            os.environ.setdefault("RANK", "0")
            os.environ.setdefault("WORLD_SIZE", "1")
            os.environ.setdefault("LOCAL_RANK", "0")
            torch.cuda.set_device(0)
            dist.init_process_group(backend="nccl")

    @classmethod
    def tearDownClass(cls):
        if cls._owns_pg:
            dist.destroy_process_group()

    def _build_sharded_model(self) -> nn.Module:
        from torch.distributed.device_mesh import init_device_mesh
        from torch.distributed.fsdp import fully_shard

        mesh = init_device_mesh("cuda", (1,), mesh_dim_names=("dp_shard",))
        model = nn.Sequential(nn.Linear(32, 32), nn.Linear(32, 32)).cuda()
        fully_shard(model, mesh=mesh)
        return model

    def test_construction_does_not_crash_on_dtensor_pin_memory(self):
        """Regression test for NYI: aten._pin_memory.default -- DTensor has
        no pin_memory() dispatch support, so construction must operate on
        the local shard, not the DTensor itself."""
        model = self._build_sharded_model()
        ema = EMAOptimizersContainer.Config(enable=True, offload_to_cpu=True).build(
            model_parts=[model]
        )
        for ema_opt in ema.optimizers:
            for param_state in ema_opt.state.values():
                t = param_state["ema_params"]
                self.assertFalse(t.is_cuda)
                self.assertTrue(t.is_pinned())

    def test_step_updates_offloaded_values_correctly(self):
        model = self._build_sharded_model()
        ema = EMAOptimizersContainer.Config(enable=True, offload_to_cpu=True).build(
            model_parts=[model]
        )
        initial = {
            id(p): st["ema_params"].clone()
            for opt in ema.optimizers
            for p, st in opt.state.items()
        }
        with torch.no_grad():
            for p in model.parameters():
                p.fill_(2.0)
        ema.step(1)
        torch.cuda.synchronize()
        beta = 2.0 ** (-1.0 / (0.05 * 1))
        for opt in ema.optimizers:
            for p, st in opt.state.items():
                expected = initial[id(p)] * beta + p.detach().to_local().cpu() * (
                    1 - beta
                )
                torch.testing.assert_close(
                    st["ema_params"], expected, atol=1e-4, rtol=0
                )

    def test_scratch_buffer_reused_across_steps(self):
        """Regression test: _get_scratch's cache key must be stable across
        step() calls -- previously keyed off id() of a freshly-constructed
        list each call, which is not a reliable identity to cache against."""
        model = self._build_sharded_model()
        ema = EMAOptimizersContainer.Config(enable=True, offload_to_cpu=True).build(
            model_parts=[model]
        )
        for step in range(1, 4):
            with torch.no_grad():
                for p in model.parameters():
                    p.fill_(float(step))
            ema.step(step)
        self.assertEqual(len(ema._offload_scratch), 1)

    def test_save_load_round_trip_stays_offloaded(self):
        import torch.distributed.checkpoint as dcp

        model = self._build_sharded_model()
        ema = EMAOptimizersContainer.Config(enable=True, offload_to_cpu=True).build(
            model_parts=[model]
        )
        with torch.no_grad():
            for p in model.parameters():
                p.fill_(3.0)
        ema.step(1)
        torch.cuda.synchronize()

        ckpt_dir = tempfile.mkdtemp()
        try:
            dcp.save({"ema_optimizer": ema}, checkpoint_id=ckpt_dir)

            model2 = self._build_sharded_model()
            ema2 = EMAOptimizersContainer.Config(
                enable=True, offload_to_cpu=True
            ).build(model_parts=[model2])
            dcp.load({"ema_optimizer": ema2}, checkpoint_id=ckpt_dir)

            for opt1, opt2 in zip(ema.optimizers, ema2.optimizers):
                for (_, st1), (_, st2) in zip(opt1.state.items(), opt2.state.items()):
                    v2 = st2["ema_params"]
                    self.assertFalse(v2.is_cuda)
                    self.assertTrue(v2.is_pinned())
                    torch.testing.assert_close(st1["ema_params"], v2, atol=1e-6, rtol=0)
        finally:
            shutil.rmtree(ckpt_dir, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
