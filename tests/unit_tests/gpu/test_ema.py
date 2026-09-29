# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import os
import shutil
import tempfile
import unittest
from unittest import mock

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed.tensor import DTensor

from torchtitan.components.optimizer import EMA


class _ModelWithExpertBias(nn.Module):
    """Toy stand-in for a module with a non-gradient-updated buffer (e.g.
    MoE's expert_bias_E, updated by a load-balancing heuristic)."""

    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(4, 4)
        self.register_buffer("expert_bias_E", torch.zeros(4))


@unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
class TestEMACpuOffload(unittest.TestCase):
    """Exercises offload_to_cpu under real FSDP2 DTensor params -- the DTensor
    pin_memory workaround (_pin_local/_materialize_dtensor in
    torchtitan/components/optimizer/ema.py) can't be tested without real
    DTensors.
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
        ema = EMA.Config(offload_to_cpu=True).build(model_parts=[model])
        for ema_opt in ema.optimizers:
            for param_state in ema_opt.state.values():
                t = param_state["ema_params"]
                self.assertFalse(t.is_cuda)
                self.assertTrue(t.is_pinned())

    def test_step_updates_offloaded_values_correctly(self):
        model = self._build_sharded_model()
        ema = EMA.Config(offload_to_cpu=True).build(model_parts=[model])
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
        """The flat scratch pool must be allocated once and reused, not
        reallocated per firing."""
        model = self._build_sharded_model()
        ema = EMA.Config(offload_to_cpu=True).build(model_parts=[model])
        pools = []
        for step in range(1, 4):
            with torch.no_grad():
                for p in model.parameters():
                    p.fill_(float(step))
            ema.step(step)
            pools.append(ema._offload_pool)
        self.assertIsNotNone(pools[0])
        # same tensor object every firing
        self.assertEqual(len({id(pool) for pool in pools}), 1)

    def test_scratch_is_bounded_well_below_parameter_memory(self):
        """Regression test: scratch used to be one cached empty_like per
        tensor, i.e. exactly 1x parameter memory -- the amount offloading is
        meant to free. It must now be bounded by the chunk budget instead of
        growing with the number of tracked tensors."""
        model = self._build_sharded_model()
        param_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
        ema = EMA.Config(offload_to_cpu=True).build(model_parts=[model])
        # shrink the budget so the model is many chunks rather than one
        ema._SCRATCH_BYTES = 4096
        torch.cuda.synchronize()
        before = torch.cuda.memory_allocated()
        for step in range(1, 4):
            ema.step(step)
        torch.cuda.synchronize()
        scratch_bytes = torch.cuda.memory_allocated() - before
        self.assertGreater(param_bytes, 0)
        self.assertLess(scratch_bytes, param_bytes)
        # and bounded by the budget itself (plus the largest single tensor)
        largest = max(p.numel() * p.element_size() for p in model.parameters())
        self.assertLessEqual(scratch_bytes, max(4096, largest) + 4096)

    def test_compute_stream_waits_for_the_offload_read(self):
        """The offload stream reads the live params; nothing may overwrite them
        until that read completes. Without the ordering, the next iteration's
        optimizer.step() lands mid-read and the EMA absorbs the new value.

        The hazard only exists once the host can run ahead of the GPU, so the
        first firing (which allocates the scratch pool and device-syncs on the
        cudaMalloc) is used as a warm-up.
        """
        model = self._build_sharded_model()
        ema = EMA.Config(offload_to_cpu=True).build(model_parts=[model])
        with torch.no_grad():
            for param in model.parameters():
                param.fill_(1.0)
        ema.step(1)  # warm up the scratch pool
        torch.cuda.synchronize()

        with torch.no_grad():
            for param in model.parameters():
                param.fill_(1.0)
        for ema_opt in ema.optimizers:
            for param_state in ema_opt.state.values():
                param_state["ema_params"].fill_(0.0)
        torch.cuda.synchronize()

        # hold the offload stream so its read of the params happens late
        with torch.cuda.stream(ema._offload_stream):
            torch.cuda._sleep(1_500_000_000)
        ema.step(2)
        with torch.no_grad():  # the conflicting write, on the compute stream
            for param in model.parameters():
                param.fill_(9.0)
        torch.cuda.synchronize()

        # firing 2 => decay 2 ** (-1 / 0.1) ~= 9.8e-4, so the EMA is ~= the
        # value the offload stream read: 1.0 if ordered, ~9.0 if it raced.
        for ema_opt in ema.optimizers:
            for param_state in ema_opt.state.values():
                mean = param_state["ema_params"].float().mean().item()
                self.assertLess(
                    mean,
                    2.0,
                    "the compute-stream write raced ahead of the offload "
                    f"stream's read of the live params (EMA mean {mean})",
                )

    def test_materialized_dtensor_is_safe_to_read_from_another_stream(self):
        """state_dict() hands these tensors to DCP, whose stager copies them
        from its own stream with no wait against the compute stream. An async
        H2D here would let the stager read the buffer before it is filled."""
        model = self._build_sharded_model()
        ema = EMA.Config(offload_to_cpu=True).build(model_parts=[model])
        param = next(iter(model.parameters()))
        pinned = torch.full_like(param.to_local(), 5.0, device="cpu").pin_memory()
        torch.cuda.synchronize()

        # hold the compute stream, so an async copy would still be pending
        torch.cuda._sleep(1_500_000_000)
        materialized = ema._materialize_dtensor(param, pinned)

        # a different stream, deliberately not ordered against the compute one
        reader = torch.cuda.Stream()
        with torch.cuda.stream(reader):
            observed = materialized.to_local().clone()
        reader.synchronize()
        torch.cuda.synchronize()
        torch.testing.assert_close(
            observed, torch.full_like(observed, 5.0), rtol=0, atol=0
        )

    def test_state_dict_leaves_the_container_pinned(self):
        """state_dict() must not swap live GPU DTensors into the container.
        It previously did, restoring them in a finally -- so any exception
        mid-save left the EMA permanently holding GPU tensors, breaking both
        further training and every later save."""
        model = self._build_sharded_model()
        ema = EMA.Config(offload_to_cpu=True).build(model_parts=[model])
        ema.step(1)
        exported = ema.state_dict()
        self.assertTrue(exported)
        for ema_opt in ema.optimizers:
            for param_state in ema_opt.state.values():
                stored = param_state["ema_params"]
                self.assertNotIsInstance(stored, DTensor)
                self.assertEqual(stored.device.type, "cpu")
                self.assertTrue(stored.is_pinned())
        # still usable afterwards
        ema.step(2)

    def test_state_dict_failure_leaves_the_container_usable(self):
        """The actual regression: state_dict() used to swap GPU DTensors into
        the container *before* its try block, so a failure part-way through a
        save left them there permanently -- breaking later training and every
        later save. A successful save restored them, so only an injected
        failure distinguishes the two implementations."""
        model = self._build_sharded_model()
        ema = EMA.Config(offload_to_cpu=True).build(model_parts=[model])
        ema.step(1)
        real = EMA._materialize_dtensor
        calls = {"n": 0}

        def flaky(self, p, local):
            calls["n"] += 1
            if calls["n"] == 3:
                raise RuntimeError("simulated failure mid-save")
            return real(self, p, local)

        with mock.patch.object(EMA, "_materialize_dtensor", flaky):
            with self.assertRaises(RuntimeError):
                ema.state_dict()
        self.assertGreaterEqual(calls["n"], 3)
        for ema_opt in ema.optimizers:
            for param_state in ema_opt.state.values():
                stored = param_state["ema_params"]
                self.assertNotIsInstance(stored, DTensor)
                self.assertTrue(stored.is_pinned())
        # both training and saving must still work afterwards
        ema.step(2)
        self.assertTrue(ema.state_dict())

    def test_offload_and_plain_load_agree_on_a_partial_state_dict(self):
        """The offload load path looks keys up itself, so it has to tolerate a
        missing one exactly as load_flat_optim_state_dict does -- otherwise the
        two paths diverge on the same input."""
        results = {}
        for offload in (False, True):
            model = self._build_sharded_model()
            ema = EMA.Config(offload_to_cpu=offload).build(model_parts=[model])
            full = ema.state_dict()
            partial = {k: v for i, (k, v) in enumerate(full.items()) if i > 0}
            try:
                ema.load_state_dict(partial)
                results[offload] = "tolerated"
            except KeyError:
                results[offload] = "KeyError"
        self.assertEqual(
            results[False], results[True], f"offload/plain disagree: {results}"
        )
        self.assertEqual(results[True], "tolerated")

    def test_offload_state_dict_key_layout_matches_super(self):
        """The offload load path looks up "state.{fqn}.ema_params" directly, so
        that layout must stay the one OptimizersContainer.state_dict()
        produces."""
        model = self._build_sharded_model()
        offloaded = EMA.Config(offload_to_cpu=True).build(model_parts=[model])
        plain = EMA.Config(offload_to_cpu=False).build(model_parts=[model])
        self.assertEqual(sorted(offloaded.state_dict()), sorted(plain.state_dict()))
        self.assertTrue(
            all(
                key.startswith("state.") and key.endswith(".ema_params")
                for key in offloaded.state_dict()
            )
        )

    def test_save_load_round_trip_stays_offloaded(self):
        import torch.distributed.checkpoint as dcp

        model = self._build_sharded_model()
        ema = EMA.Config(offload_to_cpu=True).build(model_parts=[model])
        with torch.no_grad():
            for p in model.parameters():
                p.fill_(3.0)
        ema.step(1)
        torch.cuda.synchronize()

        ckpt_dir = tempfile.mkdtemp()
        try:
            dcp.save({"ema": ema}, checkpoint_id=ckpt_dir)

            model2 = self._build_sharded_model()
            ema2 = EMA.Config(offload_to_cpu=True).build(model_parts=[model2])
            dcp.load({"ema": ema2}, checkpoint_id=ckpt_dir)

            for opt1, opt2 in zip(ema.optimizers, ema2.optimizers):
                for (_, st1), (_, st2) in zip(opt1.state.items(), opt2.state.items()):
                    v2 = st2["ema_params"]
                    self.assertFalse(v2.is_cuda)
                    self.assertTrue(v2.is_pinned())
                    torch.testing.assert_close(st1["ema_params"], v2, atol=1e-6, rtol=0)
        finally:
            shutil.rmtree(ckpt_dir, ignore_errors=True)

    def _build_sharded_model_with_buffer(self) -> nn.Module:
        from torch.distributed.device_mesh import init_device_mesh
        from torch.distributed.fsdp import fully_shard

        mesh = init_device_mesh("cuda", (1,), mesh_dim_names=("dp_shard",))
        model = _ModelWithExpertBias().cuda()
        fully_shard(model, mesh=mesh)
        return model

    def test_buffer_offload_round_trip_under_fsdp2(self):
        """Regression test for offload+buffer parity: FSDP2's fully_shard only
        shards parameters, not buffers, so expert_bias_E-like buffers stay
        plain (non-DTensor) tensors even on a DTensor-param model --
        _pin_local/_materialize_dtensor must handle that mixed case, not just
        the all-DTensor or all-plain-tensor cases exercised above."""
        import torch.distributed.checkpoint as dcp

        model = self._build_sharded_model_with_buffer()
        self.assertFalse(isinstance(model.expert_bias_E, DTensor))
        ema = EMA.Config(
            offload_to_cpu=True, buffer_patterns=[r"expert_bias_E$"]
        ).build(model_parts=[model])
        bias_state = ema._buffer_optimizers[0].state[model.expert_bias_E]["ema_params"]
        self.assertFalse(bias_state.is_cuda)
        self.assertTrue(bias_state.is_pinned())

        with torch.no_grad():
            model.expert_bias_E.fill_(2.0)
        ema.step(1)
        torch.cuda.synchronize()
        bias_state = ema._buffer_optimizers[0].state[model.expert_bias_E]["ema_params"]

        ckpt_dir = tempfile.mkdtemp()
        try:
            dcp.save({"ema": ema}, checkpoint_id=ckpt_dir)

            model2 = self._build_sharded_model_with_buffer()
            ema2 = EMA.Config(
                offload_to_cpu=True, buffer_patterns=[r"expert_bias_E$"]
            ).build(model_parts=[model2])
            dcp.load({"ema": ema2}, checkpoint_id=ckpt_dir)

            v2 = ema2._buffer_optimizers[0].state[model2.expert_bias_E]["ema_params"]
            self.assertFalse(v2.is_cuda)
            self.assertTrue(v2.is_pinned())
            torch.testing.assert_close(bias_state, v2, atol=1e-6, rtol=0)
        finally:
            shutil.rmtree(ckpt_dir, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
