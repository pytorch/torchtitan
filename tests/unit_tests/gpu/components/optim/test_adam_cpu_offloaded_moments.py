# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import os
from unittest.mock import patch

import pytest
import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
import torch.nn as nn
from torch.distributed.checkpoint.staging import DefaultStager, StagingOptions
from torch.distributed.fsdp import fully_shard
from torch.testing._internal.distributed._tensor.common_dtensor import (
    DTensorTestBase,
    with_comms,
)
from torch.testing._internal.distributed.checkpoint_utils import with_temp_dir

from torchtitan.components.optim import AdamW, offloading, OptimizersContainer
from torchtitan.distributed.utils import get_local_tensor

pytestmark = pytest.mark.multi_gpu

_MOMENT_KEYS = ("exp_avg", "exp_avg_sq")


class TestAdamCPUOffloadedMoments(DTensorTestBase):
    """AdamW with CPU-offloaded moments on FSDP2 over two GPUs.

    Values are compared with ``torch.equal``: the same fused kernel runs on the
    same values, only from a different memory location.
    """

    @property
    def world_size(self) -> int:
        return 2

    def _build(
        self, *, moment_dtype: str, fraction: float, hidden: int = 1024
    ) -> tuple[nn.Module, OptimizersContainer]:
        torch.manual_seed(0)
        # The last weight has one row, so its shard on rank 1 is empty.
        model = nn.Sequential(
            nn.Linear(1024, hidden),
            nn.Linear(hidden, 1024),
            nn.Linear(1024, 1, bias=False),
        ).cuda()
        fully_shard(model, mesh=self.build_device_mesh())
        config = OptimizersContainer.Config(
            optimizers=[
                AdamW.Config(
                    pattern=r".*",
                    lr=1e-2,
                    moment_dtype=moment_dtype,
                    moment_cpu_offload_fraction=fraction,
                )
            ]
        )
        return model, config.build(model_parts=[model])

    def _train(
        self, model: nn.Module, optimizers: OptimizersContainer, *, seed: int
    ) -> None:
        for step in range(2):
            torch.manual_seed(seed + step)
            model(torch.randn(8, 1024, device="cuda")).square().mean().backward()
            optimizers.step()
            optimizers.zero_grad()

    def _states(self, optimizers: OptimizersContainer) -> list[torch.Tensor]:
        optimizer = optimizers.optimizers[0]
        return [
            get_local_tensor(optimizer.state[param][key])
            for param in optimizer.param_groups[0]["params"]
            for key in (*_MOMENT_KEYS, "step")
        ]

    def _pinned_param_names(self, optimizers: OptimizersContainer) -> set[str]:
        return {
            key.removeprefix("state.").removesuffix(".exp_avg")
            for key, value in optimizers.state_dict().items()
            if key.endswith(".exp_avg") and get_local_tensor(value).device.type == "cpu"
        }

    def _moment_data_ptrs(self, optimizers: OptimizersContainer) -> list[int]:
        optimizer = optimizers.optimizers[0]
        return [
            get_local_tensor(optimizer.state[param][key]).data_ptr()
            for param in optimizer.param_groups[0]["params"]
            for key in _MOMENT_KEYS
        ]

    def _assert_equal(
        self,
        model: nn.Module,
        optimizers: OptimizersContainer,
        ref_model: nn.Module,
        ref_optimizers: OptimizersContainer,
    ) -> None:
        for param, ref_param in zip(model.parameters(), ref_model.parameters()):
            self.assertTrue(
                torch.equal(get_local_tensor(param), get_local_tensor(ref_param))
            )
        for state, ref_state in zip(
            self._states(optimizers), self._states(ref_optimizers), strict=True
        ):
            self.assertTrue(torch.equal(state.cpu(), ref_state.cpu()))

    @with_comms
    def test_matches_device_moments(self) -> None:
        """Catches pinned moments that diverge from device moments, use device
        memory, break on an empty shard, or a fraction that pins the wrong
        parameters. 0.25 pins the first weight on both ranks. 0.5 would fall
        between the two equal weights, where the ranks' totals differ."""
        for moment_dtype in ("parameter", "bfloat16"):
            ref_model, ref_optimizers = self._build(
                moment_dtype=moment_dtype, fraction=0.0
            )
            self._train(ref_model, ref_optimizers, seed=1)
            for fraction in (0.25, 1.0):
                with self.subTest(moment_dtype=moment_dtype, fraction=fraction):
                    allocated = torch.cuda.memory_allocated()
                    model, optimizers = self._build(
                        moment_dtype=moment_dtype, fraction=fraction
                    )
                    self._train(model, optimizers, seed=1)
                    if fraction == 1.0:
                        # Every non-empty shard is pinned. Device memory holds
                        # the model and step counters.
                        expected = {
                            name
                            for name, param in model.named_parameters()
                            if get_local_tensor(param).numel() > 0
                        }
                        param_bytes = sum(
                            get_local_tensor(p).nbytes for p in model.parameters()
                        )
                        self.assertLessEqual(
                            torch.cuda.memory_allocated() - allocated,
                            param_bytes + 2**16,
                        )
                    else:
                        expected = {"0.weight"}
                    self.assertEqual(self._pinned_param_names(optimizers), expected)
                    self._assert_equal(model, optimizers, ref_model, ref_optimizers)
                    del model, optimizers

    @with_comms
    def test_rounded_pinned_memory_raises(self) -> None:
        """Catches pinned moments silently rounded up to a power of two, which
        can double their host memory."""
        # 3 MiB moment shards. The check allows only the rounding of the biases.
        with patch.object(offloading, "_PINNED_ROUNDING_SLACK_BYTES", 2**16):
            torch._C._accelerator_setAllocatorSettings(
                "pinned_max_round_threshold_mb:1"
            )
            # Kept alive so the second build cannot reuse its unrounded blocks.
            _first = self._build(moment_dtype="parameter", fraction=1.0, hidden=1536)
            torch._C._accelerator_setAllocatorSettings(
                "pinned_max_round_threshold_mb:1024"
            )
            with self.assertRaisesRegex(
                ValueError, "Add pinned_max_round_threshold_mb:1 to PYTORCH_ALLOC_CONF"
            ):
                self._build(moment_dtype="parameter", fraction=1.0, hidden=1536)

    @with_comms
    @with_temp_dir
    def test_checkpoint_loads_into_pinned_buffers(self) -> None:
        """Catches a DCP save of mixed pinned and device moments, or a load into
        all-pinned or all-device moments, that loses moment values, replaces the
        pinned buffers, or makes a temporary device copy of the moments (e.g.
        when Optimizer.load_state_dict casts bfloat16 moments to fp32). Also
        catches a state dict that does not expose pinned moments as CPU tensors
        over the pinned memory."""
        # The checkpointer's async_with_pinned_mem stager options.
        stager = DefaultStager(
            StagingOptions(
                use_pinned_memory=True,
                use_shared_memory=True,
                use_async_staging=True,
                use_non_blocking_copy=True,
            )
        )
        gloo_group = dist.new_group(backend="gloo")
        for moment_dtype in ("parameter", "bfloat16"):
            with self.subTest(moment_dtype=moment_dtype):
                checkpoint_dir = os.path.join(self.temp_dir, moment_dtype)
                # Pins the first weight; see test_matches_device_moments.
                ref_model, ref_optimizers = self._build(
                    moment_dtype=moment_dtype, fraction=0.25
                )
                self._train(ref_model, ref_optimizers, seed=1)
                ref_state_dict = ref_optimizers.state_dict()
                ref_optimizer = ref_optimizers.optimizers[0]
                for name, param in zip(
                    ref_optimizer.param_groups[0]["param_names"],
                    ref_optimizer.param_groups[0]["params"],
                    strict=True,
                ):
                    for key in _MOMENT_KEYS:
                        moment = get_local_tensor(ref_optimizer.state[param][key])
                        saved = get_local_tensor(ref_state_dict[f"state.{name}.{key}"])
                        device_type = "cpu" if name == "0.weight" else "cuda"
                        self.assertEqual(saved.device.type, device_type)
                        if moment.numel() > 0:
                            self.assertEqual(saved.data_ptr(), moment.data_ptr())
                response = dcp.async_save(
                    ref_state_dict,
                    checkpoint_id=checkpoint_dir,
                    process_group=gloo_group,
                    async_stager=stager,
                )
                response.staging_completion.result()
                response.upload_completion.result()

                # Diverge first, so the load must overwrite every value.
                model, optimizers = self._build(moment_dtype=moment_dtype, fraction=1.0)
                self._train(model, optimizers, seed=2)
                data_ptrs = [state.data_ptr() for state in self._states(optimizers)]
                with torch.no_grad():
                    for param, ref_param in zip(
                        model.parameters(), ref_model.parameters()
                    ):
                        param.copy_(ref_param)

                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
                allocated = torch.cuda.memory_allocated()
                state_dict = optimizers.state_dict()
                dcp.load(state_dict, checkpoint_id=checkpoint_dir)
                optimizers.load_state_dict(state_dict)
                moment_bytes = sum(
                    state.nbytes
                    for state in self._states(optimizers)
                    if state.dim() > 0
                )
                # A temporary fp32 copy of bf16 moments would take 2 x moment_bytes.
                self.assertLess(
                    torch.cuda.max_memory_allocated() - allocated, moment_bytes // 4
                )
                self.assertEqual(
                    data_ptrs, [state.data_ptr() for state in self._states(optimizers)]
                )
                self._assert_equal(model, optimizers, ref_model, ref_optimizers)

                # Loading other tensors (another optimizer's moments) copies them
                # into the pinned memory.
                other_model, other_optimizers = self._build(
                    moment_dtype=moment_dtype, fraction=1.0
                )
                self._train(other_model, other_optimizers, seed=2)
                moment_data_ptrs = self._moment_data_ptrs(other_optimizers)
                other_optimizers.load_state_dict(ref_optimizers.state_dict())
                self.assertEqual(
                    moment_data_ptrs, self._moment_data_ptrs(other_optimizers)
                )
                for state, ref_state in zip(
                    self._states(other_optimizers),
                    self._states(ref_optimizers),
                    strict=True,
                ):
                    self.assertTrue(torch.equal(state.cpu(), ref_state.cpu()))

                # The checkpoint also loads into device moments.
                _, device_optimizers = self._build(
                    moment_dtype=moment_dtype, fraction=0.0
                )
                state_dict = device_optimizers.state_dict()
                dcp.load(state_dict, checkpoint_id=checkpoint_dir)
                device_optimizers.load_state_dict(state_dict)
                for state, device_state in zip(
                    self._states(optimizers),
                    self._states(device_optimizers),
                    strict=True,
                ):
                    self.assertEqual(state.dtype, device_state.dtype)
                    self.assertTrue(torch.equal(state.cpu(), device_state.cpu()))

                # Training continues identically from the loaded state.
                self._train(ref_model, ref_optimizers, seed=3)
                self._train(model, optimizers, seed=3)
                self._assert_equal(model, optimizers, ref_model, ref_optimizers)
        stager.close()
