# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Optimizer-state CPU offload: bitwise parity with the resident optimizer under FSDP2.

Every assertion on values is ``torch.equal``: the offloaded step runs the same fused kernel on
the same tensors, only from a different storage location.
"""

import os
from unittest import mock

import pytest
import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
import torch.nn as nn
from torch.distributed.checkpoint.staging import DefaultStager, StagingOptions
from torch.distributed.fsdp import fully_shard
from torch.distributed.tensor import DTensor
from torch.testing._internal.distributed._tensor.common_dtensor import (
    DTensorTestBase,
    with_comms,
)
from torch.testing._internal.distributed.checkpoint_utils import with_temp_dir
from torchtitan.components.optim import (
    AdamW,
    LRSchedulersContainer,
    offload as offload_module,
    OptimizersContainer,
    OptimizerStateOffloader,
)
from torchtitan.distributed import utils as dist_utils
from torchtitan.distributed.utils import get_local_tensor


pytestmark = pytest.mark.multi_gpu

_DIM = 256
_MOMENT_KEYS = ("exp_avg", "exp_avg_sq")


class _Model(nn.Module):
    """Four small blocks, one wide block (an oversized chunk at 1 MiB), one unused block (no grad),
    and a one-row block whose shard is empty on rank 1."""

    def __init__(self) -> None:
        super().__init__()
        self.blocks = nn.ModuleList([nn.Linear(_DIM, _DIM) for _ in range(4)])
        self.wide = nn.Linear(_DIM, 8 * _DIM, bias=False)
        self.unused = nn.Linear(_DIM, _DIM)
        self.tiny = nn.Linear(_DIM, 1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            x = torch.tanh(block(x))
        return self.wide(x) + self.tiny(x)


class TestOptimizerStateOffload(DTensorTestBase):
    @property
    def world_size(self) -> int:
        return 2

    def _build(self, *, offload: bool) -> tuple[_Model, OptimizersContainer]:
        torch.manual_seed(0)
        model = _Model().cuda()
        mesh = self.build_device_mesh()
        for module in (*model.blocks, model.wide, model.unused, model.tiny):
            fully_shard(module, mesh=mesh)
        fully_shard(model, mesh=mesh)
        state_offload = OptimizerStateOffloader.Config() if offload else None
        # optimizers[0] owns tiny (an empty slab on rank 1), [1] the biases, [2] the weights.
        config = OptimizersContainer.Config(
            optimizers=[
                AdamW.Config(pattern=r"^tiny\.", state_offload=state_offload),
                AdamW.Config(
                    pattern=r"\.bias$",
                    lr=1e-2,
                    weight_decay=0.0,
                    state_offload=state_offload,
                ),
                AdamW.Config(
                    pattern=r".*",
                    lr=1e-3,
                    weight_decay=0.1,
                    moment_dtype="bfloat16",
                    state_offload=state_offload,
                ),
            ],
        )
        # 1 MiB chunks: the four 256x256 blocks pack several per chunk, the wide block is oversized.
        with mock.patch.object(offload_module, "_CHUNK_SIZE_BYTES", 2**20):
            return model, config.build(model_parts=[model])

    def _train(
        self, model: _Model, optimizers: OptimizersContainer, *, steps: int, seed: int
    ) -> None:
        """`steps` optimizer steps, each accumulating two microbatches and clipping at 0.5."""
        params = list(model.parameters())
        for step in range(steps):
            for microbatch in range(2):
                torch.manual_seed(seed + 100 * step + microbatch)
                x = torch.randn(8, _DIM, device="cuda")
                model(x).square().mean().backward()
            dist_utils.clip_grad_norm_(params, max_norm=0.5, foreach=True)
            optimizers.step()
            optimizers.zero_grad()

    def _assert_same(
        self, resident: OptimizersContainer, offloaded: OptimizersContainer
    ) -> None:
        # step() leaves the stores into the CPU moments pending on the host.
        torch.cuda.synchronize()
        for res_opt, off_opt in zip(resident.optimizers, offloaded.optimizers):
            for res_group, off_group in zip(res_opt.param_groups, off_opt.param_groups):
                for res_param, off_param in zip(
                    res_group["params"], off_group["params"]
                ):
                    assert torch.equal(
                        get_local_tensor(res_param), get_local_tensor(off_param)
                    )
                    res_state, off_state = (
                        res_opt.state[res_param],
                        off_opt.state[off_param],
                    )
                    if not res_state:
                        # Never received a grad: torch AdamW has no state yet; ours is eager and untouched.
                        assert off_state["step"].item() == 0
                        assert all(
                            not get_local_tensor(off_state[key]).any()
                            for key in _MOMENT_KEYS
                        )
                        continue
                    for key in (*_MOMENT_KEYS, "step"):
                        assert torch.equal(
                            get_local_tensor(res_state[key]).cpu(),
                            get_local_tensor(off_state[key]).cpu(),
                        ), key

    def _assert_cpu_canonical(self, offloaded: OptimizersContainer) -> None:
        for optimizer in offloaded.optimizers:
            assert isinstance(optimizer, OptimizerStateOffloader)
            for param, state in optimizer.state.items():
                for key in _MOMENT_KEYS:
                    assert isinstance(state[key], DTensor)
                    assert state[key].device.type == "cpu"
                    # An empty shard has nothing to pin.
                    local = get_local_tensor(state[key])
                    assert local.is_pinned() or local.numel() == 0
                    assert state[key].placements == param.placements
                assert state["step"].device.type == "cuda"

    @with_comms
    def test_parity_with_resident_optimizer(self) -> None:
        resident_model, resident = self._build(offload=False)
        offloaded_model, offloaded = self._build(offload=True)
        assert len(offloaded.optimizers[2]._chunks) >= 3
        self._train(resident_model, resident, steps=3, seed=1)
        self._train(offloaded_model, offloaded, steps=3, seed=1)
        self._assert_same(resident, offloaded)
        self._assert_cpu_canonical(offloaded)
        expected_dtypes = [torch.float32, torch.float32, torch.bfloat16]
        for optimizer, expected_dtype in zip(offloaded.optimizers, expected_dtypes):
            for state in optimizer.state.values():
                assert state["exp_avg"].dtype == expected_dtype
                assert get_local_tensor(state["exp_avg"]).dtype == expected_dtype
        # LR metrics keep the wrapped optimizer's name.
        lr_schedulers = LRSchedulersContainer.Config().build(
            optimizers=offloaded, training_steps=3
        )
        assert set(lr_schedulers.get_metrics()) == {
            "lr/AdamW/0",
            "lr/AdamW/1",
            "lr/AdamW/2",
        }

    @with_comms
    def test_slot_memory_is_bounded_by_two_chunks(self) -> None:
        model, offloaded = self._build(offload=True)
        self._train(model, offloaded, steps=1, seed=3)
        params = list(model.parameters())
        model(torch.randn(8, _DIM, device="cuda")).square().mean().backward()
        dist_utils.clip_grad_norm_(params, max_norm=0.5, foreach=True)
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        before = torch.cuda.memory_allocated()
        offloaded.step()
        peak = torch.cuda.max_memory_allocated()
        after = torch.cuda.memory_allocated()
        # The offloaders step one after another, so the largest pair of slots sets the peak.
        two_slots = max(
            2
            * len(_MOMENT_KEYS)
            * sum(get_local_tensor(p).numel() for p in chunk)
            * offloader._slabs["exp_avg"].element_size()
            for offloader in offloaded.optimizers
            for chunk in offloader._chunks
        )
        assert peak - before <= two_slots + (1 << 20), (peak - before, two_slots)
        assert after - before <= 1 << 20
        offloaded.zero_grad()

    @with_comms
    @with_temp_dir
    def test_dcp_round_trip_loads_in_place_and_training_continues_identically(
        self,
    ) -> None:
        checkpoint_dir = os.path.join(self.temp_dir, "after_two_steps")
        pre_step_checkpoint_dir = os.path.join(self.temp_dir, "before_first_step")

        # Saving before the first step works: state is allocated eagerly.
        model_a, opt_a = self._build(offload=True)
        assert all(
            key.startswith(("state.", "param_groups.")) for key in opt_a.state_dict()
        )
        dcp.save(opt_a.state_dict(), checkpoint_id=pre_step_checkpoint_dir)
        self._train(model_a, opt_a, steps=2, seed=4)
        # Async save with the stager options of CheckpointManager's async_with_pinned_mem:
        # the CPU state is copied to pinned shared memory on a background thread.
        stager = DefaultStager(
            StagingOptions(
                use_pinned_memory=True,
                use_shared_memory=True,
                use_async_staging=True,
                use_non_blocking_copy=True,
            )
        )
        response = dcp.async_save(
            opt_a.state_dict(),
            checkpoint_id=checkpoint_dir,
            process_group=dist.new_group(backend="gloo"),
            async_stager=stager,
        )
        response.staging_completion.result()
        response.upload_completion.result()
        stager.close()

        # The checkpoint has the resident optimizer's keys and loads into it bitwise.
        _, resident = self._build(offload=False)
        resident_state_dict = resident.state_dict()
        assert set(resident_state_dict) == set(opt_a.state_dict())
        dcp.load(resident_state_dict, checkpoint_id=checkpoint_dir)
        resident.load_state_dict(resident_state_dict)
        for optimizer_a, resident_optimizer in zip(
            opt_a.optimizers, resident.optimizers
        ):
            params_a = optimizer_a.param_groups[0]["params"]
            resident_params = resident_optimizer.param_groups[0]["params"]
            for pa, pr in zip(params_a, resident_params, strict=True):
                for key in (*_MOMENT_KEYS, "step"):
                    assert torch.equal(
                        get_local_tensor(optimizer_a.state[pa][key]).cpu(),
                        get_local_tensor(resident_optimizer.state[pr][key]).cpu(),
                    ), key

        model_b, opt_b = self._build(offload=True)
        self._train(model_b, opt_b, steps=1, seed=99)  # diverge, then load over it
        slab_ptrs = {
            id(p): get_local_tensor(state["exp_avg"]).data_ptr()
            for optimizer in opt_b.optimizers
            for p, state in optimizer.state.items()
        }
        state_dict_b = opt_b.state_dict()
        dcp.load(state_dict_b, checkpoint_id=checkpoint_dir)
        opt_b.load_state_dict(state_dict_b)

        for optimizer in opt_b.optimizers:
            for p, state in optimizer.state.items():
                assert get_local_tensor(state["exp_avg"]).data_ptr() == slab_ptrs[id(p)]
        self._assert_cpu_canonical(opt_b)
        for optimizer_a, optimizer_b in zip(opt_a.optimizers, opt_b.optimizers):
            for pa, pb in zip(optimizer_a.state, optimizer_b.state):
                sa, sb = optimizer_a.state[pa], optimizer_b.state[pb]
                for key in (*_MOMENT_KEYS, "step"):
                    assert torch.equal(
                        get_local_tensor(sa[key]).cpu(), get_local_tensor(sb[key]).cpu()
                    ), key

        # Continued training from the loaded state matches, given equal params.
        with torch.no_grad():
            for pa, pb in zip(model_a.parameters(), model_b.parameters()):
                get_local_tensor(pb).copy_(get_local_tensor(pa))
        self._train(model_a, opt_a, steps=1, seed=5)
        self._train(model_b, opt_b, steps=1, seed=5)
        self._assert_same(opt_a, opt_b)
