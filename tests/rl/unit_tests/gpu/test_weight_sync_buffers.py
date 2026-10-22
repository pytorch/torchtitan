# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import tempfile

import pytest
import torch
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import DTensor, Shard

from torchtitan.rl.generator.vllm.model_wrapper import _create_pinned_cpu_state_dict

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


def test_weight_sync_buffers_pin_exactly_the_bytes_needed():
    with tempfile.TemporaryDirectory() as rendezvous:
        dist.init_process_group(
            "gloo", init_method=f"file://{rendezvous}/store", rank=0, world_size=1
        )
        try:
            mesh = init_device_mesh("cuda", (1,), mesh_dim_names=("dp",))
            # Sizes just over powers of two, where a power-of-two allocator pins
            # nearly twice the bytes.
            state_dict = {
                "expert": torch.randn(3, 1024 * 1024 + 7, device="cuda").bfloat16(),
                "dense": DTensor.from_local(
                    torch.randn(1025, 1024, device="cuda"), mesh, (Shard(0),)
                ),
                "scalar": torch.tensor(1.5, device="cuda"),
            }
            pinned_before = torch.cuda.host_memory_stats().get(
                "allocated_bytes.current", 0
            )

            buffers, storage = _create_pinned_cpu_state_dict(state_dict)

            needed = sum(
                (t._local_tensor if isinstance(t, DTensor) else t).nbytes
                for t in state_dict.values()
            )
            assert needed <= storage.numel() <= needed + 64 * len(state_dict)
            # Nothing went through the caching host allocator.
            assert (
                torch.cuda.host_memory_stats().get("allocated_bytes.current", 0)
                == pinned_before
            )
            for key, value in state_dict.items():
                buffer = buffers[key]
                assert buffer.shape == value.shape and buffer.dtype == value.dtype
                local = buffer._local_tensor if isinstance(buffer, DTensor) else buffer
                assert local.device.type == "cpu" and local.is_pinned()
            assert buffers["dense"].placements == (Shard(0),)

            # The buffers round-trip weights like the generator's load does.
            for key, value in state_dict.items():
                buffers[key].copy_(value)
                torch.testing.assert_close(buffers[key].to("cuda"), value)
        finally:
            dist.destroy_process_group()
