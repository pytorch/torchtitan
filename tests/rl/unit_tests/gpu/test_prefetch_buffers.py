# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import pytest
import torch
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import DTensor, Shard
from torchtitan.rl.generator.vllm.model_wrapper import _empty_pinned_like


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_empty_pinned_like_keeps_spec_without_reading_weights(tmp_path) -> None:
    """Build a pinned CPU buffer from the weight's shapes alone, with the weight's DTensor spec."""
    dist.init_process_group(
        "nccl", init_method=f"file://{tmp_path / 'store'}", rank=0, world_size=1
    )
    try:
        mesh = init_device_mesh("cuda", (1,))
        # A [2, 8] shard of a [4, 8] weight, with its storage freed: a copy would fail.
        local_tensor = torch.empty(2, 8, dtype=torch.bfloat16, device="cuda")
        weight = DTensor.from_local(
            local_tensor,
            mesh,
            [Shard(0)],
            shape=torch.Size([4, 8]),
            stride=(8, 1),
            run_check=False,
        )
        local_tensor.untyped_storage().resize_(0)

        buffer = _empty_pinned_like(weight)

        assert isinstance(buffer, DTensor)
        assert buffer._spec == weight._spec
        assert buffer.to_local().shape == (2, 8)
        assert buffer.to_local().dtype == torch.bfloat16
        assert buffer.to_local().device.type == "cpu"
        assert buffer.to_local().is_pinned()

        plain_buffer = _empty_pinned_like(torch.zeros(4, device="cuda"))
        assert plain_buffer.shape == (4,) and plain_buffer.device.type == "cpu"
        assert plain_buffer.is_pinned()
    finally:
        dist.destroy_process_group()
