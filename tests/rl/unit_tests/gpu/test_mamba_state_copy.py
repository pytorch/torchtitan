# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import pytest
import torch

from torchtitan.rl.generator.vllm.mamba_state_copy import chunked_mamba_copy_block


class _Buffer:
    """The parts of vLLM's CpuGpuBuffer the copy reads: ``np`` and ``copy_to_gpu``."""

    def __init__(self, values: np.ndarray) -> None:
        self.np = values

    def copy_to_gpu(self, n: int) -> torch.Tensor:
        return torch.from_numpy(self.np[:n].copy()).cuda()


class _CopyBuffers:
    """vLLM's MambaCopyBuffers with its dtypes: uint64 pointers, int32 sizes."""

    def __init__(self, src: list[int], dst: list[int], sizes: list[int]) -> None:
        self.src_ptrs = _Buffer(np.array(src, dtype=np.uint64))
        self.dst_ptrs = _Buffer(np.array(dst, dtype=np.uint64))
        self.sizes = _Buffer(np.array(sizes, dtype=np.int32))
        self.offset = len(src)


def _copies(
    sizes: list[int],
) -> tuple[list[torch.Tensor], list[torch.Tensor], _CopyBuffers]:
    src = [torch.randint(0, 256, (n,), dtype=torch.uint8, device="cuda") for n in sizes]
    dst = [torch.zeros_like(t) for t in src]
    buffers = _CopyBuffers(
        [t.data_ptr() for t in src], [t.data_ptr() for t in dst], sizes
    )
    return src, dst, buffers


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_chunked_copy_matches_the_source_bytes():
    # A 2 MiB SSM state spans 32 chunks; a 48 KiB conv state fits in one.
    src, dst, buffers = _copies([2 * 1024 * 1024, 48 * 1024, 16])
    fallback_calls = []
    chunked_mamba_copy_block(buffers, fallback=fallback_calls.append)
    torch.cuda.synchronize()
    assert fallback_calls == []
    for s, d in zip(src, dst, strict=True):
        assert torch.equal(s, d)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_unaligned_or_overlapping_copies_use_the_fallback():
    _, _, unaligned = _copies([1000])
    calls = []
    chunked_mamba_copy_block(unaligned, fallback=calls.append)
    assert calls == [unaligned]

    state = torch.zeros(4096, dtype=torch.uint8, device="cuda")
    overlapping = _CopyBuffers([state.data_ptr() + 1024], [state.data_ptr()], [2048])
    calls = []
    chunked_mamba_copy_block(overlapping, fallback=calls.append)
    assert calls == [overlapping]


def test_no_pending_copies_launch_nothing():
    buffers = _CopyBuffers([], [], [])
    calls = []
    chunked_mamba_copy_block(buffers, fallback=calls.append)
    assert calls == []
