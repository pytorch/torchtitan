# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""A faster copy for vLLM's mamba prefix-cache state blocks (V1 runner, ``mamba_cache_mode="align"``).

Before each step, vLLM copies the GDN state of each request that crosses a block boundary into its
new block (``mamba_utils.do_mamba_copy_block``). One request on Qwen3.5-35B-A3B (30 GDN layers,
60 copies of 2 MiB or 48 KiB), GB300:

    vLLM's kernel:  one program per state, 1 KiB per loop iteration    1112 us
    this kernel:    one program per 64 KiB chunk, 16-byte words          27 us

TODO: send this chunked copy to vLLM's ``batch_memcpy``, then delete this file and its patch in
``worker.py``. vLLM main still runs one program per copy.
"""

import numpy as np
import triton
import triton.language as tl

_CHUNK_BYTES = 64 * 1024


@triton.jit
def _chunked_memcpy_kernel(
    src_ptrs, dst_ptrs, sizes, CHUNK_WORDS: tl.constexpr, BLOCK: tl.constexpr
):
    pid = tl.program_id(0)
    num_words = tl.load(sizes + pid) // 16
    start = tl.program_id(1) * CHUNK_WORDS
    if start < num_words:
        src = tl.load(src_ptrs + pid)
        dst = tl.load(dst_ptrs + pid)
        offsets = tl.arange(0, BLOCK)
        for i in range(start, tl.minimum(start + CHUNK_WORDS, num_words), BLOCK):
            # Each 16-byte word moves as two int64 lanes.
            idx = i + offsets
            mask = idx < num_words
            lo_src = (src + 16 * idx).to(tl.pointer_type(tl.int64))
            lo_dst = (dst + 16 * idx).to(tl.pointer_type(tl.int64))
            lo = tl.load(lo_src, mask=mask)
            hi = tl.load(lo_src + 1, mask=mask)
            tl.store(lo_dst, lo, mask=mask)
            tl.store(lo_dst + 1, hi, mask=mask)


def chunked_mamba_copy_block(copy_bufs, fallback) -> None:
    """Run vLLM's pending state copies (``copy_bufs``) in 64 KiB chunks; else call ``fallback``.

    Falls back to vLLM's kernel when a copy is not 16-byte aligned or overlaps itself (the
    speculative-decoding shift within a block), which this chunked copy can't order safely.

    Example:
        >>> mamba_utils.do_mamba_copy_block = functools.partial(
        ...     chunked_mamba_copy_block, fallback=mamba_utils.do_mamba_copy_block
        ... )
    """
    n = copy_bufs.offset
    if n == 0:
        return
    # vLLM keeps the pointers as uint64 and the sizes as int32; device addresses fit in int64.
    src = copy_bufs.src_ptrs.np[:n].astype(np.int64)
    dst = copy_bufs.dst_ptrs.np[:n].astype(np.int64)
    sizes = copy_bufs.sizes.np[:n].astype(np.int64)
    aligned = not ((src | dst | sizes) & 15).any()
    overlaps = ((dst < src + sizes) & (src < dst + sizes)).any()
    if not aligned or overlaps:
        fallback(copy_bufs)
        return
    chunk_words = _CHUNK_BYTES // 16
    grid = (n, triton.cdiv(int(sizes.max()), _CHUNK_BYTES))
    _chunked_memcpy_kernel[grid](
        copy_bufs.src_ptrs.copy_to_gpu(n),
        copy_bufs.dst_ptrs.copy_to_gpu(n),
        copy_bufs.sizes.copy_to_gpu(n),
        CHUNK_WORDS=chunk_words,
        BLOCK=1024,
    )
