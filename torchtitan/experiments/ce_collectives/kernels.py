# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""One-CTA rendezvous and fixed-order local reduction kernels."""

import triton
import triton.language as tl


@triton.jit(do_not_specialize=["rank"])
def barrier_kernel(
    signal_pad_ptrs,
    rank,
    world_size: tl.constexpr,
    channel: tl.constexpr,
    block_size: tl.constexpr,
):
    peers = tl.arange(0, block_size)
    active = (peers < world_size) & (peers != rank)
    peer_base = tl.load(signal_pad_ptrs + peers, mask=active, other=0)
    local_base = tl.load(signal_pad_ptrs + rank)
    send_ptr = peer_base + 4 * (channel * world_size + rank)
    wait_ptr = local_base + 4 * (channel * world_size + peers)
    tl.debug_barrier()

    # Each active lane publishes its arrival and consumes the peer's arrival.
    # Consuming with CAS also resets the slot for the next graph replay.
    tl.inline_asm_elementwise(
        """
        {
            .reg .pred active, retry;
            .reg .b32 previous;
            setp.ne.u32 active, $3, 0;
            @!active bra done;
        send:
            atom.global.release.sys.cas.b32 previous, [$1], 0, 1;
            setp.ne.u32 retry, previous, 0;
            @retry bra send;
        wait:
            atom.global.acquire.sys.cas.b32 previous, [$2], 1, 0;
            setp.ne.u32 retry, previous, 1;
            @retry bra wait;
        done:
            mov.u32 $0, 0;
        }
        """,
        constraints="=r,l,l,r",
        # pyrefly: ignore[missing-attribute]
        args=[send_ptr, wait_ptr, active.to(tl.int32)],
        dtype=tl.int32,
        is_pure=False,
        pack=1,
    )
    tl.debug_barrier()


@triton.jit
def local_sum_kernel(
    input_ptr,
    output_ptr,
    numel: tl.constexpr,
    num_sources: tl.constexpr,
    divisor: tl.constexpr,
    block_size: tl.constexpr,
):
    offsets = tl.program_id(0) * block_size + tl.arange(0, block_size)
    total = tl.full((block_size,), 0, tl.float32)
    correction = tl.full((block_size,), 0, tl.float32)
    for source in range(num_sources):
        value = tl.load(
            input_ptr + source * numel + offsets, mask=offsets < numel, other=0
        ).to(tl.float32)
        adjusted = value - correction
        updated = total + adjusted
        correction = (updated - total) - adjusted
        total = updated
    tl.store(output_ptr + offsets, total / divisor, mask=offsets < numel)
