# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import math

import torch
import triton.testing as tt

from torchtitan.experiments.helion_kernels.rope import (
    apply_rotary_emb_cos_sin_helion,
)


def apply_rotary_emb_cos_sin_reference(
    xq: torch.Tensor,
    xk: torch.Tensor,
    rope_cache: torch.Tensor,
    positions: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    head_dim = xq.shape[-1]
    rope_cache = rope_cache[None, :, None, :].expand(xq.shape[0], -1, -1, -1)
    rope_cache = torch.gather(
        rope_cache,
        dim=1,
        index=positions.view(xq.shape[0], xq.shape[1], 1, 1).expand(
            xq.shape[0], xq.shape[1], 1, head_dim * 2
        ),
    )
    cos = rope_cache[..., :head_dim]
    sin = rope_cache[..., head_dim:]

    def rotate_half(x: torch.Tensor) -> torch.Tensor:
        half = x.shape[-1] // 2
        return torch.cat((-x[..., half:], x[..., :half]), dim=-1)

    xq_out = (xq.float() * cos) + (rotate_half(xq.float()) * sin)
    xk_out = (xk.float() * cos) + (rotate_half(xk.float()) * sin)
    return xq_out.type_as(xq), xk_out.type_as(xk)


def bench_shape(
    batch_size: int,
    seq_len: int,
    n_heads: int,
    n_kv_heads: int,
    head_dim: int,
) -> float:
    xq = torch.randn(
        batch_size,
        seq_len,
        n_heads,
        head_dim,
        device="cuda",
        dtype=torch.bfloat16,
    )
    xk = torch.randn(
        batch_size,
        seq_len,
        n_kv_heads,
        head_dim,
        device="cuda",
        dtype=torch.bfloat16,
    )
    rope_cache = torch.randn(
        seq_len,
        head_dim * 2,
        device="cuda",
        dtype=torch.float32,
    )
    positions = torch.arange(seq_len, device="cuda", dtype=torch.int32).repeat(
        batch_size, 1
    )

    apply_rotary_emb_cos_sin_helion(xq, xk, rope_cache, positions)
    apply_rotary_emb_cos_sin_reference(xq, xk, rope_cache, positions)
    torch.cuda.synchronize()

    helion_ms = tt.do_bench(
        lambda: apply_rotary_emb_cos_sin_helion(xq, xk, rope_cache, positions),
        warmup=25,
        rep=100,
        return_mode="median",
    )
    torch_ms = tt.do_bench(
        lambda: apply_rotary_emb_cos_sin_reference(xq, xk, rope_cache, positions),
        warmup=25,
        rep=100,
        return_mode="median",
    )
    speedup = torch_ms / helion_ms if helion_ms > 0 else float("nan")
    print(
        f"{batch_size:>3d} {seq_len:>5d} {n_heads:>3d} {n_kv_heads:>3d} "
        f"{head_dim:>4d} {helion_ms * 1000:>12.2f} "
        f"{torch_ms * 1000:>12.2f} {speedup:>8.3f}x"
    )
    return speedup


def main() -> None:
    assert torch.cuda.is_available(), "CUDA is required"
    shapes = [
        (1, 2048, 32, 8, 128),
        (2, 2048, 32, 8, 128),
        (4, 2048, 32, 8, 128),
        (1, 4096, 32, 8, 128),
        (2, 4096, 32, 8, 128),
        (1, 8192, 32, 8, 128),
    ]
    print(f"GPU: {torch.cuda.get_device_name()}")
    print(
        f"{'B':>3s} {'S':>5s} {'HQ':>3s} {'HK':>3s} {'D':>4s} "
        f"{'helion (us)':>12s} {'torch (us)':>12s} {'speedup':>8s}"
    )
    print("-" * 65)
    speedups = [bench_shape(*shape) for shape in shapes]
    geomean = math.exp(
        sum(math.log(speedup) for speedup in speedups if speedup > 0)
        / max(len(speedups), 1)
    )
    print(f"SUMMARY: geomean={geomean:.4f}")


if __name__ == "__main__":
    main()
