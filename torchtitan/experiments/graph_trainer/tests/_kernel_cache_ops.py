# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Custom ops used by test_kernel_cache.py.

They live in their own module so that a generated ``reference.py`` can import
it from a fresh process.
"""

import torch


@torch.library.custom_op("kc_test::scale", mutates_args=())
def scale(x: torch.Tensor, s: float) -> torch.Tensor:
    return x * s


@scale.register_fake
def _(x: torch.Tensor, s: float) -> torch.Tensor:
    return torch.empty_like(x)


try:
    import triton
    import triton.language as tl
    from torch.library import triton_op, wrap_triton
except ImportError:
    triton = None
else:

    @triton.jit
    def _add_kernel(x_ptr, y_ptr, out_ptr, n, BLOCK: tl.constexpr):
        offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        mask = offs < n
        x = tl.load(x_ptr + offs, mask=mask)
        y = tl.load(y_ptr + offs, mask=mask)
        tl.store(out_ptr + offs, x + y, mask=mask)

    @triton_op("kc_test::add", mutates_args=())
    def add(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        out = torch.empty_like(x)
        n = x.numel()
        wrap_triton(_add_kernel)[(triton.cdiv(n, 128),)](x, y, out, n, BLOCK=128)
        return out
