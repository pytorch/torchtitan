# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Shim, NOT a fix: give _grouped_mm's empty operands a stride its validator accepts.

torch._grouped_mm rejects any 2D operand whose contraction dim is 0 -- the
natural row-major stride fails `stride >= max(1, size)` and the stride
at::empty produces fails the 16-byte alignment check. The tensor has no
elements, so re-striding it describes the same (absent) data; this is the
Python-side equivalent of the proposed skip-validation-when-numel-is-0 patch.

Installed only to prove the diagnosis end to end and to unblock the compiled
EP legs. The real fix belongs in ATen.
"""
import torch
from torch._inductor.select_algorithm import extern_kernels

_orig = getattr(extern_kernels, "_grouped_mm", torch._grouped_mm)
COUNT = {"patched": 0}


def _restride(t):
    if not torch.is_tensor(t) or t.dim() != 2 or t.numel() != 0:
        return t
    align = max(1, 16 // t.element_size())
    if t.stride(1) == 1 and t.stride(0) % align == 0 and t.stride(0) >= 1:
        return t
    COUNT["patched"] += 1
    return t.as_strided(t.shape, (align, 1))


def _grouped_mm(a, b, offs=None, **kw):
    return _orig(_restride(a), _restride(b), offs, **kw)


extern_kernels._grouped_mm = _grouped_mm
torch._grouped_mm = lambda a, b, offs=None, **kw: torch.ops.aten._grouped_mm(
    _restride(a), _restride(b), offs, **kw
)
