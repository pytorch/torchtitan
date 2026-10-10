# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyrefly: ignore-errors

"""Two-lane form of the validated libdevice FP32 exp instruction sequence.

GB300 supports packed FP32 arithmetic. The constants, rounding modes, FTZ flags,
saturation, range reduction, EX2, and final subnormal-preserving multiplication
must match the validated runtime's libdevice.exp. The directed
rounding FMAs and EX2 remain scalar; RN FMAs/add/multiply execute in pairs.
"""

import triton
import triton.language as tl


EXP_PAIR = tl.constexpr(
    r"""{
    .reg .b64 X, H, C, A, B, R, J, E, P;
    .reg .b32 a0, a1, b0, b1, p0, p1;
    mov.b64 X, {$2, $3};
    mov.b64 H, 0x3F0000003F000000;
    mov.b64 C, 0x3BBB989D3BBB989D;
    fma.rn.ftz.f32x2 R, X, C, H;
    mov.b64 {a0, a1}, R;
    cvt.ftz.sat.f32.f32 a0, a0;
    cvt.ftz.sat.f32.f32 a1, a1;
    fma.rm.ftz.f32 b0, a0, 0f437C0000, 0f4B400001;
    fma.rm.ftz.f32 b1, a1, 0f437C0000, 0f4B400001;
    mov.b64 B, {b0, b1};
    mov.b64 C, 0xCB40007FCB40007F;
    add.f32x2 J, B, C;
    xor.b64 J, J, 0x8000000080000000;
    mov.b64 C, 0x3FB8AA3B3FB8AA3B;
    fma.rn.ftz.f32x2 R, X, C, J;
    mov.b64 C, 0x32A5706032A57060;
    fma.rn.ftz.f32x2 R, X, C, R;
    mov.b64 {a0, a1}, R;
    ex2.approx.ftz.f32 p0, a0;
    ex2.approx.ftz.f32 p1, a1;
    shl.b32 b0, b0, 23;
    shl.b32 b1, b1, 23;
    mov.b64 E, {b0, b1};
    mov.b64 P, {p0, p1};
    mul.f32x2 R, P, E;
    mov.b64 {$0, $1}, R;
}"""
)


@triton.jit
def exp(values):
    return tl.inline_asm_elementwise(
        EXP_PAIR,
        constraints="=f,=f,f,f",
        args=[values],
        dtype=tl.float32,
        is_pure=True,
        pack=2,
    )
