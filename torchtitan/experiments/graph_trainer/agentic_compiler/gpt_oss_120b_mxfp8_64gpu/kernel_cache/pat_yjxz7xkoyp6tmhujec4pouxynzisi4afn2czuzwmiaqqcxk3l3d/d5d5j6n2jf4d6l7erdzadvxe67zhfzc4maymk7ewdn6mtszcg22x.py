"""GPT-OSS SwiGLU-with-clamp backward (G-007), one CuTe DSL kernel, bitwise equal to reference.py.

reference (per element, every aten op rounds its fp32 opmath result to bf16 with RNE):
    v9  = bf16(in1 * bf16(in0 * 1))
    v10 = bf16(in2 * in3);  v11 = bf16(in2 * in4)
    v13 = sigmoid_backward(v10, in5) = bf16(bf16(v10 * bf16(1 - in5)) * in5)
          (the bf16 CUDA kernel computes `a * (scalar_t(1) - b) * b` with c10::BFloat16
          operators, so each of the three ops rounds to bf16)
    v14 = bf16(v13 * fl32(1.702))   (CPU scalar read as float opmath)
    v15 = bf16(v11 + v14)
    out[:, 0] = in7 < 7 ? v15 : +0;  out[:, 1] = (-7 < in6 < 7) ? v9 : +0
All fp32 arithmetic is explicit .rn PTX (no FMA contraction, no ftz); bf16 rounding is
cvt.rn.bf16x2.f32 (what __float2bfloat16 emits, NaN -> 0x7fff).

Layout: in0..in5 are contiguous [R, 2880]; in6/in7 are [R, 2880] with any row stride that is a
multiple of 8 elements (5760 in training: views into the forward input); out is [R, 2, 2880].
The grid is flat over R * 360 16-byte vectors; R is a runtime value (0 allowed).
"""

import cutlass
import cutlass.cute as cute
import torch
from cutlass import Int32, Int64, Uint32
from cutlass._mlir import ir
from cutlass._mlir.dialects import llvm
from cutlass.cutlass_dsl import T, dsl_user_op

D = 2880
NV = D // 8  # 16-byte vectors per row
THREADS = 256
U = 2  # vectors per thread


def _vals(args, loc, ip):
    return [a.ir_value(loc=loc, ip=ip) for a in args]


def _asm1(ty, args, text, cons, loc, ip, side=False):
    return llvm.inline_asm(
        ty, _vals(args, loc, ip), text, cons,
        has_side_effects=side, is_align_stack=False, asm_dialect=llvm.AsmDialect.AD_ATT, loc=loc, ip=ip,
    )


def _asmN(n, ty, args, text, cons, loc, ip, side=False):
    st = ir.Type.parse("!llvm.struct<(" + ", ".join([str(ty)] * n) + ")>")
    res = llvm.inline_asm(
        st, _vals(args, loc, ip), text, cons,
        has_side_effects=side, is_align_stack=False, asm_dialect=llvm.AsmDialect.AD_ATT, loc=loc, ip=ip,
    )
    return [llvm.extractvalue(ty, res, [k], loc=loc, ip=ip) for k in range(n)]


@dsl_user_op
def ld_v4(addr, *, loc=None, ip=None):
    r = _asmN(4, T.i32(), [Int64(addr)], "ld.global.nc.L1::no_allocate.v4.b32 {$0, $1, $2, $3}, [$4];",
              "=r,=r,=r,=r,l", loc, ip, side=True)
    return [Uint32(v) for v in r]


@dsl_user_op
def st_v4(addr, w, *, loc=None, ip=None):
    _asm1(None, [Int64(addr)] + [Uint32(x) for x in w],
          "st.global.v4.b32 [$0], {$1, $2, $3, $4};", "l,r,r,r,r", loc, ip, side=True)


# One 32-bit word (two bf16) of each input -> one word of out[:, 0] and of out[:, 1].
_BWD = """{
.reg .b32 l0, h0, l1, h1, l2, h2, l3, h3, l4, h4, l5, h5, l6, h6, l7, h7;
.reg .b32 tl, th, w9, w10, w11, ws, wt, w13, w14, w15, ul, uh;
.reg .pred pl, ph, ql, qh;
shl.b32 l0, $2, 16; and.b32 h0, $2, 0xFFFF0000;
shl.b32 l1, $3, 16; and.b32 h1, $3, 0xFFFF0000;
shl.b32 l2, $4, 16; and.b32 h2, $4, 0xFFFF0000;
shl.b32 l3, $5, 16; and.b32 h3, $5, 0xFFFF0000;
shl.b32 l4, $6, 16; and.b32 h4, $6, 0xFFFF0000;
shl.b32 l5, $7, 16; and.b32 h5, $7, 0xFFFF0000;
shl.b32 l6, $8, 16; and.b32 h6, $8, 0xFFFF0000;
shl.b32 l7, $9, 16; and.b32 h7, $9, 0xFFFF0000;
// v8 = bf16(in0 * 1)
mul.rn.f32 tl, l0, 0f3F800000; mul.rn.f32 th, h0, 0f3F800000;
cvt.rn.bf16x2.f32 w9, th, tl;
shl.b32 l0, w9, 16; and.b32 h0, w9, 0xFFFF0000;
// v9 = bf16(in1 * v8)
mul.rn.f32 tl, l1, l0; mul.rn.f32 th, h1, h0;
cvt.rn.bf16x2.f32 w9, th, tl;
// v10 = bf16(in2 * in3), v11 = bf16(in2 * in4)
mul.rn.f32 tl, l2, l3; mul.rn.f32 th, h2, h3;
cvt.rn.bf16x2.f32 w10, th, tl;
mul.rn.f32 tl, l2, l4; mul.rn.f32 th, h2, h4;
cvt.rn.bf16x2.f32 w11, th, tl;
// s = bf16(1 - in5)
sub.rn.f32 tl, 0f3F800000, l5; sub.rn.f32 th, 0f3F800000, h5;
cvt.rn.bf16x2.f32 ws, th, tl;
// t = bf16(v10 * s)
shl.b32 l3, w10, 16; and.b32 h3, w10, 0xFFFF0000;
shl.b32 l4, ws, 16; and.b32 h4, ws, 0xFFFF0000;
mul.rn.f32 tl, l3, l4; mul.rn.f32 th, h3, h4;
cvt.rn.bf16x2.f32 wt, th, tl;
// v13 = bf16(t * in5)
shl.b32 l3, wt, 16; and.b32 h3, wt, 0xFFFF0000;
mul.rn.f32 tl, l3, l5; mul.rn.f32 th, h3, h5;
cvt.rn.bf16x2.f32 w13, th, tl;
// v14 = bf16(v13 * fl32(1.702))
shl.b32 l3, w13, 16; and.b32 h3, w13, 0xFFFF0000;
mul.rn.f32 tl, l3, 0f3FD9DB23; mul.rn.f32 th, h3, 0f3FD9DB23;
cvt.rn.bf16x2.f32 w14, th, tl;
// v15 = bf16(v11 + v14)
shl.b32 l3, w11, 16; and.b32 h3, w11, 0xFFFF0000;
shl.b32 l4, w14, 16; and.b32 h4, w14, 0xFFFF0000;
add.rn.f32 tl, l3, l4; add.rn.f32 th, h3, h4;
cvt.rn.bf16x2.f32 w15, th, tl;
// masks
setp.lt.f32 pl, l7, 0f40E00000; setp.lt.f32 ph, h7, 0f40E00000;
setp.gt.f32 ql, l6, 0fC0E00000; setp.gt.f32 qh, h6, 0fC0E00000;
setp.lt.and.f32 ql, l6, 0f40E00000, ql; setp.lt.and.f32 qh, h6, 0f40E00000, qh;
selp.b32 ul, 0x0000FFFF, 0, pl; selp.b32 uh, 0xFFFF0000, 0, ph;
or.b32 ul, ul, uh; and.b32 $0, w15, ul;
selp.b32 ul, 0x0000FFFF, 0, ql; selp.b32 uh, 0xFFFF0000, 0, qh;
or.b32 ul, ul, uh; and.b32 $1, w9, ul;
}"""


@dsl_user_op
def bwd_word(w, *, loc=None, ip=None):
    r = _asmN(2, T.i32(), [Uint32(x) for x in w], _BWD, "=r,=r,r,r,r,r,r,r,r,r", loc, ip)
    return Uint32(r[0]), Uint32(r[1])


class SwigluBwd:
    @cute.jit
    def __call__(self, a0, a1, a2, a3, a4, a5, a6, a7, out, stream):
        nvec = a0.shape[0] * NV
        nblk = (nvec + THREADS * U - 1) // (THREADS * U)
        self.kernel(a0, a1, a2, a3, a4, a5, a6, a7, out, nvec).launch(
            grid=[nblk, 1, 1], block=[THREADS, 1, 1], stream=stream)

    @cute.kernel
    def kernel(self, a0, a1, a2, a3, a4, a5, a6, a7, out, nvec: Int32):
        tid, _, _ = cute.arch.thread_idx()
        bid, _, _ = cute.arch.block_idx()
        s6 = Int64(a6.stride[0]) * 2
        s7 = Int64(a7.stride[0]) * 2
        p = [a0.iterator.toint(), a1.iterator.toint(), a2.iterator.toint(), a3.iterator.toint(),
             a4.iterator.toint(), a5.iterator.toint()]
        p6 = a6.iterator.toint()
        p7 = a7.iterator.toint()
        po = out.iterator.toint()
        base = bid * (THREADS * U) + tid
        # Issue every load of this thread before any math or store (the asm loads/stores are
        # side-effecting, so the compiler keeps their program order).
        xs = []
        for u in cutlass.range_constexpr(U):
            i = base + u * THREADS
            x = [[Uint32(0)] * 4 for _ in range(8)]
            if i < nvec:
                row = i // NV
                col16 = Int64(i - row * NV) * 16
                off = Int64(i) * 16
                row64 = Int64(row)
                for k in cutlass.range_constexpr(6):
                    x[k] = ld_v4(p[k] + off)
                x[6] = ld_v4(p6 + row64 * s6 + col16)
                x[7] = ld_v4(p7 + row64 * s7 + col16)
            xs.append(x)
        for u in cutlass.range_constexpr(U):
            i = base + u * THREADS
            if i < nvec:
                row = i // NV
                col16 = Int64(i - row * NV) * 16
                x = xs[u]
                o0 = [Uint32(0)] * 4
                o1 = [Uint32(0)] * 4
                for j in cutlass.range_constexpr(4):
                    o0[j], o1[j] = bwd_word([x[k][j] for k in range(8)])
                ob = po + Int64(row) * (2 * D * 2) + col16
                st_v4(ob, o0)
                st_v4(ob + D * 2, o1)


_SYM_R = cute.sym_int(symbol="R")
_SYM_S6 = cute.sym_int64(divisibility=8, symbol="S6")
_SYM_S7 = cute.sym_int64(divisibility=8, symbol="S7")
_bf = cutlass.BFloat16
_fn = cute.compile(
    SwigluBwd(),
    *[cute.runtime.make_fake_tensor(_bf, (_SYM_R, D), stride=(D, 1), assumed_align=16) for _ in range(6)],
    cute.runtime.make_fake_tensor(_bf, (_SYM_R, D), stride=(_SYM_S6, 1), assumed_align=16),
    cute.runtime.make_fake_tensor(_bf, (_SYM_R, D), stride=(_SYM_S7, 1), assumed_align=16),
    cute.runtime.make_fake_tensor(_bf, (_SYM_R, 2, D), stride=(2 * D, D, 1), assumed_align=16),
    cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
    options="--enable-tvm-ffi",
)


def _dense(t):
    return t if t.is_contiguous() and t.data_ptr() % 16 == 0 else t.clone(memory_format=torch.contiguous_format)


def run(in0, in1, in2, in3, in4, in5, in6, in7):
    rows = in0.shape[0]
    out = torch.empty((rows, 2, D), dtype=torch.bfloat16, device=in0.device)
    if rows:
        try:
            # The compiled entry point validates dtype, device, shapes, strides (in0..in5
            # contiguous, in6/in7 row stride % 8 == 0) and 16-byte alignment.
            _fn(in0, in1, in2, in3, in4, in5, in6, in7, out)
        except ValueError:
            # Unexpected layouts (e.g. misaligned or column-major views): dense copies.
            _fn(*[_dense(t) for t in (in0, in1, in2, in3, in4, in5, in6, in7)], out)
    return out
