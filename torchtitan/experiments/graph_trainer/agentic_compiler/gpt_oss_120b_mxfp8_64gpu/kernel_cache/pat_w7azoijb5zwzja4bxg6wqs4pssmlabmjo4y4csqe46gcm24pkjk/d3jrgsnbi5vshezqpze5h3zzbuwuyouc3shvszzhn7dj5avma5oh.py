"""GPT-OSS SwiGLU-with-clamp forward (G-017), one CuTe DSL kernel, bitwise equal to reference.py.

reference (in0 = [R, 2, 2880] bf16, g = in0[:, 0], u = in0[:, 1]; every aten op rounds its fp32
opmath result to bf16 with RNE):
    v4  = clamp(g, max=7)        NaN -> g unchanged; g > 7 -> 7; else g
    v5  = clamp(u, -7, 7)        NaN -> u unchanged; u < -7 -> -7; u > 7 -> 7; else u
    v6  = bf16(v4 * fl32(1.702))  (CPU scalar read as float opmath)
    v7  = sigmoid(v6) = bf16(1 / (1 + expf(-v6)))   (div.rn, non-fastmath expf)
    v9  = bf16(v4 * v7)
    v10 = addcmul(v9, v9, v5) = bf16(fmaf(v9, v5, v9))   (alpha == 1 -> std::fma)
    returns (g, u, v4, v5, v7, v7 alias, v9, v10); g and u are views of in0.
All fp32 arithmetic is explicit .rn PTX (no FMA contraction, no ftz); bf16 rounding is
cvt.rn.bf16x2.f32 (what __float2bfloat16 emits, NaN -> 0x7fff).
The grid is flat over R * 360 16-byte vectors; R is a runtime value (0 allowed).
"""

import cutlass
import cutlass.cute as cute
import torch
from cutlass import Float32, Int32, Int64, Uint32
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


def _asmN(tys, args, text, cons, loc, ip, side=False):
    st = ir.Type.parse("!llvm.struct<(" + ", ".join(str(t) for t in tys) + ")>")
    res = llvm.inline_asm(
        st, _vals(args, loc, ip), text, cons,
        has_side_effects=side, is_align_stack=False, asm_dialect=llvm.AsmDialect.AD_ATT, loc=loc, ip=ip,
    )
    return [llvm.extractvalue(t, res, [k], loc=loc, ip=ip) for k, t in enumerate(tys)]


@dsl_user_op
def ld_v4(addr, *, loc=None, ip=None):
    r = _asmN([T.i32()] * 4, [Int64(addr)], "ld.global.nc.L1::no_allocate.v4.b32 {$0, $1, $2, $3}, [$4];",
              "=r,=r,=r,=r,l", loc, ip, side=True)
    return [Uint32(v) for v in r]


@dsl_user_op
def st_v4(addr, w, *, loc=None, ip=None):
    _asm1(None, [Int64(addr)] + [Uint32(x) for x in w],
          "st.global.v4.b32 [$0], {$1, $2, $3, $4};", "l,r,r,r,r", loc, ip, side=True)


# words g, u -> v4 word, v5 word, -v6 as two floats (lo, hi)
_PRE = """{
.reg .b32 gl, gh, ul, uh, w4, tl, th;
.reg .pred p, q;
.reg .b16 x, y, z;
mov.b32 {x, y}, $4;
shl.b32 gl, $4, 16; and.b32 gh, $4, 0xFFFF0000;
setp.gt.f32 p, gl, 0f40E00000; selp.b16 x, 0x40E0, x, p;
setp.gt.f32 p, gh, 0f40E00000; selp.b16 y, 0x40E0, y, p;
mov.b32 w4, {x, y};
mov.b32 $0, w4;
mov.b32 {x, y}, $5;
shl.b32 ul, $5, 16; and.b32 uh, $5, 0xFFFF0000;
setp.lt.f32 p, ul, 0fC0E00000; setp.gt.f32 q, ul, 0f40E00000;
selp.b16 x, 0xC0E0, x, p; selp.b16 x, 0x40E0, x, q;
setp.lt.f32 p, uh, 0fC0E00000; setp.gt.f32 q, uh, 0f40E00000;
selp.b16 y, 0xC0E0, y, p; selp.b16 y, 0x40E0, y, q;
mov.b32 $1, {x, y};
shl.b32 gl, w4, 16; and.b32 gh, w4, 0xFFFF0000;
mul.rn.f32 tl, gl, 0f3FD9DB23; mul.rn.f32 th, gh, 0f3FD9DB23;
cvt.rn.bf16x2.f32 w4, th, tl;
shl.b32 tl, w4, 16; and.b32 th, w4, 0xFFFF0000;
neg.f32 $2, tl; neg.f32 $3, th;
}"""

# v4 word, v5 word, sigmoid floats (lo, hi) -> v7 word, v9 word, v10 word
_POST = """{
.reg .b32 al, ah, bl, bh, w7, w9, tl, th;
cvt.rn.bf16x2.f32 w7, $6, $5;
mov.b32 $0, w7;
shl.b32 al, $3, 16; and.b32 ah, $3, 0xFFFF0000;
shl.b32 bl, w7, 16; and.b32 bh, w7, 0xFFFF0000;
mul.rn.f32 tl, al, bl; mul.rn.f32 th, ah, bh;
cvt.rn.bf16x2.f32 w9, th, tl;
mov.b32 $1, w9;
shl.b32 al, w9, 16; and.b32 ah, w9, 0xFFFF0000;
shl.b32 bl, $4, 16; and.b32 bh, $4, 0xFFFF0000;
fma.rn.f32 tl, al, bl, al; fma.rn.f32 th, ah, bh, ah;
cvt.rn.bf16x2.f32 $2, th, tl;
}"""


@dsl_user_op
def pre_word(g, u, *, loc=None, ip=None):
    r = _asmN([T.i32(), T.i32(), T.f32(), T.f32()], [Uint32(g), Uint32(u)], _PRE, "=r,=r,=f,=f,r,r", loc, ip)
    return Uint32(r[0]), Uint32(r[1]), Float32(r[2]), Float32(r[3])


@dsl_user_op
def post_word(w4, w5, sl, sh, *, loc=None, ip=None):
    r = _asmN([T.i32()] * 3, [Uint32(w4), Uint32(w5), Float32(sl), Float32(sh)], _POST, "=r,=r,=r,r,r,f,f", loc, ip)
    return Uint32(r[0]), Uint32(r[1]), Uint32(r[2])


@dsl_user_op
def sigmoid_of_neg(nx, *, loc=None, ip=None):
    """rcp.rn(1 +rn expf(nx)) with nx = -x; rcp.rn == div.rn(1, .) (both correctly rounded)."""
    e = cute.math.exp(nx, fastmath=False)
    return Float32(_asm1(T.f32(), [Float32(e)],
                         "{ .reg .f32 d; add.rn.f32 d, $1, 0f3F800000; rcp.rn.f32 $0, d; }", "=f,f", loc, ip))


class SwigluFwd:
    @cute.jit
    def __call__(self, x, o4, o5, o7, o9, o10, stream):
        nvec = x.shape[0] * NV
        nblk = (nvec + THREADS * U - 1) // (THREADS * U)
        self.kernel(x, o4, o5, o7, o9, o10, nvec).launch(grid=[nblk, 1, 1], block=[THREADS, 1, 1], stream=stream)

    @cute.kernel
    def kernel(self, x, o4, o5, o7, o9, o10, nvec: Int32):
        tid, _, _ = cute.arch.thread_idx()
        bid, _, _ = cute.arch.block_idx()
        px = x.iterator.toint()
        po = [o4.iterator.toint(), o5.iterator.toint(), o7.iterator.toint(), o9.iterator.toint(),
              o10.iterator.toint()]
        base = bid * (THREADS * U) + tid
        # Issue every load of this thread before any math or store (the asm loads/stores are
        # side-effecting, so the compiler keeps their program order).
        gws = []
        uws = []
        for u in cutlass.range_constexpr(U):
            i = base + u * THREADS
            gw = [Uint32(0)] * 4
            uw = [Uint32(0)] * 4
            if i < nvec:
                row = i // NV
                gaddr = px + Int64(row) * (2 * D * 2) + Int64(i - row * NV) * 16
                gw = ld_v4(gaddr)
                uw = ld_v4(gaddr + D * 2)
            gws.append(gw)
            uws.append(uw)
        for u in cutlass.range_constexpr(U):
            i = base + u * THREADS
            if i < nvec:
                r4 = [Uint32(0)] * 4
                r5 = [Uint32(0)] * 4
                r7 = [Uint32(0)] * 4
                r9 = [Uint32(0)] * 4
                r10 = [Uint32(0)] * 4
                for j in cutlass.range_constexpr(4):
                    w4, w5, nl, nh = pre_word(gws[u][j], uws[u][j])
                    sl = sigmoid_of_neg(nl)
                    sh = sigmoid_of_neg(nh)
                    r7[j], r9[j], r10[j] = post_word(w4, w5, sl, sh)
                    r4[j] = w4
                    r5[j] = w5
                off = Int64(i) * 16
                st_v4(po[0] + off, r4)
                st_v4(po[1] + off, r5)
                st_v4(po[2] + off, r7)
                st_v4(po[3] + off, r9)
                st_v4(po[4] + off, r10)


_SYM_R = cute.sym_int(symbol="R")
_bf = cutlass.BFloat16
_fn = cute.compile(
    SwigluFwd(),
    cute.runtime.make_fake_tensor(_bf, (_SYM_R, 2, D), stride=(2 * D, D, 1), assumed_align=16),
    *[cute.runtime.make_fake_tensor(_bf, (_SYM_R, D), stride=(D, 1), assumed_align=16) for _ in range(5)],
    cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
    options="--enable-tvm-ffi",
)


def run(in0):
    rows = in0.shape[0]
    dev = in0.device
    bf = torch.bfloat16
    o4 = torch.empty((rows, D), dtype=bf, device=dev)
    o5 = torch.empty((rows, D), dtype=bf, device=dev)
    o7 = torch.empty((rows, D), dtype=bf, device=dev)
    o9 = torch.empty((rows, D), dtype=bf, device=dev)
    o10 = torch.empty((rows, D), dtype=bf, device=dev)
    if rows:
        try:
            # The compiled entry point validates dtype, device, shape, strides and alignment.
            _fn(in0, o4, o5, o7, o9, o10)
        except ValueError:
            # Unexpected layout (e.g. a non-contiguous or misaligned view): dense copy.
            _fn(in0.clone(memory_format=torch.contiguous_format), o4, o5, o7, o9, o10)
    v2, v3 = in0.unbind(-2)
    return (v2, v3, o4, o5, o7, torch.ops.aten.alias.default(o7), o9, o10)
