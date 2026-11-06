"""GPT-OSS RoPE backward (G-019, k [T, 8, 64]), one CuTe DSL kernel, bitwise equal to reference.py.

reference (g = in0 [T, 8, 64] bf16, a = in1, b = in2 [T, 1, 64] fp32, broadcast over heads):
    gf = fp32(g); m = gf * a
    v10 = slice_backward(-m[..., :32], start 32) + slice_backward(m[..., 32:], end 32)
    out = bf16(v10 + gf * b)
slice_backward pads with +0, so v10 = (+0) + m[d + 32] for d < 32 and (-m[d - 32]) + (+0) for
d >= 32 (the +0 turns a -0 into +0). With lo = dims [0, 32), hi = dims [32, 64):
    out_lo = bf16((rn(g_hi * a_hi) +rn 0) +rn rn(g_lo * b_lo))
    out_hi = bf16((0 -rn rn(g_lo * a_lo)) +rn rn(g_hi * b_hi))     (0 - y == (-y) + 0 in IEEE)
All fp32 arithmetic is explicit .rn PTX (packed f32x2, no FMA contraction, no ftz); bf16
rounding is cvt.rn.bf16x2.f32 (what __float2bfloat16 emits).

Layout: x and out [T, 8, 64] contiguous; the two fp32 tables [T, 1, 64] with any row stride that
is a multiple of 4 floats (128 in training). Thread (t, hg, c) owns the 16-byte chunk c (dims
[8c, 8c + 8)) of heads h = k * 4 + hg (k < 2) of token t, so a warp reads and writes 512
contiguous bytes per head step; the partner chunk c ^ 4 (dims d -+ 32) comes from lane ^ 4 by
shuffle. The formula is made lane-uniform by flipping the sign of the partner value on one half
(rn((-y) * s) == -rn(y * s); the table values a lane needs are loaded once for 2 heads).
T is a runtime value (0 allowed).
"""

import cutlass
import cutlass.cute as cute
import torch
from cutlass import Int32, Int64, Uint32
from cutlass._mlir import ir
from cutlass._mlir.dialects import llvm
from cutlass.cutlass_dsl import T, dsl_user_op

H = 8  # heads
HPT = 2  # heads per thread
HG = H // HPT  # head groups
TPT = HG * 8  # threads per token (one per 16-byte chunk per head group)
assert TPT % 32 == 0, "whole warps per token (the partner shuffle needs all 32 lanes)"
THREADS = 256
FWD = False


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
def ld_stream_v4(addr, *, loc=None, ip=None):
    r = _asmN(4, T.i32(), [Int64(addr)], "ld.global.nc.L1::no_allocate.v4.b32 {$0, $1, $2, $3}, [$4];",
              "=r,=r,=r,=r,l", loc, ip, side=True)
    return [Uint32(v) for v in r]


@dsl_user_op
def ld_cached_v4(addr, *, loc=None, ip=None):
    r = _asmN(4, T.i32(), [Int64(addr)], "ld.global.nc.v4.b32 {$0, $1, $2, $3}, [$4];",
              "=r,=r,=r,=r,l", loc, ip, side=True)
    return [Uint32(v) for v in r]


@dsl_user_op
def st_v4(addr, w, *, loc=None, ip=None):
    _asm1(None, [Int64(addr)] + [Uint32(x) for x in w],
          "st.global.L1::no_allocate.v4.b32 [$0], {$1, $2, $3, $4};", "l,r,r,r,r", loc, ip, side=True)


# One bf16x2 word of the own chunk (o) and of the partner chunk (q, sign already flipped where the
# formula subtracts), and the fp32 table pairs t0 t1 (multiplies the own value in the forward,
# the partner value in the backward) and u0 u1 (the other table). Returns the output word.
#   forward:  out = rn(o * t) +rn rn(q * u)            (t = c_own, u = s_own)
#   backward: out = (rn(q * t) +rn 0) +rn rn(o * u)    (t = a_partner, u = b_own)
_FWD = """{
.reg .b64 xo, xq, t, u, p, r;
.reg .b32 a, b;
shl.b32 a, $1, 16; and.b32 b, $1, 0xFFFF0000; mov.b64 xo, {a, b};
shl.b32 a, $2, 16; and.b32 b, $2, 0xFFFF0000; mov.b64 xq, {a, b};
mov.b64 t, {$3, $4}; mov.b64 u, {$5, $6};
mul.rn.f32x2 p, xo, t;
mul.rn.f32x2 r, xq, u;
add.rn.f32x2 p, p, r;
mov.b64 {a, b}, p;
cvt.rn.bf16x2.f32 $0, b, a;
}"""

_BWD = """{
.reg .b64 xo, xq, t, u, p, r, z;
.reg .b32 a, b;
shl.b32 a, $1, 16; and.b32 b, $1, 0xFFFF0000; mov.b64 xo, {a, b};
shl.b32 a, $2, 16; and.b32 b, $2, 0xFFFF0000; mov.b64 xq, {a, b};
mov.b64 t, {$3, $4}; mov.b64 u, {$5, $6};
mov.b64 z, 0;
mul.rn.f32x2 p, xq, t;
add.rn.f32x2 p, p, z;
mul.rn.f32x2 r, xo, u;
add.rn.f32x2 p, p, r;
mov.b64 {a, b}, p;
cvt.rn.bf16x2.f32 $0, b, a;
}"""


@dsl_user_op
def rope_word(o, q, t, *, loc=None, ip=None):
    return Uint32(_asm1(T.i32(), [Uint32(o), Uint32(q)] + [Uint32(v) for v in t], _FWD if FWD else _BWD,
                        "=r," + ",".join(["r"] * 6), loc, ip))


class Rope:
    @cute.jit
    def __call__(self, x, ct, st, out, stream):
        nthr = x.shape[0] * TPT
        self.kernel(x, ct, st, out, nthr).launch(
            grid=[(nthr + THREADS - 1) // THREADS, 1, 1], block=[THREADS, 1, 1], stream=stream)

    @cute.kernel
    def kernel(self, x, ct, st, out, nthr: Int32):
        tid, _, _ = cute.arch.thread_idx()
        bid, _, _ = cute.arch.block_idx()
        g = bid * THREADS + tid
        # TPT is a multiple of 32, so a warp never straddles tokens and either all or none of its
        # lanes pass this guard (the shuffles below need the full warp).
        if g < nthr:
            t = g // TPT
            r = g - t * TPT
            c = r % 8
            hg = r // 8
            t64 = Int64(t)
            cb = ct.iterator.toint() + t64 * (Int64(ct.stride[0]) * 4)
            sb = st.iterator.toint() + t64 * (Int64(st.stride[0]) * 4)
            own = Int64(c * 32)
            par = Int64((c ^ 4) * 32)
            if cutlass.const_expr(FWD):
                ta = cb + own  # c of the own dims
                ua = sb + own  # s of the own dims
            else:
                ta = cb + par  # a of the partner dims
                ua = sb + own  # b of the own dims
            tv = ld_cached_v4(ta) + ld_cached_v4(ta + 16)
            uv = ld_cached_v4(ua) + ld_cached_v4(ua + 16)
            # sign flip of the partner value: forward lo lanes (-x_hi * s), backward hi lanes (-g_lo * a)
            if cutlass.const_expr(FWD):
                flip = Uint32(1 - (c >> 2)) * Uint32(0x80008000)
            else:
                flip = Uint32(c >> 2) * Uint32(0x80008000)
            xb = t64 * (H * 128) + Int64(hg * 128 + c * 16)
            px = x.iterator.toint() + xb
            po = out.iterator.toint() + xb
            xs = []
            for k in cutlass.range_constexpr(HPT):
                xs.append(ld_stream_v4(px + k * HG * 128))
            for k in cutlass.range_constexpr(HPT):
                xo = xs[k]
                ow = [Uint32(0)] * 4
                for w in cutlass.range_constexpr(4):
                    q = Uint32(cute.arch.shuffle_sync_bfly(xo[w], offset=4)) ^ flip
                    ow[w] = rope_word(xo[w], q, [tv[2 * w], tv[2 * w + 1], uv[2 * w], uv[2 * w + 1]])
                st_v4(po + k * HG * 128, ow)


_SYM_T = cute.sym_int(symbol="T")
_bf = cutlass.BFloat16
_f32 = cutlass.Float32


def _table():
    return cute.runtime.make_fake_tensor(
        _f32, (_SYM_T, 1, 64), stride=(cute.sym_int64(divisibility=4), cute.sym_int64(), 1), assumed_align=16)


_fn = cute.compile(
    Rope(),
    cute.runtime.make_fake_tensor(_bf, (_SYM_T, H, 64), stride=(H * 64, 64, 1), assumed_align=16),
    _table(),
    _table(),
    cute.runtime.make_fake_tensor(_bf, (_SYM_T, H, 64), stride=(H * 64, 64, 1), assumed_align=16),
    cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
    options="--enable-tvm-ffi",
)


def _dense(t):
    return t if t.is_contiguous() and t.data_ptr() % 16 == 0 else t.clone(memory_format=torch.contiguous_format)


def run(in0, in1, in2):
    n = in0.shape[0]
    out = torch.empty((n, H, 64), dtype=torch.bfloat16, device=in0.device)
    if n:
        try:
            # The compiled entry point validates dtypes, devices, shapes, strides (x contiguous,
            # table row stride % 4 == 0, unit last stride) and 16-byte alignment.
            _fn(in0, in1, in2, out)
        except ValueError:
            # Unexpected layouts (misaligned / non-unit inner stride views): dense copies.
            _fn(_dense(in0), _dense(in1), _dense(in2), out)
    return out
