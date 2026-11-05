"""Fused RMSNorm backward + residual-grad add + MXFP8 rowwise/colwise quantize (DSv3 671B, DistMoE MXFP8).

Reference (pat_3ybb; pat_a6rv is the same math with another output order). run() returns
(dw, g, qrow, srow, scol, qcol.t()).
aten._fused_rms_norm_backward(grad_out=in1, x=in2, [7168], rstd=in0, w=in3, [True, True]) -> (dx, dw);
g = in4 + dx; torchao mxfp8_quantize(g, rowwise + colwise, 32, e4m3, rceil) + triton_mx_block_rearrange of both
scale tensors.

On this build aten._fused_rms_norm_backward runs the vendored quack CuTe DSL kernel
(torch/_native/ops/norm/rmsnorm_impl.py) for N=7168 bf16 on sm_103: RMSNormBackward with 128 threads/row,
one row per CTA iteration, grid = B = min(M, 2 * num_SMs) persistent CTAs (CTA b handles rows b, b + B, ...),
then reduce_rows(partial[B, N], dw, cols=16, threads=256). Per element (read off the quack PTX/SASS; ptxas
contracts its unrounded PTX mul/sub/add):
    x_hat = rn(x * rstd);  w' = bf16(w + 0)  (bf16 add: -0 -> +0)
    S_thread = sequential fma(x_hat_i, rn(dout_i * w'_i), acc) from +0 over the thread's 56 elements,
               i = j + 8 k for cols 8 t + 1024 k + j
    S = warp butterfly (16, 8, 4, 2, 1), then ((W0 + W2) + (W1 + W3)) after +0 on each W
    mean = div.rn(S, 7168)
    dx = bf16(rn(fma(dout, w', -rn(x_hat * mean)) * rstd))
    partial[b] = fma(x_hat, dout, acc) over the CTA's rows in order, from +0
    dw[c] = bf16((((S0 + S1) + S2) ... + S7)), S_w = A_2w + A_2w+1, A_m = 0 + p[m] + p[m+16] + ...
bf16 x bf16 products are exact in fp32, so rn(dout * w') and fma(dout, w', c) are computed with the mixed-precision
fma.rn.f32.bf16 (FHFMA; verified bitwise == unpack + fma.rn.f32 exhaustively for c = +-0 and on 2^32 random c).
g = bf16(in4 + dx) via add.rn.bf16x2 (== fp32 add + RNE).
MXFP8 (torchao, verified by dex/eve): amax = NaN-propagating max |g| over the 32-block; byte =
cvt.rp.ue8m0(amax * fl(1/448)); recip = 2^(127 - byte) (byte 255 -> NaN); q = cvt.rn.satfinite.e4m3(g * recip)
(g * recip again via fma.rn.f32.bf16: recip is exactly a bf16).

Kernels (all bytes of every output are written by the kernels):
  K0: zero the 32 per-128-row-block counters.
  K1 (B CTAs x 128 threads, quack's row schedule): dx, g, rowwise e4m3 + swizzled rowwise scales, fp32 dw
     partials. Per-thread cp.async 2-stage ring (each thread reads only bytes it copied itself), one CTA barrier
     per row for the row reduction. After each row is complete, thread 0 does red.release.gpu on its block's
     counter. Triggers programmatic dependent launch at its start.
  KC (num_SMs CTAs x 128 threads, PDL secondary of K1, runs concurrently): persistent 128 x 128 colwise tiles in
     row-block order; waits (ld.acquire.gpu) until all 128 rows of the block are signaled, reads g while it is
     still in L2 (g is stored with an L2 evict_last hint), quantizes colwise, transposes through swizzled smem into
     full 128 B lines of the column-major e4m3 output and one 512 B swizzled scale block per tile. Then
     griddepcontrol.wait (K1 complete) and the dw reduction in reduce_rows order.
  Spin-wait safety: the PDL secondary can only launch after every K1 CTA has executed launch_dependents, i.e.
  all K1 CTAs are resident; K1 never waits on KC, so every wait terminates. Without PDL, KC simply runs after K1.
"""

import cutlass
import cutlass.cute as cute
import torch
from cutlass import Float32, Int32, Int64, Uint32
from cutlass._mlir import ir
from cutlass._mlir.dialects import llvm
from cutlass.cutlass_dsl import T, dsl_user_op

M = 4096
N = 7168
NCH = 7  # 1024-column chunks per row (thread t owns cols 8 t + 1024 k + j)
ROWB = N * 2  # 14336 bytes per bf16 row
STAGE = 3 * ROWB  # x, dout, in4 rows
OFF_RED = 2 * STAGE
SMEM1 = OFF_RED + 64
NSC = N // 32  # 224 rowwise scale columns
NDW = N // 16  # 448 dw-reduce CTAs
OFF_SCL2 = 128 * 128  # colwise transposed tile [128 cols][128 B]
SMEM2 = 128 * 128 + 512 + 64

_INV448 = "0f3B124925"  # fl32(1/448)


def _vals(args, loc, ip):
    return [a.ir_value(loc=loc, ip=ip) for a in args]


def _asm1(ty, args, text, cons, loc, ip, side=False):
    return llvm.inline_asm(
        ty, _vals(args, loc, ip), text, cons,
        has_side_effects=side, is_align_stack=False, asm_dialect=llvm.AsmDialect.AD_ATT, loc=loc, ip=ip,
    )


def _asmN(n, ty, args, text, cons, loc, ip, side=False):
    tys = ty if isinstance(ty, (list, tuple)) else [ty] * n
    st = ir.Type.parse("!llvm.struct<(" + ", ".join(str(t) for t in tys) + ")>")
    res = llvm.inline_asm(
        st, _vals(args, loc, ip), text, cons,
        has_side_effects=side, is_align_stack=False, asm_dialect=llvm.AsmDialect.AD_ATT, loc=loc, ip=ip,
    )
    return [llvm.extractvalue(t, res, [k], loc=loc, ip=ip) for k, t in enumerate(tys)]


# ---------------------------------------------------------------- memory / sync primitives
@dsl_user_op
def ld_f32(addr, *, loc=None, ip=None):
    return Float32(_asm1(T.f32(), [Int64(addr)], "ld.global.nc.f32 $0, [$1];", "=f,l", loc, ip, side=True))


@dsl_user_op
def ld_v4(addr, *, loc=None, ip=None):
    r = _asmN(4, T.i32(), [Int64(addr)], "ld.global.nc.v4.b32 {$0, $1, $2, $3}, [$4];", "=r,=r,=r,=r,l", loc, ip,
              side=True)
    return [Uint32(v) for v in r]


@dsl_user_op
def st_u8(addr, a, *, loc=None, ip=None):
    _asm1(None, [Int64(addr), Uint32(a)], "st.global.u8 [$0], $1;", "l,r", loc, ip, side=True)


@dsl_user_op
def st_b16(addr, a, *, loc=None, ip=None):
    _asm1(None, [Int64(addr), Uint32(a)], "{ .reg .b16 h; cvt.u16.u32 h, $1; st.global.b16 [$0], h; }", "l,r", loc,
          ip, side=True)


@dsl_user_op
def st_b32(addr, a, *, loc=None, ip=None):
    _asm1(None, [Int64(addr), Uint32(a)], "st.global.b32 [$0], $1;", "l,r", loc, ip, side=True)


@dsl_user_op
def st_v4(addr, a, b, c, d, *, loc=None, ip=None):
    _asm1(None, [Int64(addr), Uint32(a), Uint32(b), Uint32(c), Uint32(d)],
          "st.global.v4.b32 [$0], {$1, $2, $3, $4};", "l,r,r,r,r", loc, ip, side=True)


@dsl_user_op
def st_f32x4(addr, a, b, c, d, *, loc=None, ip=None):
    _asm1(None, [Int64(addr), Float32(a), Float32(b), Float32(c), Float32(d)],
          "st.global.v4.f32 [$0], {$1, $2, $3, $4};", "l,f,f,f,f", loc, ip, side=True)


@dsl_user_op
def sts_v4(addr, a, b, c, d, *, loc=None, ip=None):
    _asm1(None, [Int32(addr), Uint32(a), Uint32(b), Uint32(c), Uint32(d)],
          "st.shared.v4.b32 [$0], {$1, $2, $3, $4};", "r,r,r,r,r", loc, ip, side=True)


@dsl_user_op
def sts_u8(addr, a, *, loc=None, ip=None):
    _asm1(None, [Int32(addr), Uint32(a)], "st.shared.u8 [$0], $1;", "r,r", loc, ip, side=True)


@dsl_user_op
def sts_f32(addr, a, *, loc=None, ip=None):
    _asm1(None, [Int32(addr), Float32(a)], "st.shared.f32 [$0], $1;", "r,f", loc, ip, side=True)


@dsl_user_op
def lds_v4(addr, *, loc=None, ip=None):
    r = _asmN(4, T.i32(), [Int32(addr)], "ld.shared.v4.b32 {$0, $1, $2, $3}, [$4];", "=r,=r,=r,=r,r", loc, ip,
              side=True)
    return [Uint32(v) for v in r]


@dsl_user_op
def lds_f32(addr, *, loc=None, ip=None):
    return Float32(_asm1(T.f32(), [Int32(addr)], "ld.shared.f32 $0, [$1];", "=f,r", loc, ip, side=True))


@dsl_user_op
def lds_f32x4(addr, *, loc=None, ip=None):
    r = _asmN(4, T.f32(), [Int32(addr)], "ld.shared.v4.f32 {$0, $1, $2, $3}, [$4];", "=f,=f,=f,=f,r", loc, ip,
              side=True)
    return [Float32(v) for v in r]


@dsl_user_op
def policy_evict_first(*, loc=None, ip=None):
    return Int64(_asm1(T.i64(), [], "createpolicy.fractional.L2::evict_first.b64 $0, 1.0;", "=l", loc, ip, side=True))


@dsl_user_op
def policy_evict_last(*, loc=None, ip=None):
    return Int64(_asm1(T.i64(), [], "createpolicy.fractional.L2::evict_last.b64 $0, 1.0;", "=l", loc, ip, side=True))


@dsl_user_op
def cp16h(dst, src, pol, *, loc=None, ip=None):
    _asm1(None, [Int32(dst), Int64(src), Int64(pol)], "cp.async.cg.shared.global.L2::cache_hint [$0], [$1], 16, $2;",
          "r,l,l", loc, ip, side=True)


@dsl_user_op
def st_v4h(addr, a, b, c, d, pol, *, loc=None, ip=None):
    _asm1(None, [Int64(addr), Uint32(a), Uint32(b), Uint32(c), Uint32(d), Int64(pol)],
          "st.global.L2::cache_hint.v4.b32 [$0], {$1, $2, $3, $4}, $5;", "l,r,r,r,r,l", loc, ip, side=True)


@dsl_user_op
def st_v2h(addr, a, b, pol, *, loc=None, ip=None):
    _asm1(None, [Int64(addr), Uint32(a), Uint32(b), Int64(pol)], "st.global.L2::cache_hint.v2.b32 [$0], {$1, $2}, $3;",
          "l,r,r,l", loc, ip, side=True)


@dsl_user_op
def red_release_add(addr, v: int, *, loc=None, ip=None):
    _asm1(None, [Int64(addr)], f"red.release.gpu.global.add.u32 [$0], {v};", "l", loc, ip, side=True)


@dsl_user_op
def spin_acquire_geq(addr, target: int, *, loc=None, ip=None):
    """Spin until the u32 at addr >= target (ld.acquire.gpu), with nanosleep backoff."""
    _asm1(
        None, [Int64(addr)],
        "{ .reg .u32 v; .reg .pred p;\n"
        "SPIN_${:uid}:\n"
        "ld.acquire.gpu.global.u32 v, [$0];\n"
        f"setp.ge.u32 p, v, {target};\n"
        "@p bra DONE_${:uid};\n"
        "nanosleep.u32 256;\n"
        "bra SPIN_${:uid};\n"
        "DONE_${:uid}: }",
        "l", loc, ip, side=True,
    )


@dsl_user_op
def ld_v2_cg(addr, pol, *, loc=None, ip=None):
    r = _asmN(2, T.i32(), [Int64(addr), Int64(pol)], "ld.global.cg.L2::cache_hint.v2.b32 {$0, $1}, [$2], $3;",
              "=r,=r,l,l", loc, ip, side=True)
    return [Uint32(v) for v in r]


@dsl_user_op
def qcol4h(w0, w1, w2, w3, rr0, rr1, *, loc=None, ip=None):
    """Rows j..j+3 (bf16x2 words w0..w3 = {col a, col b}) scaled by recips rr0 (col a), rr1 (col b) (f32 bits).

    Returns (word for column a: bytes = rows j..j+3, word for column b). fma.rn.f32.bf16(g, bf16(recip), -0)."""
    r = _asmN(
        2, T.i32(), [Uint32(w0), Uint32(w1), Uint32(w2), Uint32(w3), Uint32(rr0), Uint32(rr1)],
        "{ .reg .b32 nz, a0, a1, a2, a3, b0, b1, b2, b3; .reg .b16 x0, y0, x1, y1, x2, y2, x3, y3, ra, rb, u, h0, h1, h2, h3;\n"
        "mov.b32 nz, 0x80000000;\n"
        "mov.b32 {u, ra}, $6; mov.b32 {u, rb}, $7;\n"
        "mov.b32 {x0, y0}, $2; mov.b32 {x1, y1}, $3; mov.b32 {x2, y2}, $4; mov.b32 {x3, y3}, $5;\n"
        "fma.rn.f32.bf16 a0, x0, ra, nz; fma.rn.f32.bf16 a1, x1, ra, nz;\n"
        "fma.rn.f32.bf16 a2, x2, ra, nz; fma.rn.f32.bf16 a3, x3, ra, nz;\n"
        "fma.rn.f32.bf16 b0, y0, rb, nz; fma.rn.f32.bf16 b1, y1, rb, nz;\n"
        "fma.rn.f32.bf16 b2, y2, rb, nz; fma.rn.f32.bf16 b3, y3, rb, nz;\n"
        "cvt.rn.satfinite.e4m3x2.f32 h0, a1, a0;\n"
        "cvt.rn.satfinite.e4m3x2.f32 h1, a3, a2;\n"
        "cvt.rn.satfinite.e4m3x2.f32 h2, b1, b0;\n"
        "cvt.rn.satfinite.e4m3x2.f32 h3, b3, b2;\n"
        "mov.b32 $0, {h0, h1};\n"
        "mov.b32 $1, {h2, h3}; }",
        "=r,=r,r,r,r,r,r,r", loc, ip,
    )
    return Uint32(r[0]), Uint32(r[1])


@dsl_user_op
def cp_commit(*, loc=None, ip=None):
    _asm1(None, [], "cp.async.commit_group;", "", loc, ip, side=True)


@dsl_user_op
def cp_wait1(*, loc=None, ip=None):
    _asm1(None, [], "cp.async.wait_group 1;", "", loc, ip, side=True)


# ---------------------------------------------------------------- math primitives
@dsl_user_op
def fadd(a, b, *, loc=None, ip=None):
    return Float32(_asm1(T.f32(), [Float32(a), Float32(b)], "add.rn.f32 $0, $1, $2;", "=f,f,f", loc, ip))


@dsl_user_op
def add0(a, *, loc=None, ip=None):
    """a + (+0.0) (turns -0 into +0)."""
    return Float32(_asm1(T.f32(), [Float32(a)], "add.rn.f32 $0, $1, 0f00000000;", "=f,f", loc, ip))


@dsl_user_op
def div7168(a, *, loc=None, ip=None):
    return Float32(_asm1(T.f32(), [Float32(a)], "div.rn.f32 $0, $1, 0f45E00000;", "=f,f", loc, ip))


@dsl_user_op
def bf16x2_add0(w, *, loc=None, ip=None):
    """bf16x2 w + 0 (quack's no-op fp32 round trip on the weight; -0 -> +0)."""
    return Uint32(_asm1(T.i32(), [Uint32(w)], "{ .reg .b32 z; mov.b32 z, 0; add.rn.bf16x2 $0, $1, z; }", "=r,r",
                        loc, ip))


@dsl_user_op
def chain2h(xw, dw, ww, rs, acc, *, loc=None, ip=None):
    """Returns (acc', XH): XH = rn(x * rs) pair; acc' = fma(xh_hi, d_hi*w_hi, fma(xh_lo, d_lo*w_lo, acc)).

    d*w via fma.rn.f32.bf16(d, w, -0) == rn(f32(d) * f32(w)) (exact product; verified exhaustively)."""
    r = _asmN(
        2, [T.f32(), T.i64()], [Uint32(xw), Uint32(dw), Uint32(ww), Float32(rs), Float32(acc)],
        "{ .reg .b32 a, b, c, nz, p0, p1; .reg .b64 X, R, XH; .reg .b16 dl, dh, wl, wh; .reg .f32 s;\n"
        "shl.b32 a, $2, 16; and.b32 b, $2, 0xFFFF0000; mov.b64 X, {a, b};\n"
        "mov.b32 c, $5; mov.b64 R, {c, c};\n"
        "mul.rn.f32x2 XH, X, R;\n"
        "mov.b32 {dl, dh}, $3; mov.b32 {wl, wh}, $4; mov.b32 nz, 0x80000000;\n"
        "fma.rn.f32.bf16 p0, dl, wl, nz;\n"
        "fma.rn.f32.bf16 p1, dh, wh, nz;\n"
        "mov.b64 {a, b}, XH;\n"
        "fma.rn.f32 s, a, p0, $6;\n"
        "fma.rn.f32 $0, b, p1, s;\n"
        "mov.b64 $1, XH; }",
        "=f,=l,r,r,r,f,f", loc, ip,
    )
    return Float32(r[0]), Int64(r[1])


@dsl_user_op
def phase_ch(xh, dw, ww, iw, rs, nm, acc, *, loc=None, ip=None):
    """Returns (g bf16x2 word, new dw-partial pair) from the kept x_hat pair XH.

    dx = rn(fma(d, w, rn(xh * nm)) * rs) (nm = -mean; fma via fma.rn.f32.bf16); g = bf16(in4 + bf16(dx));
    acc = fma(xh, d, acc)."""
    r = _asmN(
        2, [T.i32(), T.i64()], [Int64(xh), Uint32(dw), Uint32(ww), Uint32(iw), Float32(rs), Float32(nm), Int64(acc)],
        "{ .reg .b32 a, b, c, e, p0, p1; .reg .b64 D, R, MM, TT, P, DX; .reg .b16 dl, dh, wl, wh;\n"
        "mov.b32 c, $6; mov.b64 R, {c, c};\n"
        "mov.b32 e, $7; mov.b64 MM, {e, e};\n"
        "mul.rn.f32x2 TT, $2, MM;\n"
        "mov.b64 {a, b}, TT;\n"
        "mov.b32 {dl, dh}, $3; mov.b32 {wl, wh}, $4;\n"
        "fma.rn.f32.bf16 p0, dl, wl, a;\n"
        "fma.rn.f32.bf16 p1, dh, wh, b;\n"
        "mov.b64 P, {p0, p1};\n"
        "mul.rn.f32x2 DX, P, R;\n"
        "mov.b64 {a, b}, DX;\n"
        "cvt.rn.bf16x2.f32 c, b, a;\n"
        "add.rn.bf16x2 $0, $5, c;\n"
        "shl.b32 a, $3, 16; and.b32 b, $3, 0xFFFF0000; mov.b64 D, {a, b};\n"
        "fma.rn.f32x2 $1, $2, D, $8; }",
        "=r,=l,l,r,r,r,f,f,l", loc, ip,
    )
    return Uint32(r[0]), Int64(r[1])


@dsl_user_op
def qrow4h(g0, g1, rr, *, loc=None, ip=None):
    """4 bytes e4m3(g * recip) for bf16x2 words g0 (cols 0,1), g1 (cols 2,3); recip = 2^k as f32 bits rr.

    g * recip via fma.rn.f32.bf16(g, bf16(recip), -0) == rn(f32(g) * recip) (recip is exactly a bf16)."""
    return Uint32(_asm1(
        T.i32(), [Uint32(g0), Uint32(g1), Uint32(rr)],
        "{ .reg .b32 nz, f0, f1, f2, f3; .reg .b16 a0, a1, a2, a3, rl, rh, h0, h1;\n"
        "mov.b32 {a0, a1}, $1; mov.b32 {a2, a3}, $2; mov.b32 {rl, rh}, $3; mov.b32 nz, 0x80000000;\n"
        "fma.rn.f32.bf16 f0, a0, rh, nz;\n"
        "fma.rn.f32.bf16 f1, a1, rh, nz;\n"
        "fma.rn.f32.bf16 f2, a2, rh, nz;\n"
        "fma.rn.f32.bf16 f3, a3, rh, nz;\n"
        "cvt.rn.satfinite.e4m3x2.f32 h0, f1, f0;\n"
        "cvt.rn.satfinite.e4m3x2.f32 h1, f3, f2;\n"
        "mov.b32 $0, {h0, h1}; }",
        "=r,r,r,r", loc, ip,
    ))


@dsl_user_op
def pair_lo(p, *, loc=None, ip=None):
    return Float32(_asm1(T.f32(), [Int64(p)], "{ .reg .b32 a, b; mov.b64 {a, b}, $1; mov.b32 $0, a; }", "=f,l", loc,
                         ip))


@dsl_user_op
def pair_hi(p, *, loc=None, ip=None):
    return Float32(_asm1(T.f32(), [Int64(p)], "{ .reg .b32 a, b; mov.b64 {a, b}, $1; mov.b32 $0, b; }", "=f,l", loc,
                         ip))


@dsl_user_op
def zero64(*, loc=None, ip=None):
    return Int64(_asm1(T.i64(), [], "mov.b64 $0, 0;", "=l", loc, ip))


@dsl_user_op
def bf16_bits(a, *, loc=None, ip=None):
    return Uint32(_asm1(T.i32(), [Float32(a)], "{ .reg .b16 h; cvt.rn.bf16.f32 h, $1; cvt.u32.u16 $0, h; }", "=r,f",
                        loc, ip))


@dsl_user_op
def hmax(a, b, *, loc=None, ip=None):
    """Per 16-bit half: max(|a|, |b|) with NaN propagation; sign bit is garbage (xor of signs)."""
    return Uint32(_asm1(T.i32(), [Uint32(a), Uint32(b)], "max.NaN.xorsign.abs.bf16x2 $0, $1, $2;", "=r,r,r", loc, ip))


@dsl_user_op
def prmt(a, b, sel: int, *, loc=None, ip=None):
    return Uint32(_asm1(T.i32(), [Uint32(a), Uint32(b)], f"prmt.b32 $0, $1, $2, {sel};", "=r,r,r", loc, ip))


@dsl_user_op
def scale2(p, *, loc=None, ip=None):
    """p = two masked bf16 amax values (lo, hi).

    Returns (bytes16 = lo byte | hi byte << 8, recip pair {recip_lo, recip_hi} as a 64-bit register pair).
    recip bits = (254 - b) << 23, byte 255 -> 0x7FC00000 (NaN).
    """
    r = _asmN(
        2, [T.i32(), T.i64()], [Uint32(p)],
        "{ .reg .b32 a0, a1, b0, b1, e0, e1, t0, t1; .reg .b64 x, s; .reg .b16 h;\n"
        "shl.b32 a0, $2, 16;\n"
        "and.b32 a1, $2, 0xFFFF0000;\n"
        "mov.b64 x, {a0, a1};\n"
        f"mov.b64 s, {{{_INV448}, {_INV448}}};\n"
        "mul.rn.f32x2 x, x, s;\n"
        "mov.b64 {a0, a1}, x;\n"
        "cvt.rp.ue8m0x2.f32 h, a1, a0;\n"
        "cvt.u32.u16 $0, h;\n"
        "and.b32 b0, $0, 255;\n"
        "shr.u32 b1, $0, 8;\n"
        "mad.lo.s32 e0, b0, -8388608, 0x7F000000;\n"
        "mad.lo.s32 e1, b1, -8388608, 0x7F000000;\n"
        "mad.lo.s32 t0, b0, 16384, 16384;\n"
        "mad.lo.s32 t1, b1, 16384, 16384;\n"
        "or.b32 e0, e0, t0;\n"
        "or.b32 e1, e1, t1;\n"
        "and.b32 e0, e0, 0x7FC00000;\n"
        "and.b32 e1, e1, 0x7FC00000;\n"
        "mov.b64 $1, {e0, e1}; }",
        "=r,=l,r", loc, ip,
    )
    return Uint32(r[0]), Int64(r[1])


@dsl_user_op
def recip_lo(rp, *, loc=None, ip=None):
    return Uint32(_asm1(T.i32(), [Int64(rp)], "{ .reg .b32 a, b; mov.b64 {a, b}, $1; mov.b32 $0, a; }", "=r,l", loc, ip))


@dsl_user_op
def recip_hi(rp, *, loc=None, ip=None):
    return Uint32(_asm1(T.i32(), [Int64(rp)], "{ .reg .b32 a, b; mov.b64 {a, b}, $1; mov.b32 $0, b; }", "=r,l", loc, ip))


# ---------------------------------------------------------------- K1: norm backward + add + row/col quant
class NormBwdQuant:
    def __init__(self, nblk: int):
        self.nblk = nblk

    @cute.jit
    def __call__(self, rstd, dout, x, w, res, g, qrow, srow, part, cnt_buf, stream):
        self.kernel(rstd, dout, x, w, res, g, qrow, srow, part, cnt_buf).launch(
            grid=[self.nblk, 1, 1], block=[128, 1, 1], stream=stream,
        )

    @cute.kernel
    def kernel(self, rstd, dout, x, w, res, g, qrow, srow, part, cnt_buf):
        B = self.nblk
        tid, _, _ = cute.arch.thread_idx()
        b, _, _ = cute.arch.block_idx()
        wid = tid // 32
        l = tid % 32

        cute.arch.griddepcontrol_launch_dependents()
        smem = cutlass.utils.SmemAllocator()
        sAll = smem.allocate_tensor(cutlass.Uint8, cute.make_layout(SMEM1), byte_alignment=128)
        sb = sAll.iterator.toint()

        rstd0 = rstd.iterator.toint()
        dout0 = dout.iterator.toint()
        x0 = x.iterator.toint()
        w0 = w.iterator.toint()
        res0 = res.iterator.toint()
        g0 = g.iterator.toint()
        qrow0 = qrow.iterator.toint()
        srow0 = srow.iterator.toint()
        part0 = part.iterator.toint()
        cnt0 = cnt_buf.iterator.toint()

        cnt = (M - 1 - b) // B + 1
        toff = Int64(tid * 16)
        pol_ef = policy_evict_first()
        pol_el = policy_evict_last()

        wv = cute.make_rmem_tensor(cute.make_layout(NCH * 4), Uint32)
        for kk in cutlass.range_constexpr(NCH):
            v = ld_v4(w0 + toff + kk * 2048)
            for q in cutlass.range_constexpr(4):
                wv[kk * 4 + q] = bf16x2_add0(v[q])
        xh = cute.make_rmem_tensor(cute.make_layout(NCH * 4), Int64)
        acc = cute.make_rmem_tensor(cute.make_layout(NCH * 4), Int64)
        for i in cutlass.range_constexpr(NCH * 4):
            acc[i] = zero64()

        def issue(r, s, sb, tid, toff, x0, dout0, res0, pol_ef):
            src = Int64(r) * ROWB + toff
            so = sb + s * STAGE + tid * 16
            for kk in cutlass.range_constexpr(NCH):
                cp16h(so + kk * 2048, x0 + src + kk * 2048, pol_ef)
            for kk in cutlass.range_constexpr(NCH):
                cp16h(so + ROWB + kk * 2048, dout0 + src + kk * 2048, pol_ef)
            for kk in cutlass.range_constexpr(NCH):
                cp16h(so + 2 * ROWB + kk * 2048, res0 + src + kk * 2048, pol_ef)

        issue(b, 0, sb, tid, toff, x0, dout0, res0, pol_ef)
        cp_commit()
        if cnt > 1:
            issue(b + B, 1, sb, tid, toff, x0, dout0, res0, pol_ef)
        cp_commit()
        rsb = cute.make_rmem_tensor(cute.make_layout(1), Float32)
        rsb[0] = ld_f32(rstd0 + Int64(b) * 4)

        for k in cutlass.range(cnt, unroll=1):
            r = b + k * B
            s = k % 2
            so = sb + s * STAGE + tid * 16
            rs = rsb[0]
            if k + 1 < cnt:
                rsb[0] = ld_f32(rstd0 + Int64(r + B) * 4)
            cp_wait1()

            # ---- phase A: per-thread sequential fma chain over 56 elements ----
            sacc = Float32(0.0)
            for kk in cutlass.range_constexpr(NCH):
                xv = lds_v4(so + kk * 2048)
                dv = lds_v4(so + ROWB + kk * 2048)
                for q in cutlass.range_constexpr(4):
                    sacc, xhq = chain2h(xv[q], dv[q], wv[kk * 4 + q], rs, sacc)
                    xh[kk * 4 + q] = xhq
            for sh in cutlass.range_constexpr(5):
                sacc = fadd(sacc, cute.arch.shuffle_sync_bfly(sacc, offset=16 >> sh))
            if l == 0:
                sts_f32(sb + OFF_RED + (s * 4 + wid) * 4, sacc)
            cute.arch.barrier()
            if k > 0:
                if tid == 0:
                    red_release_add(cnt0 + Int64((r - B) // 128) * 4, 1)
            wsum = lds_f32x4(sb + OFF_RED + s * 16)
            tot = fadd(fadd(add0(wsum[0]), add0(wsum[2])), fadd(add0(wsum[1]), add0(wsum[3])))
            nm = -div7168(tot)

            # ---- phase C: dx, g, dw partial, rowwise quant (chunk pairs) ----
            r64 = Int64(r)
            for cp in cutlass.range_constexpr(4):
                kks = [[0, 1], [2, 3], [4, 5], [6]][cp]
                mx = []
                gall = []
                for kk in kks:
                    dv = lds_v4(so + ROWB + kk * 2048)
                    iv = lds_v4(so + 2 * ROWB + kk * 2048)
                    gw = []
                    for q in cutlass.range_constexpr(4):
                        gq, aq = phase_ch(xh[kk * 4 + q], dv[q], wv[kk * 4 + q], iv[q], rs, nm, acc[kk * 4 + q])
                        acc[kk * 4 + q] = aq
                        gw.append(gq)
                    st_v4h(g0 + r64 * ROWB + toff + kk * 2048, gw[0], gw[1], gw[2], gw[3], pol_el)
                    mx.append(hmax(hmax(gw[0], gw[1]), hmax(gw[2], gw[3])))
                    gall.append(gw)
                mxb = mx[len(mx) - 1]
                pp = hmax(prmt(mx[0], mxb, 0x5410), prmt(mx[0], mxb, 0x7632))
                pp = hmax(pp, cute.arch.shuffle_sync_bfly(pp, offset=1))
                pp = hmax(pp, cute.arch.shuffle_sync_bfly(pp, offset=2))
                pp = pp & Uint32(0x7FFF7FFF)
                b16, rp = scale2(pp)
                rr = [recip_lo(rp), recip_hi(rp)]
                for ii in cutlass.range_constexpr(len(kks)):
                    kk = kks[ii]
                    gw = gall[ii]
                    y0 = qrow4h(gw[0], gw[1], rr[ii])
                    y1 = qrow4h(gw[2], gw[3], rr[ii])
                    st_v2h(qrow0 + r64 * N + Int64(tid * 8 + kk * 1024), y0, y1, pol_ef)
                    if l % 4 == 0:
                        sc = kk * 32 + tid // 4
                        soff = ((r // 128) * (NSC // 4) + sc // 4) * 512 + (r % 32) * 16 + ((r % 128) // 32) * 4 + sc % 4
                        st_u8(srow0 + Int64(soff), (b16 >> Uint32(8 * ii)) & Uint32(255))

            if k + 2 < cnt:
                issue(r + 2 * B, s, sb, tid, toff, x0, dout0, res0, pol_ef)
            cp_commit()

        cute.arch.barrier()
        if tid == 0:
            red_release_add(cnt0 + Int64((b + (cnt - 1) * B) // 128) * 4, 1)
        # ---- dw partial row b ----
        pb = part0 + Int64(b) * (N * 4) + Int64(tid * 32)
        for kk in cutlass.range_constexpr(NCH):
            st_f32x4(pb + kk * 4096, pair_lo(acc[kk * 4]), pair_hi(acc[kk * 4]),
                     pair_lo(acc[kk * 4 + 1]), pair_hi(acc[kk * 4 + 1]))
            st_f32x4(pb + kk * 4096 + 16, pair_lo(acc[kk * 4 + 2]), pair_hi(acc[kk * 4 + 2]),
                     pair_lo(acc[kk * 4 + 3]), pair_hi(acc[kk * 4 + 3]))
# ---------------------------------------------------------------- K2: colwise quant (concurrent with K1 via PDL)
class ColQuant:
    def __init__(self, nctas: int, nblk: int):
        self.nctas = nctas
        self.nblk = nblk

    @cute.jit
    def __call__(self, g, qcolt, scol, cnt_buf, part, dw, stream):
        self.kernel(g, qcolt, scol, cnt_buf, part, dw).launch(grid=[self.nctas, 1, 1], block=[128, 1, 1],
                                                              stream=stream, use_pdl=True)

    @cute.kernel
    def kernel(self, g, qcolt, scol, cnt_buf, part, dw):
        """Persistent; tile t = 128 rows (block T) x 128 cols (cb), t = cta + nctas * j in block order.
        Waits until K1 has signaled all 128 rows of block T (acquire), so g is read while still in L2."""
        tid, _, _ = cute.arch.thread_idx()
        cta, _, _ = cute.arch.block_idx()
        wid = tid // 32
        l = tid % 32
        smem = cutlass.utils.SmemAllocator()
        sAll = smem.allocate_tensor(cutlass.Uint8, cute.make_layout(SMEM2), byte_alignment=128)
        sb = sAll.iterator.toint()
        g0 = g.iterator.toint()
        q0 = qcolt.iterator.toint()
        sc0 = scol.iterator.toint()
        cnt0 = cnt_buf.iterator.toint()
        pol_ef = policy_evict_first()
        ntiles = (M // 128) * (N // 128)
        for t in cutlass.range(cta, ntiles, self.nctas, unroll=1):
            T = t // (N // 128)
            cb = t % (N // 128)
            if tid == 0:
                spin_acquire_geq(cnt0 + Int64(T) * 4, 128)
            cute.arch.barrier()
            r0 = T * 128 + wid * 32
            c0 = cb * 128 + l * 4
            v = []
            for i in cutlass.range_constexpr(32):
                v.append(ld_v2_cg(g0 + (Int64(r0 + i) * N + Int64(c0)) * 2, pol_ef))
            m0 = hmax(v[0][0], v[1][0])
            m1 = hmax(v[0][1], v[1][1])
            for i in cutlass.range_constexpr(2, 32):
                m0 = hmax(m0, v[i][0])
                m1 = hmax(m1, v[i][1])
            b0, rp0 = scale2(m0 & Uint32(0x7FFF7FFF))
            b1, rp1 = scale2(m1 & Uint32(0x7FFF7FFF))
            ra = [recip_lo(rp0), recip_hi(rp0), recip_lo(rp1), recip_hi(rp1)]
            words = [[None] * 8 for _ in range(4)]
            for jj in cutlass.range_constexpr(8):
                wa, wb = qcol4h(v[4 * jj][0], v[4 * jj + 1][0], v[4 * jj + 2][0], v[4 * jj + 3][0], ra[0], ra[1])
                wc, wd = qcol4h(v[4 * jj][1], v[4 * jj + 1][1], v[4 * jj + 2][1], v[4 * jj + 3][1], ra[2], ra[3])
                words[0][jj] = wa
                words[1][jj] = wb
                words[2][jj] = wc
                words[3][jj] = wd
            sw = l & 7
            for c in cutlass.range_constexpr(4):
                cl = l * 4 + c
                sts_v4(sb + cl * 128 + (((2 * wid) ^ sw) * 16), words[c][0], words[c][1], words[c][2], words[c][3])
                sts_v4(sb + cl * 128 + (((2 * wid + 1) ^ sw) * 16), words[c][4], words[c][5], words[c][6], words[c][7])
                byte = ([b0, b0, b1, b1][c] >> Uint32(8 * (c % 2))) & Uint32(255)
                sts_u8(sb + OFF_SCL2 + (cl % 32) * 16 + (cl // 32) * 4 + wid, byte)
            cute.arch.barrier()
            for mm in cutlass.range_constexpr(8):
                idx = tid + 128 * mm
                cl = idx >> 3
                ch = idx & 7
                vt = lds_v4(sb + cl * 128 + ((ch ^ ((cl >> 2) & 7)) * 16))
                st_v4h(q0 + Int64(cb * 128 + cl) * M + Int64(T * 128 + ch * 16), vt[0], vt[1], vt[2], vt[3], pol_ef)
            if tid < 32:
                vs = lds_v4(sb + OFF_SCL2 + tid * 16)
                st_v4(sc0 + Int64((cb * (M // 128) + T) * 512 + tid * 16), vs[0], vs[1], vs[2], vs[3])
            cute.arch.barrier()
        # ---- dw = reduce_rows(partial, cols=16, threads=256) order, after K1 completed ----
        cute.arch.griddepcontrol_wait()
        B = self.nblk
        part0 = part.iterator.toint()
        cl16 = tid % 16
        wp = tid // 16  # = warp w of the 256-thread reduce_rows block: row lanes m = 2 wp, 2 wp + 1
        for grp in cutlass.range(cta, NDW, self.nctas, unroll=1):
            col = grp * 16 + cl16
            pv = []
            for h in cutlass.range_constexpr(2):
                m = 2 * wp + h
                for i in cutlass.range_constexpr((B + 15) // 16):
                    vi = Float32(0.0)
                    if m + 16 * i < B:
                        vi = ld_f32(part0 + (Int64(m + 16 * i) * N + Int64(col)) * 4)
                    pv.append(vi)
            nrow = (B + 15) // 16
            aa = []
            for h in cutlass.range_constexpr(2):
                m = 2 * wp + h
                a = Float32(0.0)
                for i in cutlass.range_constexpr(nrow):
                    if m + 16 * i < B:
                        a = fadd(a, pv[h * nrow + i])
                aa.append(a)
            sw_ = fadd(aa[0], aa[1])
            cute.arch.barrier()
            sts_f32(sb + (wp * 16 + cl16) * 4, sw_)
            cute.arch.barrier()
            if tid < 16:
                a = lds_f32(sb + tid * 4)
                for i in cutlass.range_constexpr(1, 8):
                    a = fadd(a, lds_f32(sb + (i * 16 + tid) * 4))
                st_b16(dw.iterator.toint() + Int64(grp * 16 + tid) * 2, bf16_bits(a))


# ---------------------------------------------------------------- K0: zero the block counters
class ZeroCounters:
    @cute.jit
    def __call__(self, cnt_buf, stream):
        self.kernel(cnt_buf).launch(grid=[1, 1, 1], block=[32, 1, 1], stream=stream)

    @cute.kernel
    def kernel(self, cnt_buf):
        tid, _, _ = cute.arch.thread_idx()
        st_b32(cnt_buf.iterator.toint() + Int64(tid) * 4, Uint32(0))


_DT = {torch.bfloat16: cutlass.BFloat16, torch.float32: cutlass.Float32, torch.uint8: cutlass.Uint8,
       torch.int32: cutlass.Int32}


def _fake(dtype, shape, stride=None, align=16):
    if stride is None:
        stride, a = [], 1
        for s in reversed(shape):
            stride.append(a)
            a *= s
        stride = tuple(reversed(stride))
    return cute.runtime.make_fake_tensor(_DT[dtype], tuple(shape), stride=tuple(stride), assumed_align=align)


def _num_blocks(device):
    # quack RMSNormBackward grid: min(M, get_sm_count(N, device)); sm_count * 2 for N > 2048 on SM100+
    props = torch.cuda.get_device_properties(device)
    if props.major < 10:
        raise ValueError("expects the SM100-family quack schedule")
    return min(M, props.multi_processor_count * 2)


_B = _num_blocks(torch.cuda.current_device())
if _B <= 32:
    raise ValueError("reduce_rows(32, 128) path not implemented")

_k1 = cute.compile(
    NormBwdQuant(_B),
    _fake(torch.float32, (M, 1), (1, 1), align=4),
    _fake(torch.bfloat16, (M, N)),
    _fake(torch.bfloat16, (M, N)),
    _fake(torch.bfloat16, (N,)),
    _fake(torch.bfloat16, (M, N)),
    _fake(torch.bfloat16, (M, N)),
    _fake(torch.uint8, (M, N)),
    _fake(torch.uint8, (M, NSC)),
    _fake(torch.float32, (_B, N)),
    _fake(torch.int32, (M // 128,)),
    cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
    options="--enable-tvm-ffi",
)
_NSM = torch.cuda.get_device_properties(torch.cuda.current_device()).multi_processor_count
_kc = cute.compile(
    ColQuant(_NSM, _B),
    _fake(torch.bfloat16, (M, N)),
    _fake(torch.uint8, (N, M)),
    _fake(torch.uint8, (N, M // 32)),
    _fake(torch.int32, (M // 128,)),
    _fake(torch.float32, (_B, N)),
    _fake(torch.bfloat16, (N,)),
    cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
    options="--enable-tvm-ffi",
)
_k0 = cute.compile(
    ZeroCounters(),
    _fake(torch.int32, (M // 128,)),
    cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
    options="--enable-tvm-ffi",
)


def _check(t, dtype, shape, stride):
    if t.dtype != dtype or tuple(t.shape) != shape or tuple(t.stride()) != stride or t.device.type != "cuda":
        raise ValueError(f"expected cuda {dtype} {shape} stride {stride}, got {t.dtype} {tuple(t.shape)} "
                         f"stride {t.stride()} on {t.device}")


def _reference(in0, in1, in2, in3, in4):
    """Exactly the reference op sequence (used only for inputs the kernels cannot take: misaligned views)."""
    import torchao.prototype.mx_formats.kernels  # noqa: F401  (registers torchao::mxfp8_quantize & co.)

    v6 = torch.ops.aten._fused_rms_norm_backward.default(in1, in2, [N], torch.ops.aten.alias.default(in0), in3,
                                                         [True, True])
    v8 = torch.ops.aten.add.Tensor(in4, v6[0])
    v9 = torch.ops.torchao.mxfp8_quantize.default(v8, True, True, 32, 32, "e4m3", "rceil")
    return (
        v8,
        v9[0],
        torch.ops.aten.t.default(v9[1]),
        torch.ops.torchao.triton_mx_block_rearrange.default(v9[2]),
        torch.ops.torchao.triton_mx_block_rearrange.default(v9[3]),
        v6[1],
    )


def _compute(in0, in1, in2, in3, in4):
    _check(in0, torch.float32, (M, 1), (1, 1))
    _check(in1, torch.bfloat16, (M, N), (N, 1))
    _check(in2, torch.bfloat16, (M, N), (N, 1))
    _check(in3, torch.bfloat16, (N,), (1,))
    _check(in4, torch.bfloat16, (M, N), (N, 1))
    if _num_blocks(in1.device) != _B:
        raise ValueError("device SM count differs from the compiled schedule")
    if in0.data_ptr() % 4 or any(t.data_ptr() % 16 for t in (in1, in2, in3, in4)):
        return _reference(in0, in1, in2, in3, in4)
    dev = in1.device
    g = torch.empty((M, N), dtype=torch.bfloat16, device=dev)
    qrow = torch.empty((M, N), dtype=torch.uint8, device=dev)
    qcol = torch.empty_strided((M, N), (1, M), dtype=torch.uint8, device=dev)
    srow = torch.empty((M, NSC), dtype=torch.uint8, device=dev)
    scol = torch.empty((N, M // 32), dtype=torch.uint8, device=dev)
    part = torch.empty((_B, N), dtype=torch.float32, device=dev)
    dw = torch.empty((N,), dtype=torch.bfloat16, device=dev)
    cnt = torch.empty((M // 128,), dtype=torch.int32, device=dev)
    _k0(cnt)
    _k1(in0, in1, in2, in3, in4, g, qrow, srow, part, cnt)
    _kc(g, qcol.t(), scol, cnt, part, dw)
    return (
        g,
        qrow.view(torch.float8_e4m3fn),
        qcol.view(torch.float8_e4m3fn).t(),
        srow.view(torch.float8_e8m0fnu),
        scol.view(torch.float8_e8m0fnu),
        dw,
    )


def run(in0, in1, in2, in3, in4):
    g, qrow, qcol_t, srow, scol, dw = _compute(in0, in1, in2, in3, in4)
    return (dw, g, qrow, srow, scol, qcol_t)
