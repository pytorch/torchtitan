"""Chunked cross-entropy loss forward, one 4096-token chunk, vocab 151936 (Qwen3 30B-A3B, G-90).

Reference: v3 = f32(in0); v4 = _log_softmax(v3, 1); (v6, v7) = nll_loss_forward(v4, in1, None, sum, -100);
v8 = v6 / in2; returns (v4, v7, v8).

torch's cunn_SoftMaxForward<4, float, float, float, LogSoftMaxForwardEpilogue> (sm_100 cubin, 1024 threads per
row, no shift since rows are 16 B aligned) computes, per row:
    max_k  = max over the row (FMNMX from -FLT_MAX, NaN ignored; any order gives the same value up to the sign
             of a zero max, which never changes an output: see below)
    s_t    = chain over thread t's elements x[4096 i + 4 t + j] (i = 0..36, j = 0..3, then x[151552 + t] if
             t < 384), from +0, each step s = fma(2^jj, ex2(f), s) (CUDA's accurate expf of x - max_k with the
             final multiply contracted into the accumulation)
    sum    = per-warp shfl_down tree (16, 8, 4, 2, 1) over 32 consecutive t, then the same tree over the 32 warp sums
    L      = CUDA's accurate logf(sum)
    y      = (x - max_k) - L
A +-0 tie in the max needs both +0 and -0 in the row, so sum >= 2, L > 0 and (x -+ 0) - L is the same value.

Kernel 1 (one 8-CTA cluster per row, 128 threads per CTA): CTA c owns torch threads t = 128 c .. 128 c + 127,
i.e. the contiguous 512-element slice [4096 i + 512 c, 4096 i + 512 c + 512) of every i plus its tail elements,
so every exp chain and every warp tree lives in one CTA. The CTA bulk-copies its bf16 slice (37.25 KB) into
shared memory once, computes its warp maxima, pushes them to all 8 CTAs' shared memory (DSMEM), cluster barrier,
runs the exact chains and warp trees, pushes the 32 warp sums, cluster barrier, finishes the tree and logf, and
writes its fp32 slice from shared memory. HBM traffic: one bf16 read, one fp32 write.
Kernel 2 (1 CTA x 256 threads, PDL secondary): nll_loss_forward_reduce_cuda_kernel_2d's exact order (thread p
subtracts rows p, p + 256, ... from +0; smem tree 128 .. 1), reading v4[i, t_i] back from kernel 1's output;
then v8 = div.rn(v6, f32(in2)).
"""

import cutlass
import cutlass.cute as cute
import torch
from cutlass import Float32, Int32, Int64, Uint32
from cutlass._mlir import ir
from cutlass._mlir.dialects import llvm
from cutlass.cutlass_dsl import T, dsl_user_op

M = 4096
V = 151936
K = 8  # CTAs per row (cluster size)
TPC = 1024 // K  # torch threads (exp chains) per CTA = physical threads per CTA
NW = TPC // 32  # warps per CTA
NI = V // 4096  # 37 full ILP-4 iterations per torch thread
TAIL = V - NI * 4096  # 384 tail elements (one each for torch threads t < 384)
CHB = 4 * TPC * 2  # bytes of one bf16 chunk slice (1 KB)
TAILB = TPC * 2  # bytes of a CTA's tail slice (256 B, CTAs with 128 c < 384)
NR = 11  # chunks 0 .. NR-1 live in registers (ld.global by each thread), the rest in shared memory (bulk copies)
MINB = 8  # CTAs per SM: 8 x (smem + 1 KB) <= 228 KB and 64 registers per thread
GROUPS = ((NR, 20), (20, 28), (28, NI))  # smem chunk ranges sharing one mbarrier (the last one also has the tail)
SM_DATA = (NI - NR) * CHB + TAILB
SM_TAIL = (NI - NR) * CHB
SM_MBAR = SM_DATA  # 8 B per group
SM_XMAX = SM_MBAR + 8 * len(GROUPS)  # mbarrier: the 32 warp maxima arrived (st.async from the 8 CTAs)
SM_XSUM = SM_XMAX + 8  # mbarrier: the 32 warp sums arrived
SM_SMAX = SM_XSUM + 8  # 32 f32 warp maxima
SM_SSUM = SM_SMAX + 4 * 32  # 32 f32 warp sums
SMEM1 = SM_SSUM + 4 * 32
NLL_T = 256  # torch's nll_loss_threads(4096)
NLL_K = M // NLL_T
assert TAIL % TPC == 0 and NI * 4096 + TAIL == V and K * NW == 32


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


# ---------------------------------------------------------------- memory, barriers, cluster
@dsl_user_op
def cluster_rank(*, loc=None, ip=None):
    return Int32(_asm1(T.i32(), [], "mov.u32 $0, %cluster_ctarank;", "=r", loc, ip))


@dsl_user_op
def cluster_arrive_relaxed(*, loc=None, ip=None):
    _asm1(None, [], "barrier.cluster.arrive.relaxed.aligned;", "", loc, ip, side=True)


@dsl_user_op
def cluster_wait(*, loc=None, ip=None):
    _asm1(None, [], "barrier.cluster.wait.aligned;", "", loc, ip, side=True)


@dsl_user_op
def cluster_sync(*, loc=None, ip=None):
    _asm1(None, [], "barrier.cluster.arrive.release.aligned; barrier.cluster.wait.acquire.aligned;", "", loc, ip,
          side=True)


@dsl_user_op
def mbar_init(addr, count: int, *, loc=None, ip=None):
    _asm1(None, [Int32(addr)], f"mbarrier.init.shared::cta.b64 [$0], {count};", "r", loc, ip, side=True)


@dsl_user_op
def fence_mbar_init(*, loc=None, ip=None):
    _asm1(None, [], "fence.mbarrier_init.release.cluster;", "", loc, ip, side=True)


@dsl_user_op
def mbar_expect_tx(addr, nbytes, *, loc=None, ip=None):
    _asm1(None, [Int32(addr), Int32(nbytes)],
          "mbarrier.arrive.expect_tx.shared::cta.b64 _, [$0], $1;", "r,r", loc, ip, side=True)


@dsl_user_op
def mbar_wait0(addr, *, loc=None, ip=None):
    """Wait for phase 0 of the mbarrier at addr to complete."""
    _asm1(
        None, [Int32(addr)],
        "{ .reg .pred p;\n"
        "WAIT_${:uid}:\n"
        "mbarrier.try_wait.parity.shared::cta.b64 p, [$0], 0;\n"
        "@!p bra WAIT_${:uid}; }",
        "r", loc, ip, side=True,
    )


@dsl_user_op
def policy_evict_first(*, loc=None, ip=None):
    return Int64(_asm1(T.i64(), [], "createpolicy.fractional.L2::evict_first.b64 $0, 1.0;", "=l", loc, ip, side=True))


@dsl_user_op
def bulk_g2s(dst, src, nbytes, mbar, pol, *, loc=None, ip=None):
    _asm1(None, [Int32(dst), Int64(src), Int32(nbytes), Int32(mbar), Int64(pol)],
          "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint [$0], [$1], $2, [$3], $4;",
          "r,l,r,r,l", loc, ip, side=True)


@dsl_user_op
def ldg_v2(addr, pol, *, loc=None, ip=None):
    r = _asmN(2, T.i32(), [Int64(addr), Int64(pol)],
              "ld.global.nc.L1::no_allocate.L2::cache_hint.v2.b32 {$0, $1}, [$2], $3;", "=r,=r,l,l", loc, ip, side=True)
    return [Uint32(v) for v in r]


@dsl_user_op
def lds_v2(addr, *, loc=None, ip=None):
    r = _asmN(2, T.i32(), [Int32(addr)], "ld.shared.v2.b32 {$0, $1}, [$2];", "=r,=r,r", loc, ip, side=True)
    return [Uint32(v) for v in r]


@dsl_user_op
def lds_bf16_as_f32(addr, *, loc=None, ip=None):
    return Float32(_asm1(T.f32(), [Int32(addr)],
                         "{ .reg .b32 u; ld.shared.u16 u, [$1]; shl.b32 u, u, 16; mov.b32 $0, u; }",
                         "=f,r", loc, ip, side=True))


@dsl_user_op
def lds_f32(addr, *, loc=None, ip=None):
    return Float32(_asm1(T.f32(), [Int32(addr)], "ld.shared.f32 $0, [$1];", "=f,r", loc, ip, side=True))


@dsl_user_op
def sts_f32(addr, v, *, loc=None, ip=None):
    _asm1(None, [Int32(addr), Float32(v)], "st.shared.f32 [$0], $1;", "r,f", loc, ip, side=True)


@dsl_user_op
def st_async_f32(addr, mbar, rank, v, *, loc=None, ip=None):
    """st.async v to shared address addr of cluster CTA `rank`, completing 4 tx bytes on its mbarrier mbar."""
    _asm1(None, [Int32(addr), Int32(mbar), Int32(rank), Float32(v)],
          "{ .reg .b32 ra, rb; mapa.shared::cluster.u32 ra, $0, $2; mapa.shared::cluster.u32 rb, $1, $2;\n"
          "st.async.shared::cluster.mbarrier::complete_tx::bytes.b32 [ra], $3, [rb]; }",
          "r,r,r,f", loc, ip, side=True)


@dsl_user_op
def st_v4(addr, a, b, c, d, *, loc=None, ip=None):
    _asm1(None, [Int64(addr), Float32(a), Float32(b), Float32(c), Float32(d)],
          "st.global.v4.f32 [$0], {$1, $2, $3, $4};", "l,f,f,f,f", loc, ip, side=True)


@dsl_user_op
def st_f32(addr, v, *, loc=None, ip=None):
    _asm1(None, [Int64(addr), Float32(v)], "st.global.f32 [$0], $1;", "l,f", loc, ip, side=True)


@dsl_user_op
def ld_f32(addr, *, loc=None, ip=None):
    """Coherent load (data written by the primary kernel)."""
    return Float32(_asm1(T.f32(), [Int64(addr)], "ld.global.f32 $0, [$1];", "=f,l", loc, ip, side=True))


@dsl_user_op
def ld_s64(addr, *, loc=None, ip=None):
    return Int64(_asm1(T.i64(), [Int64(addr)], "ld.global.s64 $0, [$1];", "=l,l", loc, ip, side=True))


# ---------------------------------------------------------------- warp primitives
@dsl_user_op
def shfl_bfly(v, off: int, *, loc=None, ip=None):
    return Float32(_asm1(T.f32(), [Float32(v)], f"shfl.sync.bfly.b32 $0, $1, {off}, 0x1f, 0xffffffff;", "=f,f",
                         loc, ip, side=True))


@dsl_user_op
def shfl_down(v, off: int, *, loc=None, ip=None):
    return Float32(_asm1(T.f32(), [Float32(v)], f"shfl.sync.down.b32 $0, $1, {off}, 0x1f, 0xffffffff;", "=f,f",
                         loc, ip, side=True))


@dsl_user_op
def shfl_idx0(v, *, loc=None, ip=None):
    return Float32(_asm1(T.f32(), [Float32(v)], "shfl.sync.idx.b32 $0, $1, 0, 0x1f, 0xffffffff;", "=f,f",
                         loc, ip, side=True))


# ---------------------------------------------------------------- math
@dsl_user_op
def fmax(a, b, *, loc=None, ip=None):
    return Float32(_asm1(T.f32(), [Float32(a), Float32(b)], "max.f32 $0, $1, $2;", "=f,f,f", loc, ip))


@dsl_user_op
def fadd(a, b, *, loc=None, ip=None):
    return Float32(_asm1(T.f32(), [Float32(a), Float32(b)], "add.rn.f32 $0, $1, $2;", "=f,f,f", loc, ip))


@dsl_user_op
def fneg(a, *, loc=None, ip=None):
    return Float32(_asm1(T.f32(), [Float32(a)], "neg.f32 $0, $1;", "=f,f", loc, ip))


@dsl_user_op
def bmax(a, b, *, loc=None, ip=None):
    """bf16x2 max (a NaN lane yields the other operand's lane)."""
    return Uint32(_asm1(T.i32(), [Uint32(a), Uint32(b)], "max.bf16x2 $0, $1, $2;", "=r,r,r", loc, ip))


@dsl_user_op
def bmax_to_f32(acc, *, loc=None, ip=None):
    """max(f32(lo), f32(hi), -FLT_MAX) of a bf16x2 max accumulator (lanes start at -inf, never NaN)."""
    return Float32(_asm1(T.f32(), [Uint32(acc)],
                         "{ .reg .b32 a, b; .reg .f32 x, y;\n"
                         "shl.b32 a, $1, 16; and.b32 b, $1, 0xffff0000; mov.b32 x, a; mov.b32 y, b;\n"
                         "max.f32 x, x, y; max.f32 $0, x, 0fFF7FFFFF; }",
                         "=f,r", loc, ip))


# exp(x - max) accumulated into the chain s, as torch: d = x + (-max); t = sat(fma(d, 0x3BBB989D, 0.5));
# j = fma.rm(t, 252, 12582913); f = fma(d, log2e_lo, fma(d, log2e_hi, -(j - 12583039))); s = fma(j << 23, ex2(f), s).
# Pairs use f32x2 (same IEEE rn ops). 12583039 - j replaces -(j - 12583039): equal except the sign of a zero
# difference (only for d = +-0, then f = +-0 and ex2(+-0) = 1 either way).
_EXP_CONSTS = (
    "mov.b64 CS, {0f437C0000, 0f437C0000};\n"
    "mov.b64 CB, {0f4B400001, 0f4B400001};\n"
    "mov.b64 CN, {0f4B40007F, 0f4B40007F};\n"
    "mov.b64 CL, {0f3FB8AA3B, 0f3FB8AA3B};\n"
    "mov.b64 CLO, {0f32A57060, 0f32A57060};\n"
)


def _exp_pair(k):
    """Pair k: X{k} = {x_lo, x_hi} (b64), NM = {-max, -max}; leaves S{k} (scales) and E{k} (ex2) as b32 pairs."""
    return (
        f"add.rn.f32x2 D{k}, X{k}, NM;\n"
        f"mov.b64 {{d{k}a, d{k}b}}, D{k};\n"
        f"fma.rn.sat.f32 t{k}a, d{k}a, 0f3BBB989D, 0f3F000000;\n"
        f"fma.rn.sat.f32 t{k}b, d{k}b, 0f3BBB989D, 0f3F000000;\n"
        f"mov.b64 TT{k}, {{t{k}a, t{k}b}};\n"
        f"fma.rm.f32x2 J{k}, TT{k}, CS, CB;\n"
        f"sub.rn.f32x2 F{k}, CN, J{k};\n"
        f"fma.rn.f32x2 F{k}, D{k}, CL, F{k};\n"
        f"fma.rn.f32x2 F{k}, D{k}, CLO, F{k};\n"
        f"mov.b64 {{s{k}a, s{k}b}}, J{k};\n"
        f"shl.b32 s{k}a, s{k}a, 23;\n"
        f"shl.b32 s{k}b, s{k}b, 23;\n"
        f"mov.b64 {{e{k}a, e{k}b}}, F{k};\n"
        f"ex2.approx.ftz.f32 e{k}a, e{k}a;\n"
        f"ex2.approx.ftz.f32 e{k}b, e{k}b;\n"
    )


_SUM4 = (
    "{ .reg .b64 X0, X1, D0, D1, TT0, TT1, J0, J1, F0, F1, NM, CS, CB, CN, CL, CLO;\n"
    ".reg .b32 x0, x1, x2, x3, nm, d0a, d0b, d1a, d1b, t0a, t0b, t1a, t1b, s0a, s0b, s1a, s1b, e0a, e0b, e1a, e1b;\n"
    ".reg .f32 acc;\n"
    + _EXP_CONSTS
    + "mov.b32 nm, $3;\n"
    "mov.b64 NM, {nm, nm};\n"
    "shl.b32 x0, $1, 16; and.b32 x1, $1, 0xffff0000; shl.b32 x2, $2, 16; and.b32 x3, $2, 0xffff0000;\n"
    "mov.b64 X0, {x0, x1};\n"
    "mov.b64 X1, {x2, x3};\n"
    + _exp_pair(0)
    + _exp_pair(1)
    + "mov.f32 acc, $4;\n"
    "fma.rn.f32 acc, s0a, e0a, acc;\n"
    "fma.rn.f32 acc, s0b, e0b, acc;\n"
    "fma.rn.f32 acc, s1a, e1a, acc;\n"
    "fma.rn.f32 acc, s1b, e1b, acc;\n"
    "mov.f32 $0, acc; }"
)


@dsl_user_op
def sum4(w0, w1, negm, s, *, loc=None, ip=None):
    """Chain s through the 4 bf16 values in words w0 = {x0, x1}, w1 = {x2, x3} (in order x0..x3)."""
    return Float32(_asm1(T.f32(), [Uint32(w0), Uint32(w1), Float32(negm), Float32(s)], _SUM4, "=f,r,r,f,f", loc, ip))


@dsl_user_op
def sum1(x, negm, s, *, loc=None, ip=None):
    """Scalar step s = fma(2^j, ex2(f), s) for one f32 value x (torch's SASS sequence)."""
    return Float32(_asm1(
        T.f32(), [Float32(x), Float32(negm), Float32(s)],
        "{ .reg .f32 d, t, j, f, e, sc; .reg .b32 jb;\n"
        "add.rn.f32 d, $1, $2;\n"
        "fma.rn.sat.f32 t, d, 0f3BBB989D, 0f3F000000;\n"
        "fma.rm.f32 j, t, 0f437C0000, 0f4B400001;\n"
        "add.rn.f32 f, j, 0fCB40007F;\n"
        "neg.f32 f, f;\n"
        "fma.rn.f32 f, d, 0f3FB8AA3B, f;\n"
        "fma.rn.f32 f, d, 0f32A57060, f;\n"
        "mov.b32 jb, j; shl.b32 jb, jb, 23; mov.b32 sc, jb;\n"
        "ex2.approx.ftz.f32 e, f;\n"
        "fma.rn.f32 $0, sc, e, $3; }",
        "=f,f,f,f", loc, ip))


@dsl_user_op
def out4(w0, w1, negm, negl, *, loc=None, ip=None):
    """(x - max) - L for the 4 bf16 values in w0, w1 (f32x2 adds, same IEEE rn ops as torch's FADDs)."""
    r = _asmN(
        4, T.f32(), [Uint32(w0), Uint32(w1), Float32(negm), Float32(negl)],
        "{ .reg .b64 X0, X1, NM, NL; .reg .b32 x0, x1, x2, x3, a, b;\n"
        "mov.b32 a, $6; mov.b32 b, $7; mov.b64 NM, {a, a}; mov.b64 NL, {b, b};\n"
        "shl.b32 x0, $4, 16; and.b32 x1, $4, 0xffff0000; shl.b32 x2, $5, 16; and.b32 x3, $5, 0xffff0000;\n"
        "mov.b64 X0, {x0, x1}; mov.b64 X1, {x2, x3};\n"
        "add.rn.f32x2 X0, X0, NM; add.rn.f32x2 X1, X1, NM;\n"
        "add.rn.f32x2 X0, X0, NL; add.rn.f32x2 X1, X1, NL;\n"
        "mov.b64 {$0, $1}, X0; mov.b64 {$2, $3}, X1; }",
        "=f,=f,=f,=f,r,r,f,f", loc, ip)
    return [Float32(v) for v in r]


@dsl_user_op
def logf_torch(s, *, loc=None, ip=None):
    """CUDA's accurate logf exactly as in torch's cunn_SoftMaxForward SASS (sm_100)."""
    return Float32(_asm1(
        T.f32(), [Float32(s)],
        "{ .reg .pred p0, p2, p3; .reg .f32 s, r4, r5, r6, r7, e; .reg .b32 b, i4, i5;\n"
        "mov.f32 s, $1;\n"
        "setp.geu.f32 p0, s, 0f00800000;\n"
        "@!p0 mul.rn.f32 s, s, 0f4B000000;\n"
        "mov.b32 b, s;\n"
        "sub.s32 i4, b, 0x3f2aaaab;\n"
        "and.b32 i5, i4, 0xff800000;\n"
        "sub.s32 i4, b, i5;\n"
        "cvt.rn.f32.s32 r5, i5;\n"
        "mov.b32 r4, i4;\n"
        "add.rn.f32 r6, r4, 0fBF800000;\n"
        "selp.f32 e, 0f00000000, 0fC1B80000, p0;\n"
        "setp.gt.u32 p3, b, 0x7f7fffff;\n"
        "setp.neu.f32 p2, s, 0f00000000;\n"
        "fma.rn.f32 r7, r6, 0fBE055027, 0f3E1039F6;\n"
        "fma.rn.f32 e, r5, 0f34000000, e;\n"
        "fma.rn.f32 r7, r6, r7, 0fBDF8CDCC;\n"
        "fma.rn.f32 r7, r6, r7, 0f3E0F2955;\n"
        "fma.rn.f32 r7, r6, r7, 0fBE2AD8B9;\n"
        "fma.rn.f32 r7, r6, r7, 0f3E4CED0B;\n"
        "fma.rn.f32 r7, r6, r7, 0fBE7FFF22;\n"
        "fma.rn.f32 r7, r6, r7, 0f3EAAAA78;\n"
        "fma.rn.f32 r7, r6, r7, 0fBF000000;\n"
        "mul.rn.f32 r7, r6, r7;\n"
        "fma.rn.f32 r7, r6, r7, r6;\n"
        "fma.rn.f32 e, e, 0f3F317218, r7;\n"
        "@p3 fma.rn.f32 e, s, 0f7F800000, 0f7F800000;\n"
        "selp.f32 $0, e, 0fFF800000, p2; }",
        "=f,f", loc, ip))


@dsl_user_op
def nll_final(loss, cw, d64, *, loc=None, ip=None):
    """(total_weight, loss / f32(in2)) with div.rn.f32 and cvt.rn.f32.s64."""
    r = _asmN(2, T.f32(), [Float32(loss), Float32(cw), Int64(d64)],
              "{ .reg .f32 d; cvt.rn.f32.s64 d, $4; div.rn.f32 $1, $2, d; mov.f32 $0, $3; }",
              "=f,=f,f,f,l", loc, ip)
    return Float32(r[0]), Float32(r[1])


@dsl_user_op
def trap_if_bad_target(t, *, loc=None, ip=None):
    """torch's CHECK_INDEX_IN_CLASS: a target outside [0, V) other than -100 is a device-side assert."""
    _asm1(None, [Int64(t)],
          "{ .reg .pred p, q; setp.ne.s64 p, $0, -100; setp.ge.u64 q, $0, " + str(V) + "; and.pred p, p, q;\n"
          "@p trap; }", "l", loc, ip, side=True)


# ---------------------------------------------------------------- kernel 1: log-softmax
class LogSoftmaxFwd:
    @cute.jit
    def __call__(self, x, out, stream):
        self.kernel(x, out).launch(grid=[M * K, 1, 1], block=[TPC, 1, 1], cluster=[K, 1, 1], stream=stream,
                                   min_blocks_per_mp=MINB)

    @cute.kernel
    def kernel(self, x, out):
        tid, _, _ = cute.arch.thread_idx()
        bid, _, _ = cute.arch.block_idx()
        q = tid // 32
        lane = tid % 32
        c = cluster_rank()
        row = bid // K

        # @T0
        cute.arch.griddepcontrol_launch_dependents()
        smem = cutlass.utils.SmemAllocator()
        sAll = smem.allocate_tensor(cutlass.Uint8, cute.make_layout(SMEM1), byte_alignment=128)
        sb = sAll.iterator.toint()
        x0 = x.iterator.toint()
        o0 = out.iterator.toint()

        tail_on = c * TPC < TAIL  # this CTA owns tail elements (uniform per CTA)
        mine_tail = c * TPC + tid < TAIL
        if tid == 0:
            for g in cutlass.range_constexpr(len(GROUPS)):
                mbar_init(sb + SM_MBAR + 8 * g, 1)
            mbar_init(sb + SM_XMAX, 1)
            mbar_init(sb + SM_XSUM, 1)
            mbar_expect_tx(sb + SM_XMAX, Int32(32 * 4))
            mbar_expect_tx(sb + SM_XSUM, Int32(32 * 4))
            fence_mbar_init()
        cute.arch.barrier()
        cluster_arrive_relaxed()  # after tid 0's mbarrier init + release fence
        pol = policy_evict_first()
        src_row = x0 + Int64(row) * (V * 2)
        rw = cute.make_rmem_tensor(cute.make_layout(2 * NR), Uint32)
        src_t = src_row + Int64(c) * CHB + Int64(tid * 8)
        for i in cutlass.range_constexpr(NR):
            w = ldg_v2(src_t + 4096 * i * 2, pol)
            rw[2 * i] = w[0]
            rw[2 * i + 1] = w[1]
        if tid == 0:
            for g in cutlass.range_constexpr(len(GROUPS)):
                lo, hi = GROUPS[g]
                nb = Int32((hi - lo) * CHB)
                if g == len(GROUPS) - 1:
                    if tail_on:
                        nb = nb + TAILB
                mb = sb + SM_MBAR + 8 * g
                mbar_expect_tx(mb, nb)
                for i in cutlass.range_constexpr(lo, hi):
                    bulk_g2s(sb + (i - NR) * CHB, src_row + (4096 * i * 2) + Int64(c) * CHB, Int32(CHB), mb, pol)
                if g == len(GROUPS) - 1:
                    if tail_on:
                        bulk_g2s(sb + SM_TAIL, src_row + (NI * 4096 * 2) + Int64(c) * TAILB, Int32(TAILB), mb, pol)

        # ---- max (order-free)
        my = sb + tid * 8
        acc0 = Uint32(0xFF80FF80)
        acc1 = Uint32(0xFF80FF80)
        for i in cutlass.range_constexpr(NR):
            acc0 = bmax(acc0, rw[2 * i])
            acc1 = bmax(acc1, rw[2 * i + 1])
        for g in cutlass.range_constexpr(len(GROUPS)):
            lo, hi = GROUPS[g]
            mbar_wait0(sb + SM_MBAR + 8 * g)
            for i in cutlass.range_constexpr(lo, hi):
                w = lds_v2(my + (i - NR) * CHB)
                acc0 = bmax(acc0, w[0])
                acc1 = bmax(acc1, w[1])
        # @T1
        m = fmax(bmax_to_f32(acc0), bmax_to_f32(acc1))
        if mine_tail:
            m = fmax(m, lds_bf16_as_f32(sb + SM_TAIL + tid * 2))
        for off in (16, 8, 4, 2, 1):
            m = fmax(m, shfl_bfly(m, off))
        cluster_wait()  # every CTA of the cluster has initialized its mbarriers: st.async pushes are safe
        if lane < K:
            st_async_f32(sb + SM_SMAX + (c * NW + q) * 4, sb + SM_XMAX, lane, m)
        mbar_wait0(sb + SM_XMAX)
        # @T2
        mk = lds_f32(sb + SM_SMAX + lane * 4)
        for off in (16, 8, 4, 2, 1):
            mk = fmax(mk, shfl_bfly(mk, off))
        negm = fneg(mk)

        # ---- sum of exp: torch thread t = c * TPC + tid, exact chain order
        s = Float32(0.0)
        for i in cutlass.range_constexpr(NR):
            s = sum4(rw[2 * i], rw[2 * i + 1], negm, s)
        for i in cutlass.range_constexpr(NR, NI):
            w = lds_v2(my + (i - NR) * CHB)
            s = sum4(w[0], w[1], negm, s)
        if mine_tail:
            s = sum1(lds_bf16_as_f32(sb + SM_TAIL + tid * 2), negm, s)
        for off in (16, 8, 4, 2, 1):
            s = fadd(shfl_down(s, off), s)
        # @T3
        ws = shfl_idx0(s)
        if lane < K:
            st_async_f32(sb + SM_SSUM + (c * NW + q) * 4, sb + SM_XSUM, lane, ws)
        mbar_wait0(sb + SM_XSUM)  # also: no CTA writes this CTA's shared memory after this point
        # @T4
        v = lds_f32(sb + SM_SSUM + lane * 4)
        for off in (16, 8, 4, 2, 1):
            v = fadd(shfl_down(v, off), v)
        tot = shfl_idx0(v)
        negl = fneg(logf_torch(tot))

        # ---- write
        orow = o0 + Int64(row) * (V * 4) + Int64(c * TPC * 16 + tid * 16)
        for i in cutlass.range_constexpr(NR):
            y = out4(rw[2 * i], rw[2 * i + 1], negm, negl)
            st_v4(orow + i * (4096 * 4), y[0], y[1], y[2], y[3])
        for i in cutlass.range_constexpr(NR, NI):
            w = lds_v2(my + (i - NR) * CHB)
            y = out4(w[0], w[1], negm, negl)
            st_v4(orow + i * (4096 * 4), y[0], y[1], y[2], y[3])
        if mine_tail:
            xt = lds_bf16_as_f32(sb + SM_TAIL + tid * 2)
            st_f32(o0 + Int64(row) * (V * 4) + Int64(NI * 4096 + c * TPC + tid) * 4, fadd(fadd(xt, negm), negl))
        # @T5
        # @TSTORE


# ---------------------------------------------------------------- kernel 2: nll_loss_forward (sum) + div
class NllFwd:
    @cute.jit
    def __call__(self, out, tgt, div, tw, loss, stream):
        self.kernel(out, tgt, div, tw, loss).launch(grid=[1, 1, 1], block=[NLL_T, 1, 1], stream=stream, use_pdl=True)

    @cute.kernel
    def kernel(self, out, tgt, div, tw, loss):
        tid, _, _ = cute.arch.thread_idx()
        smem = cutlass.utils.SmemAllocator()
        sAll = smem.allocate_tensor(cutlass.Uint8, cute.make_layout(NLL_T * 8), byte_alignment=16)
        sb = sAll.iterator.toint()
        o0 = out.iterator.toint()
        t0 = tgt.iterator.toint()
        tg = cute.make_rmem_tensor(cute.make_layout(NLL_K), Int64)
        xv = cute.make_rmem_tensor(cute.make_layout(NLL_K), Float32)
        for k in cutlass.range_constexpr(NLL_K):
            tg[k] = ld_s64(t0 + Int64(tid + k * NLL_T) * 8)
        cute.arch.griddepcontrol_wait()
        for k in cutlass.range_constexpr(NLL_K):
            trap_if_bad_target(tg[k])
            xv[k] = Float32(0.0)
            if tg[k] != -100:
                xv[k] = ld_f32(o0 + (Int64(tid + k * NLL_T) * V + tg[k]) * 4)
        sh = Float32(0.0)
        cw = Float32(0.0)
        for k in cutlass.range_constexpr(NLL_K):
            if tg[k] != -100:
                sh = fadd(sh, fneg(xv[k]))
                cw = fadd(cw, Float32(1.0))
        sts_f32(sb + tid * 4, sh)
        sts_f32(sb + NLL_T * 4 + tid * 4, cw)
        cute.arch.barrier()
        for e in cutlass.range_constexpr(7, -1, -1):
            stride = 1 << e
            if tid < stride:
                sts_f32(sb + tid * 4, fadd(lds_f32(sb + tid * 4), lds_f32(sb + (tid + stride) * 4)))
                sts_f32(sb + NLL_T * 4 + tid * 4,
                        fadd(lds_f32(sb + NLL_T * 4 + tid * 4), lds_f32(sb + NLL_T * 4 + (tid + stride) * 4)))
            cute.arch.barrier()
        if tid == 0:
            d64 = ld_s64(div.iterator.toint())
            a, b = nll_final(lds_f32(sb), lds_f32(sb + NLL_T * 4), d64)
            st_f32(tw.iterator.toint(), a)
            st_f32(loss.iterator.toint(), b)


_DT = {torch.bfloat16: cutlass.BFloat16, torch.float32: cutlass.Float32, torch.int64: cutlass.Int64}


def _fake(dtype, shape, align=16):
    stride, s = [], 1
    for d in reversed(shape):
        stride.append(s)
        s *= d
    return cute.runtime.make_fake_tensor(_DT[dtype], tuple(shape), stride=tuple(reversed(stride)), assumed_align=align)


_k1 = cute.compile(
    LogSoftmaxFwd(),
    _fake(torch.bfloat16, (M, V), align=16),
    _fake(torch.float32, (M, V), align=16),
    cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
    options="--enable-tvm-ffi",
)
_k2 = cute.compile(
    NllFwd(),
    _fake(torch.float32, (M, V), align=16),
    _fake(torch.int64, (M,), align=8),
    _fake(torch.int64, (1,), align=8),
    _fake(torch.float32, (1,), align=4),
    _fake(torch.float32, (1,), align=4),
    cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
    options="--enable-tvm-ffi",
)


def _reference(in0, in1, in2):
    v3 = torch.ops.aten._to_copy.default(in0, dtype=torch.float32)
    v4 = torch.ops.aten._log_softmax.default(v3, 1, False)
    v5 = torch.ops.aten.nll_loss_forward.default(v4, in1, None, 2, -100)
    return (v4, v5[1], torch.ops.aten.div.Tensor(v5[0], in2))


def _ok(t, dtype, shape):
    return (t.dtype == dtype and tuple(t.shape) == shape and t.is_cuda
            and (t.is_contiguous() or t.numel() <= 1))


def run(in0, in1, in2):
    if not (_ok(in0, torch.bfloat16, (M, V)) and _ok(in1, torch.int64, (M,)) and _ok(in2, torch.int64, ())
            and in0.data_ptr() % 16 == 0 and in1.data_ptr() % 8 == 0 and in2.data_ptr() % 8 == 0
            and in0.device == in1.device == in2.device):
        return _reference(in0, in1, in2)
    out = torch.empty((M, V), dtype=torch.float32, device=in0.device)
    tw = torch.empty((), dtype=torch.float32, device=in0.device)
    loss = torch.empty((), dtype=torch.float32, device=in0.device)
    _k1(in0, out)
    _k2(out, in1, in2.view(1), tw.view(1), loss.view(1))
    return (out, tw, loss)
