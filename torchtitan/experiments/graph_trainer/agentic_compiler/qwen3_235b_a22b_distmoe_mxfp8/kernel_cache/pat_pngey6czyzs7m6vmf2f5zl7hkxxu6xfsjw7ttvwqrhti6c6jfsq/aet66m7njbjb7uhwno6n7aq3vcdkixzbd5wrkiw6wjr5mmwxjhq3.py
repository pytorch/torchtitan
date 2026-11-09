"""Qwen3 235B-A22B residual add + ffn_norm (pat_pnge / aet66m, fusion region G45), one CuTe DSL kernel.

reference: h = in0 + in1 (bf16 [8192, 4096]); (y, rstd) = aten._fused_rms_norm(h, [4096], w = in2, eps=1e-6)
           -> returns (h, y, rstd [8192, 1]).
Adapted from the Qwen3 30B-A3B G45 kernel (ada t7_final, N=2048); the row split and reduction tree re-derived for
N=4096.

RMSNorm reproduces the kernel aten._fused_rms_norm dispatches to on this build (torch._native override ->
vendored quack RMSNorm, N=4096 bf16: RmsNormFwdConfig = 128 threads/CTA, 64 threads per row (2 warps, 2 rows per
CTA), vecsize 8, cluster 1, no reload; decoded from its SASS): thread t = 32 p + c of a row (quack warp p in
{0, 1}, lane c) owns columns 8 t + 512 k + j (k = 0..7, j = 0..7); its sum of squares is a sequential
fma.rn.f32.bf16 chain (FHFMA.BF16) in (k, j) order from +0; warp butterfly FADD 16, 8, 4, 2, 1 -> W_p; block
reduce (lanes < 2 load W_p, others +0, butterfly again) = W0 + W1; rstd = rsqrt.approx.ftz(fma.rn(sum, 2^-12,
fl32(1e-6))); y = bf16_rn((x *rn rstd) *rn w).

Here one CTA per row with 2 * SEG warps: warp (q, p) = tid // 64, (tid // 32) % 2 handles chain p's vectors
k in [q * KPS, (q + 1) * KPS) (KPS = 8 / SEG); the per-lane chain is continued segment to segment through smem
(warp (q, p) starts from warp (q - 1, p)'s partial), the last segment's warps do the butterflies.

Numerics (all explicit PTX, no contraction):
* add: h = add.rn.bf16x2(a, b), bitwise equal to torch's bf16 add (fp32 add + one RNE to bf16) for all 2^32
  bf16 pairs (exhaustively checked on GB300 for DSv3 pat_dre3).
"""

import cutlass
import cutlass.cute as cute
import torch
from cutlass import Float32, Int32, Int64, Uint32
from cutlass._mlir import ir
from cutlass._mlir.dialects import llvm
from cutlass.cutlass_dsl import T, dsl_user_op

M = 8192
N = 4096
SEG = 8  # chain segments per row (warps per row = 2 * SEG; 8 / 4 / 2 / 1 measured 40.6 / 41.0 / 41.1 / 41.7 us)
KPS = 8 // SEG  # quack vectors k per thread per segment
NTHR = 64 * SEG
EPS_HEX = "0f358637BD"  # fl32(1e-6)
INVN_HEX = "0f39800000"  # 2^-12 = 1/4096


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


# ---------------------------------------------------------------- memory
@dsl_user_op
def ld_stream_v4(addr, *, loc=None, ip=None):
    r = _asmN(4, T.i32(), [Int64(addr)], "ld.global.v4.b32 {$0, $1, $2, $3}, [$4];",
              "=r,=r,=r,=r,l", loc, ip, side=True)
    return [Uint32(v) for v in r]


@dsl_user_op
def ld_cached_v4(addr, *, loc=None, ip=None):
    r = _asmN(4, T.i32(), [Int64(addr)], "ld.global.nc.v4.b32 {$0, $1, $2, $3}, [$4];",
              "=r,=r,=r,=r,l", loc, ip, side=True)
    return [Uint32(v) for v in r]


@dsl_user_op
def st_v4(addr, a, b, c, d, *, loc=None, ip=None):
    _asm1(None, [Int64(addr), Uint32(a), Uint32(b), Uint32(c), Uint32(d)],
          "st.global.v4.b32 [$0], {$1, $2, $3, $4};", "l,r,r,r,r", loc, ip, side=True)


@dsl_user_op
def sts_f32(addr, a, *, loc=None, ip=None):
    _asm1(None, [Int32(addr), Float32(a)], "st.shared.f32 [$0], $1;", "r,f", loc, ip, side=True)


@dsl_user_op
def lds_f32(addr, *, loc=None, ip=None):
    return Float32(_asm1(T.f32(), [Int32(addr)], "ld.shared.f32 $0, [$1];", "=f,r", loc, ip, side=True))


@dsl_user_op
def st_f32(addr, a, *, loc=None, ip=None):
    _asm1(None, [Int64(addr), Float32(a)], "st.global.f32 [$0], $1;", "l,f", loc, ip, side=True)


# ---------------------------------------------------------------- math
@dsl_user_op
def add_bf16x2(a, b, *, loc=None, ip=None):
    """bf16x2 word a + b: one rounding of the exact sum (== torch's bf16 add for all pairs)."""
    return Uint32(_asm1(T.i32(), [Uint32(a), Uint32(b)], "add.rn.bf16x2 $0, $1, $2;", "=r,r,r", loc, ip))


@dsl_user_op
def sumsq8(acc, w0, w1, w2, w3, *, loc=None, ip=None):
    """acc = fma(x_j, x_j, acc) for the 8 bf16 of words w0..w3 in order (lo half first)."""
    return Float32(_asm1(
        T.f32(), [Float32(acc), Uint32(w0), Uint32(w1), Uint32(w2), Uint32(w3)],
        "{ .reg .f32 a; .reg .b16 l, h;\n"
        "mov.f32 a, $1;\n"
        "mov.b32 {l, h}, $2; fma.rn.f32.bf16 a, l, l, a; fma.rn.f32.bf16 a, h, h, a;\n"
        "mov.b32 {l, h}, $3; fma.rn.f32.bf16 a, l, l, a; fma.rn.f32.bf16 a, h, h, a;\n"
        "mov.b32 {l, h}, $4; fma.rn.f32.bf16 a, l, l, a; fma.rn.f32.bf16 a, h, h, a;\n"
        "mov.b32 {l, h}, $5; fma.rn.f32.bf16 a, l, l, a; fma.rn.f32.bf16 a, h, h, a;\n"
        "mov.f32 $0, a; }",
        "=f,f,r,r,r,r", loc, ip))


@dsl_user_op
def bfly_add(v, off: int, *, loc=None, ip=None):
    return Float32(_asm1(
        T.f32(), [Float32(v)],
        "{ .reg .f32 o;\n"
        f"shfl.sync.bfly.b32 o, $1, {off}, 31, -1;\n"
        "add.rn.f32 $0, $1, o; }",
        "=f,f", loc, ip, side=True))


@dsl_user_op
def fadd(a, b, *, loc=None, ip=None):
    return Float32(_asm1(T.f32(), [Float32(a), Float32(b)], "add.rn.f32 $0, $1, $2;", "=f,f,f", loc, ip))


@dsl_user_op
def rstd_of(total, *, loc=None, ip=None):
    """rsqrt.approx.ftz(fma.rn(total, 2^-12, fl32(1e-6)))."""
    return Float32(_asm1(
        T.f32(), [Float32(total)],
        "{ .reg .f32 m;\n"
        f"fma.rn.f32 m, $1, {INVN_HEX}, {EPS_HEX};\n"
        "rsqrt.approx.ftz.f32 $0, m; }",
        "=f,f", loc, ip))


@dsl_user_op
def norm_bf16x2(x, w, r, *, loc=None, ip=None):
    """bf16x2 word: bf16_rn((x *rn r) *rn w) per half."""
    return Uint32(_asm1(
        T.i32(), [Uint32(x), Uint32(w), Float32(r)],
        "{ .reg .b32 xl, xh, wl, wh; .reg .b64 xp, wp, rp;\n"
        "shl.b32 xl, $1, 16; and.b32 xh, $1, 0xFFFF0000;\n"
        "shl.b32 wl, $2, 16; and.b32 wh, $2, 0xFFFF0000;\n"
        "mov.b64 xp, {xl, xh}; mov.b64 wp, {wl, wh}; mov.b64 rp, {$3, $3};\n"
        "mul.rn.f32x2 xp, xp, rp;\n"
        "mul.rn.f32x2 xp, xp, wp;\n"
        "mov.b64 {xl, xh}, xp;\n"
        "cvt.rn.bf16x2.f32 $0, xh, xl; }",
        "=r,r,r,f", loc, ip))


class AddNorm:
    @cute.jit
    def __call__(self, a, b, w, h, y, rstd, stream):
        self.kernel(a, b, w, h, y, rstd).launch(grid=[M, 1, 1], block=[NTHR, 1, 1], stream=stream)

    @cute.kernel
    def kernel(self, a, b, w, h, y, rstd):
        tid, _, _ = cute.arch.thread_idx()
        row, _, _ = cute.arch.block_idx()
        c = tid % 32
        p = (tid // 32) % 2
        q = tid // 64
        row64 = Int64(row)

        smem = cutlass.utils.SmemAllocator()
        sP = smem.allocate_tensor(cutlass.Float32, cute.make_layout(80), byte_alignment=16)
        sPb = sP.iterator.toint()

        # lane c of quack warp p owns columns 8 (32 p + c) + 512 k + j; segment q holds k = q*KPS + kk
        eoff2 = (row64 * N + Int64(c * 8 + p * 256 + q * KPS * 512)) * 2
        aa = a.iterator.toint() + eoff2
        ba = b.iterator.toint() + eoff2
        av = [ld_stream_v4(aa + kk * 1024) for kk in range(KPS)]
        bv = [ld_stream_v4(ba + kk * 1024) for kk in range(KPS)]
        wa = w.iterator.toint() + Int64(c * 16 + p * 512 + q * KPS * 1024)
        wv = [ld_cached_v4(wa + kk * 1024) for kk in range(KPS)]
        hv = [[add_bf16x2(av[kk][i], bv[kk][i]) for i in range(4)] for kk in range(KPS)]
        ha = h.iterator.toint() + eoff2
        for kk in cutlass.range_constexpr(KPS):
            st_v4(ha + kk * 1024, hv[kk][0], hv[kk][1], hv[kk][2], hv[kk][3])

        acc = Float32(0.0)
        for seg in cutlass.range_constexpr(SEG):
            if q == seg:
                if cutlass.const_expr(seg > 0):
                    acc = lds_f32(sPb + (p * 32 + c) * 4)
                for kk in cutlass.range_constexpr(KPS):
                    acc = sumsq8(acc, hv[kk][0], hv[kk][1], hv[kk][2], hv[kk][3])
                if cutlass.const_expr(seg < SEG - 1):
                    sts_f32(sPb + (p * 32 + c) * 4, acc)
                else:
                    for off in (16, 8, 4, 2, 1):
                        acc = bfly_add(acc, off)
                    if c == 0:
                        sts_f32(sPb + (64 + p) * 4, acc)
            cute.arch.barrier()
        # quack block reduce of the two warp sums: fl(W0 + W1)
        rs = rstd_of(fadd(lds_f32(sPb + 64 * 4), lds_f32(sPb + 65 * 4)))
        if tid == 0:
            st_f32(rstd.iterator.toint() + row64 * 4, rs)
        ya = y.iterator.toint() + eoff2
        for kk in cutlass.range_constexpr(KPS):
            yv = [norm_bf16x2(hv[kk][i], wv[kk][i], rs) for i in range(4)]
            st_v4(ya + kk * 1024, yv[0], yv[1], yv[2], yv[3])


_DT = {torch.bfloat16: cutlass.BFloat16, torch.float32: cutlass.Float32}


def _fake(dtype, shape):
    stride, acc = [], 1
    for s in reversed(shape):
        stride.append(acc)
        acc *= s
    return cute.runtime.make_fake_tensor(_DT[dtype], tuple(shape), stride=tuple(reversed(stride)), assumed_align=16)


_fn = cute.compile(
    AddNorm(),
    _fake(torch.bfloat16, (M, N)),
    _fake(torch.bfloat16, (M, N)),
    _fake(torch.bfloat16, (N,)),
    _fake(torch.bfloat16, (M, N)),
    _fake(torch.bfloat16, (M, N)),
    _fake(torch.float32, (M,)),
    cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
    options="--enable-tvm-ffi",
)


def _check(t, shape):
    if (t.dtype != torch.bfloat16 or tuple(t.shape) != shape or not t.is_contiguous() or not t.is_cuda
            or t.data_ptr() % 16):
        raise ValueError(f"expected a contiguous 16-byte aligned cuda bf16 {shape} tensor, got {t.dtype} "
                         f"{tuple(t.shape)} stride {t.stride()}")


def run(in0, in1, in2):
    _check(in0, (M, N))
    _check(in1, (M, N))
    _check(in2, (N,))
    a, b, w = in0, in1, in2
    dev = a.device
    h = torch.empty((M, N), dtype=torch.bfloat16, device=dev)
    y = torch.empty((M, N), dtype=torch.bfloat16, device=dev)
    rstd = torch.empty((M, 1), dtype=torch.float32, device=dev)
    _fn(a, b, w, h, y, rstd.view(M))
    return (h, y, rstd)
