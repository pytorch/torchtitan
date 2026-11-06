"""Qwen3 30B-A3B residual add + ffn_norm (pat_na2k4 / a2aqmc, fusion region G45), one CuTe DSL kernel.

reference: h = in0 + in1 (bf16 [32768, 2048]); (y, rstd) = aten._fused_rms_norm(h, [2048], w = in2, eps=1e-6)
           -> returns (h, y, rstd [32768, 1]).
One CTA of WPR = 8 warps per row: lane c of warp q owns columns 8c + 256q + j (j = 0..7), i.e. vector k = q of
quack's lane c. quack's per-lane chain over k = 0..7 is continued warp to warp through smem (warp q starts from
warp q-1's partial sum), and warp 7 does the butterfly. Splitting the row over 8 warps runs at the add+norm
streaming rate (6.98 TB/s, same as a plain Triton add+norm), one warp per row reached only 6.36 TB/s.

Numerics (all explicit PTX, no contraction):
* add: h = add.rn.bf16x2(a, b), bitwise equal to torch's bf16 add (fp32 add + one RNE to bf16) for all 2^32
  bf16 pairs (exhaustively checked on GB300 for DSv3 pat_dre3).
* RMSNorm reproduces the kernel aten._fused_rms_norm dispatches to on this build (torch._native override ->
  vendored quack RMSNorm, N=2048 bf16: 32 threads per row, vecsize 8, cluster 1; decoded from its SASS):
  per-lane sum of squares is a sequential fma.rn.f32.bf16 chain (FHFMA.BF16) in (k, j) order from +0;
  warp butterfly FADD with offsets 16, 8, 4, 2, 1; rstd = rsqrt.approx.ftz(fma.rn(sum, 2^-11, fl32(1e-6)));
  y = bf16_rn((x *rn rstd) *rn w).
"""

import cutlass
import cutlass.cute as cute
import torch
from cutlass import Float32, Int32, Int64, Uint32
from cutlass._mlir import ir
from cutlass._mlir.dialects import llvm
from cutlass.cutlass_dsl import T, dsl_user_op

M = 32768
N = 2048
WPR = 8  # warps per row (measured best of 1, 2, 4, 8); warp q owns k in [q*8/WPR, (q+1)*8/WPR)
KPW = 8 // WPR
NTHR = 32 * WPR
EPS_HEX = "0f358637BD"  # fl32(1e-6)
INVN_HEX = "0f3A000000"  # 2^-11 = 1/2048


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
def rstd_of(total, *, loc=None, ip=None):
    """rsqrt.approx.ftz(fma.rn(total, 2^-11, fl32(1e-6)))."""
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
        q = tid // 32
        row64 = Int64(row)

        smem = cutlass.utils.SmemAllocator()
        sP = smem.allocate_tensor(cutlass.Float32, cute.make_layout(64), byte_alignment=16)
        sPb = sP.iterator.toint()

        # lane c of warp q owns columns 8c + 256k + j for k = q*KPW + kk: the quack per-lane chain of 64
        # elements is split into WPR consecutive segments, continued warp to warp through smem
        eoff2 = (row64 * N + Int64(c * 8 + q * KPW * 256)) * 2
        aa = a.iterator.toint() + eoff2
        ba = b.iterator.toint() + eoff2
        av = [ld_stream_v4(aa + kk * 512) for kk in range(KPW)]
        bv = [ld_stream_v4(ba + kk * 512) for kk in range(KPW)]
        wa = w.iterator.toint() + Int64(c * 16 + q * KPW * 512)
        wv = [ld_cached_v4(wa + kk * 512) for kk in range(KPW)]
        hv = [[add_bf16x2(av[kk][i], bv[kk][i]) for i in range(4)] for kk in range(KPW)]
        ha = h.iterator.toint() + eoff2
        for kk in cutlass.range_constexpr(KPW):
            st_v4(ha + kk * 512, hv[kk][0], hv[kk][1], hv[kk][2], hv[kk][3])

        acc = Float32(0.0)
        for seg in cutlass.range_constexpr(WPR):
            if q == seg:
                if cutlass.const_expr(seg > 0):
                    acc = lds_f32(sPb + c * 4)
                for kk in cutlass.range_constexpr(KPW):
                    acc = sumsq8(acc, hv[kk][0], hv[kk][1], hv[kk][2], hv[kk][3])
                if cutlass.const_expr(seg < WPR - 1):
                    sts_f32(sPb + c * 4, acc)
                else:
                    for off in (16, 8, 4, 2, 1):
                        acc = bfly_add(acc, off)
                    rs0 = rstd_of(acc)
                    if c == 0:
                        sts_f32(sPb + 128, rs0)
                        st_f32(rstd.iterator.toint() + row64 * 4, rs0)
            cute.arch.barrier()
        rs = lds_f32(sPb + 128)
        ya = y.iterator.toint() + eoff2
        for kk in cutlass.range_constexpr(KPW):
            yv = [norm_bf16x2(hv[kk][i], wv[kk][i], rs) for i in range(4)]
            st_v4(ya + kk * 512, yv[0], yv[1], yv[2], yv[3])


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


def _dense(t):
    """The kernel uses 16-byte vector accesses on contiguous rows."""
    return t if t.is_contiguous() and t.data_ptr() % 16 == 0 else t.clone(memory_format=torch.contiguous_format)


def run(in0, in1, in2):
    a, b, w = _dense(in0), _dense(in1), _dense(in2)
    dev = a.device
    h = torch.empty((M, N), dtype=torch.bfloat16, device=dev)
    y = torch.empty((M, N), dtype=torch.bfloat16, device=dev)
    rstd = torch.empty((M, 1), dtype=torch.float32, device=dev)
    _fn(a, b, w, h, y, rstd.view(M))
    return (h, y, rstd)
