"""wqkv dgrad GEMM (unchanged) + attention_norm RMSNorm backward + residual-grad add (Qwen3 235B-A22B, G15).

Reference (pat_4j6e): d = aten._scaled_mm_v2(in0, in1, [in2], [3], [1], [in3], [3], [1], None, bf16) (MXFP8 GEMM,
[8192, 9216] x [9216, 4096]); (dx, dw) = aten._fused_rms_norm_backward(d, x=in5, [4096], rstd=in4, w=in6,
[True, True]); g = bf16(in7 + dx). run() returns (dw, g).
Adapted from the Qwen3 30B-A3B G15 kernel (dex t4, N=2048); shapes, grid and the dx formula re-derived for N=4096.

The GEMM is the identical aten call (same op, arguments and operands, so the same cuBLAS kernel and bits); only the
part after it is replaced. On this build (sm_103) aten._fused_rms_norm_backward runs the vendored quack CuTe DSL
kernel: N=4096 bf16 Blackwell config = 128 threads/row, cluster 1, cp.async, 2 stages, reload_wdy="smem";
grid B = min(M, get_sm_count(4096)) = 2 * num_SMs persistent CTAs (CTA b handles rows b, b + B, ...), then
reduce_rows(partial[B, N], dw, cols=8, threads=256). Per element (read off the N=4096 quack SASS; same derivation
as the G19 kernel):
    x_hat = rn(x * rstd);  w' = bf16(w + 0);  wdy = rn(d * w') (exact)
    S_thread = sequential fma(x_hat_i, wdy_i, acc) from +0 over the thread's 32 elements (i = j + 8 k for
               cols 8 t + 1024 k + j, k = 0..3); S = warp butterfly (16, 8, 4, 2, 1), then ((W0 + W2) + (W1 + W3))
               after +0
    mean = S * 2^-12;  dx = bf16(rn(fma(d, w', -rn(x_hat * mean)) * rstd));  partial[b] = fma(x_hat, d, acc)
    dw[c] = bf16(((W0 + W1) + ... + W7)), W_w = (A_4w + A_4w+1) + (A_4w+2 + A_4w+3), A_m = 0 + p[m] + p[m + 32] + ...
g = bf16(in7 + dx) via add.rn.bf16x2 (== fp32 add + RNE).

Kernels after the GEMM:
  K1 (B CTAs x 128 threads, quack's row schedule): reads d, x, in7 once (per-thread cp.async ring), writes g once,
     fp32 dw partials (the quack/aten intermediate dx is never materialized). Triggers programmatic dependent launch
     at its start.
  K2 (N/8 CTAs x 256 threads, PDL secondary of K1): griddepcontrol.wait, then exactly reduce_rows(8, 256).
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
K = 9216
NCH = 4  # 1024-column chunks per row (thread t owns cols 8 t + 1024 k + j)
ROWB = N * 2  # 8192 bytes per bf16 row
NSTG = 3  # cp.async ring stages (2 rows in flight; 2 -> 93 us, 3 -> 89 us, 4 -> 91 us for the tail alone)
STAGE = 3 * ROWB  # x, d, in7 rows
OFF_RED = NSTG * STAGE
SMEM1 = OFF_RED + 64


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
def ld_f32_plain(addr, *, loc=None, ip=None):
    return Float32(_asm1(T.f32(), [Int64(addr)], "ld.global.f32 $0, [$1];", "=f,l", loc, ip, side=True))


@dsl_user_op
def ld_v4(addr, *, loc=None, ip=None):
    r = _asmN(4, T.i32(), [Int64(addr)], "ld.global.nc.v4.b32 {$0, $1, $2, $3}, [$4];", "=r,=r,=r,=r,l", loc, ip,
              side=True)
    return [Uint32(v) for v in r]


@dsl_user_op
def st_b16(addr, a, *, loc=None, ip=None):
    _asm1(None, [Int64(addr), Uint32(a)], "{ .reg .b16 h; cvt.u16.u32 h, $1; st.global.b16 [$0], h; }", "l,r", loc,
          ip, side=True)


@dsl_user_op
def st_v4(addr, a, b, c, d, *, loc=None, ip=None):
    _asm1(None, [Int64(addr), Uint32(a), Uint32(b), Uint32(c), Uint32(d)],
          "st.global.v4.b32 [$0], {$1, $2, $3, $4};", "l,r,r,r,r", loc, ip, side=True)


@dsl_user_op
def st_f32x4(addr, a, b, c, d, *, loc=None, ip=None):
    _asm1(None, [Int64(addr), Float32(a), Float32(b), Float32(c), Float32(d)],
          "st.global.v4.f32 [$0], {$1, $2, $3, $4};", "l,f,f,f,f", loc, ip, side=True)


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
def cp16h(dst, src, pol, *, loc=None, ip=None):
    _asm1(None, [Int32(dst), Int64(src), Int64(pol)], "cp.async.cg.shared.global.L2::cache_hint [$0], [$1], 16, $2;",
          "r,l,l", loc, ip, side=True)


@dsl_user_op
def cp_commit(*, loc=None, ip=None):
    _asm1(None, [], "cp.async.commit_group;", "", loc, ip, side=True)


@dsl_user_op
def cp_wait(n: int, *, loc=None, ip=None):
    _asm1(None, [], f"cp.async.wait_group {n};", "", loc, ip, side=True)


# ---------------------------------------------------------------- math primitives
@dsl_user_op
def fadd(a, b, *, loc=None, ip=None):
    return Float32(_asm1(T.f32(), [Float32(a), Float32(b)], "add.rn.f32 $0, $1, $2;", "=f,f,f", loc, ip))


@dsl_user_op
def add0(a, *, loc=None, ip=None):
    """a + (+0.0) (turns -0 into +0)."""
    return Float32(_asm1(T.f32(), [Float32(a)], "add.rn.f32 $0, $1, 0f00000000;", "=f,f", loc, ip))


@dsl_user_op
def mul_inv4096(a, *, loc=None, ip=None):
    """a * 2^-12 (quack N=4096: FMUL by 0.000244140625 == div.rn by 4096)."""
    return Float32(_asm1(T.f32(), [Float32(a)], "mul.rn.f32 $0, $1, 0f39800000;", "=f,f", loc, ip))


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

    quack N=4096 SASS (reload_wdy="smem"): p = rn(x_hat * mean) (FMUL2), dx = bf16(rn(fma(d, w', -p) * rstd))
    (FFMA2 + FMUL2); here -p = rn(x_hat * -mean) (nm = -mean; RN is sign-symmetric); g = bf16(in7 + dx);
    acc = fma(x_hat, d, acc)."""
    r = _asmN(
        2, [T.i32(), T.i64()], [Int64(xh), Uint32(dw), Uint32(ww), Uint32(iw), Float32(rs), Float32(nm), Int64(acc)],
        "{ .reg .b32 a, b, c, e; .reg .b64 D, R, MM, W, NP, TT, DX;\n"
        "mov.b32 c, $6; mov.b64 R, {c, c};\n"
        "mov.b32 e, $7; mov.b64 MM, {e, e};\n"
        "mul.rn.f32x2 NP, $2, MM;\n"
        "shl.b32 a, $3, 16; and.b32 b, $3, 0xFFFF0000; mov.b64 D, {a, b};\n"
        "shl.b32 a, $4, 16; and.b32 b, $4, 0xFFFF0000; mov.b64 W, {a, b};\n"
        "fma.rn.f32x2 TT, D, W, NP;\n"
        "mul.rn.f32x2 DX, TT, R;\n"
        "mov.b64 {a, b}, DX;\n"
        "cvt.rn.bf16x2.f32 c, b, a;\n"
        "add.rn.bf16x2 $0, $5, c;\n"
        "fma.rn.f32x2 $1, $2, D, $8; }",
        "=r,=l,l,r,r,r,f,f,l", loc, ip,
    )
    return Uint32(r[0]), Int64(r[1])


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


# ---------------------------------------------------------------- K1: norm backward + residual add
class NormBwdAdd:
    def __init__(self, nblk: int):
        self.nblk = nblk

    @cute.jit
    def __call__(self, rstd, dd, x, w, res, g, part, stream):
        self.kernel(rstd, dd, x, w, res, g, part).launch(grid=[self.nblk, 1, 1], block=[128, 1, 1], stream=stream)

    @cute.kernel
    def kernel(self, rstd, dd, x, w, res, g, part):
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
        d0 = dd.iterator.toint()
        x0 = x.iterator.toint()
        w0 = w.iterator.toint()
        res0 = res.iterator.toint()
        g0 = g.iterator.toint()
        part0 = part.iterator.toint()

        cnt = (M - 1 - b) // B + 1
        toff = Int64(tid * 16)
        pol_ef = policy_evict_first()

        wv = cute.make_rmem_tensor(cute.make_layout(NCH * 4), Uint32)
        for kk in cutlass.range_constexpr(NCH):
            v = ld_v4(w0 + toff + kk * 2048)
            for q in cutlass.range_constexpr(4):
                wv[kk * 4 + q] = bf16x2_add0(v[q])
        xh = cute.make_rmem_tensor(cute.make_layout(NCH * 4), Int64)
        acc = cute.make_rmem_tensor(cute.make_layout(NCH * 4), Int64)
        for i in cutlass.range_constexpr(NCH * 4):
            acc[i] = zero64()

        def issue(r, s, sb, tid, toff, x0, d0, res0, pol_ef):
            src = Int64(r) * ROWB + toff
            so = sb + s * STAGE + tid * 16
            for kk in cutlass.range_constexpr(NCH):
                cp16h(so + kk * 2048, x0 + src + kk * 2048, pol_ef)
            for kk in cutlass.range_constexpr(NCH):
                cp16h(so + ROWB + kk * 2048, d0 + src + kk * 2048, pol_ef)
            for kk in cutlass.range_constexpr(NCH):
                cp16h(so + 2 * ROWB + kk * 2048, res0 + src + kk * 2048, pol_ef)

        for s0 in cutlass.range_constexpr(NSTG - 1):
            if s0 < cnt:
                issue(b + s0 * B, s0, sb, tid, toff, x0, d0, res0, pol_ef)
            cp_commit()
        rsb = cute.make_rmem_tensor(cute.make_layout(1), Float32)
        rsb[0] = ld_f32(rstd0 + Int64(b) * 4)

        for k in cutlass.range(cnt, unroll=1):
            r = b + k * B
            s = k % NSTG
            so = sb + s * STAGE + tid * 16
            # keep NSTG - 1 rows in flight: issue row k + NSTG - 1 into the stage freed at the end of row k - 1
            if k + NSTG - 1 < cnt:
                issue(r + (NSTG - 1) * B, (k + NSTG - 1) % NSTG, sb, tid, toff, x0, d0, res0, pol_ef)
            cp_commit()
            rs = rsb[0]
            if k + 1 < cnt:
                rsb[0] = ld_f32(rstd0 + Int64(r + B) * 4)
            cp_wait(NSTG - 1)

            # ---- phase A: per-thread sequential fma chain over 16 elements ----
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
                sts_f32(sb + OFF_RED + (k % 2 * 4 + wid) * 4, sacc)
            cute.arch.barrier()
            wsum = lds_f32x4(sb + OFF_RED + k % 2 * 16)
            tot = fadd(fadd(add0(wsum[0]), add0(wsum[2])), fadd(add0(wsum[1]), add0(wsum[3])))
            nm = -mul_inv4096(tot)

            # ---- phase C: dx, g = in7 + dx, dw partial ----
            r64 = Int64(r)
            for kk in cutlass.range_constexpr(NCH):
                dv = lds_v4(so + ROWB + kk * 2048)
                iv = lds_v4(so + 2 * ROWB + kk * 2048)
                gw = []
                for q in cutlass.range_constexpr(4):
                    gq, aq = phase_ch(xh[kk * 4 + q], dv[q], wv[kk * 4 + q], iv[q], rs, nm, acc[kk * 4 + q])
                    acc[kk * 4 + q] = aq
                    gw.append(gq)
                st_v4(g0 + r64 * ROWB + toff + kk * 2048, gw[0], gw[1], gw[2], gw[3])

        # ---- dw partial row b ----
        pb = part0 + Int64(b) * (N * 4) + Int64(tid * 32)
        for kk in cutlass.range_constexpr(NCH):
            st_f32x4(pb + kk * 4096, pair_lo(acc[kk * 4]), pair_hi(acc[kk * 4]),
                     pair_lo(acc[kk * 4 + 1]), pair_hi(acc[kk * 4 + 1]))
            st_f32x4(pb + kk * 4096 + 16, pair_lo(acc[kk * 4 + 2]), pair_hi(acc[kk * 4 + 2]),
                     pair_lo(acc[kk * 4 + 3]), pair_hi(acc[kk * 4 + 3]))


# ---------------------------------------------------------------- K2: reduce_rows(partial, dw, cols=8, threads=256)
class ReduceRows:
    def __init__(self, nblk: int):
        self.nblk = nblk

    @cute.jit
    def __call__(self, part, dw, stream):
        self.kernel(part, dw).launch(grid=[N // 8, 1, 1], block=[256, 1, 1], stream=stream, use_pdl=True)

    @cute.kernel
    def kernel(self, part, dw):
        """torch/_native/cutedsl/reduce.py reduce_rows with cols=8, threads=256, rows = B (PDL secondary of K1)."""
        B = self.nblk
        tid, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        lane = tid % 32
        warp = tid // 32
        smem = cutlass.utils.SmemAllocator()
        sAll = smem.allocate_tensor(cutlass.Uint8, cute.make_layout(256), byte_alignment=16)
        sb = sAll.iterator.toint()
        cute.arch.griddepcontrol_wait()
        part0 = part.iterator.toint()
        col = bidx * 8 + tid % 8
        m = tid // 8
        a = Float32(0.0)
        for i in cutlass.range_constexpr((B + 31) // 32):
            if m + 32 * i < B:
                a = fadd(a, ld_f32_plain(part0 + (Int64(m + 32 * i) * N + Int64(col)) * 4))
        a = fadd(a, cute.arch.shuffle_sync_bfly(a, offset=8))
        a = fadd(a, cute.arch.shuffle_sync_bfly(a, offset=16))
        if lane < 8:
            sts_f32(sb + (warp * 8 + lane) * 4, a)
        cute.arch.barrier()
        if tid < 8:
            for i in cutlass.range_constexpr(1, 8):
                a = fadd(a, lds_f32(sb + (i * 8 + tid) * 4))
            st_b16(dw.iterator.toint() + Int64(bidx * 8 + tid) * 2, bf16_bits(a))


_DT = {torch.bfloat16: cutlass.BFloat16, torch.float32: cutlass.Float32}


def _fake(dtype, shape, stride=None, align=16):
    if stride is None:
        stride, a = [], 1
        for s in reversed(shape):
            stride.append(a)
            a *= s
        stride = tuple(reversed(stride))
    return cute.runtime.make_fake_tensor(_DT[dtype], tuple(shape), stride=tuple(stride), assumed_align=align)


def _num_blocks(device):
    # quack RMSNormBackward grid on sm_103 (not in the AOT launch table): min(M, get_sm_count(4096, device)),
    # get_sm_count = sm_count * 2 for N > 2048 on SM100+
    props = torch.cuda.get_device_properties(device)
    if (props.major, props.minor) != (10, 3):
        raise ValueError("expects sm_103: the quack launch schedule and SASS this kernel reproduces were read there")
    return min(M, props.multi_processor_count * 2)


_B = _num_blocks(torch.cuda.current_device())
if _B <= 32:
    raise ValueError("reduce_rows(32, 128) path not implemented")

_k1 = cute.compile(
    NormBwdAdd(_B),
    _fake(torch.float32, (M, 1), (1, 1), align=4),
    _fake(torch.bfloat16, (M, N)),
    _fake(torch.bfloat16, (M, N)),
    _fake(torch.bfloat16, (N,)),
    _fake(torch.bfloat16, (M, N)),
    _fake(torch.bfloat16, (M, N)),
    _fake(torch.float32, (_B, N)),
    cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
    options="--enable-tvm-ffi",
)
_k2 = cute.compile(
    ReduceRows(_B),
    _fake(torch.float32, (_B, N)),
    _fake(torch.bfloat16, (N,)),
    cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
    options="--enable-tvm-ffi",
)


def _gemm(in0, in1, in2, in3):
    """The reference's GEMM call, verbatim."""
    return torch.ops.aten._scaled_mm_v2.default(in0, in1, [in2], [3], [1], [in3], [3], [1], None, torch.bfloat16)


def _ok(t, dtype, shape, stride):
    return t.dtype == dtype and tuple(t.shape) == shape and tuple(t.stride()) == stride and t.is_cuda


def run(in0, in1, in2, in3, in4, in5, in6, in7):
    d = _gemm(in0, in1, in2, in3)
    if not (_ok(d, torch.bfloat16, (M, N), (N, 1)) and _ok(in4, torch.float32, (M, 1), (1, 1))
            and _ok(in5, torch.bfloat16, (M, N), (N, 1)) and _ok(in6, torch.bfloat16, (N,), (1,))
            and _ok(in7, torch.bfloat16, (M, N), (N, 1)) and _num_blocks(d.device) == _B
            and in4.data_ptr() % 4 == 0 and all(t.data_ptr() % 16 == 0 for t in (d, in5, in6, in7))):
        raise ValueError("unsupported inputs: expected the spec's contiguous shapes/dtypes, 16-byte aligned "
                         "(rstd 4-byte aligned), on an sm_103 device with the compiled SM count")
    g = torch.empty((M, N), dtype=torch.bfloat16, device=d.device)
    part = torch.empty((_B, N), dtype=torch.float32, device=d.device)
    dw = torch.empty((N,), dtype=torch.bfloat16, device=d.device)
    _k1(in4, d, in5, in6, in7, g, part)
    _k2(part, dw)
    return dw, g
