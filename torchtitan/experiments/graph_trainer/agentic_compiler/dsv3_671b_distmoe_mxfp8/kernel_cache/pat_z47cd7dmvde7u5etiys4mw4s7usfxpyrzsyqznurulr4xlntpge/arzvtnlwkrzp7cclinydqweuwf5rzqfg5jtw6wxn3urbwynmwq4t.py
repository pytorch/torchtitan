"""Fused RMSNorm + MXFP8 rowwise quantize (x2) for DSv3 671B (pat_z47c), one CuTe DSL kernel.

reference: y = aten._fused_rms_norm(in1 [4096, 7168] bf16, w = in0.view(7168), eps=1e-6)[0];
           (q, s) = torchao.mxfp8_quantize(y, rowwise, 1x32, e4m3, rceil), s swizzled by
           torchao.triton_mx_block_rearrange; done twice -> returns (q1, s1, q2, s2).
One CTA (128 threads) per row; y never touches HBM. q1/q2 (and s1/s2) are separate buffers
with identical bytes (the two reference quantize calls are identical and deterministic).

Numerics (all explicit PTX, no contraction):
* add (if present): h = add.rn.bf16x2(a, b), bitwise equal to torch's bf16 add (fp32 add + one RNE
  to bf16) for all 2^32 bf16 pairs.
* RMSNorm reproduces the kernel aten._fused_rms_norm dispatches to on this build (torch._native
  override -> vendored quack RMSNorm, N=7168 bf16: 128 threads/row, vecsize 8, cluster 1):
  thread t owns columns 8t + 1024k + j (k = 0..6, j = 0..7); per-thread sum of squares is a
  sequential fma.rn.f32.bf16 chain in (k, j) order starting at +0; warp butterfly add
  (shfl.bfly 16, 8, 4, 2, 1); block: (W0 + W2) + (W1 + W3); ms = div.rn(sum, 7168);
  rstd = rsqrt.approx.ftz(ms +rn fl32(1e-6)); y = bf16_rn((x *rn rstd) *rn w).
* MXFP8 (torchao mxfp8_quantize_kernel<bf16, e4m3, 1, 32, RCEIL>, decoded from its SASS):
  amax = NaN-propagating max |y| over the 1x32 block; v = amax *rn fl32(1/448);
  b = cvt.rp.bf16(|v|); byte = exp(b) + 1 if (b >= 0x41 and mant(b) != 0 and exp(b) != 0xff)
  else exp(b) -- for bf16 amax bits a this equals 255 if a >= 0x7f80 else
  max(((a + 31) >> 7) - 8, 0) (all 2^16 patterns checked); recip = 2^(127 - byte) as bits
  ((254 - byte) & 0xff) << 23, byte 254 -> 2^-127 (0x00400000), byte 255 -> 0x7fffffff;
  q = cvt.rn.satfinite.e4m3(y *rn recip).
* triton_mx_block_rearrange: scale (r, c) -> byte offset (c // 4) * 512 + (r // 128) * 28672
  + (r % 32) * 16 + ((r % 128) // 32) * 4 + c % 4 of a contiguous [4096, 224] buffer.
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
NTHR = 128
NVEC = 7  # 16-byte vectors per thread per row
NSB = N // 32  # 224 scale bytes per row

# ---- problem flags (pat_z47c) ----
HAS_ADD = False
WRITE_H = False
WRITE_Y = False
WRITE_RSTD = False
NQ = 2


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
    r = _asmN(4, T.i32(), [Int64(addr)], "ld.global.nc.L1::no_allocate.v4.b32 {$0, $1, $2, $3}, [$4];",
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
          "st.global.L1::no_allocate.v4.b32 [$0], {$1, $2, $3, $4};", "l,r,r,r,r", loc, ip, side=True)


@dsl_user_op
def st_v2(addr, a, b, *, loc=None, ip=None):
    _asm1(None, [Int64(addr), Uint32(a), Uint32(b)], "st.global.v2.b32 [$0], {$1, $2};", "l,r,r", loc, ip, side=True)


@dsl_user_op
def st_b32(addr, a, *, loc=None, ip=None):
    _asm1(None, [Int64(addr), Uint32(a)], "st.global.b32 [$0], $1;", "l,r", loc, ip, side=True)


@dsl_user_op
def st_f32(addr, a, *, loc=None, ip=None):
    _asm1(None, [Int64(addr), Float32(a)], "st.global.f32 [$0], $1;", "l,f", loc, ip, side=True)


@dsl_user_op
def sts_f32(addr, a, *, loc=None, ip=None):
    _asm1(None, [Int32(addr), Float32(a)], "st.shared.f32 [$0], $1;", "r,f", loc, ip, side=True)


@dsl_user_op
def lds_f32x4(addr, *, loc=None, ip=None):
    r = _asmN(4, T.f32(), [Int32(addr)], "ld.shared.v4.f32 {$0, $1, $2, $3}, [$4];", "=f,=f,=f,=f,r", loc, ip, side=True)
    return [Float32(v) for v in r]


# ---------------------------------------------------------------- math
@dsl_user_op
def add_bf16x2(a, b, *, loc=None, ip=None):
    """bf16x2 word a + b. add.rn.bf16x2 (one rounding of the exact sum) equals torch's bf16 add
    (fp32 add, then RNE to bf16) for all 2^32 bf16 pairs (exhaustively checked on GB300)."""
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
def fadd(a, b, *, loc=None, ip=None):
    return Float32(_asm1(T.f32(), [Float32(a), Float32(b)], "add.rn.f32 $0, $1, $2;", "=f,f,f", loc, ip))


@dsl_user_op
def rstd_of(total, *, loc=None, ip=None):
    """rsqrt.approx.ftz(div.rn(total, 7168) +rn fl32(1e-6))."""
    return Float32(_asm1(
        T.f32(), [Float32(total)],
        "{ .reg .f32 m;\n"
        "div.rn.f32 m, $1, 0f45E00000;\n"
        "add.rn.f32 m, m, 0f358637BD;\n"
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


@dsl_user_op
def hmax4(y0, y1, y2, y3, *, loc=None, ip=None):
    """Per 16-bit half: NaN-propagating max |.| of the bf16 halves (sign bit is garbage)."""
    return Uint32(_asm1(
        T.i32(), [Uint32(y0), Uint32(y1), Uint32(y2), Uint32(y3)],
        "{ .reg .b32 a, b;\n"
        "max.NaN.xorsign.abs.bf16x2 a, $1, $2;\n"
        "max.NaN.xorsign.abs.bf16x2 b, $3, $4;\n"
        "max.NaN.xorsign.abs.bf16x2 $0, a, b; }",
        "=r,r,r,r,r", loc, ip))


@dsl_user_op
def hmax(a, b, *, loc=None, ip=None):
    return Uint32(_asm1(T.i32(), [Uint32(a), Uint32(b)], "max.NaN.xorsign.abs.bf16x2 $0, $1, $2;", "=r,r,r", loc, ip))


@dsl_user_op
def scale_of(m, *, loc=None, ip=None):
    """m = packed per-half block |max| (NaN-propagated). Returns (e8m0 byte, recip f32 bits).

    a = max of the two halves of (m & 0x7fff7fff) (bf16 bits of amax; > 0x7f80 is NaN).
    byte = 255 if a >= 0x7f80 else max(((a + 31) >> 7) - 8, 0)
      (== torchao's rceil byte for every bf16 amax, checked against the emulation of its SASS);
    recip = 0x7fffffff if byte == 255 else (254 - byte) << 23 (bytes 248..254 are unreachable).
    """
    r = _asmN(
        2, T.i32(), [Uint32(m)],
        "{ .reg .b32 a, b, e, r; .reg .pred p;\n"
        "and.b32 a, $2, 0x7FFF7FFF;\n"
        "shr.u32 b, a, 16;\n"
        "and.b32 a, a, 0xFFFF;\n"
        "max.u32 a, a, b;\n"
        "add.s32 e, a, -993;\n"
        "shr.s32 e, e, 7;\n"
        "max.s32 e, e, 0;\n"
        "setp.ge.u32 p, a, 0x7F80;\n"
        "sub.s32 r, 254, e;\n"
        "shl.b32 r, r, 23;\n"
        "selp.b32 $0, 255, e, p;\n"
        "selp.b32 $1, 0x7FFFFFFF, r, p; }",
        "=r,=r,r", loc, ip,
    )
    return Uint32(r[0]), Uint32(r[1])


@dsl_user_op
def quant4(y0, y1, r, *, loc=None, ip=None):
    """4 e4m3 bytes (byte k = element k) of the bf16 pairs y0, y1 scaled by recip r (f32 bits)."""
    return Uint32(_asm1(
        T.i32(), [Uint32(y0), Uint32(y1), Uint32(r)],
        "{ .reg .b32 a, b, c, d; .reg .b64 p, q, s; .reg .b16 h0, h1;\n"
        "shl.b32 a, $1, 16; and.b32 b, $1, 0xFFFF0000;\n"
        "shl.b32 c, $2, 16; and.b32 d, $2, 0xFFFF0000;\n"
        "mov.b64 p, {a, b}; mov.b64 q, {c, d}; mov.b64 s, {$3, $3};\n"
        "mul.rn.f32x2 p, p, s;\n"
        "mul.rn.f32x2 q, q, s;\n"
        "mov.b64 {a, b}, p; mov.b64 {c, d}, q;\n"
        "cvt.rn.satfinite.e4m3x2.f32 h0, b, a;\n"
        "cvt.rn.satfinite.e4m3x2.f32 h1, d, c;\n"
        "mov.b32 $0, {h0, h1}; }",
        "=r,r,r,r", loc, ip))


class NormQuant:
    @cute.jit
    def __call__(self, a, b, w, h, y, rstd, q1, s1, q2, s2, stream):
        self.kernel(a, b, w, h, y, rstd, q1, s1, q2, s2).launch(grid=[M, 1, 1], block=[NTHR, 1, 1], stream=stream)

    @cute.kernel
    def kernel(self, a, b, w, h, y, rstd, q1, s1, q2, s2):
        tid, _, _ = cute.arch.thread_idx()
        row, _, _ = cute.arch.block_idx()
        wid = tid // 32
        lane = tid % 32

        smem = cutlass.utils.SmemAllocator()
        sred = smem.allocate_tensor(cutlass.Float32, cute.make_layout(4), byte_alignment=16)
        sb = sred.iterator.toint()

        row64 = Int64(row)
        eoff = row64 * N + Int64(tid * 8)  # element offset of vector 0
        xa = a.iterator.toint() + eoff * 2
        xv = []
        for k in cutlass.range_constexpr(NVEC):
            xv.append(ld_stream_v4(xa + k * 2048))
        if cutlass.const_expr(HAS_ADD):
            xb = b.iterator.toint() + eoff * 2
            bv = []
            for k in cutlass.range_constexpr(NVEC):
                bv.append(ld_stream_v4(xb + k * 2048))
            for k in cutlass.range_constexpr(NVEC):
                xv[k] = [add_bf16x2(xv[k][i], bv[k][i]) for i in range(4)]
            if cutlass.const_expr(WRITE_H):
                ha = h.iterator.toint() + eoff * 2
                for k in cutlass.range_constexpr(NVEC):
                    st_v4(ha + k * 2048, xv[k][0], xv[k][1], xv[k][2], xv[k][3])

        acc = Float32(0.0)
        for k in cutlass.range_constexpr(NVEC):
            acc = sumsq8(acc, xv[k][0], xv[k][1], xv[k][2], xv[k][3])
        for off in (16, 8, 4, 2, 1):
            acc = fadd(acc, cute.arch.shuffle_sync_bfly(acc, offset=off))
        if lane == 0:
            sts_f32(sb + wid * 4, acc)
        cute.arch.barrier()
        wsum = lds_f32x4(sb)
        total = fadd(fadd(wsum[0], wsum[2]), fadd(wsum[1], wsum[3]))
        rs = rstd_of(total)
        if cutlass.const_expr(WRITE_RSTD):
            if tid == 0:
                st_f32(rstd.iterator.toint() + row64 * 4, rs)

        qa1 = q1.iterator.toint() + eoff
        if cutlass.const_expr(NQ == 2):
            qa2 = q2.iterator.toint() + eoff
        if cutlass.const_expr(WRITE_Y):
            ya = y.iterator.toint() + eoff * 2
        # swizzled scale offset for this row; block c = 32 k + tid // 4
        srow = Int64((row // 128) * 28672 + (row % 32) * 16 + ((row % 128) // 32) * 4)
        sa1 = s1.iterator.toint() + srow + Int64((tid // 16) * 512)
        if cutlass.const_expr(NQ == 2):
            sa2 = s2.iterator.toint() + srow + Int64((tid // 16) * 512)
        wa = w.iterator.toint() + Int64(tid * 16)
        for k in cutlass.range_constexpr(NVEC):
            wk = ld_cached_v4(wa + k * 2048)
            yv = [norm_bf16x2(xv[k][i], wk[i], rs) for i in range(4)]
            if cutlass.const_expr(WRITE_Y):
                st_v4(ya + k * 2048, yv[0], yv[1], yv[2], yv[3])
            am = hmax4(yv[0], yv[1], yv[2], yv[3])
            am = hmax(am, cute.arch.shuffle_sync_bfly(am, offset=1))
            am = hmax(am, cute.arch.shuffle_sync_bfly(am, offset=2))
            byte, rcp = scale_of(am)
            lo = quant4(yv[0], yv[1], rcp)
            hi = quant4(yv[2], yv[3], rcp)
            st_v2(qa1 + k * 1024, lo, hi)
            if cutlass.const_expr(NQ == 2):
                st_v2(qa2 + k * 1024, lo, hi)
            b1 = cute.arch.shuffle_sync_down(byte, offset=4)
            b2 = cute.arch.shuffle_sync_down(byte, offset=8)
            b3 = cute.arch.shuffle_sync_down(byte, offset=12)
            word = byte | (b1 << Uint32(8)) | (b2 << Uint32(16)) | (b3 << Uint32(24))
            if lane % 16 == 0:
                st_b32(sa1 + k * 4096, word)
                if cutlass.const_expr(NQ == 2):
                    st_b32(sa2 + k * 4096, word)


_DT = {torch.bfloat16: cutlass.BFloat16, torch.float32: cutlass.Float32, torch.uint8: cutlass.Uint8}


def _fake(dtype, shape):
    stride, acc = [], 1
    for s in reversed(shape):
        stride.append(acc)
        acc *= s
    return cute.runtime.make_fake_tensor(_DT[dtype], tuple(shape), stride=tuple(reversed(stride)), assumed_align=16)


_fn = cute.compile(
    NormQuant(),
    _fake(torch.bfloat16, (M, N)),
    _fake(torch.bfloat16, (M, N)) if HAS_ADD else None,
    _fake(torch.bfloat16, (N,)),
    _fake(torch.bfloat16, (M, N)) if WRITE_H else None,
    _fake(torch.bfloat16, (M, N)) if WRITE_Y else None,
    _fake(torch.float32, (M,)) if WRITE_RSTD else None,
    _fake(torch.uint8, (M, N)),
    _fake(torch.uint8, (M, NSB)),
    _fake(torch.uint8, (M, N)) if NQ == 2 else None,
    _fake(torch.uint8, (M, NSB)) if NQ == 2 else None,
    cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
    options="--enable-tvm-ffi",
)


def _dense(t):
    """The kernel uses 16-byte vector accesses on contiguous rows."""
    return t if t.is_contiguous() and t.data_ptr() % 16 == 0 else t.clone(memory_format=torch.contiguous_format)


def _alloc(dev, specs):
    return [torch.empty(shape, dtype=dt, device=dev) for shape, dt in specs]


def run(in0, in1):
    w, x = _dense(in0), _dense(in1)
    q1, s1, q2, s2 = _alloc(x.device, [((M, N), torch.uint8), ((M, NSB), torch.uint8),
                                       ((M, N), torch.uint8), ((M, NSB), torch.uint8)])
    _fn(x, None, w.view(N), None, None, None, q1, s1, q2, s2)
    e4m3, e8m0 = torch.float8_e4m3fn, torch.float8_e8m0fnu
    return (q1.view(e4m3), s1.view(e8m0), q2.view(e4m3), s2.view(e8m0))
