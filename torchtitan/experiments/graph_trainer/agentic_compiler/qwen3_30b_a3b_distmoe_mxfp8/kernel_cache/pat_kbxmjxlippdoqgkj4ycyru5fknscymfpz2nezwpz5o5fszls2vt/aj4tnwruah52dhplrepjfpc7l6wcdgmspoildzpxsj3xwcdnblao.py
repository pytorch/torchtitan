"""Qwen3 30B-A3B attention_norm + MXFP8 rowwise quantize (pat_kbxmj / aj4tnw, fusion region G41), one CuTe DSL
kernel.

reference: (y, rstd) = aten._fused_rms_norm(in0 [32768, 2048] bf16, [2048], w = in1, eps=1e-6);
           (q, s) = torchao.mxfp8_quantize(y, rowwise, 1x32, e4m3, rceil), s swizzled by
           torchao.triton_mx_block_rearrange -> returns (y, rstd [32768, 1], q, s [32768, 64]).
One warp per row (lane c owns columns 8c + 256k + j, k, j = 0..7, the quack layout), RPC = 4 J rows per CTA:
rows g*128 + i*32 + p*J + j (i < 4, j < J), so a CTA's scale bytes fill whole 16 J-byte runs of the swizzled
layout; they are gathered in smem and written as 16-byte units (partial-sector scale stores cost ~1 us).

Numerics (all explicit PTX, no contraction):
* RMSNorm reproduces the kernel aten._fused_rms_norm dispatches to on this build (torch._native override ->
  vendored quack RMSNorm, N=2048 bf16: 32 threads per row, vecsize 8, cluster 1; decoded from its SASS):
  per-lane sum of squares is a sequential fma.rn.f32.bf16 chain (FHFMA.BF16) in (k, j) order from +0;
  warp butterfly FADD with offsets 16, 8, 4, 2, 1; rstd = rsqrt.approx.ftz(fma.rn(sum, 2^-11, fl32(1e-6)));
  y = bf16_rn((x *rn rstd) *rn w).
* MXFP8 (torchao mxfp8_quantize_kernel<bf16, e4m3, 1, 32, RCEIL>, decoded from its SASS for DSv3 pat_use7):
  amax = NaN-propagating max |y| over the 1x32 block (4 lanes x 8 columns here);
  byte = 255 if a >= 0x7f80 else max(((a + 31) >> 7) - 8, 0) for the bf16 bits a of amax;
  recip = 2^(127 - byte) as bits ((254 - byte) & 0xff) << 23 (byte 255 -> 0x7fffffff);
  q = cvt.rn.satfinite.e4m3(y *rn recip).
* triton_mx_block_rearrange: scale (r, c) -> byte offset (c // 4) * 512 + (r // 128) * 8192
  + (r % 32) * 16 + ((r % 128) // 32) * 4 + c % 4 of a contiguous [32768, 64] buffer.
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
NSB = N // 32  # 64 scale bytes per row
J = 2  # consecutive rows per 32-row group in a CTA (measured best of 1, 2, 4)
RPC = 4 * J  # rows (warps) per CTA: rows g*128 + i*32 + p*J + j, i < 4, j < J
NTHR = 32 * RPC
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
          "st.global.v4.b32 [$0], {$1, $2, $3, $4};", "l,r,r,r,r", loc, ip, side=True)


@dsl_user_op
def st_v2(addr, a, b, *, loc=None, ip=None):
    _asm1(None, [Int64(addr), Uint32(a), Uint32(b)], "st.global.v2.b32 [$0], {$1, $2};", "l,r,r", loc, ip, side=True)


@dsl_user_op
def st_b32(addr, a, *, loc=None, ip=None):
    _asm1(None, [Int64(addr), Uint32(a)], "st.global.b32 [$0], $1;", "l,r", loc, ip, side=True)


@dsl_user_op
def sts_b32(addr, a, *, loc=None, ip=None):
    _asm1(None, [Int32(addr), Uint32(a)], "st.shared.b32 [$0], $1;", "r,r", loc, ip, side=True)


@dsl_user_op
def lds_v4(addr, *, loc=None, ip=None):
    r = _asmN(4, T.i32(), [Int32(addr)], "ld.shared.v4.b32 {$0, $1, $2, $3}, [$4];",
              "=r,=r,=r,=r,r", loc, ip, side=True)
    return [Uint32(v) for v in r]


@dsl_user_op
def st_f32(addr, a, *, loc=None, ip=None):
    _asm1(None, [Int64(addr), Float32(a)], "st.global.f32 [$0], $1;", "l,f", loc, ip, side=True)


# ---------------------------------------------------------------- math
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
def hmax_bfly(a, off: int, *, loc=None, ip=None):
    return Uint32(_asm1(
        T.i32(), [Uint32(a)],
        "{ .reg .b32 o;\n"
        f"shfl.sync.bfly.b32 o, $1, {off}, 31, -1;\n"
        "max.NaN.xorsign.abs.bf16x2 $0, $1, o; }",
        "=r,r", loc, ip, side=True))


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
    def __call__(self, x, w, y, rstd, q, s, stream):
        self.kernel(x, w, y, rstd, q, s).launch(grid=[M // RPC, 1, 1], block=[NTHR, 1, 1], stream=stream)

    @cute.kernel
    def kernel(self, x, w, y, rstd, q, s):
        tid, _, _ = cute.arch.thread_idx()
        bid, _, _ = cute.arch.block_idx()
        c = tid % 32
        wid = tid // 32
        # this CTA's rows complete whole 16*J-byte runs of the swizzled scale layout
        g = bid // (32 // J)
        p = bid % (32 // J)
        i_ = wid // J
        j_ = wid % J
        row = g * 128 + i_ * 32 + p * J + j_
        row64 = Int64(row)

        smem = cutlass.utils.SmemAllocator()
        sS = smem.allocate_tensor(cutlass.Uint32, cute.make_layout(64 * J), byte_alignment=16)
        sSb = sS.iterator.toint()

        eoff = row64 * N + Int64(c * 8)  # element offset of this lane's vector k = 0
        xa = x.iterator.toint() + eoff * 2
        xv = [ld_stream_v4(xa + k * 512) for k in range(8)]
        wa = w.iterator.toint() + Int64(c * 16)
        wv = [ld_cached_v4(wa + k * 512) for k in range(8)]

        acc = Float32(0.0)
        for k in cutlass.range_constexpr(8):
            acc = sumsq8(acc, xv[k][0], xv[k][1], xv[k][2], xv[k][3])
        for off in (16, 8, 4, 2, 1):
            acc = bfly_add(acc, off)
        rs = rstd_of(acc)
        if c == 0:
            st_f32(rstd.iterator.toint() + row64 * 4, rs)

        ya = y.iterator.toint() + eoff * 2
        qa = q.iterator.toint() + eoff
        # scale word of column group cg = 2k + c // 16 -> smem [cg][j][i]
        sw = sSb + ((c // 16) * (4 * J) + j_ * 4 + i_) * 4
        for k in cutlass.range_constexpr(8):
            yv = [norm_bf16x2(xv[k][i], wv[k][i], rs) for i in range(4)]
            st_v4(ya + k * 512, yv[0], yv[1], yv[2], yv[3])
            am = hmax4(yv[0], yv[1], yv[2], yv[3])
            am = hmax_bfly(am, 1)
            am = hmax_bfly(am, 2)
            byte, rcp = scale_of(am)
            lo = quant4(yv[0], yv[1], rcp)
            hi = quant4(yv[2], yv[3], rcp)
            st_v2(qa + k * 256, lo, hi)
            b1 = cute.arch.shuffle_sync_down(byte, offset=4)
            b2 = cute.arch.shuffle_sync_down(byte, offset=8)
            b3 = cute.arch.shuffle_sync_down(byte, offset=12)
            word = byte | (b1 << Uint32(8)) | (b2 << Uint32(16)) | (b3 << Uint32(24))
            if c % 16 == 0:
                sts_b32(sw + k * (2 * 4 * J * 4), word)
        cute.arch.barrier()
        # thread t < 16 J writes 16-byte unit (cg = t // J, jj = t % J) of the scale layout
        if tid < 16 * J:
            cg = tid // J
            jj = tid % J
            u = lds_v4(sSb + (cg * 4 * J + jj * 4) * 4)
            sa = s.iterator.toint() + Int64(cg * 512 + g * 8192 + (p * J + jj) * 16)
            st_v4(sa, u[0], u[1], u[2], u[3])


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
    _fake(torch.bfloat16, (N,)),
    _fake(torch.bfloat16, (M, N)),
    _fake(torch.float32, (M,)),
    _fake(torch.uint8, (M, N)),
    _fake(torch.uint8, (M, NSB)),
    cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
    options="--enable-tvm-ffi",
)


def _dense(t):
    """The kernel uses 16-byte vector accesses on contiguous rows."""
    return t if t.is_contiguous() and t.data_ptr() % 16 == 0 else t.clone(memory_format=torch.contiguous_format)


def run(in0, in1):
    x, w = _dense(in0), _dense(in1)
    dev = x.device
    y = torch.empty((M, N), dtype=torch.bfloat16, device=dev)
    rstd = torch.empty((M, 1), dtype=torch.float32, device=dev)
    q = torch.empty((M, N), dtype=torch.uint8, device=dev)
    s = torch.empty((M, NSB), dtype=torch.uint8, device=dev)
    _fn(x, w, y, rstd.view(M), q, s)
    return (y, rstd, q.view(torch.float8_e4m3fn), s.view(torch.float8_e8m0fnu))
