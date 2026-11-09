"""Rowwise torchao mxfp8_quantize + triton_mx_block_rearrange as one CuTe DSL kernel, then the reference's
own aten._scaled_mm_v2 call.

Reference: q, _, s, _ = torchao.mxfp8_quantize(in0, rowwise=True, colwise=False, 32, 1, 'e4m3', 'rceil');
out = _scaled_mm_v2(q, in1, [triton_mx_block_rearrange(s)], [3], [1], [in2], [3], [1], None, bf16).
The GEMM call is unchanged: same op and arguments, operands with the reference's shapes, strides and
dtypes, freshly allocated by the caching allocator like the reference's (same alignment).

Quantize numerics (torchao mxfp8_quantize e4m3 rceil, same emulation as the verified DSv3 lfu5 kernel):
  amax = NaN-propagating max |x| over each 1 x 32 block; for bf16 amax bits a:
  byte = 255 if a >= 0x7f80 else max(((a + 31) >> 7) - 8, 0);
  recip = 0x7fffffff if byte == 255 else (254 - byte) << 23; q = cvt.rn.satfinite.e4m3(x *rn recip).
triton_mx_block_rearrange of the [R, C / 32] scales (no padding here: R % 128 == 0, (C / 32) % 4 == 0):
  (r, c) -> (r // 128) * 128 * (C / 32) + (c // 4) * 512 + (r % 32) * 16 + ((r % 128) // 32) * 4 + c % 4.
Each 128-row x 128-column tile of x owns one 512-byte atom of the rearranged scales.

Work split: CTA = 128 rows x TC columns. Lane l owns VB consecutive bf16 (VB * 2 bytes, LDW-byte loads)
of RPT rows (rows RPI apart); a 32-column block spans 32 / VB lanes.
"""

import cutlass
import cutlass.cute as cute
import torch
from cutlass import Int32, Int64, Uint32
from cutlass._mlir import ir
from cutlass._mlir.dialects import llvm
from cutlass.cutlass_dsl import T, dsl_user_op

R = 8192
C = 8192
# (bf16 per lane, rows per thread, CTA columns)
CFG = (16, 2, 128)

TR = 128
NCB = C // 32  # scale columns
assert R % 128 == 0 and NCB % 4 == 0, "no rearrange padding in this specialization"


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


@dsl_user_op
def ld_v8(addr, *, loc=None, ip=None):
    r = _asmN(8, T.i32(), [Int64(addr)],
              "ld.global.nc.L1::no_allocate.v8.b32 {$0, $1, $2, $3, $4, $5, $6, $7}, [$8];",
              "=r,=r,=r,=r,=r,=r,=r,=r,l", loc, ip)
    return [Uint32(v) for v in r]


@dsl_user_op
def ld_v4(addr, *, loc=None, ip=None):
    r = _asmN(4, T.i32(), [Int64(addr)], "ld.global.nc.L1::no_allocate.v4.b32 {$0, $1, $2, $3}, [$4];",
              "=r,=r,=r,=r,l", loc, ip)
    return [Uint32(v) for v in r]


def ld_words(addr, n, ldw):
    """n consecutive 32-bit words from addr with ldw-byte loads (unrolled at trace time)."""
    out = []
    for k in range(0, n, ldw // 4):
        out += ld_v8(addr + k * 4) if ldw == 32 else ld_v4(addr + k * 4)
    return out


@dsl_user_op
def st_v2(addr, a, b, *, loc=None, ip=None):
    _asm1(None, [Int64(addr), Uint32(a), Uint32(b)], "st.global.v2.b32 [$0], {$1, $2};", "l,r,r", loc, ip, side=True)


@dsl_user_op
def st_v4(addr, a, b, c, d, *, loc=None, ip=None):
    _asm1(None, [Int64(addr), Uint32(a), Uint32(b), Uint32(c), Uint32(d)],
          "st.global.v4.b32 [$0], {$1, $2, $3, $4};", "l,r,r,r,r", loc, ip, side=True)


@dsl_user_op
def sts_u8(addr, a, *, loc=None, ip=None):
    _asm1(None, [Int32(addr), Uint32(a)], "st.shared.u8 [$0], $1;", "r,r", loc, ip, side=True)


@dsl_user_op
def lds_v4(addr, *, loc=None, ip=None):
    r = _asmN(4, T.i32(), [Int32(addr)], "ld.shared.v4.b32 {$0, $1, $2, $3}, [$4];", "=r,=r,=r,=r,r", loc, ip, side=True)
    return [Uint32(v) for v in r]


@dsl_user_op
def hmax(a, b, *, loc=None, ip=None):
    """Per 16-bit half: NaN-propagating max |.| of bf16 (sign bit is garbage)."""
    return Uint32(_asm1(T.i32(), [Uint32(a), Uint32(b)], "max.NaN.xorsign.abs.bf16x2 $0, $1, $2;", "=r,r,r", loc, ip))


@dsl_user_op
def shfl_xor(a, off: int, *, loc=None, ip=None):
    return Uint32(_asm1(T.i32(), [Uint32(a)], f"shfl.sync.bfly.b32 $0, $1, {off}, 31, -1;", "=r,r", loc, ip))


@dsl_user_op
def scale_of(m, *, loc=None, ip=None):
    """m = packed per-half block |max|; the block amax is the max of both halves.
    Returns (e8m0 byte, recip f32 bits): a = bf16 bits of amax (> 0x7f80 is NaN);
    byte = 255 if a >= 0x7f80 else max(((a + 31) >> 7) - 8, 0) (torchao rceil);
    recip = 0x7fffffff if byte == 255 else (254 - byte) << 23."""
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
        "mad.lo.s32 r, e, -8388608, 0x7F000000;\n"
        "selp.b32 $0, 255, e, p;\n"
        "selp.b32 $1, 0x7FFFFFFF, r, p; }",
        "=r,=r,r", loc, ip,
    )
    return Uint32(r[0]), Uint32(r[1])


@dsl_user_op
def quant_row4(y0, y1, r, *, loc=None, ip=None):
    """4 e4m3 bytes (byte k = column k) of the bf16 pairs y0, y1 scaled by recip r (f32 bits)."""
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


def _tree_hmax(words):
    t = list(words)
    while len(t) > 1:
        t = [hmax(t[k], t[k + 1]) for k in range(0, len(t), 2)]
    return t[0]


class RowQuant:
    def __init__(self, vb: int, rpt: int, tc: int, ldw: int):
        self.vb, self.rpt, self.tc, self.ldw = vb, rpt, tc, ldw
        self.lpr = tc // vb  # lanes per row
        self.rpi = 32 // self.lpr  # rows per warp load instruction
        self.rpw = rpt * self.rpi  # rows per warp
        self.nw = TR // self.rpw
        self.lpb = 32 // vb  # lanes per 32-column block
        self.natoms = tc // 128  # scale atoms per tile
        assert vb in (8, 16) and (vb * 2) % ldw == 0 and self.lpr <= 32 and TR % self.rpw == 0 and C % tc == 0 and tc % 128 == 0
        assert self.natoms * 32 <= self.nw * 32
        self.ncta = (R // TR) * (C // tc)

    @cute.jit
    def __call__(self, x, q, s, stream):
        # min_blocks_per_mp=1 (.minnctapersm 1) keeps ptxas from sinking row loads below the first math.
        self.kernel(x, q, s).launch(grid=[self.ncta, 1, 1], block=[self.nw * 32, 1, 1], stream=stream,
                                    min_blocks_per_mp=1)

    @cute.kernel
    def kernel(self, x, q, s):
        VB, RPT, LPR, RPI, RPW, LPB, TC = self.vb, self.rpt, self.lpr, self.rpi, self.rpw, self.lpb, self.tc
        tid, _, _ = cute.arch.thread_idx()
        b, _, _ = cute.arch.block_idx()
        w = tid // 32
        l = tid % 32
        smem = cutlass.utils.SmemAllocator()
        sAll = smem.allocate_tensor(cutlass.Uint8, cute.make_layout(512 * self.natoms), byte_alignment=16)
        sb = sAll.iterator.toint()

        tr = b // (C // TC)
        tc = b - tr * (C // TC)
        c = l % LPR
        sub = l // LPR
        rl0 = w * RPW + sub  # tile-local row of i = 0
        row0 = Int64(tr * TR + rl0)
        col0 = Int64(tc * TC + c * VB)
        src = x.iterator.toint() + (row0 * C + col0) * 2
        v = []
        for i in cutlass.range_constexpr(RPT):
            v.append(ld_words(src + i * (RPI * C * 2), VB // 2, self.ldw))
        dst = q.iterator.toint() + row0 * C + col0
        kb = c // LPB  # 32-column block within the tile
        for i in cutlass.range_constexpr(RPT):
            m = _tree_hmax(v[i])
            for k in cutlass.range_constexpr(LPB.bit_length() - 1):
                m = hmax(m, shfl_xor(m, 1 << k))
            byte, rcp = scale_of(m)
            y = [quant_row4(v[i][k], v[i][k + 1], rcp) for k in range(0, VB // 2, 2)]
            if cutlass.const_expr(VB == 16):
                st_v4(dst + i * (RPI * C), y[0], y[1], y[2], y[3])
            else:
                st_v2(dst + i * (RPI * C), y[0], y[1])
            if c % LPB == 0:
                rl = rl0 + i * RPI
                sts_u8(sb + (kb // 4) * 512 + (rl % 32) * 16 + (rl // 32) * 4 + kb % 4, byte)
        cute.arch.barrier()
        if tid < 32 * self.natoms:
            a = tid // 32
            chunk = tid % 32
            pat = lds_v4(sb + a * 512 + chunk * 16)
            atom = Int64(tr * (128 * NCB) + (tc * self.natoms + a) * 512 + chunk * 16)
            st_v4(s.iterator.toint() + atom, pat[0], pat[1], pat[2], pat[3])


def _compile(vb, rpt, tc, ldw):
    def fake(dtype, n, align):
        return cute.runtime.make_fake_tensor(dtype, (n,), stride=(1,), assumed_align=align)

    return cute.compile(
        RowQuant(vb, rpt, tc, ldw),
        fake(cutlass.BFloat16, R * C, ldw),
        fake(cutlass.Uint8, R * C, 16),
        fake(cutlass.Uint8, R * NCB, 16),
        cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
        options="--enable-tvm-ffi",
    )


# 32-byte aligned inputs use 32-byte loads, 16-byte aligned ones 16-byte loads; anything else is
# copied to a fresh (allocator-aligned) buffer first.
_FN32 = _compile(CFG[0], CFG[1], CFG[2], 32)
_FN16 = _compile(CFG[0], CFG[1], CFG[2], 16)


def _quantize(in0):
    """Rowwise e4m3 data [R, C] and the rearranged (cuBLAS blocked) e8m0 scales [R, C / 32], with the
    shapes, strides and dtypes of the reference's mxfp8_quantize / triton_mx_block_rearrange outputs."""
    if in0.dtype != torch.bfloat16 or tuple(in0.shape) != (R, C) or in0.device.type != "cuda":
        raise ValueError(f"expected cuda bf16 [{R}, {C}], got {in0.dtype} {tuple(in0.shape)} on {in0.device}")
    if not in0.is_contiguous() or in0.data_ptr() % 16:
        in0 = in0.clone(memory_format=torch.contiguous_format)
    dev = in0.device
    qrow = torch.empty((R, C), dtype=torch.float8_e4m3fn, device=dev)
    srow = torch.empty((R, NCB), dtype=torch.float8_e8m0fnu, device=dev)
    fn = _FN32 if in0.data_ptr() % 32 == 0 else _FN16
    fn(in0.view(-1), qrow.view(-1).view(torch.uint8), srow.view(-1).view(torch.uint8))
    return qrow, srow


def run(in0, in1, in2):
    qrow, srow = _quantize(in0)
    return torch.ops.aten._scaled_mm_v2.default(qrow, in1, [srow], [3], [1], [in2], [3], [1], None, torch.bfloat16)
