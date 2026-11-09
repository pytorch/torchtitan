"""Chunked cross-entropy loss backward, one 1024-token chunk, vocab 151936 (Qwen3 235B-A22B, G07).

Adapted from the verified Qwen3 30B-A3B kernel (dex t5, 4096-token chunks): only the row count M changed.
The softmax-backward row sum has one nonzero term (-g) so it is order independent, and the elementwise
op sequence (cunn_SoftMaxBackward, independent of the row count) is unchanged.

Reference: g = in0 / in1; v7 = nll_loss_backward(g, in2, in3, None, sum, -100, in4);
out = bf16(_log_softmax_backward_data(v7, in5, 1, float32)).

v7 is zero except v7[i, t_i] = -g for rows with t_i != -100, so (in2 and in4 are not read):
    sum_i = 0 + ... + (-g) + ... + 0 = (-g) + 0     (rows with t_i == -100: +0)
    out[i, j] = bf16(fma(exp(in5[i, j]), -sum_i, v7[i, j]))
exp is CUDA's accurate expf exactly as in torch's cunn_SoftMaxBackward SASS (sm_100 cubin):
    t = sat(fma(x, 0x3BBB989D, 0.5)); j = fma.rm(t, 252, 12582913); f = fma(x, log2e_hi, -(j - 12583039));
    f = fma(x, log2e_lo, f); e = (j << 23) * ex2.approx(f)
and the epilogue is FFMA(e, -sum, gO) (contracted), then cvt.rn.bf16 (torch's _to_copy).
The main loop evaluates pairs with packed f32x2 FMA/MUL (same IEEE rn ops); there 12583039 - j replaces
-(j - 12583039): equal (exact integers) except the sign of a zero difference, which only occurs for x = +-0
and then f = +-0 and ex2(+-0) = 1 either way.

One streaming kernel (one CTA per 512 groups of 8 elements, 256 threads, 2 groups per thread): reads in5
(622 MB) once with 32 B loads, writes the bf16 output (311 MB) once with 16 B stores. Each thread computes
8-element groups with v7 = 0 everywhere; the thread whose group holds column t_i then rewrites that one element
(scalar path, same op sequence as the reference) after its group store (same thread, program order).
"""

import cutlass
import cutlass.cute as cute
import torch
from cutlass import Float32, Int32, Int64, Uint32
from cutlass._mlir import ir
from cutlass._mlir.dialects import llvm
from cutlass.cutlass_dsl import T, dsl_user_op

M = 1024
V = 151936
GPR = V // 8  # 18992 groups of 8 elements per row
NG = M * GPR  # 19447808 groups
THREADS = 256
U = 2  # groups per thread
MINB = 5  # min CTAs per SM (register cap 51; tuned: 518 us vs 539 us at 6)
CHUNK = THREADS * U
NCHUNK = NG // CHUNK
assert NG % CHUNK == 0


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
def ld_f32(addr, *, loc=None, ip=None):
    return Float32(_asm1(T.f32(), [Int64(addr)], "ld.global.nc.f32 $0, [$1];", "=f,l", loc, ip, side=True))


@dsl_user_op
def ld_s64(addr, *, loc=None, ip=None):
    return Int64(_asm1(T.i64(), [Int64(addr)], "ld.global.nc.s64 $0, [$1];", "=l,l", loc, ip, side=True))


@dsl_user_op
def ld_s32(addr, *, loc=None, ip=None):
    return Int32(_asm1(T.i32(), [Int64(addr)], "ld.global.nc.s32 $0, [$1];", "=r,l", loc, ip, side=True))


@dsl_user_op
def ld_v8(addr, *, loc=None, ip=None):
    r = _asmN(8, T.i32(), [Int64(addr)], "ld.global.nc.L1::no_allocate.v8.b32 {$0, $1, $2, $3, $4, $5, $6, $7}, [$8];",
              "=r,=r,=r,=r,=r,=r,=r,=r,l", loc, ip, side=True)
    return [Uint32(v) for v in r]


@dsl_user_op
def st_v4(addr, a, b, c, d, *, loc=None, ip=None):
    _asm1(None, [Int64(addr), Uint32(a), Uint32(b), Uint32(c), Uint32(d)], "st.global.v4.b32 [$0], {$1, $2, $3, $4};",
          "l,r,r,r,r", loc, ip, side=True)


@dsl_user_op
def grad_scalars(a, b64, *, loc=None, ip=None):
    """Returns (gneg, nsum) with gneg = -(a / f32(b)), nsum = -(gneg + 0)."""
    r = _asmN(2, T.f32(), [Float32(a), Int64(b64)],
              "{ .reg .f32 bf, q, s;\n"
              "cvt.rn.f32.s64 bf, $3;\n"
              "div.rn.f32 q, $2, bf;\n"
              "neg.f32 $0, q;\n"
              "add.rn.f32 s, $0, 0f00000000;\n"
              "neg.f32 $1, s; }",
              "=f,=f,f,l", loc, ip)
    return Float32(r[0]), Float32(r[1])


@dsl_user_op
def sel_nsum(t, nsum, *, loc=None, ip=None):
    """t == -100 (ignored row): sum = +0, so -sum = -0."""
    return Float32(_asm1(T.f32(), [Int32(t), Float32(nsum)],
                         "{ .reg .pred p; setp.eq.s32 p, $1, -100; selp.f32 $0, 0f80000000, $2, p; }",
                         "=f,r,f", loc, ip))


def _body8():
    # $4..$11 = 8 raw f32 words, $12 = -sum; outputs $0..$3 = bf16x2 words (v7 = 0)
    txt = ("{ .reg .b64 X<4>, T<4>, J<4>, F<4>, E<4>, S<4>, R<4>, CS, CB, CN, CL, CLO, NS, Z;\n"
           ".reg .b32 a<8>, b<8>, n;\n"
           "mov.b64 CS, {0f437C0000, 0f437C0000};\n"
           "mov.b64 CB, {0f4B400001, 0f4B400001};\n"
           "mov.b64 CN, {0f4B40007F, 0f4B40007F};\n"
           "mov.b64 CL, {0f3FB8AA3B, 0f3FB8AA3B};\n"
           "mov.b64 CLO, {0f32A57060, 0f32A57060};\n"
           "mov.b32 n, $12;\n"
           "mov.b64 NS, {n, n};\n"
           "mov.b64 Z, 0;\n")
    for k in range(4):
        lo, hi = 2 * k, 2 * k + 1
        txt += (
            f"mov.b64 X{k}, {{${4 + lo}, ${4 + hi}}};\n"
            f"fma.rn.sat.f32 a{lo}, ${4 + lo}, 0f3BBB989D, 0f3F000000;\n"
            f"fma.rn.sat.f32 a{hi}, ${4 + hi}, 0f3BBB989D, 0f3F000000;\n"
            f"mov.b64 T{k}, {{a{lo}, a{hi}}};\n"
            f"fma.rm.f32x2 J{k}, T{k}, CS, CB;\n"
            f"sub.rn.f32x2 F{k}, CN, J{k};\n"
            f"fma.rn.f32x2 F{k}, X{k}, CL, F{k};\n"
            f"fma.rn.f32x2 F{k}, X{k}, CLO, F{k};\n"
            f"mov.b64 {{a{lo}, a{hi}}}, J{k};\n"
            f"shl.b32 a{lo}, a{lo}, 23;\n"
            f"shl.b32 a{hi}, a{hi}, 23;\n"
            f"mov.b64 S{k}, {{a{lo}, a{hi}}};\n"
            f"mov.b64 {{b{lo}, b{hi}}}, F{k};\n"
            f"ex2.approx.ftz.f32 b{lo}, b{lo};\n"
            f"ex2.approx.ftz.f32 b{hi}, b{hi};\n"
            f"mov.b64 E{k}, {{b{lo}, b{hi}}};\n"
            f"mul.rn.f32x2 E{k}, S{k}, E{k};\n"
            f"fma.rn.f32x2 R{k}, E{k}, NS, Z;\n"
            f"mov.b64 {{a{lo}, a{hi}}}, R{k};\n"
            f"cvt.rn.bf16x2.f32 ${k}, a{hi}, a{lo};\n"
        )
    return txt + "}"


_BODY8 = _body8()


@dsl_user_op
def compute8(w, nsum, *, loc=None, ip=None):
    r = _asmN(4, T.i32(), [Uint32(v) for v in w] + [Float32(nsum)], _BODY8,
              "=r,=r,=r,=r," + ",".join(["r"] * 8) + ",f", loc, ip)
    return [Uint32(v) for v in r]


@dsl_user_op
def fixup_target(kt, xaddr, oaddr, nsum, gneg, *, loc=None, ip=None):
    """If 0 <= kt < 8: out[t] = bf16(fma(exp(x[t]), -sum, -g)), reference op sequence (scalar).

    Warp-uniform skip (vote.any + bra.uni) so the rare path is not if-converted into the main loop."""
    _asm1(
        None, [Int32(kt), Int64(xaddr), Int64(oaddr), Float32(nsum), Float32(gneg)],
        "{ .reg .pred p, q; .reg .f32 x, t, j, f, e, r; .reg .b32 s; .reg .b16 h;\n"
        "setp.lt.u32 p, $0, 8;\n"
        "vote.sync.any.pred q, p, 0xffffffff;\n"
        "@!q bra.uni FIX_DONE_${:uid};\n"
        "@!p bra FIX_SKIP_${:uid};\n"
        "ld.global.nc.f32 x, [$1];\n"
        "fma.rn.sat.f32 t, x, 0f3BBB989D, 0f3F000000;\n"
        "fma.rm.f32 j, t, 0f437C0000, 0f4B400001;\n"
        "add.rn.f32 f, j, 0fCB40007F;\n"
        "neg.f32 f, f;\n"
        "fma.rn.f32 f, x, 0f3FB8AA3B, f;\n"
        "fma.rn.f32 f, x, 0f32A57060, f;\n"
        "mov.b32 s, j;\n"
        "shl.b32 s, s, 23;\n"
        "mov.b32 e, s;\n"
        "ex2.approx.ftz.f32 f, f;\n"
        "mul.rn.f32 e, e, f;\n"
        "fma.rn.f32 r, e, $3, $4;\n"
        "cvt.rn.bf16.f32 h, r;\n"
        "st.global.b16 [$2], h;\n"
        "FIX_SKIP_${:uid}:\n"
        "bar.warp.sync 0xffffffff;\n"
        "FIX_DONE_${:uid}: }",
        "r,l,l,f,f", loc, ip, side=True,
    )


class LossBwd:
    @cute.jit
    def __call__(self, in0, in1, tgt, x, out, stream):
        self.kernel(in0, in1, tgt, x, out).launch(grid=[NCHUNK, 1, 1], block=[THREADS, 1, 1], stream=stream,
                                                  min_blocks_per_mp=MINB)

    @cute.kernel
    def kernel(self, in0, in1, tgt, x, out):
        tid, _, _ = cute.arch.thread_idx()
        cta, _, _ = cute.arch.block_idx()
        a = ld_f32(in0.iterator.toint())
        b = ld_s64(in1.iterator.toint())
        gneg, nsum_t = grad_scalars(a, b)
        x0 = x.iterator.toint()
        o0 = out.iterator.toint()
        t0 = tgt.iterator.toint()
        ws = []
        ts = []
        rows = []
        for u in cutlass.range_constexpr(U):
            grp = Int64(cta * CHUNK + u * THREADS + tid)
            ws.append(ld_v8(x0 + grp * 32))
        for u in cutlass.range_constexpr(U):
            grp32 = Uint32(cta * CHUNK + u * THREADS + tid)
            row = grp32 // Uint32(GPR)
            rows.append(row)
            ts.append(ld_s32(t0 + Int64(row) * 8))
        for u in cutlass.range_constexpr(U):
            grp32 = Uint32(cta * CHUNK + u * THREADS + tid)
            row = rows[u]
            t = ts[u]
            nsum = sel_nsum(t, nsum_t)
            r = compute8(ws[u], nsum)
            st_v4(o0 + Int64(grp32) * 16, r[0], r[1], r[2], r[3])
            kt = t - Int32(grp32 - row * Uint32(GPR)) * 8
            eidx = Int64(row) * V + Int64(t)
            fixup_target(kt, x0 + eidx * 4, o0 + eidx * 2, nsum, gneg)


_DT = {torch.bfloat16: cutlass.BFloat16, torch.float32: cutlass.Float32, torch.int64: cutlass.Int64}


def _fake(dtype, shape, align=16):
    stride, s = [], 1
    for d in reversed(shape):
        stride.append(s)
        s *= d
    return cute.runtime.make_fake_tensor(_DT[dtype], tuple(shape), stride=tuple(reversed(stride)), assumed_align=align)


_kernel = cute.compile(
    LossBwd(),
    _fake(torch.float32, (1,), align=4),
    _fake(torch.int64, (1,), align=8),
    _fake(torch.int64, (M,), align=8),
    _fake(torch.float32, (M, V), align=32),
    _fake(torch.bfloat16, (M, V), align=16),
    cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
    options="--enable-tvm-ffi",
)


def _ok(t, dtype, shape):
    return (t.dtype == dtype and tuple(t.shape) == shape and t.is_cuda
            and (t.is_contiguous() or t.numel() <= 1))


def run(in0, in1, in2, in3, in4, in5):
    if not (_ok(in0, torch.float32, ()) and _ok(in1, torch.int64, ()) and _ok(in3, torch.int64, (M,))
            and _ok(in5, torch.float32, (M, V)) and tuple(in2.shape) == (M, V) and in5.data_ptr() % 32 == 0
            and in3.data_ptr() % 8 == 0):
        raise ValueError(
            "unsupported inputs: need cuda f32 scalar in0, int64 scalar in1, contiguous 8B-aligned int64 in3 "
            f"[{M}], contiguous 32B-aligned f32 in5 [{M}, {V}] (in2 [{M}, {V}] is not read); got "
            + ", ".join(f"in{i} {t.dtype} {tuple(t.shape)} stride {t.stride()}"
                        for i, t in enumerate((in0, in1, in2, in3, in4, in5))))
    out = torch.empty((M, V), dtype=torch.bfloat16, device=in5.device)
    _kernel(in0.view(1), in1.view(1), in3, in5, out)
    return out
