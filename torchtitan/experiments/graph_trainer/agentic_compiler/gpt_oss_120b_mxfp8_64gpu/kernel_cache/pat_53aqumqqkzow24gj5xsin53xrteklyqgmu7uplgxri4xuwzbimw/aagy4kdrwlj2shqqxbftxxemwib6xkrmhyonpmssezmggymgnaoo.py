"""GPT-OSS loss backward (G-037): div + nll_loss_backward + _log_softmax_backward_data + bf16 cast,
one CuTe DSL kernel, bitwise equal to reference.py.

reference (in0 = grad fp32 [], in1 = count int64 [], in2 = logp [B, C] fp32 (only its shape is
used), in3 = target [B] int64, in4 = total_weight (unused for reduction=sum), in5 = logp [B, C]):
    go  = in0 / fp32(in1)                      div.rn (DivFunctor), cvt.rn.f32.s64
    v7  = nll_loss_backward(sum): zeros, v7[i, t_i] = -go for t_i != -100
    s_i = sum_j v7[i, j]                       one nonzero term: (-go) + (+0 terms) = add.rn(-go, 0);
                                               +0 for ignored rows
    v9  = v7 - exp(in5) * s                    cunn_SoftMaxBackward epilogue, compiled as
                                               fma.rn(-expf(o), s, v7) (checked against reference.py
                                               with cancellation-sensitive inputs)
    out = bf16(v9)                             cvt.rn.bf16 (RNE, NaN -> 0x7fff)
expf is cute.math.exp(fastmath=False), bitwise equal to torch.exp (CUDA expf) on all 2^32 fp32
inputs. Reads only in5 (+ a few scalars): 1.65 GB in, 0.82 GB out per call at B=2048, C=201088.
Grid: flat over B * C / 8 vectors of 8 fp32; B is a runtime value.
"""

import cutlass
import cutlass.cute as cute
import torch
from cutlass import Float32, Int32, Int64, Uint32
from cutlass._mlir import ir
from cutlass._mlir.dialects import llvm
from cutlass.cutlass_dsl import T, dsl_user_op

C = 201088  # classes
NV = C // 8  # 32-byte vectors per row
THREADS = 256
U = 2  # vectors per thread


def _vals(args, loc, ip):
    return [a.ir_value(loc=loc, ip=ip) for a in args]


def _asm1(ty, args, text, cons, loc, ip, side=False):
    return llvm.inline_asm(
        ty, _vals(args, loc, ip), text, cons,
        has_side_effects=side, is_align_stack=False, asm_dialect=llvm.AsmDialect.AD_ATT, loc=loc, ip=ip,
    )


def _asmN(tys, args, text, cons, loc, ip, side=False):
    st = ir.Type.parse("!llvm.struct<(" + ", ".join(str(t) for t in tys) + ")>")
    res = llvm.inline_asm(
        st, _vals(args, loc, ip), text, cons,
        has_side_effects=side, is_align_stack=False, asm_dialect=llvm.AsmDialect.AD_ATT, loc=loc, ip=ip,
    )
    return [llvm.extractvalue(t, res, [k], loc=loc, ip=ip) for k, t in enumerate(tys)]


@dsl_user_op
def ld_stream_f4(addr, *, loc=None, ip=None):
    r = _asmN([T.f32()] * 4, [Int64(addr)], "ld.global.nc.L1::no_allocate.v4.f32 {$0, $1, $2, $3}, [$4];",
              "=f,=f,=f,=f,l", loc, ip, side=True)
    return [Float32(v) for v in r]


@dsl_user_op
def st_v4(addr, w, *, loc=None, ip=None):
    _asm1(None, [Int64(addr)] + [Uint32(x) for x in w],
          "st.global.L1::no_allocate.v4.b32 [$0], {$1, $2, $3, $4};", "l,r,r,r,r", loc, ip, side=True)


@dsl_user_op
def ld_f32(addr, *, loc=None, ip=None):
    return Float32(_asm1(T.f32(), [Int64(addr)], "ld.global.nc.f32 $0, [$1];", "=f,l", loc, ip, side=True))


@dsl_user_op
def ld_s64(addr, *, loc=None, ip=None):
    return Int64(_asm1(T.i64(), [Int64(addr)], "ld.global.nc.s64 $0, [$1];", "=l,l", loc, ip, side=True))


@dsl_user_op
def neg_go(a, n, *, loc=None, ip=None):
    """-(a / fp32(n)): div.rn.f32 with n converted by cvt.rn.f32.s64."""
    return Float32(_asm1(T.f32(), [Float32(a), Int64(n)],
                         "{ .reg .f32 b, q; cvt.rn.f32.s64 b, $2; div.rn.f32 q, $1, b; neg.f32 $0, q; }",
                         "=f,f,l", loc, ip))


@dsl_user_op
def row_sum(g, t, *, loc=None, ip=None):
    """t != -100 ? g + 0 : +0 (the only nonzero term of the row sum, accumulated from +0)."""
    return Float32(_asm1(T.f32(), [Float32(g), Int64(t)],
                         "{ .reg .pred p; .reg .f32 x; add.rn.f32 x, $1, 0f00000000;\n"
                         "setp.ne.s64 p, $2, -100; selp.f32 $0, x, 0f00000000, p; }",
                         "=f,f,l", loc, ip))


@dsl_user_op
def bwd_pair(e0, e1, s, g, k0, *, loc=None, ip=None):
    """bf16x2 of fma.rn(-e, s, v7) for two elements; v7 = g for the element whose offset k (k0, k0 + 1)
    is 0 (k = target - column), else +0."""
    return Uint32(_asm1(T.i32(), [Float32(e0), Float32(e1), Float32(s), Float32(g), Int64(k0)],
                        "{ .reg .pred p, q; .reg .f32 a, b, x, y;\n"
                        "setp.eq.s64 p, $5, 0; setp.eq.s64 q, $5, 1;\n"
                        "selp.f32 a, $4, 0f00000000, p; selp.f32 b, $4, 0f00000000, q;\n"
                        "neg.f32 x, $1; neg.f32 y, $2;\n"
                        "fma.rn.f32 x, x, $3, a; fma.rn.f32 y, y, $3, b;\n"
                        "cvt.rn.bf16x2.f32 $0, y, x; }",
                        "=r,f,f,f,f,l", loc, ip))


class LossBwd:
    @cute.jit
    def __call__(self, gr, cnt, tgt, logp, out, stream):
        nvec = logp.shape[0] * NV
        self.kernel(gr, cnt, tgt, logp, out, nvec).launch(
            grid=[(nvec + THREADS * U - 1) // (THREADS * U), 1, 1], block=[THREADS, 1, 1], stream=stream)

    @cute.kernel
    def kernel(self, gr, cnt, tgt, logp, out, nvec: Int32):
        tid, _, _ = cute.arch.thread_idx()
        bid, _, _ = cute.arch.block_idx()
        g = neg_go(ld_f32(gr.iterator.toint()), ld_s64(cnt.iterator.toint()))
        pl = logp.iterator.toint()
        po = out.iterator.toint()
        pt = tgt.iterator.toint()
        base = bid * (THREADS * U) + tid
        xs = []
        for u in cutlass.range_constexpr(U):
            i = base + u * THREADS
            x = [Float32(0.0)] * 8
            if i < nvec:
                off = Int64(i) * 32
                x = ld_stream_f4(pl + off) + ld_stream_f4(pl + off + 16)
            xs.append(x)
        for u in cutlass.range_constexpr(U):
            i = base + u * THREADS
            if i < nvec:
                row = i // NV
                col = (i - row * NV) * 8
                t = ld_s64(pt + Int64(row) * 8)
                s = row_sum(g, t)
                kk = t - Int64(col)
                e = [cute.math.exp(xs[u][k], fastmath=False) for k in range(8)]
                w = [bwd_pair(e[2 * p], e[2 * p + 1], s, g, kk - 2 * p) for p in range(4)]
                st_v4(po + Int64(i) * 16, w)


_SYM_B = cute.sym_int(symbol="B")
_stream = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)
_fn = cute.compile(
    LossBwd(),
    cute.runtime.make_fake_tensor(cutlass.Float32, (), stride=(), assumed_align=4),
    cute.runtime.make_fake_tensor(cutlass.Int64, (), stride=(), assumed_align=8),
    cute.runtime.make_fake_tensor(cutlass.Int64, (_SYM_B,), stride=(1,), assumed_align=8),
    cute.runtime.make_fake_tensor(cutlass.Float32, (_SYM_B, C), stride=(C, 1), assumed_align=16),
    cute.runtime.make_fake_tensor(cutlass.BFloat16, (_SYM_B, C), stride=(C, 1), assumed_align=16),
    _stream,
    options="--enable-tvm-ffi",
)


def run(in0, in1, in2, in3, in4, in5):
    b = in5.shape[0]
    out = torch.empty((b, C), dtype=torch.bfloat16, device=in5.device)
    if b:
        try:
            # The compiled entry point validates dtypes, devices, shapes, strides and alignment.
            _fn(in0, in1, in3, in5, out)
        except ValueError:
            # Unexpected layouts (non-contiguous / misaligned logp or target): dense copies.
            _fn(in0, in1, in3.contiguous(), in5.clone(memory_format=torch.contiguous_format), out)
    return out
