"""Routed-expert FSDP reduce-scatter bucket pack (pat_czaf..., DSv3 671B DistMoE MXFP8).

Reference: bucketing._pre_bucket_reduce_scatter([in1, in0.reshape(4, 2, 2048, 7168)], 4)
= torch._chunk_cat([in1, v2], 0, 4).flatten(). Both inputs have dim 0 == 4 == num_chunks, so
there is no padding: output chunk c (88080384 B) = in1[c] (29360128 B) ++ in0[c] (58720256 B).
That is 8 contiguous, 16 B aligned segment copies; one CuTe DSL kernel copies them with 16 B
vector loads/stores (one tile of 256 threads x 4 vectors per CTA, every tile inside one input).
"""

import cutlass
import cutlass.cute as cute
import torch
from cutlass import Int32, Int64, Uint32
from cutlass._mlir import ir
from cutlass._mlir.dialects import llvm
from cutlass.cutlass_dsl import T, dsl_user_op

NCH = 4
IN0_SHAPE = (4, 4096, 7168)  # w13 grad
IN1_SHAPE = (4, 7168, 2048)  # w2 grad
V1 = 7168 * 2048 * 2 // 16  # 16 B vectors of in1 per chunk: 1835008
V0 = 4096 * 7168 * 2 // 16  # 16 B vectors of in0 per chunk: 3670016
CHUNK_VEC = V1 + V0
NOUT = NCH * CHUNK_VEC * 8  # 176160768 bf16
NTHR = 256
UNROLL = 4
TILE = NTHR * UNROLL
MIN_CTAS = 8  # keep <= 32 registers/thread: 8 CTAs (2048 threads) per SM
# Regions (source, vectors per chunk, destination vector offset in a chunk), in0 scheduled first.
REG = []
_t = 0
for _src, _lv, _doff in ((0, V0, V1), (1, V1, 0)):
    assert (NCH * _lv) % TILE == 0
    _nt = NCH * _lv // TILE
    REG.append((_src, _lv, _doff, _t, _nt))
    _t += _nt
NTILES = _t


def _vals(args, loc, ip):
    return [a.ir_value(loc=loc, ip=ip) for a in args]


@dsl_user_op
def ld_v4(addr, *, loc=None, ip=None):
    st = ir.Type.parse("!llvm.struct<(i32, i32, i32, i32)>")
    res = llvm.inline_asm(
        st, _vals([Int64(addr)], loc, ip), "ld.global.nc.L1::no_allocate.v4.b32 {$0, $1, $2, $3}, [$4];",
        "=r,=r,=r,=r,l", has_side_effects=False, is_align_stack=False, asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc, ip=ip,
    )
    return [Uint32(llvm.extractvalue(T.i32(), res, [k], loc=loc, ip=ip)) for k in range(4)]


@dsl_user_op
def st_v4(addr, v, *, loc=None, ip=None):
    llvm.inline_asm(
        None, _vals([Int64(addr)] + [Uint32(x) for x in v], loc, ip),
        "st.global.v4.b32 [$0], {$1, $2, $3, $4};", "l,r,r,r,r",
        has_side_effects=True, is_align_stack=False, asm_dialect=llvm.AsmDialect.AD_ATT, loc=loc, ip=ip,
    )


@cute.kernel
def _pack_kernel(in0: cute.Tensor, in1: cute.Tensor, out: cute.Tensor):
    tid, _, _ = cute.arch.thread_idx()
    bid, _, _ = cute.arch.block_idx()
    srcs = [in0.iterator.toint(), in1.iterator.toint()]
    o0 = out.iterator.toint()
    for r in cutlass.range_constexpr(len(REG)):
        si, lv, doff, t0, nt = REG[r]
        if bid >= t0 and bid < t0 + nt:
            vals = []
            for u in cutlass.range_constexpr(UNROLL):
                g = (bid - t0) * TILE + u * NTHR + tid
                vals.append(ld_v4(srcs[si] + Int64(g) * 16))
            for u in cutlass.range_constexpr(UNROLL):
                g = (bid - t0) * TILE + u * NTHR + tid
                c = g // lv
                st_v4(o0 + (Int64(c) * CHUNK_VEC + doff + Int64(g - c * lv)) * 16, vals[u])


@cute.jit
def _launch(in0: cute.Tensor, in1: cute.Tensor, out: cute.Tensor, stream):
    _pack_kernel(in0, in1, out).launch(grid=[NTILES, 1, 1], block=[NTHR, 1, 1], min_blocks_per_mp=MIN_CTAS,
                                       stream=stream)


def _fake(shape):
    stride, acc = [], 1
    for s in reversed(shape):
        stride.append(acc)
        acc *= s
    return cute.runtime.make_fake_tensor(cutlass.BFloat16, tuple(shape), stride=tuple(reversed(stride)),
                                         assumed_align=16)


_fn = cute.compile(
    _launch, _fake(IN0_SHAPE), _fake(IN1_SHAPE), _fake((NOUT,)),
    cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
    options="--enable-tvm-ffi",
)


def run(in0, in1):
    for name, t, shape in (("in0", in0, IN0_SHAPE), ("in1", in1, IN1_SHAPE)):
        if (t.dtype != torch.bfloat16 or tuple(t.shape) != shape or not t.is_contiguous()
                or t.data_ptr() % 16 or t.device != in0.device):
            raise ValueError(f"{name}: expected contiguous 16 B aligned bf16 {shape} on {in0.device}, got "
                             f"{t.dtype} {tuple(t.shape)} stride {t.stride()} ptr % 16 = {t.data_ptr() % 16}")
    out = torch.empty((NOUT,), dtype=torch.bfloat16, device=in0.device)
    _fn(in0, in1, out)
    return out
