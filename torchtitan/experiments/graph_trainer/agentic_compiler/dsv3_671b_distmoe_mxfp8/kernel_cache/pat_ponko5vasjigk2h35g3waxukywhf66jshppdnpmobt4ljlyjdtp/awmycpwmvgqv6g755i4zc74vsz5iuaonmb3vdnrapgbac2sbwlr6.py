"""MXFP8 wo weight-gradient GEMM + FSDP reduce-scatter bucket pack (pat_ponk..., DSv3 671B MXFP8).

Reference: v15 = aten._scaled_mm_v2(in0, in1, [in2], [3], [1], [in3], [3], [1], None, bf16)
-> [7168, 16384] bf16, then bucketing._pre_bucket_reduce_scatter([in4..in9, v15, in10..in14],
256, [3]) = torch._chunk_cat(inputs, 0, 256).flatten(): every input is split into 256 chunks
along dim 0 (in7 [576, 7168]: 3 rows per chunk, so chunks 192..255 are zero rows), and output
chunk c (1831040 B) is the concatenation of every input's chunk c.

The GEMM is the identical aten call on the identical input tensors (same overload, same
argument list, output allocated by the op), so cuBLASLt picks the same algorithm. The pack is
one CuTe DSL kernel. Byte layout of output chunk c (offset +size):
  in4 0 +86016 | in5 86016 +12 | in6 86028 +294912 | in7 380940 +43008 | in8 423948 +4 |
  in9 423952 +131072 | v15 555024 +917504 | in10 1472528 +56 | in11 1472584 +56 |
  in12 1472640 +14336 | in13 1486976 +229376 | in14 1716352 +114688
Regions whose destination and source are 16 B aligned (in4, in9, v15, in12, in13, in14) are
plain 16 B vector copies (contiguous source, one destination run per chunk). [in5 in6 in7 in8]
and [in10 in11] start and end 16 B aligned in the output but their pieces do not: each 16 B
output vector there is assembled from 4 B source words. Interior vectors of in6 / in7 (region
word offset 3 mod 4) are source words 4q-3..4q, i.e. words 1..3 of aligned vector q-1 plus
word 0 of vector q (two 16 B loads, the second mostly an L1 hit); the few vectors that straddle
a piece boundary load 4 single words. Zero padding (in7, chunks >= 192) is a predicated-off
load that yields +0.0 words. v15 tiles are scheduled first: the GEMM just wrote v15, so part
of it is still in L2.
"""

import cutlass
import cutlass.cute as cute
import torch
from cutlass import Int32, Int64, Uint32
from cutlass._mlir import ir
from cutlass._mlir.dialects import llvm
from cutlass.cutlass_dsl import T, dsl_user_op

NCH = 256  # num_chunks (reduce-scatter group size)
NTHR = 256
UNROLL = 4
TILE = NTHR * UNROLL  # 16 B output vectors per CTA
MIN_CTAS = 8  # 32 registers per thread -> 8 CTAs (2048 threads) per SM; 34 registers (6 CTAs) cost ~1.5 us

# Bucket inputs in reference order.
SHAPES = [
    ("in4", (1536, 7168)), ("in5", (1536,)), ("in6", (24576, 1536)), ("in7", (576, 7168)),
    ("in8", (512,)), ("in9", (32768, 512)), ("v15", (7168, 16384)), ("in10", (7168,)),
    ("in11", (7168,)), ("in12", (256, 7168)), ("in13", (512, 8, 7168)), ("in14", (7168, 2048)),
]
# Required source alignment: 16 B where the kernel issues 16 B loads, 4 B for word pieces.
ALIGN = [16, 4, 16, 16, 4, 16, 16, 4, 4, 16, 16, 16]

PIECES = []  # (dst offset in chunk, bytes per chunk, chunks backed by source data)
_off = 0
for _name, _shape in SHAPES:
    _trail = 1
    for _s in _shape[1:]:
        _trail *= _s
    _rpc = -(-_shape[0] // NCH)  # rows per chunk (rounded up, zero padded)
    _L = _rpc * _trail * 2
    _src = _shape[0] * _trail * 2
    assert _src % _L == 0  # padding is whole chunks only (true for every input here)
    PIECES.append((_off, _L, _src // _L))
    _off += _L
CHUNK = _off
assert CHUNK == 1831040 and CHUNK % 16 == 0
CHUNK_VEC = CHUNK // 16
NOUT = CHUNK * NCH // 2  # 234373120 bf16

# Regions (16 B aligned output ranges), scheduled in this order: v15 first.
REGIONS = [("vec", [6]), ("vec", [0]), ("gather", [1, 2, 3, 4]), ("vec", [5]), ("gather", [7, 8]),
           ("vec", [9]), ("vec", [10]), ("vec", [11])]
REG = []  # (kind, pieces, dst vector offset in chunk, vectors per chunk, first tile, num tiles, fast)
_t = 0
for _kind, _p in REGIONS:
    _o = PIECES[_p[0]][0]
    _L = sum(PIECES[i][1] for i in _p)
    assert _o % 16 == 0 and _L % 16 == 0
    assert [PIECES[i][0] for i in _p] == [_o + sum(PIECES[i][1] for i in _p[:k]) for k in range(len(_p))]
    _fast = []  # pieces whose region word offset is 3 mod 4: (piece, region word offset, q0, q1)
    _w0 = 0
    for _i in _p:
        _nw = PIECES[_i][1] // 4
        if _kind == "gather" and _w0 % 4 == 3 and _nw >= 8:
            _fast.append((_i, _w0, (_w0 + 1) // 4, (_w0 + _nw) // 4))
        _w0 += _nw
    if _kind == "vec":
        assert len(_p) == 1 and PIECES[_p[0]][2] == NCH and ALIGN[_p[0]] == 16
    _nt = -(-(NCH * _L // 16) // TILE)
    REG.append((_kind, _p, _o // 16, _L // 16, _t, _nt, _fast))
    _t += _nt
NTILES = _t


def _vals(args, loc, ip):
    return [a.ir_value(loc=loc, ip=ip) for a in args]


@dsl_user_op
def ld_v4(addr, pred, cache: str = ".L1::no_allocate", *, loc=None, ip=None):
    """16 B load; all-zero words when pred == 0."""
    st = ir.Type.parse("!llvm.struct<(i32, i32, i32, i32)>")
    res = llvm.inline_asm(
        st, _vals([Int64(addr), Int32(pred)], loc, ip),
        "{ .reg .pred p; setp.ne.b32 p, $5, 0; mov.b32 $0, 0; mov.b32 $1, 0; mov.b32 $2, 0; mov.b32 $3, 0;\n"
        f"@p ld.global.nc{cache}.v4.b32 {{$0, $1, $2, $3}}, [$4]; }}",
        "=r,=r,=r,=r,l,r", has_side_effects=False, is_align_stack=False, asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc, ip=ip,
    )
    return [Uint32(llvm.extractvalue(T.i32(), res, [k], loc=loc, ip=ip)) for k in range(4)]


@dsl_user_op
def ld_b32(addr, pred, *, loc=None, ip=None):
    """4 B load; zero when pred == 0."""
    return Uint32(llvm.inline_asm(
        T.i32(), _vals([Int64(addr), Int32(pred)], loc, ip),
        "{ .reg .pred p; setp.ne.b32 p, $2, 0; mov.b32 $0, 0;\n@p ld.global.nc.b32 $0, [$1]; }",
        "=r,l,r", has_side_effects=False, is_align_stack=False, asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc, ip=ip,
    ))


@dsl_user_op
def st_v4(addr, v, pred, *, loc=None, ip=None):
    llvm.inline_asm(
        None, _vals([Int64(addr)] + [Uint32(x) for x in v] + [Int32(pred)], loc, ip),
        "{ .reg .pred p; setp.ne.b32 p, $5, 0;\n@p st.global.v4.b32 [$0], {$1, $2, $3, $4}; }",
        "l,r,r,r,r,r", has_side_effects=True, is_align_stack=False, asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc, ip=ip,
    )


@cute.kernel
def _pack_kernel(s0: cute.Tensor, s1: cute.Tensor, s2: cute.Tensor, s3: cute.Tensor, s4: cute.Tensor,
                 s5: cute.Tensor, s6: cute.Tensor, s7: cute.Tensor, s8: cute.Tensor, s9: cute.Tensor,
                 s10: cute.Tensor, s11: cute.Tensor, out: cute.Tensor):
    tid, _, _ = cute.arch.thread_idx()
    bid, _, _ = cute.arch.block_idx()
    srcs = [s.iterator.toint() for s in (s0, s1, s2, s3, s4, s5, s6, s7, s8, s9, s10, s11)]
    o0 = out.iterator.toint()
    for r in cutlass.range_constexpr(len(REG)):
        kind, pieces, doff, lv, t0, nt, fast = REG[r]
        nv = NCH * lv  # output vectors of this region
        if bid >= t0 and bid < t0 + nt:
            if cutlass.const_expr(kind == "vec"):
                src = srcs[pieces[0]]
                vals = []
                for u in cutlass.range_constexpr(UNROLL):
                    g = (bid - t0) * TILE + u * NTHR + tid
                    vals.append(ld_v4(src + Int64(g) * 16, Int32(g < nv)))
                for u in cutlass.range_constexpr(UNROLL):
                    g = (bid - t0) * TILE + u * NTHR + tid
                    c = g // lv
                    st_v4(o0 + (Int64(c) * CHUNK_VEC + doff + Int64(g - c * lv)) * 16, vals[u], Int32(g < nv))
            else:
                for u in cutlass.range_constexpr(UNROLL):
                    g = (bid - t0) * TILE + u * NTHR + tid
                    inb = Int32(g < nv)
                    c = g // lv
                    q = g - c * lv  # output vector within the region's chunk run
                    dst = o0 + (Int64(c) * CHUNK_VEC + doff + Int64(q)) * 16
                    hit = Int32(0)
                    for i, w0, q0, q1 in fast:
                        hit = hit | Int32((q >= q0) & (q < q1))
                    if hit != 0:
                        # Output words 4q..4q+3 = piece words 4q-w0 .. 4q-w0+3 (4q-w0 = 1 mod 4).
                        a = Int64(0)
                        ok = Int32(0)
                        for i, w0, q0, q1 in fast:
                            m = (q >= q0) & (q < q1)
                            base = srcs[i] + Int64(c) * PIECES[i][1] + Int64(4 * q - w0 - 1) * 4
                            a = Int64(cutlass.select_(m, base, a))
                            if cutlass.const_expr(PIECES[i][2] < NCH):
                                m = m & (c < PIECES[i][2])
                            ok = ok | Int32(m)
                        ok = ok & inb
                        x = ld_v4(a, ok, "")
                        y = ld_v4(a + 16, ok, "")
                        st_v4(dst, [x[1], x[2], x[3], y[0]], inb)
                    else:
                        words = []
                        for t in cutlass.range_constexpr(4):
                            k = 4 * q + t  # region word
                            a = Int64(0)
                            ok = Int32(0)
                            w0 = 0
                            for i in pieces:
                                nw = PIECES[i][1] // 4
                                m = (k >= w0) & (k < w0 + nw)
                                base = srcs[i] + Int64(c) * PIECES[i][1] + Int64(k - w0) * 4
                                a = Int64(cutlass.select_(m, base, a))
                                if cutlass.const_expr(PIECES[i][2] < NCH):
                                    m = m & (c < PIECES[i][2])
                                ok = ok | Int32(m)
                                w0 += nw
                            words.append(ld_b32(a, ok & inb))
                        st_v4(dst, words, inb)


@cute.jit
def _launch(s0: cute.Tensor, s1: cute.Tensor, s2: cute.Tensor, s3: cute.Tensor, s4: cute.Tensor,
            s5: cute.Tensor, s6: cute.Tensor, s7: cute.Tensor, s8: cute.Tensor, s9: cute.Tensor,
            s10: cute.Tensor, s11: cute.Tensor, out: cute.Tensor, stream):
    _pack_kernel(s0, s1, s2, s3, s4, s5, s6, s7, s8, s9, s10, s11, out).launch(
        grid=[NTILES, 1, 1], block=[NTHR, 1, 1], min_blocks_per_mp=MIN_CTAS, stream=stream)


def _fake(shape, align):
    stride, acc = [], 1
    for s in reversed(shape):
        stride.append(acc)
        acc *= s
    return cute.runtime.make_fake_tensor(cutlass.BFloat16, tuple(shape), stride=tuple(reversed(stride)),
                                         assumed_align=align)


_fn = cute.compile(
    _launch, *[_fake(s, a) for (_, s), a in zip(SHAPES, ALIGN)], _fake((NOUT,), 16),
    cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
    options="--enable-tvm-ffi",
)


def run(in0, in1, in2, in3, in4, in5, in6, in7, in8, in9, in10, in11, in12, in13, in14):
    # Byte-for-byte the reference's GEMM call (same overload, arguments and input tensors).
    v15 = torch.ops.aten._scaled_mm_v2.default(in0, in1, [in2], [3], [1], [in3], [3], [1], None, torch.bfloat16)
    srcs = [in4, in5, in6, in7, in8, in9, v15, in10, in11, in12, in13, in14]
    for t, (name, shape), a in zip(srcs, SHAPES, ALIGN):
        if (t.dtype != torch.bfloat16 or tuple(t.shape) != shape or not t.is_contiguous()
                or t.data_ptr() % a or t.device != v15.device):
            raise ValueError(f"{name}: expected contiguous {a} B aligned bf16 {shape} on {v15.device}, got "
                             f"{t.dtype} {tuple(t.shape)} stride {t.stride()} ptr % 16 = {t.data_ptr() % 16}")
    out = torch.empty((NOUT,), dtype=torch.bfloat16, device=v15.device)
    _fn(*srcs, out)
    return out
