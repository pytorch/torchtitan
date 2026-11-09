"""FSDP all-gather bucket prep (fp32 params -> bf16, rank-0 slot of the bucket) as one CuTe DSL kernel.

Reference: v8 = bf16(in0), v10 = bf16(in1), v12 = bf16(in2) (_to_copy, then zero-width constant_pad_nd
copies), then bucketing._pre_bucket_all_gather([in3, in4, v8, v10, v12, in5, in6], group_size=256, bf16,
out dtypes all bf16 (_ALL_DTYPES[1] on this build), rank=0, groups [[0, 1, 5, 6], [2, 3, 4]]):
  buf = torch.empty(282658 * 256, bf16); slot = buf[0:282658]
  slot = cat(bf16(in3).flat, bf16(in4).flat, v8, v10, v12.flat, bf16(in5), bf16(in6))   (_foreach_copy_)
Outputs: (buf, buf[0:282658]); the second is a view of the first (same storage, offset 0). Only the rank-0
slot (the first 282658 elements) is written; the remaining 72077790 elements are left exactly as
torch.empty returned them (the all-gather fills them later), as in the reference.

Every fp32 -> bf16 conversion is cvt.rn.bf16x2.f32 (round to nearest even, subnormals kept, overflow to inf,
NaN -> 0x7FFF), which is what aten's CUDA copy (c10::BFloat16(float) -> __float2bfloat16 on sm_80+) does.

Slot layout (bf16 element offsets): in3 [0, 147456), in4 [147456, 278528), in0 278528, in1 278529,
in2 [278530, 282626), in5 [282626, 282642), in6 [282642, 282658).
Work split: blocks [0, 136) convert 8 elements per thread over [0, 278528) (two 16-byte loads of in3 or in4,
one 16-byte store); blocks [136, 145) write the tail [278528, 282658) as bf16 pairs (one 4-byte store per
thread; the pair (in0, in1) is the first, every other pair lies inside one tensor).
"""

import cutlass
import cutlass.cute as cute
import torch
from cutlass import Int32, Int64, Uint32
from cutlass._mlir import ir
from cutlass._mlir.dialects import llvm
from cutlass.cutlass_dsl import T, dsl_user_op

GROUP_SIZE = 256
N0, N1, N2, N3, N4, N5, N6 = 1, 1, 4096, 36 * 4096, 16 * 8192, 16, 16
SLOT = N3 + N4 + N0 + N1 + N2 + N5 + N6  # 282658
TOTAL = SLOT * GROUP_SIZE  # 72360448
MAIN = N3 + N4  # 278528 elements converted 8 per thread
TAIL = SLOT - MAIN  # 4130 elements written as pairs
NTH = 256
MAIN_BLOCKS = MAIN // 8 // NTH  # 136
TAIL_PAIRS = TAIL // 2  # 2065
TAIL_BLOCKS = (TAIL_PAIRS + NTH - 1) // NTH  # 9
assert MAIN % (8 * NTH) == 0 and N3 % 8 == 0 and TAIL % 2 == 0 and N2 % 2 == 0 and N5 % 2 == 0
IN_SHAPES = ((1,), (1,), (1, 4096), (36, 4096), (16, 8192), (16,), (16,))
# Tail segment starts (relative to MAIN): in2 at 2, in5 at 2 + N2, in6 at 2 + N2 + N5.
T2, T5, T6 = 2, 2 + N2, 2 + N2 + N5


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
def ld_v4(addr, *, loc=None, ip=None):
    r = _asmN(4, T.i32(), [Int64(addr)], "ld.global.v4.b32 {$0, $1, $2, $3}, [$4];", "=r,=r,=r,=r,l", loc, ip)
    return [Uint32(v) for v in r]


@dsl_user_op
def ld_b32(addr, *, loc=None, ip=None):
    return Uint32(_asm1(T.i32(), [Int64(addr)], "ld.global.b32 $0, [$1];", "=r,l", loc, ip))


@dsl_user_op
def st_b32(addr, a, *, loc=None, ip=None):
    _asm1(None, [Int64(addr), Uint32(a)], "st.global.b32 [$0], $1;", "l,r", loc, ip, side=True)


@dsl_user_op
def st_v4(addr, a, b, c, d, *, loc=None, ip=None):
    _asm1(None, [Int64(addr), Uint32(a), Uint32(b), Uint32(c), Uint32(d)],
          "st.global.v4.b32 [$0], {$1, $2, $3, $4};", "l,r,r,r,r", loc, ip, side=True)


@dsl_user_op
def cvt_bf16x2(lo, hi, *, loc=None, ip=None):
    """Two fp32 words -> one bf16x2 word (lo in the low half), round to nearest even."""
    return Uint32(_asm1(T.i32(), [Uint32(lo), Uint32(hi)],
                        "{ .reg .f32 a, b; mov.b32 a, $1; mov.b32 b, $2; cvt.rn.bf16x2.f32 $0, b, a; }",
                        "=r,r,r", loc, ip))


class BucketPrep:
    @cute.jit
    def __call__(self, a0, a1, a2, a3, a4, a5, a6, out, stream):
        self.kernel(a0, a1, a2, a3, a4, a5, a6, out).launch(
            grid=[MAIN_BLOCKS + TAIL_BLOCKS, 1, 1], block=[NTH, 1, 1], stream=stream)

    @cute.kernel
    def kernel(self, a0, a1, a2, a3, a4, a5, a6, out):
        tid, _, _ = cute.arch.thread_idx()
        b, _, _ = cute.arch.block_idx()
        o = out.iterator.toint()
        if b < MAIN_BLOCKS:
            t = b * NTH + tid
            e0 = Int64(t) * 8
            src = a3.iterator.toint() + e0 * 4
            if t >= N3 // 8:
                src = a4.iterator.toint() + (e0 - N3) * 4
            w = ld_v4(src) + ld_v4(src + 16)
            y = [cvt_bf16x2(w[2 * k], w[2 * k + 1]) for k in range(4)]
            st_v4(o + e0 * 2, y[0], y[1], y[2], y[3])
        else:
            d = (b - MAIN_BLOCKS) * NTH * 2 + tid * 2  # element offset inside the tail
            if d < TAIL:
                plo = a2.iterator.toint() + Int64(d - T2) * 4
                if d >= T5:
                    plo = a5.iterator.toint() + Int64(d - T5) * 4
                if d >= T6:
                    plo = a6.iterator.toint() + Int64(d - T6) * 4
                phi = plo + 4
                if d == 0:
                    plo = a0.iterator.toint()
                    phi = a1.iterator.toint()
                st_b32(o + Int64(MAIN + d) * 2, cvt_bf16x2(ld_b32(plo), ld_b32(phi)))


def _compile():
    def fake(dtype, n, align):
        return cute.runtime.make_fake_tensor(dtype, (n,), stride=(1,), assumed_align=align)

    f32 = cutlass.Float32
    return cute.compile(
        BucketPrep(),
        fake(f32, N0, 4), fake(f32, N1, 4), fake(f32, N2, 4), fake(f32, N3, 16), fake(f32, N4, 16),
        fake(f32, N5, 4), fake(f32, N6, 4),
        fake(cutlass.BFloat16, SLOT, 16),
        cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
        options="--enable-tvm-ffi",
    )


_fn = _compile()


def run(in0, in1, in2, in3, in4, in5, in6):
    ins = (in0, in1, in2, in3, in4, in5, in6)
    for k, (x, shp) in enumerate(zip(ins, IN_SHAPES)):
        if x.dtype != torch.float32 or tuple(x.shape) != shp or x.device.type != "cuda":
            raise ValueError(f"in{k}: expected cuda fp32 {shp}, got {x.dtype} {tuple(x.shape)} on {x.device}")
    flat = [x.view(-1) if x.is_contiguous() else x.reshape(-1) for x in ins]
    # The kernel takes unit-stride rows, and reads in3 / in4 with 16-byte loads. Strided or misaligned
    # inputs are copied first (not reached by the spec: fresh contiguous fp32 parameter shards).
    for k, x in enumerate(flat):
        if x.stride(0) != 1 or (k in (3, 4) and x.data_ptr() % 16):
            flat[k] = x.clone(memory_format=torch.contiguous_format)
    buf = torch.empty(TOTAL, dtype=torch.bfloat16, device=in0.device)
    slot = buf[:SLOT]
    _fn(*flat, slot)
    return buf, slot
