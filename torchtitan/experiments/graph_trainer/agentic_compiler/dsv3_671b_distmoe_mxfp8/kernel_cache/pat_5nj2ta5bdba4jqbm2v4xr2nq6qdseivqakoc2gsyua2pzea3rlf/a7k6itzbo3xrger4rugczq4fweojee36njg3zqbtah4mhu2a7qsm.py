"""MXFP8 GEMM + 8-row block permutation (pat_5nj2..., DSv3 671B DistMoE MXFP8).

Reference: v4 = aten._scaled_mm_v2(in0, in1, [in2], [3], [1], [in3], [3], [1], None, bf16)
-> [4096, 7168] bf16, then cat(split(v4.reshape(2, 2048, 7168), 8, dim=1), dim=0) -> [512, 8, 7168]:
out block 2 i + b (8 rows, 114688 B) = v4 block 256 b + i.

The GEMM is the identical aten call on the identical input tensors (same op overload, same
argument list, output allocated by the op itself), so cuBLASLt sees the same problem and picks
the same algorithm. The aten.cat (256 strided inputs, ~183 us) is replaced by an in-place block
permutation of the GEMM's own output, which is then returned as the [512, 8, 7168] result
(v4 is a temporary of this region; nothing else references it).

Permutation: source block j goes to block p(j) = 2 (j mod 256) + j div 256 = 2 j mod 511
(j < 511), p(511) = 511. Multiplication by 2 mod 511 has order 9 (2^9 = 512 = 1 mod 511), so the
cycles are {0}, {511}, two 3-cycles ({73, 146, 292}, {219, 438, 365}) and 56 9-cycles. One thread
owns one 16 B column of every block of one cycle: it loads the 9 blocks j_k = r 2^k mod 511
(a 3-cycle simply repeats), then stores block j_k's data to block 2 j_k mod 511 = j_{k+1}. All
of a thread's loads precede its stores and no two threads touch the same address, so the
in-place permutation needs no synchronization. Working set: v4 only (58.7 MB, L2 resident right
after the GEMM), no second 58.7 MB buffer.
"""


import cutlass
import cutlass.cute as cute
import torch
from cutlass import Int32, Int64, Uint32
from cutlass._mlir import ir
from cutlass._mlir.dialects import llvm
from cutlass.cutlass_dsl import T, dsl_user_op

M, N = 4096, 7168
BLK_VEC = 8 * N * 2 // 16  # 16 B vectors per 8-row block: 7168
NBLK = M // 8  # 512
MOD = NBLK - 1  # 511
NTHR = 128  # 128 vs 256 vs 512 threads: 13.1 vs 13.4 vs 14.0 us after the GEMM (nsys)
assert BLK_VEC % NTHR == 0


def _cycle_reps():
    seen, reps = {0, MOD}, []
    for j in range(1, MOD):
        if j in seen:
            continue
        reps.append(j)
        k = j
        while k not in seen:
            seen.add(k)
            k = 2 * k % MOD
    return reps


REPS = _cycle_reps()  # 58 cycle representatives (56 9-cycles, 2 3-cycles)
assert len(REPS) == 58 and all(pow(2, 9, MOD) * r % MOD == r for r in REPS)
NCYC = len(REPS)


def _vals(args, loc, ip):
    return [a.ir_value(loc=loc, ip=ip) for a in args]


@dsl_user_op
def ld_v4(addr, *, loc=None, ip=None):
    st = ir.Type.parse("!llvm.struct<(i32, i32, i32, i32)>")
    # Plain (coherent) load: this kernel writes the same buffer, so no .nc.
    res = llvm.inline_asm(
        st, _vals([Int64(addr)], loc, ip), "ld.global.L1::no_allocate.v4.b32 {$0, $1, $2, $3}, [$4];",
        "=r,=r,=r,=r,l", has_side_effects=True, is_align_stack=False, asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc, ip=ip,
    )
    return [Uint32(llvm.extractvalue(T.i32(), res, [k], loc=loc, ip=ip)) for k in range(4)]


@dsl_user_op
def ld_b32(addr, *, loc=None, ip=None):
    return Int32(llvm.inline_asm(
        T.i32(), _vals([Int64(addr)], loc, ip), "ld.global.nc.b32 $0, [$1];", "=r,l",
        has_side_effects=False, is_align_stack=False, asm_dialect=llvm.AsmDialect.AD_ATT, loc=loc, ip=ip,
    ))


@dsl_user_op
def st_v4(addr, v, *, loc=None, ip=None):
    llvm.inline_asm(
        None, _vals([Int64(addr)] + [Uint32(x) for x in v], loc, ip),
        "st.global.v4.b32 [$0], {$1, $2, $3, $4};", "l,r,r,r,r",
        has_side_effects=True, is_align_stack=False, asm_dialect=llvm.AsmDialect.AD_ATT, loc=loc, ip=ip,
    )


@cute.kernel
def _shuffle_kernel(buf: cute.Tensor, reps: cute.Tensor):
    tid, _, _ = cute.arch.thread_idx()
    bx, cy, _ = cute.arch.block_idx()
    b0 = buf.iterator.toint()
    r = ld_b32(reps.iterator.toint() + Int64(cy) * 4)
    col = Int64(bx * NTHR + tid) * 16
    blocks = []
    j = r
    for k in cutlass.range_constexpr(9):
        blocks.append(j)
        j = (j * 2) % MOD
    vals = []
    for k in cutlass.range_constexpr(9):
        vals.append(ld_v4(b0 + Int64(blocks[k]) * (BLK_VEC * 16) + col))
    for k in cutlass.range_constexpr(9):
        st_v4(b0 + Int64(blocks[(k + 1) % 9]) * (BLK_VEC * 16) + col, vals[k])


@cute.jit
def _launch(buf: cute.Tensor, reps: cute.Tensor, stream):
    _shuffle_kernel(buf, reps).launch(grid=[BLK_VEC // NTHR, NCYC, 1], block=[NTHR, 1, 1], stream=stream)


_fn = cute.compile(
    _launch,
    cute.runtime.make_fake_tensor(cutlass.BFloat16, (M, N), stride=(N, 1), assumed_align=16),
    cute.runtime.make_fake_tensor(cutlass.Int32, (NCYC,), stride=(1,), assumed_align=4),
    cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
    options="--enable-tvm-ffi",
)
_REPS = {}


def run(in0, in1, in2, in3):
    # Byte-for-byte the reference's GEMM call (same overload, arguments and input tensors).
    v4 = torch.ops.aten._scaled_mm_v2.default(in0, in1, [in2], [3], [1], [in3], [3], [1], None, torch.bfloat16)
    if v4.shape != (M, N) or v4.stride() != (N, 1) or v4.data_ptr() % 16:
        raise ValueError(f"unexpected GEMM output {tuple(v4.shape)} {v4.stride()}")
    reps = _REPS.get(v4.device)
    if reps is None:
        reps = _REPS[v4.device] = torch.tensor(REPS, dtype=torch.int32, device=v4.device)
    _fn(v4, reps)
    return v4.view(NBLK, 8, N)
