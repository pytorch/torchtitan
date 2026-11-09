"""FSDP reduce-scatter bucket pack of the routed-expert grads (Qwen3 235B-A22B) as one CuTe DSL copy.

reference: v12 = cat(split(w2g [4,4096,1536], 512, dim=1)) -> [32,512,1536] (row s*4+e = w2g[e, s*512:+512]),
v22 = cat(split(w13g.view(4,2,1536,4096), 192, dim=2)) -> [32,2,192,4096] (row s*4+e = w13g[e, h*1536+s*192:+192]
for h=0,1), out = _pre_bucket_reduce_scatter([v12, v22], 8) = torch._chunk_cat([v12, v22], 0, 8).flatten():
shard s of 8 is [v12[4s:4s+4].flatten(), v22[4s:4s+4].flatten()].

So the output is 8 shards x 12 contiguous blocks of 786432 bf16 (1.5 MiB), and every block is one contiguous
1.5 MiB range of an input:
  block j < 4 (e = j):            w2g  elements [e*6291456 + s*786432, +786432)
  block j >= 4 (e, h = divmod(j-4, 2)): w13g elements [e*12582912 + h*6291456 + s*786432, +786432)
A pure byte copy (bitwise exact by construction): CTA (q, b) copies bytes [q*32 KiB, +32 KiB) of output block b;
each of 256 threads issues UNROLL 32-byte loads, then UNROLL 32-byte stores.
"""

import cutlass
import cutlass.cute as cute
import torch
from cutlass import Int32, Int64, Uint32
from cutlass._mlir import ir
from cutlass._mlir.dialects import llvm
from cutlass.cutlass_dsl import T, dsl_user_op

E = 4
W13_SHAPE = (4, 3072, 4096)
W2_SHAPE = (4, 4096, 1536)
NSHARD = 8
BLK_BYTES = 786432 * 2  # one contiguous block (1.5 MiB)
BLOCKS_PER_SHARD = 12  # 4 from w2g, 8 from w13g
NBLK = NSHARD * BLOCKS_PER_SHARD
OUT_NUMEL = NBLK * BLK_BYTES // 2
THREADS = 256
VEC = 32  # bytes per access
UNROLL = 4
CTA_BYTES = THREADS * VEC * UNROLL
CTAS_PER_BLK = BLK_BYTES // CTA_BYTES
assert BLK_BYTES % CTA_BYTES == 0
assert OUT_NUMEL == 75497472


def _vals(args, loc, ip):
    return [a.ir_value(loc=loc, ip=ip) for a in args]


@dsl_user_op
def ld_v8(addr, *, loc=None, ip=None):
    st = ir.Type.parse("!llvm.struct<(" + ", ".join(["i32"] * 8) + ")>")
    res = llvm.inline_asm(
        st, _vals([Int64(addr)], loc, ip),
        "ld.global.nc.L1::no_allocate.v8.b32 {$0, $1, $2, $3, $4, $5, $6, $7}, [$8];",
        "=r,=r,=r,=r,=r,=r,=r,=r,l",
        has_side_effects=False, is_align_stack=False, asm_dialect=llvm.AsmDialect.AD_ATT, loc=loc, ip=ip,
    )
    return [Uint32(llvm.extractvalue(T.i32(), res, [k], loc=loc, ip=ip)) for k in range(8)]


@dsl_user_op
def st_v8(addr, v, *, loc=None, ip=None):
    llvm.inline_asm(
        None, _vals([Int64(addr)] + [Uint32(x) for x in v], loc, ip),
        "st.global.v8.b32 [$0], {$1, $2, $3, $4, $5, $6, $7, $8};",
        "l,r,r,r,r,r,r,r,r",
        has_side_effects=True, is_align_stack=False, asm_dialect=llvm.AsmDialect.AD_ATT, loc=loc, ip=ip,
    )


@cute.kernel
def _pack_kernel(mW13: cute.Tensor, mW2: cute.Tensor, mOut: cute.Tensor):
    tidx, _, _ = cute.arch.thread_idx()
    q, b, _ = cute.arch.block_idx()
    s = b // BLOCKS_PER_SHARD
    j = b - s * BLOCKS_PER_SHARD
    w13 = mW13.iterator.toint()
    w2 = mW2.iterator.toint()
    src = Int64(0)
    if j < E:
        src = w2 + Int64(j) * (W2_SHAPE[1] * W2_SHAPE[2] * 2) + Int64(s) * BLK_BYTES
    else:
        k = j - E
        src = w13 + Int64(k) * (W13_SHAPE[1] // 2 * W13_SHAPE[2] * 2) + Int64(s) * BLK_BYTES  # k = 2e + h
    off = Int64(q) * CTA_BYTES + Int64(tidx) * VEC
    src = src + off
    dst = mOut.iterator.toint() + Int64(b) * BLK_BYTES + off
    vals = [ld_v8(src + u * THREADS * VEC) for u in range(UNROLL)]
    for u in cutlass.range_constexpr(UNROLL):
        st_v8(dst + u * THREADS * VEC, vals[u])


@cute.jit
def _launch(mW13: cute.Tensor, mW2: cute.Tensor, mOut: cute.Tensor, stream):
    _pack_kernel(mW13, mW2, mOut).launch(grid=[CTAS_PER_BLK, NBLK, 1], block=[THREADS, 1, 1], stream=stream)


_CACHE = {}


def _fake(n):
    return cute.runtime.make_fake_tensor(cutlass.BFloat16, (n,), stride=(1,), assumed_align=32)


def _check(t, shape, name):
    if not (t.is_cuda and t.dtype == torch.bfloat16 and tuple(t.shape) == shape and t.is_contiguous()
            and t.data_ptr() % 32 == 0):
        raise ValueError(f"{name}: expected contiguous 32B-aligned cuda bf16 {shape}, got {t.dtype} "
                         f"{tuple(t.shape)} stride {t.stride()} on {t.device}")


def run(in0, in1):
    _check(in0, W13_SHAPE, "in0 (w13 grad)")
    _check(in1, W2_SHAPE, "in1 (w2 grad)")
    dev = in0.device
    out = torch.empty(OUT_NUMEL, dtype=torch.bfloat16, device=dev)
    fn = _CACHE.get(dev.index)
    if fn is None:
        fn = cute.compile(
            _launch, _fake(in0.numel()), _fake(in1.numel()), _fake(OUT_NUMEL),
            cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
            options="--enable-tvm-ffi",
        )
        _CACHE[dev.index] = fn
    with torch.cuda.device(dev):
        fn(in0.view(-1), in1.view(-1), out)
    return out
