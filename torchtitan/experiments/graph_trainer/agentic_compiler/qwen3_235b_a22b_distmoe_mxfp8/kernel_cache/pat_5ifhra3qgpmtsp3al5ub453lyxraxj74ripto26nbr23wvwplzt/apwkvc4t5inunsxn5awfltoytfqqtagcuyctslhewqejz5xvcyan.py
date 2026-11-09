"""MoE router forward: softmax + top-k gather + renormalization (Qwen3 235B-A22B, 8192 x 128, top-8).

reference: v2 = softmax(in0, -1); v3 = alias(v2); v4 = gather(v2, -1, in1); v6 = sum(v4, -1) + 1e-20;
v8 = (v4 / v6) * 1.0.

One CuTe DSL kernel. Each CTA is one warp that owns NGW = 1 group of GR = 4 consecutive rows
(2048 CTAs, a single wave); lane L = 8 r + t works on row r of a group
(t = L & 7).

softmax (torch's softmax_warp_forward<float, float, float, 7>): torch lane l (one warp per row)
holds columns l + 32 it (it = 0..3), takes the row max, e = expf(x - max) (CUDA's accurate expf op
sequence, see _EXP_PAIR), sums (((0 + e[l]) + e[l+32]) + e[l+64]) + e[l+96], xor butterfly
16, 8, 4, 2, 1, and stores e / sum (IEEE div.rn). Here lane t holds torch lanes l = 4 t + s
(s = 0..3) as four 16-byte loads (columns 4 t + 32 it .. 4 t + 32 it + 3), so the per-torch-lane
sums are in-thread, and the butterfly's offsets 16, 8, 4 (bits of t) and 2, 1 (bits of s) run as a
reduce-scatter over the row's 8 lanes that adds exactly the same pairs in the same tree (fp add is
commutative). The 0 + seed is exact (e >= +0 or NaN).
The max uses max.f32 (NaN ignored) instead of torch's compare-select chain: a NaN anywhere in the
row makes its e NaN, the sum NaN and every output the canonical NaN for any max; otherwise both
give the row max (the sign of a zero max cannot matter: x - (+-0) differs only for x = +-0, and
expf(+-0) = 1).
v2 is written to shared memory and leaves as one 1-D bulk store (cp.async.bulk) per group.

gather: lane L handles slot t of its row; v4 is the v2 element just written to shared memory at
column in1[row, t]. sum over 8 (Reduce.cuh, [8192, 8] fp32, last dim): thread k holds slot k,
shfl_down 4, 2, 1, done as the equivalent xor butterfly inside the 8-lane group.
v6 = sum + float32(1e-20); v8 = v4 / v6 (div.rn); "* 1.0" only canonicalizes NaN payloads, and
every NaN produced here is already canonical.

Division: the softmax divisor (row sum) is in [1, 128] or NaN (the max element gives e = 1, every
e <= 1), so e / sum uses ptxas's div.rn.f32 fast-path sequence with a hoisted refined reciprocal
whenever every dividend of the warp's group is 0 or >= 2^-60 (checked as min(x) - max >= -41 per
row); otherwise div.rn.f32. NaN rows give canonical NaN on both paths.
"""

import cutlass
import cutlass.cute as cute
import torch
from cutlass import Float32, Int32, Int64
from cutlass._mlir import ir
from cutlass._mlir.dialects import llvm
from cutlass.cutlass_dsl import T, dsl_user_op

ROWS = 8192
NE = 128
TOPK = 8
GR = 4  # rows per warp
NCTA = ROWS // GR
NGW = 1  # groups per warp (T=8192: 2048 one-warp CTAs beat NGW = 2 by ~0.4 us)


def _asm(ret_type, args, text, cons, loc=None, ip=None):
    return llvm.inline_asm(
        ret_type,
        [a.ir_value(loc=loc, ip=ip) for a in args],
        text,
        cons,
        has_side_effects=False,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def fadd(a, b, *, loc=None, ip=None):
    return Float32(_asm(T.f32(), [Float32(a), Float32(b)], "add.rn.f32 $0, $1, $2;", "=f,f,f", loc, ip))


@dsl_user_op
def fsub(a, b, *, loc=None, ip=None):
    return Float32(_asm(T.f32(), [Float32(a), Float32(b)], "sub.rn.f32 $0, $1, $2;", "=f,f,f", loc, ip))


@dsl_user_op
def fdiv(a, b, *, loc=None, ip=None):
    return Float32(_asm(T.f32(), [Float32(a), Float32(b)], "div.rn.f32 $0, $1, $2;", "=f,f,f", loc, ip))


@dsl_user_op
def fadd_1em20(a, *, loc=None, ip=None):
    """a + float32(1e-20)"""
    return Float32(_asm(T.f32(), [Float32(a)], "add.rn.f32 $0, $1, 0f1E3CE508;", "=f,f", loc, ip))


@dsl_user_op
def fmax(a, b, *, loc=None, ip=None):
    return Float32(_asm(T.f32(), [Float32(a), Float32(b)], "max.f32 $0, $1, $2;", "=f,f,f", loc, ip))


@dsl_user_op
def fmin(a, b, *, loc=None, ip=None):
    return Float32(_asm(T.f32(), [Float32(a), Float32(b)], "min.f32 $0, $1, $2;", "=f,f,f", loc, ip))


@dsl_user_op
def ge_m41(x, *, loc=None, ip=None):
    """1 if x >= -41.0 else 0 (NaN -> 0)"""
    return Int32(
        _asm(T.i32(), [Float32(x)], "{ .reg .pred p; setp.ge.f32 p, $1, 0fC2240000; selp.u32 $0, 1, 0, p; }", "=r,f", loc, ip)
    )


@dsl_user_op
def frcp_refined(n, *, loc=None, ip=None):
    """Divisor-only part of ptxas's div.rn.f32 fast path: r0 = MUFU.RCP(n); r = r0 + r0*(1 - n*r0)."""
    return Float32(
        _asm(
            T.f32(),
            [Float32(n)],
            "{ .reg .f32 r0, e, nn;\n"
            "neg.f32 nn, $1;\n"
            "rcp.approx.ftz.f32 r0, $1;\n"
            "fma.rn.f32 e, nn, r0, 0f3F800000;\n"
            "fma.rn.f32 $0, r0, e, r0; }",
            "=f,f",
            loc,
            ip,
        )
    )


@dsl_user_op
def fsel_nz(c, a, b, *, loc=None, ip=None):
    """c != 0 ? a : b (c is an Int32)."""
    return Float32(
        _asm(T.f32(), [Int32(c), Float32(a), Float32(b)], "{ .reg .pred p; setp.ne.s32 p, $1, 0; selp.f32 $0, $2, $3, p; }",
             "=f,r,f,f", loc, ip)
    )


# CUDA's accurate expf (libdevice __nv_expf, as compiled into torch's softmax kernels):
#   t = sat(fma(x, 0x3BBB989D, 0.5)); j = fma.rm(t, 252, 12582913); f = fma(x, log2e_hi, 12583039 - j);
#   f = fma(x, log2e_lo, f); e = float(j << 23) * ex2.approx.ftz(f)
# (12583039 - j replaces -(j - 12583039); both are exact integers and differ only in the sign of a
# zero, which only occurs for x = +-0, where f = +-0 and ex2 gives 1 either way.)
_EXP_PAIR = (
    "{ .reg .b64 D, TT, J, F, E, S, CS, CB, CN, CL, CLO;\n"
    ".reg .b32 t0, t1, j0, j1, f0, f1;\n"
    "mov.b64 CS, {0f437C0000, 0f437C0000};\n"
    "mov.b64 CB, {0f4B400001, 0f4B400001};\n"
    "mov.b64 CN, {0f4B40007F, 0f4B40007F};\n"
    "mov.b64 CL, {0f3FB8AA3B, 0f3FB8AA3B};\n"
    "mov.b64 CLO, {0f32A57060, 0f32A57060};\n"
    "mov.b64 D, {$2, $3};\n"
    "fma.rn.sat.f32 t0, $2, 0f3BBB989D, 0f3F000000;\n"
    "fma.rn.sat.f32 t1, $3, 0f3BBB989D, 0f3F000000;\n"
    "mov.b64 TT, {t0, t1};\n"
    "fma.rm.f32x2 J, TT, CS, CB;\n"
    "sub.rn.f32x2 F, CN, J;\n"
    "fma.rn.f32x2 F, D, CL, F;\n"
    "fma.rn.f32x2 F, D, CLO, F;\n"
    "mov.b64 {j0, j1}, J;\n"
    "shl.b32 j0, j0, 23;\n"
    "shl.b32 j1, j1, 23;\n"
    "mov.b64 S, {j0, j1};\n"
    "mov.b64 {f0, f1}, F;\n"
    "ex2.approx.ftz.f32 f0, f0;\n"
    "ex2.approx.ftz.f32 f1, f1;\n"
    "mov.b64 E, {f0, f1};\n"
    "mul.rn.f32x2 E, S, E;\n"
    "mov.b64 {$0, $1}, E; }"
)

@dsl_user_op
def exp_pair(d0, d1, *, loc=None, ip=None):
    res = _asm(ir.Type.parse("!llvm.struct<(f32, f32)>"), [Float32(d0), Float32(d1)], _EXP_PAIR, "=f,=f,f,f", loc, ip)
    return (
        Float32(llvm.extractvalue(T.f32(), res, [0], loc=loc, ip=ip)),
        Float32(llvm.extractvalue(T.f32(), res, [1], loc=loc, ip=ip)),
    )


def p_sub(a, b):
    return cute.arch.sub_packed_f32x2(a, b, rnd="rn")


def p_mul(a, b):
    return cute.arch.mul_packed_f32x2(a, b, rnd="rn")


def p_add(a, b):
    return cute.arch.add_packed_f32x2(a, b, rnd="rn")


def p_fma(a, b, c):
    return cute.arch.fma_packed_f32x2(a, b, c, rnd="rn")


@dsl_user_op
def _bulk_s2g(dst, src, nbytes, *, loc=None, ip=None):
    """1-D bulk async copy this CTA's smem -> global, committed as one bulk group."""
    llvm.inline_asm(
        None,
        [dst.toint(loc=loc, ip=ip).ir_value(loc=loc, ip=ip), src.toint(loc=loc, ip=ip).ir_value(loc=loc, ip=ip)],
        f"cp.async.bulk.global.shared::cta.bulk_group [$0], [$1], {nbytes};\n"
        "cp.async.bulk.commit_group;",
        "l,r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def _bulk_wait_read(*, loc=None, ip=None):
    llvm.inline_asm(None, [], "cp.async.bulk.wait_group.read 0;", "", has_side_effects=True, is_align_stack=False,
                    asm_dialect=llvm.AsmDialect.AD_ATT, loc=loc, ip=ip)


@dsl_user_op
def _fence_proxy_async(*, loc=None, ip=None):
    llvm.inline_asm(None, [], "fence.proxy.async.shared::cta;", "", has_side_effects=True, is_align_stack=False,
                    asm_dialect=llvm.AsmDialect.AD_ATT, loc=loc, ip=ip)



def tree_max(v):
    while len(v) > 1:
        v = [fmax(v[i], v[i + 1]) for i in range(0, len(v), 2)]
    return v[0]


def tree_min(v):
    while len(v) > 1:
        v = [fmin(v[i], v[i + 1]) for i in range(0, len(v), 2)]
    return v[0]


class RouterFwd:
    @cute.jit
    def __call__(self, mX, mIdx, mP, mG, mS, mQ, stream):
        self.kernel(mX, mIdx, mP, mG, mS, mQ).launch(grid=[NCTA // NGW, 1, 1], block=[32, 1, 1], stream=stream)

    @cute.kernel
    def kernel(self, mX: cute.Tensor, mIdx: cute.Tensor, mP: cute.Tensor, mG: cute.Tensor, mS: cute.Tensor, mQ: cute.Tensor):
        lane, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        t = lane & 7
        rl = lane >> 3
        b2 = Int32(lane & 4)
        b1 = Int32(lane & 2)
        rowL = cute.make_layout((4, 4, 8, ROWS), stride=(1, 32, 4, NE))
        gX = cute.make_tensor(mX.iterator, rowL)
        # v2 rows go out through shared memory and one 1-D bulk store per group (GR rows = 2 KB)
        smem = cutlass.utils.SmemAllocator()
        sP = smem.allocate_tensor(Float32, cute.make_layout((GR * NE, NGW)), byte_alignment=128)

        # loads of all NGW groups precede the math in program order (group 1's rows arrive during group 0's math)
        rows, slots, cols, xs = [], [], [], []
        for g in cutlass.range_constexpr(NGW):
            row = (bidx * NGW + g) * GR + rl
            slot = row * TOPK + t
            iw = cute.make_rmem_tensor(cute.make_layout(2), Int32)
            cute.autovec_copy(cute.make_tensor(cute.recast_ptr(mIdx.iterator + slot, dtype=Int32).align(8), cute.make_layout(2)), iw)
            cols.append(iw[0] & Int32(NE - 1))
            x = cute.make_rmem_tensor(cute.make_layout((4, 4), stride=(1, 4)), Float32)
            for it in cutlass.range_constexpr(4):
                cute.autovec_copy(gX[None, it, t, row], x[None, it])
            rows.append(row)
            slots.append(slot)
            xs.append(x)

        for g in cutlass.range_constexpr(NGW):
            x = xs[g]
            row = rows[g]
            slot = slots[g]
            sL = cute.make_tensor(sP[None, g].iterator, cute.make_layout((4, 4, 8, GR), stride=(1, 32, 4, NE)))
            xl = [x[s, it] for it in range(4) for s in range(4)]
            mx = tree_max(xl)
            for off in cutlass.range_constexpr(3):
                mx = fmax(mx, cute.arch.shuffle_sync_bfly(mx, 4 >> off))
            ok = ge_m41(fsub(tree_min(xl), mx))

            e = cute.make_rmem_tensor(cute.make_layout((4, 4), stride=(1, 4)), Float32)
            for it in cutlass.range_constexpr(4):
                for h in cutlass.range_constexpr(2):
                    d = p_sub((x[2 * h, it], x[2 * h + 1, it]), (mx, mx))
                    e[2 * h, it], e[2 * h + 1, it] = exp_pair(d[0], d[1])
            p01 = p_add(p_add(p_add((e[0, 0], e[1, 0]), (e[0, 1], e[1, 1])), (e[0, 2], e[1, 2])), (e[0, 3], e[1, 3]))
            p23 = p_add(p_add(p_add((e[2, 0], e[3, 0]), (e[2, 1], e[3, 1])), (e[2, 2], e[3, 2])), (e[2, 3], e[3, 3]))
            u0 = fsel_nz(b2, p01[0], p23[0])
            u1 = fsel_nz(b2, p01[1], p23[1])
            k0 = fsel_nz(b2, p23[0], p01[0])
            k1 = fsel_nz(b2, p23[1], p01[1])
            a = p_add((k0, k1), (cute.arch.shuffle_sync_bfly(u0, 4), cute.arch.shuffle_sync_bfly(u1, 4)))
            u = fsel_nz(b1, a[0], a[1])
            kk = fsel_nz(b1, a[1], a[0])
            c = fadd(kk, cute.arch.shuffle_sync_bfly(u, 2))
            c = fadd(c, cute.arch.shuffle_sync_bfly(c, 1))
            c = fadd(c, cute.arch.shuffle_sync_bfly(c, 4))
            sm = fadd(c, cute.arch.shuffle_sync_bfly(c, 2))

            q = cute.make_rmem_tensor(cute.make_layout(4), Float32)
            if cute.arch.vote_all_sync(ok != Int32(0)):
                r = frcp_refined(sm)
                nn = Float32(0.0) - sm
                for it in cutlass.range_constexpr(4):
                    for h in cutlass.range_constexpr(2):
                        ee = (e[2 * h, it], e[2 * h + 1, it])
                        qq = p_mul(ee, (r, r))
                        q[2 * h], q[2 * h + 1] = p_fma((r, r), p_fma((nn, nn), qq, ee), qq)
                    cute.autovec_copy(q, sL[None, it, t, rl])
            else:
                for it in cutlass.range_constexpr(4):
                    for s in cutlass.range_constexpr(4):
                        q[s] = fdiv(e[s, it], sm)
                    cute.autovec_copy(q, sL[None, it, t, rl])

            _fence_proxy_async()  # make this lane's smem writes visible to the bulk-copy (async) proxy
            cute.arch.sync_warp()
            if lane == 0:
                _bulk_s2g(mP.iterator + (bidx * NGW + g) * (GR * NE), sP[None, g].iterator, GR * NE * 4)
            v4 = sP[rl * NE + cols[g], g]  # gather: the stored v2 element

            v5 = v4
            for off in cutlass.range_constexpr(3):
                v5 = fadd(v5, cute.arch.shuffle_sync_bfly(v5, 4 >> off))
            v6 = fadd_1em20(v5)
            mG[slot] = v4
            mQ[slot] = fdiv(v4, v6)
            if t == 0:
                mS[row] = v6
        if lane == 0:
            _bulk_wait_read()  # the bulk stores have read sP before the CTA's shared memory is released


_CACHE = {}


def _fake(dtype, n, align=16):
    return cute.runtime.make_fake_tensor(dtype, (n,), stride=(1,), assumed_align=align)


def run(in0, in1):
    dev = in0.device
    x = in0 if in0.is_contiguous() else in0.contiguous()
    idx = in1 if in1.is_contiguous() else in1.contiguous()
    if x.data_ptr() % 16:
        x = x.clone()
    v2 = torch.empty((ROWS, NE), dtype=torch.float32, device=dev)
    v4 = torch.empty((ROWS, TOPK), dtype=torch.float32, device=dev)
    v6 = torch.empty((ROWS, 1), dtype=torch.float32, device=dev)
    v8 = torch.empty((ROWS, TOPK), dtype=torch.float32, device=dev)
    fn = _CACHE.get(dev.index)
    if fn is None:
        fn = cute.compile(
            RouterFwd(),
            _fake(Float32, ROWS * NE),
            _fake(Int64, ROWS * TOPK, 8),
            _fake(Float32, ROWS * NE),
            _fake(Float32, ROWS * TOPK),
            _fake(Float32, ROWS),
            _fake(Float32, ROWS * TOPK),
            cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
            options="--enable-tvm-ffi",
        )
        _CACHE[dev.index] = fn
    fn(x.view(-1), idx.view(-1), v2.view(-1), v4.view(-1), v6.view(-1), v8.view(-1))
    return (v2, v2.view(ROWS, NE), v4, v6, v8)
