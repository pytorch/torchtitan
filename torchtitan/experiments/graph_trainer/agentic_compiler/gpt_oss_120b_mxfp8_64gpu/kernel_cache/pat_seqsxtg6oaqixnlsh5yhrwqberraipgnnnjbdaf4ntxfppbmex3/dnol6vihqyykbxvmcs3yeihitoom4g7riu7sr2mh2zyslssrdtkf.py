"""G-005 expert-bias gradient: new_zeros([9, O]).index_add(0, index, src)[:8] (bf16) in CuTe DSL.

Reference semantics (deterministic CUDA index_add, aten/src/ATen/native/cuda/Indexing.cu):
index_add -> index_put_(accumulate=True) -> stable radix sort of the indices, then
indexing_backward_kernel walks the rows of each destination in original order and does
dst = bf16(float(dst) + float(src_row)) after every row (source * alpha only canonicalizes NaN
payloads, which the add does anyway). So out[d, c] is the left fold
acc = bf16_rn(float(acc) + float(src[i, c])) over the rows i with index[i] == d (index -9..-1
wraps to 0..8) in increasing i, starting from +0. ``add.rn.bf16x2`` (and the HFMA2 x*1+y form
ptxas may emit for it) equals that step bit for bit for every pair of bf16 inputs (exhaustively
checked over all 2^32 pairs incl. NaN, inf, -0, denormals; scratch/exh_add.py), so each chain is
a sequence of add.rn.bf16x2 in row order. bf16 -0 is the identity of that step for every
accumulator value the chain can hold (+0 + -0 = +0; a NaN accumulator is already canonical).

Kernels (one jit launch):
1. bitmap: bm[d, w] = 32-bit mask of rows 32w..32w+31 with index == d (d = 0..7); zeroes the
   per-destination progress counters. Triggers the main kernel's programmatic launch at once.
2. main (PDL; its producer runs griddepcontrol.wait before any global access): one CTA per
   (destination d, column tile of NH boxes of BOXC columns), 8 x 18 = 144 CTAs. A producer warp
   walks bm[d] in row order (2048 words staged per round; empty and full 32-word batches found
   with one ballot each) and fills a 4-stage smem ring. A stage holds SB sub-blocks of 32 row
   slots; each nonzero bitmap word fills one sub-block: a full word with one 2D TMA box per
   half tile, a partial word with one cp.async.bulk per row (and half) plus -0 in the unused
   slots. A batch of 32 full words (the sorted training case) goes as whole stages of SB*32
   consecutive rows, one TMA box per half each. Every producer lane arrives on a stage's full
   barrier, so each lane releases its own -0 smem stores. Consumer warps own one bf16x2 column pair each
   and fold every row slot of every sub-block in order (two register buffers: smem loads of the
   next sub-block overlap the add chain). The producers of one destination keep within LEAD
   stages of their average progress (global counters), because CTAs on some SMs get less DRAM
   bandwidth and the slowest CTA ends the kernel.
Destination 8 (padding row) is never read: the reference slices it off.
"""

import cutlass
import cutlass.cute as cute
import torch
from cutlass import Int32, Int64, Uint32
from cutlass._mlir import ir
from cutlass._mlir.dialects import llvm
from cutlass.cute.nvgpu import cpasync
from cutlass.cutlass_dsl import T, dsl_user_op

ND = 8  # destinations kept by the slice [:8]
NTILES = 18  # column tiles; 8 * 18 = 144 CTAs (GB300: 152 SMs); O / 18 is a 32 B multiple for O = 2880, 5760
G = 32  # rows per sub-block (one bitmap word)
STAGE_MAX = 40960  # smem bytes per stage at most (bigger stages amortize the per-stage producer work)
RING = 163840  # smem ring bytes per CTA (4 stages); more stages did not help
ROUND = 2048  # bitmap words staged per producer round (64 per lane)
LEAD = 1  # stages a CTA may run ahead of its destination's average (CTAs on some SMs get less bandwidth)
TERM = 0xFFFF  # count of the first sub-block of the terminal stage
FULLW = 0xFFFFFFFF


def _asm(ret, args, text, cons, side, loc, ip):
    return llvm.inline_asm(
        ret, [a.ir_value(loc=loc, ip=ip) for a in args], text, cons,
        has_side_effects=side, is_align_stack=False, asm_dialect=llvm.AsmDialect.AD_ATT, loc=loc, ip=ip,
    )


@dsl_user_op
def hadd2(a, b, *, loc=None, ip=None):
    return Uint32(_asm(T.i32(), [Uint32(a), Uint32(b)], "add.rn.bf16x2 $0, $1, $2;", "=r,r,r", False, loc, ip))


@dsl_user_op
def lds32(addr, *, loc=None, ip=None):
    return Uint32(_asm(T.i32(), [Int32(addr)], "ld.shared.b32 $0, [$1];", "=r,r,~{memory}", True, loc, ip))


@dsl_user_op
def sts32(addr, v, *, loc=None, ip=None):
    _asm(None, [Int32(addr), Uint32(v)], "st.shared.b32 [$0], $1;", "r,r,~{memory}", True, loc, ip)


@dsl_user_op
def ldg32(addr, *, loc=None, ip=None):
    return Uint32(_asm(T.i32(), [Int64(addr)], "ld.global.b32 $0, [$1];", "=r,l,~{memory}", True, loc, ip))


@dsl_user_op
def stg32(addr, v, *, loc=None, ip=None):
    _asm(None, [Int64(addr), Uint32(v)], "st.global.b32 [$0], $1;", "l,r,~{memory}", True, loc, ip)


@dsl_user_op
def mbar_init(addr, cnt, *, loc=None, ip=None):
    _asm(None, [Int32(addr), Int32(cnt)], "mbarrier.init.shared::cta.b64 [$0], $1;", "r,r,~{memory}", True, loc, ip)


@dsl_user_op
def mbar_wait(addr, parity, *, loc=None, ip=None):
    _asm(
        None, [Int32(addr), Int32(parity)],
        "{ .reg .pred p; LAB_WAIT: mbarrier.try_wait.parity.shared::cta.b64 p, [$0], $1; @!p bra LAB_WAIT; }",
        "r,r,~{memory}", True, loc, ip,
    )


@dsl_user_op
def mbar_arrive(addr, *, loc=None, ip=None):
    _asm(None, [Int32(addr)], "mbarrier.arrive.shared::cta.b64 _, [$0];", "r,~{memory}", True, loc, ip)


@dsl_user_op
def mbar_arrive_tx(addr, tx, *, loc=None, ip=None):
    _asm(None, [Int32(addr), Int32(tx)], "mbarrier.arrive.expect_tx.shared::cta.b64 _, [$0], $1;", "r,r,~{memory}",
         True, loc, ip)


@dsl_user_op
def mbar_expect_tx(addr, tx, *, loc=None, ip=None):
    _asm(None, [Int32(addr), Int32(tx)], "mbarrier.expect_tx.relaxed.cta.shared::cta.b64 [$0], $1;", "r,r,~{memory}",
         True, loc, ip)


@dsl_user_op
def bulk_g2s(dst, src, nbytes, mbar, *, loc=None, ip=None):
    _asm(
        None, [Int32(dst), Int64(src), Int32(nbytes), Int32(mbar)],
        "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [$0], [$1], $2, [$3];",
        "r,l,r,r,~{memory}", True, loc, ip,
    )


@dsl_user_op
def ldg128(addr, *, loc=None, ip=None):
    res = _asm(ir.Type.parse("!llvm.struct<(i32, i32, i32, i32)>"), [Int64(addr)],
               "ld.global.v4.b32 {$0, $1, $2, $3}, [$4];", "=r,=r,=r,=r,l,~{memory}", True, loc, ip)
    return [Uint32(llvm.extractvalue(T.i32(), res, [i], loc=loc, ip=ip)) for i in range(4)]


@dsl_user_op
def sts128(addr, v0, v1, v2, v3, *, loc=None, ip=None):
    _asm(None, [Int32(addr), Uint32(v0), Uint32(v1), Uint32(v2), Uint32(v3)],
         "st.shared.v4.b32 [$0], {$1, $2, $3, $4};", "r,r,r,r,r,~{memory}", True, loc, ip)


@dsl_user_op
def ld_ctr(addr, *, loc=None, ip=None):
    """(started, progress) pair of a destination, relaxed gpu-scope load."""
    res = _asm(ir.Type.parse("!llvm.struct<(i32, i32)>"), [Int64(addr)],
               "ld.relaxed.gpu.global.v2.u32 {$0, $1}, [$2];", "=r,=r,l,~{memory}", True, loc, ip)
    return Int32(llvm.extractvalue(T.i32(), res, [0], loc=loc, ip=ip)), Int32(
        llvm.extractvalue(T.i32(), res, [1], loc=loc, ip=ip))


@dsl_user_op
def red_add(addr, v, *, loc=None, ip=None):
    _asm(None, [Int64(addr), Int32(v)], "red.relaxed.gpu.global.add.u32 [$0], $1;", "l,r,~{memory}", True, loc, ip)


@dsl_user_op
def nanosleep(ns, *, loc=None, ip=None):
    _asm(None, [Int32(ns)], "nanosleep.u32 $0;", "r,~{memory}", True, loc, ip)


@dsl_user_op
def pick(x, jj, cnum, *, loc=None, ip=None):
    """jj < cnum ? x : bf16x2 -0 (0x80008000)."""
    return Uint32(_asm(T.i32(), [Uint32(x), Int32(jj), Int32(cnum)],
                       "{ .reg .pred p; setp.lt.s32 p, $2, $3; selp.b32 $0, $1, 0x80008000, p; }", "=r,r,r,r", False,
                       loc, ip))


@dsl_user_op
def sts128c(addr, *, loc=None, ip=None):
    """16 bytes of bf16 -0 (0x8000) to smem."""
    _asm(None, [Int32(addr)], "{ .reg .b32 z; mov.b32 z, 0x80008000; st.shared.v4.b32 [$0], {z, z, z, z}; }",
         "r,~{memory}", True, loc, ip)


@dsl_user_op
def lanemask_lt(*, loc=None, ip=None):
    return Uint32(_asm(T.i32(), [], "mov.u32 $0, %lanemask_lt;", "=r", False, loc, ip))


@dsl_user_op
def ffs0(x, *, loc=None, ip=None):
    """Index of the lowest set bit of a nonzero word."""
    return Int32(_asm(T.i32(), [Uint32(x)], "{ .reg .b32 t; brev.b32 t, $1; clz.b32 $0, t; }", "=r,r", False, loc, ip))


@dsl_user_op
def popc32(x, *, loc=None, ip=None):
    return Int32(_asm(T.i32(), [Uint32(x)], "popc.b32 $0, $1;", "=r,r", False, loc, ip))


def _config(o):
    """Column tiling: NTILES tiles of NH boxes of BOXC columns (BOXC <= 256, multiple of 8)."""
    per_tile = -(-o // NTILES)
    nh = -(-per_tile // 256)
    boxc = -(-per_tile // nh)
    boxc = -(-boxc // 8) * 8
    tile = nh * boxc
    assert (NTILES - 1) * tile < o <= NTILES * tile and o % 8 == 0
    sb = 1  # sub-blocks (bitmap words) per stage: the largest power of 2 with a stage <= STAGE_MAX bytes
    while sb < 8 and 2 * sb * nh * G * boxc * 2 <= STAGE_MAX:
        sb *= 2
    nstage = RING // (sb * nh * G * boxc * 2)
    ncw = -(-(tile // 2) // 32)  # consumer warps: one bf16x2 word (2 columns) per thread
    return dict(O=o, BOXC=boxc, NH=nh, TILE=tile, SB=sb, NCW=ncw, NSTAGE=nstage)


class IndexAddBias:
    def __init__(self, o):
        self.c = _config(o)

    @cute.kernel
    def bitmap_kernel(self, mIdx: cute.Tensor, mBm: cute.Tensor, R: Int32, NWP: Int32):
        """bm[d * NWP + w] = rows 32w..32w+31 with index == d (zero for w >= NW); zero the counters."""
        tid, _, _ = cute.arch.thread_idx()
        bid, _, _ = cute.arch.block_idx()
        cute.arch.griddepcontrol_launch_dependents()
        lane = tid % 32
        r = bid * 256 + tid
        v = Int64(-100)
        if r < R:
            v = Int64(mIdx[r])
        if v < 0:
            v = v + 9
        w = bid * 8 + tid // 32
        for d in cutlass.range_constexpr(ND):
            m = cute.arch.vote_ballot_sync(v == d)
            if lane == d and w < NWP:
                mBm[d * NWP + w] = m
        if bid == 0 and tid < 2 * ND:
            mBm[ND * NWP + tid] = Int32(0)

    @cute.kernel
    def main_kernel(self, tma32: cute.CopyAtom, tmt32: cute.Tensor, tmaS: cute.CopyAtom, tmtS: cute.Tensor,
                    mSrc: cute.Tensor, mBm: cute.Tensor, mOut: cute.Tensor, NWP: Int32):
        c = self.c
        O, BOXC, NH, TILE, SB, NCW, NSTAGE = (c[k] for k in ("O", "BOXC", "NH", "TILE", "SB", "NCW", "NSTAGE"))
        ROWB = BOXC * 2  # smem bytes per row slot of one half
        SUBB = G * ROWB  # one sub-block (32 rows) of one half
        HALFB = SB * SUBB  # one half of a stage
        NCONS = NCW * 32

        tid, _, _ = cute.arch.thread_idx()
        bid, _, _ = cute.arch.block_idx()
        lane = tid % 32
        warp = tid // 32
        d = bid % ND
        t = bid // ND
        c0 = t * TILE  # first column of the tile

        smem = cutlass.utils.SmemAllocator()
        sXS = smem.allocate_tensor(cutlass.BFloat16,
                                   cute.make_layout((SB * G, BOXC, NSTAGE * NH), stride=(BOXC, 1, SB * G * BOXC)),
                                   byte_alignment=1024)
        sX32 = cute.make_tensor(sXS.iterator,
                                cute.make_layout((G, BOXC, NSTAGE * NH * SB), stride=(BOXC, 1, G * BOXC)))
        bars = smem.allocate_array(cutlass.Int64, 2 * NSTAGE, byte_alignment=8)
        cnts = smem.allocate_array(cutlass.Int32, NSTAGE * SB, byte_alignment=16).toint()
        bmbuf = smem.allocate_array(cutlass.Uint32, ROUND, byte_alignment=16).toint()
        ring = sXS.iterator.toint()
        full = bars.toint()
        empty = full + NSTAGE * 8

        if tid == 0:
            cpasync.prefetch_descriptor(tma32)
            cpasync.prefetch_descriptor(tmaS)
            for s in cutlass.range_constexpr(NSTAGE):
                mbar_init(full + s * 8, 32)  # every producer lane arrives: each releases its own smem writes
                mbar_init(empty + s * 8, NCONS)
            cute.arch.mbarrier_init_fence()
        cute.arch.barrier()

        if warp == 0:
            # ---------------- producer ----------------
            g32 = cute.local_tile(tmt32, (G, BOXC), (None, None))
            s32, d32 = cpasync.tma_partition(tma32, 0, cute.make_layout(1), cute.group_modes(sX32, 0, 2),
                                             cute.group_modes(g32, 0, 2))
            gS = cute.local_tile(tmtS, (SB * G, BOXC), (None, None))
            sS, dS = cpasync.tma_partition(tmaS, 0, cute.make_layout(1), cute.group_modes(sXS, 0, 2),
                                           cute.group_modes(gS, 0, 2))
            src0 = mSrc.iterator.toint()
            bm0 = mBm.iterator.toint() + Int64(d) * Int64(NWP) * 4
            ctr = mBm.iterator.toint() + (Int64(ND) * Int64(NWP) + Int64(2 * d)) * 4  # (started, progress)
            lmlt = lanemask_lt()
            # valid bytes of one row's slice per half (the last tile runs past column O)
            rowb = Int32(0)
            for h in cutlass.range_constexpr(NH):
                bh = (O - (c0 + h * BOXC)) * 2
                if bh > ROWB:
                    bh = Int32(ROWB)
                if bh > 0:
                    rowb = rowb + bh
            # PDL: the bitmap kernel (bitmap, zeroed counters) must be complete before any global access;
            # it was launched without PDL, so every kernel before it is complete as well.
            cute.arch.griddepcontrol_wait()
            if lane == 0:
                red_add(ctr, 1)
            st_n, st_sum = ld_ctr(ctr)
            k = Int32(0)  # current stage
            j = Int32(0)  # next free sub-block of stage k
            for rb in cutlass.range(0, NWP, ROUND, unroll=1):
                # lane l holds batch l (words rb + 32l ..) and batch 32 + l (words rb + 1024 + 32l ..)
                va = cute.make_rmem_tensor(cute.make_layout(32), Uint32)
                vb = cute.make_rmem_tensor(cute.make_layout(32), Uint32)
                for half in cutlass.range_constexpr(2):
                    vx = va if half == 0 else vb
                    for q4 in cutlass.range_constexpr(8):
                        wq = rb + half * 1024 + lane * 32 + q4 * 4
                        for e in cutlass.range_constexpr(4):
                            vx[q4 * 4 + e] = Uint32(0)
                        if wq < NWP:
                            x4 = ldg128(bm0 + Int64(wq) * 4)
                            for e in cutlass.range_constexpr(4):
                                vx[q4 * 4 + e] = x4[e]
                ora = Uint32(0)
                anda = Uint32(FULLW)
                orb = Uint32(0)
                andb = Uint32(FULLW)
                for e in cutlass.range_constexpr(32):
                    ora = ora | va[e]
                    anda = anda & va[e]
                    orb = orb | vb[e]
                    andb = andb & vb[e]
                nzA = Uint32(cute.arch.vote_ballot_sync(ora != Uint32(0)))
                fuA = Uint32(cute.arch.vote_ballot_sync(anda == Uint32(FULLW)))
                nzB = Uint32(cute.arch.vote_ballot_sync(orb != Uint32(0)))
                fuB = Uint32(cute.arch.vote_ballot_sync(andb == Uint32(FULLW)))
                if (nzA | nzB) != Uint32(0):
                    for q4 in cutlass.range_constexpr(8):
                        sts128(bmbuf + (lane * 32 + q4 * 4) * 4, va[q4 * 4], va[q4 * 4 + 1], va[q4 * 4 + 2],
                               va[q4 * 4 + 3])
                        sts128(bmbuf + (1024 + lane * 32 + q4 * 4) * 4, vb[q4 * 4], vb[q4 * 4 + 1], vb[q4 * 4 + 2],
                               vb[q4 * 4 + 3])
                cute.arch.sync_warp()
                for half in cutlass.range_constexpr(2):
                    act = nzA if half == 0 else nzB
                    fum = fuA if half == 0 else fuB
                    while act != Uint32(0):
                        bi = ffs0(act)
                        act = act & (act - Uint32(1))
                        w0 = rb + half * 1024 + bi * 32  # first bitmap word of the batch
                        if ((fum >> Uint32(bi)) & Uint32(1)) != Uint32(0):
                            # 32 full words = 1024 consecutive rows: stages of SB*32 rows, one box per half
                            if j > 0:
                                self.fill_neg0(ring + (k % NSTAGE) * (NH * HALFB), j * G, SB * G, lane)
                                mbar_arrive(full + (k % NSTAGE) * 8)
                                if lane == 0:
                                    red_add(ctr + 4, 1)
                                k = k + 1
                                j = Int32(0)
                            for g in cutlass.range(32 // SB, unroll=1):
                                slot = k % NSTAGE
                                if LEAD > 0:
                                    # keep within LEAD stages of the destination's average progress
                                    # (counters loaded 4 stages earlier, so the load latency is hidden)
                                    if (k & 3) == 0:
                                        while k * st_n > st_sum + LEAD * st_n:
                                            nanosleep(200)
                                            st_n, st_sum = ld_ctr(ctr)
                                        st_n, st_sum = ld_ctr(ctr)
                                mbar_wait(empty + slot * 8, ((k // NSTAGE) & 1) ^ 1)
                                if lane == 0:
                                    sts32(cnts + slot * SB * 4, Uint32(G))
                                    mbar_arrive_tx(full + slot * 8, NH * HALFB)
                                else:
                                    mbar_arrive(full + slot * 8)
                                cute.arch.sync_warp()
                                for h in cutlass.range_constexpr(NH):
                                    cute.copy(tmaS, dS[(None, w0 // SB + g, t * NH + h)], sS[(None, slot * NH + h)],
                                              tma_bar_ptr=bars + slot)
                                if lane == 0:
                                    red_add(ctr + 4, 1)
                                k = k + 1
                        else:
                            wv = lds32(bmbuf + (half * 1024 + bi * 32 + lane) * 4)
                            wact = Uint32(cute.arch.vote_ballot_sync(wv != Uint32(0)))
                            while wact != Uint32(0):
                                jw = ffs0(wact)
                                wact = wact & (wact - Uint32(1))
                                m = Uint32(cute.arch.shuffle_sync(wv, jw))
                                wrd = w0 + jw
                                slot = k % NSTAGE
                                if j == 0:
                                    mbar_wait(empty + slot * 8, ((k // NSTAGE) & 1) ^ 1)
                                if m == Uint32(FULLW):
                                    if lane == 0:
                                        sts32(cnts + (slot * SB + j) * 4, Uint32(G))
                                        mbar_expect_tx(full + slot * 8, NH * SUBB)
                                    cute.arch.sync_warp()
                                    for h in cutlass.range_constexpr(NH):
                                        cute.copy(tma32, d32[(None, wrd, t * NH + h)],
                                                  s32[(None, (slot * NH + h) * SB + j)], tma_bar_ptr=bars + slot)
                                else:
                                    cnt = popc32(m)
                                    self.fill_neg0(ring + slot * (NH * HALFB), j * G + cnt, j * G + G, lane)
                                    if lane == 0:
                                        sts32(cnts + (slot * SB + j) * 4, Uint32(cnt))
                                        mbar_expect_tx(full + slot * 8, cnt * rowb)
                                    cute.arch.sync_warp()
                                    if ((m >> Uint32(lane)) & Uint32(1)) != Uint32(0):
                                        pos = popc32(m & lmlt)
                                        row = wrd * 32 + lane
                                        for h in cutlass.range_constexpr(NH):
                                            bh = (O - (c0 + h * BOXC)) * 2
                                            if bh > ROWB:
                                                bh = Int32(ROWB)
                                            if bh > 0:
                                                dst = ring + slot * (NH * HALFB) + h * HALFB + (j * G + pos) * ROWB
                                                srca = src0 + (Int64(row) * O + Int64(c0 + h * BOXC)) * 2
                                                bulk_g2s(dst, srca, bh, full + slot * 8)
                                cute.arch.sync_warp()
                                j = j + 1
                                if j == SB:
                                    mbar_arrive(full + slot * 8)
                                    if lane == 0:
                                        red_add(ctr + 4, 1)
                                    k = k + 1
                                    j = Int32(0)
                cute.arch.sync_warp()
            if j > 0:
                self.fill_neg0(ring + (k % NSTAGE) * (NH * HALFB), j * G, SB * G, lane)
                mbar_arrive(full + (k % NSTAGE) * 8)
                if lane == 0:
                    red_add(ctr + 4, 1)
                k = k + 1
            mbar_wait(empty + (k % NSTAGE) * 8, ((k // NSTAGE) & 1) ^ 1)
            if lane == 0:
                sts32(cnts + (k % NSTAGE) * SB * 4, Uint32(TERM))
            mbar_arrive(full + (k % NSTAGE) * 8)
        else:
            # ---------------- consumers ----------------
            # Sub-blocks q = 0, 1, ... (stage q // SB, sub-block q % SB) in order; every row slot
            # holds either a source row or bf16 -0 (written by the producer), and -0 is the
            # identity of the bf16 add for every accumulator value (acc starts at +0 and only holds
            # add results, so a NaN accumulator is already the canonical NaN). Two register
            # buffers: the smem loads of sub-block q+1 overlap the add chain of q.
            cw = tid - 32  # bf16x2 word of the tile owned by this thread
            h = cw // (BOXC // 2)
            wi = cw - h * (BOXC // 2)
            active = (cw < TILE // 2) & (c0 + 2 * cw < O)
            col = ring  # idle threads read (and fold) a valid smem word; they never store
            if active:
                col = ring + h * HALFB + wi * 4
            xa = cute.make_rmem_tensor(cute.make_layout(G), Uint32)
            xb = cute.make_rmem_tensor(cute.make_layout(G), Uint32)
            acc = Uint32(0)
            j = Int32(0)  # sub-block q = (slot, j) whose rows are in xa
            slot = Int32(0)
            ph = Int32(0)
            mbar_wait(full, 0)
            go = Int32(lds32(cnts)) != TERM
            for jj in cutlass.range_constexpr(G):
                xa[jj] = lds32(col + jj * ROWB)
            while go:
                # step A: load sub-block q+1 into xb, fold xa
                j1 = j + 1
                slot1 = slot
                ph1 = ph
                term = False
                if j1 == SB:
                    j1 = Int32(0)
                    slot1 = slot + 1
                    if slot1 == NSTAGE:
                        slot1 = Int32(0)
                        ph1 = ph ^ 1
                    mbar_wait(full + slot1 * 8, ph1)
                    term = Int32(lds32(cnts + slot1 * SB * 4)) == TERM
                addr = col + slot1 * (NH * HALFB) + j1 * SUBB
                for jj in cutlass.range_constexpr(G):
                    xb[jj] = lds32(addr + jj * ROWB)
                for jj in cutlass.range_constexpr(G):
                    acc = hadd2(acc, xa[jj])
                if j1 == 0:
                    mbar_arrive(empty + slot * 8)
                if term:
                    go = False
                else:
                    # step B: load sub-block q+2 into xa, fold xb
                    j2 = j1 + 1
                    slot2 = slot1
                    ph2 = ph1
                    if j2 == SB:
                        j2 = Int32(0)
                        slot2 = slot1 + 1
                        if slot2 == NSTAGE:
                            slot2 = Int32(0)
                            ph2 = ph1 ^ 1
                        mbar_wait(full + slot2 * 8, ph2)
                        term = Int32(lds32(cnts + slot2 * SB * 4)) == TERM
                    addr = col + slot2 * (NH * HALFB) + j2 * SUBB
                    for jj in cutlass.range_constexpr(G):
                        xa[jj] = lds32(addr + jj * ROWB)
                    for jj in cutlass.range_constexpr(G):
                        acc = hadd2(acc, xb[jj])
                    if j2 == 0:
                        mbar_arrive(empty + slot1 * 8)
                    j = j2
                    slot = slot2
                    ph = ph2
                    go = not term
            if active:
                out0 = mOut.iterator.toint()
                stg32(out0 + (Int64(d) * O + Int64(c0) + Int64(cw) * 2) * 2, acc)

    @cute.jit
    def fill_neg0(self, stage: Int32, r0: Int32, r1: Int32, lane: Int32):
        """Producer warp: rows r0..r1-1 of every half of the stage at smem address stage := bf16 -0."""
        BOXC, NH, SB = self.c["BOXC"], self.c["NH"], self.c["SB"]
        ROWB = BOXC * 2
        V = ROWB // 16  # 16 B vectors per row slot
        n = (r1 - r0) * V
        for h in cutlass.range_constexpr(NH):
            base = stage + h * (SB * G * ROWB) + r0 * ROWB
            for e in cutlass.range(lane, n, 32, unroll=1):
                sts128c(base + e * 16)

    @cute.jit
    def __call__(self, mSrc: cute.Tensor, mIdx: cute.Tensor, mBm: cute.Tensor, mOut: cute.Tensor, stream):
        c = self.c
        BOXC, SB = c["BOXC"], c["SB"]
        R = mIdx.shape[0]
        NWP = (R + 127) // 128 * 4  # bitmap words per destination, padded to 16 B
        tma32, tmt32 = cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileG2SOp(), mSrc, cute.make_layout((G, BOXC), stride=(BOXC, 1)), (G, BOXC))
        tmaS, tmtS = cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileG2SOp(), mSrc, cute.make_layout((SB * G, BOXC), stride=(BOXC, 1)),
            (SB * G, BOXC))
        self.bitmap_kernel(mIdx, mBm, R, NWP).launch(grid=[R // 256 + 2, 1, 1], block=[256, 1, 1], stream=stream)
        self.main_kernel(tma32, tmt32, tmaS, tmtS, mSrc, mBm, mOut, NWP).launch(
            grid=[ND * NTILES, 1, 1], block=[32 * (1 + c["NCW"]), 1, 1], stream=stream, use_pdl=True
        )


_CACHE = {}


def _compiled(o):
    fn = _CACHE.get(o)
    if fn is None:
        r = cute.sym_int()
        nb = cute.sym_int()
        fn = cute.compile(
            IndexAddBias(o),
            cute.runtime.make_fake_tensor(cutlass.BFloat16, (r, o), stride=(o, 1), assumed_align=16),
            cute.runtime.make_fake_tensor(cutlass.Int64, (r,), stride=(1,), assumed_align=8),
            cute.runtime.make_fake_tensor(cutlass.Int32, (nb,), stride=(1,), assumed_align=16),
            cute.runtime.make_fake_tensor(cutlass.BFloat16, (ND, o), stride=(o, 1), assumed_align=16),
            cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
            options="--enable-tvm-ffi",
        )
        _CACHE[o] = fn
    return fn


def run(in0, in1):
    r, o = in0.shape
    if (in0.dtype != torch.bfloat16 or in1.dtype != torch.int64 or in0.dim() != 2 or in1.dim() != 1
            or in1.shape[0] != r or o % 8 or in0.stride() != (o, 1) or in1.stride(0) != 1
            or in0.data_ptr() % 16 or in1.data_ptr() % 8 or in1.device != in0.device):
        raise ValueError(f"unsupported inputs: {in0.dtype} {tuple(in0.shape)} {in0.stride()} ptr%16="
                         f"{in0.data_ptr() % 16}, {in1.dtype} {tuple(in1.shape)} {in1.stride()}")
    if r == 0:
        # no rows: every destination keeps its new_zeros value (a TMA map needs a nonzero extent)
        return torch.zeros((ND, o), dtype=torch.bfloat16, device=in0.device)
    nbm = ND * ((r + 127) // 128 * 4) + 2 * ND  # bitmap words + (started, progress) per destination
    fill = torch.utils.deterministic.fill_uninitialized_memory
    if fill:
        torch.utils.deterministic.fill_uninitialized_memory = False  # both buffers are fully overwritten
    try:
        out = torch.empty((ND, o), dtype=torch.bfloat16, device=in0.device)
        bm = torch.empty((nbm,), dtype=torch.int32, device=in0.device)
    finally:
        if fill:
            torch.utils.deterministic.fill_uninitialized_memory = True
    _compiled(o)(in0, in1, bm, out)
    return out
