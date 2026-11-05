"""Fused MoE router forward recompute (DSv3 671B, 4096 tokens x 256 experts, top-8).

CuTe DSL implementation of reference.py, bitwise identical to torch's CUDA kernels.

One launch: 128 CTAs (64 clusters of 2) x 1024 threads, one token row per warp.
  * per row: sigmoid, row L1 norm, per-row outputs (v6, v15, v17, v20), top-k gather and
    normalization (v7, v9, v12), and the masked row-normalized scores v21 kept in smem.
  * per CTA: torch's per-thread partial column sums of v21 for its two row classes; the CTA
    pair combines them through DSMEM into torch's per-CTA value C_b; packed routing counts.
  * the last cluster to finish (generation counter) runs the cross-cluster tail: column sums,
    routing counts, v27, and the scalar loss terms.

Reduction orders reproduce torch's Reduce.cuh on this exact shape (GB300, fp32):
  * sum(v7, -1) over 8: shfl_down tree (4, 2, 1).
  * vector_norm(v6, 1, -1) over 256: vec-4 input, 32 lanes, two float4 per lane summed per
    vector slot, slots combined in order, shfl_down tree (16..1).
  * sum(v21, 0) over 4096 rows: output_vec 4, block (32, 4), 64 CTAs split over rows; row
    r = w + 4 * b + 256 * m. Per torch thread (w, b): accumulators A_i over m = i + 4 j in j
    order, P = ((A0 + A1) + A2) + A3; per torch CTA b: C_b = (P_w0 + P_w2) + (P_w1 + P_w3);
    final: T_y = C_y + C_{y+4} + ... + C_{y+60} sequentially, v22 = (T0 + T2) + (T1 + T3).
  * sum(v28) over 256: vec-4 input, 64 threads, x-reduce through smem then shfl tree.
Counts of booleans are exact in any order.

Rounding: every float add/mul is an explicit round-to-nearest op (no FMA contraction).
  * sigmoid = rcp.rn(1 + exp(-x)) with exp the non-fastmath expansion. rcp.rn and div.rn(1, .)
    are both IEEE correctly rounded, hence identical. The reciprocal is ptxas's own rcp.rn
    fast-path sequence (MUFU.RCP + one FMA refinement); rows with any x < -69 or NaN
    (1 + exp(-x) > 2^99) are recomputed with rcp.rn. Checked bit-exact against torch.sigmoid
    on all 2^32 fp32 inputs (in order and randomly permuted).
  * v6 / v17 (per-row divisor) uses ptxas's div.rn fast-path sequence with the divisor's
    refined reciprocal computed once per row; rows outside the range where that sequence is
    exact (any dividend < 2^-99.6 or NaN, divisor outside [2^-100, 2^20]) use div.rn.
"""

import cutlass
import cutlass.cute as cute
import numpy as np
import torch
from cutlass import Float32, Int32, Int64, Uint8
from cutlass._mlir_helpers.math import RoundingMode

_RN = RoundingMode.NEAREST_EVEN
_T = 4096  # tokens
_E = 256  # experts
_K = 8  # top-k
_NB = 64  # torch's ctas_per_output for sum(v21, 0); one cluster per b
_NCTA = 2 * _NB
_WARPS = 32  # rows per CTA, one per warp

# Scalar constants exactly as torch casts the python scalars to fp32.
_EPS20 = float(np.float32(1e-20))
_EPS12 = float(np.float32(1e-12))
_MILLI = float(np.float32(0.001))
_X_FAST_MIN = -69.0  # x >= -69: 1 + exp(-x) < 2^100 and sigmoid(x) > 2^-100
_B_FAST_MIN = 2.0**-100
_B_FAST_MAX = 2.0**20


def _add(a, b):
    return cute.math.add(a, b, rounding=_RN)


def _mul(a, b):
    return cute.math.mul(a, b, rounding=_RN)


def _fma(a, b, c):
    return cute.math.fma(a, b, c, rounding=_RN)


def _div(a, b):
    return cute.math.div(a, b, rounding=_RN)


def _rcp_rn(a):
    return cute.math.rcp(a, rounding=_RN)


def _vec4(ptr, off):
    return cute.make_tensor((ptr + off).align(16), cute.make_layout(4))


def _cluster_addr(smem_ptr, rank):
    return Int32(cute.arch.map_dsmem_ptr(smem_ptr, rank).toint())


def _st_async(val, remote_addr, remote_mbar_addr):
    """st.async to a peer CTA's smem, completing tx bytes on the peer's mbarrier."""
    suffix = "f32" if isinstance(val, Float32) else "s32"
    cute.arch.inline_ptx(
        "st.async.shared::cluster.mbarrier::complete_tx::bytes." + suffix + " [{$r0}], {$r1}, [{$r2}];",
        read_only_args=[remote_addr, val, remote_mbar_addr],
    )


@cute.kernel
def _kernel(
    mIn0: cute.Tensor,
    mIn1: cute.Tensor,
    mIn2: cute.Tensor,
    mIn3w: cute.Tensor,
    mIn4: cute.Tensor,
    mV6: cute.Tensor,
    mV7: cute.Tensor,
    mV9: cute.Tensor,
    mV12: cute.Tensor,
    mV15: cute.Tensor,
    mV17: cute.Tensor,
    mV20: cute.Tensor,
    mWsC: cute.Tensor,
    mWsCnt: cute.Tensor,
):
    tidx, _, _ = cute.arch.thread_idx()
    bidx, _, _ = cute.arch.block_idx()
    lane = tidx % 32
    warp = tidx // 32
    h = bidx % 2  # rank in the CTA pair
    b = bidx // 2  # torch's reduction CTA index for sum(v21, 0)
    rl = warp // 16
    m = warp % 16
    row = b * 4 + h * 2 + rl + 256 * m  # row class r0 = 4 b + 2 h + rl, rows r0 + 256 m

    smem = cutlass.utils.SmemAllocator()
    # sY: per-warp row of sigmoid values (top-k gather), then of v21 (column partials).
    # The tail reuses sY..sStage as staging for C and the packed counts.
    sY = smem.allocate_tensor(Float32, cute.make_layout((_E, _WARPS)), byte_alignment=16)
    sP = smem.allocate_tensor(Float32, cute.make_layout((_E, 2)), byte_alignment=16)
    sPeer = smem.allocate_tensor(Float32, cute.make_layout((_E, 2)), byte_alignment=16)
    sCnt = smem.allocate_tensor(Int32, cute.make_layout(64), byte_alignment=16)
    sCntPeer = smem.allocate_tensor(Int32, cute.make_layout(64), byte_alignment=16)
    mbar = smem.allocate_array(cutlass.Int64, 1, byte_alignment=8)

    # ---- issue this warp's loads ----
    p0 = mIn0.iterator
    xa = cute.make_rmem_tensor(cute.make_layout(4), Float32)
    xb = cute.make_rmem_tensor(cute.make_layout(4), Float32)
    cute.autovec_copy(_vec4(p0, row * _E + lane * 4), xa)
    cute.autovec_copy(_vec4(p0, row * _E + 128 + lane * 4), xb)
    wa = mIn3w[row, lane]
    wb = mIn3w[row, 32 + lane]
    nmb = mIn4[row]
    gk = lane % 8
    gidx = Int32(mIn1[row, gk])
    gmask = mIn2[row]

    if tidx < 64:
        sCnt[tidx] = Int32(0)
    if tidx == 0:
        cute.arch.mbarrier_init(mbar, 1)
        cute.arch.mbarrier_init_fence()
    # Peer's mbarrier init must be visible before rank 0's st.async; waited on below.
    cute.arch.cluster_arrive_relaxed()
    # Let the tail kernel launch now; it waits for this grid with griddepcontrol.wait.
    cute.arch.griddepcontrol_launch_dependents()

    # ---- sigmoid: 1 + exp(-x), then rcp.rn's fast-path sequence ----
    ya = cute.make_rmem_tensor(cute.make_layout(4), Float32)
    yb = cute.make_rmem_tensor(cute.make_layout(4), Float32)
    lane_ok = (xa[0] >= Float32(_X_FAST_MIN)) & (xb[0] >= Float32(_X_FAST_MIN))
    for k in cutlass.range_constexpr(1, 4):
        lane_ok = lane_ok & (xa[k] >= Float32(_X_FAST_MIN))
        lane_ok = lane_ok & (xb[k] >= Float32(_X_FAST_MIN))
    for k in cutlass.range_constexpr(4):
        da = _add(Float32(1.0), cute.math.exp(-xa[k], fastmath=False))
        db = _add(Float32(1.0), cute.math.exp(-xb[k], fastmath=False))
        ra = cute.arch.rcp_approx(da)
        rb = cute.arch.rcp_approx(db)
        ya[k] = _fma(ra, _fma(-da, ra, Float32(1.0)), ra)
        yb[k] = _fma(rb, _fma(-db, rb, Float32(1.0)), rb)
    if not lane_ok:
        for k in cutlass.range_constexpr(4):
            ya[k] = _rcp_rn(_add(Float32(1.0), cute.math.exp(-xa[k], fastmath=False)))
            yb[k] = _rcp_rn(_add(Float32(1.0), cute.math.exp(-xb[k], fastmath=False)))
    p6 = mV6.iterator
    cute.autovec_copy(ya, _vec4(p6, row * _E + lane * 4))
    cute.autovec_copy(yb, _vec4(p6, row * _E + 128 + lane * 4))
    py = sY.iterator + warp * _E
    cute.autovec_copy(ya, _vec4(py, lane * 4))
    cute.autovec_copy(yb, _vec4(py, 128 + lane * 4))

    # ---- L1 norm over the row in torch's order ----
    a0 = _add(ya[0], yb[0])
    a1 = _add(ya[1], yb[1])
    a2 = _add(ya[2], yb[2])
    a3 = _add(ya[3], yb[3])
    s = _add(_add(_add(a0, a1), a2), a3)
    for off in cutlass.range_constexpr(5):
        s = _add(s, cute.arch.shuffle_sync_bfly(s, 16 >> off))
    v15 = s
    v16 = v15
    if v15 < Float32(_EPS12):
        v16 = Float32(_EPS12)
    notm = Uint8(1) - nmb
    nmf = Float32(notm)
    if lane == 0:
        mV15[row] = v15
        mV17[row] = v16
        mV20[row] = notm

    # ---- top-k gather + normalization (lanes 0..7 own slot k = lane) ----
    cute.arch.sync_warp()
    val = sY[gidx, warp]
    t = val
    t = _add(t, cute.arch.shuffle_sync_bfly(t, 4))
    t = _add(t, cute.arch.shuffle_sync_bfly(t, 2))
    t = _add(t, cute.arch.shuffle_sync_bfly(t, 1))
    if lane < 8:
        v9 = _add(t, Float32(_EPS20))
        v11 = _mul(_div(val, v9), Float32(2.5))
        v12 = v11
        if gmask != Uint8(0):
            v12 = Float32(0.0)
        mV7[row, gk] = val
        mV12[row, gk] = v12
        if lane == 0:
            mV9[row] = v9

    # ---- v21 = (v6 / v17) * ~in4: div.rn fast path with the row's refined reciprocal ----
    rr = cute.arch.rcp_approx(v16)
    rr = _fma(rr, _fma(-v16, rr, Float32(1.0)), rr)
    qa = cute.make_rmem_tensor(cute.make_layout(4), Float32)
    qb = cute.make_rmem_tensor(cute.make_layout(4), Float32)
    for k in cutlass.range_constexpr(4):
        q = _mul(ya[k], rr)
        qa[k] = _fma(rr, _fma(-v16, q, ya[k]), q)
        q = _mul(yb[k], rr)
        qb[k] = _fma(rr, _fma(-v16, q, yb[k]), q)
    row_ok = (v16 >= Float32(_B_FAST_MIN)) & (v16 <= Float32(_B_FAST_MAX))
    if not (lane_ok & row_ok):
        for k in cutlass.range_constexpr(4):
            qa[k] = _div(ya[k], v16)
            qb[k] = _div(yb[k], v16)
    for k in cutlass.range_constexpr(4):
        qa[k] = _mul(qa[k], nmf)
        qb[k] = _mul(qb[k], nmf)
    cute.arch.sync_warp()  # gather reads of sY done before v21 overwrites it
    cute.autovec_copy(qa, _vec4(py, lane * 4))
    cute.autovec_copy(qb, _vec4(py, 128 + lane * 4))
    cute.arch.sync_threads()  # sCnt init, sY v21 rows
    cute.arch.atomic_add(sCnt.iterator + lane, wa)
    cute.arch.atomic_add(sCnt.iterator + 32 + lane, wb)

    # ---- torch's per-thread partial P for this CTA's two row classes ----
    if tidx < 2 * _E:
        col = tidx % _E
        q2 = tidx // _E
        acc = []
        for i in cutlass.range_constexpr(4):
            a = sY[col, 16 * q2 + i]
            for j in cutlass.range_constexpr(1, 4):
                a = _add(a, sY[col, 16 * q2 + i + 4 * j])
            acc.append(a)
        sP[col, q2] = _add(_add(_add(acc[0], acc[1]), acc[2]), acc[3])
    cute.arch.sync_threads()

    # ---- CTA pair: rank 0 pushes P[4b], P[4b+1] and its counts into rank 1's smem ----
    cute.arch.cluster_wait()
    if h == 0:
        if tidx < 2 * _E:
            peer_mbar = _cluster_addr(mbar, 1)
            _st_async(sP[tidx % _E, tidx // _E], _cluster_addr(sPeer.iterator + tidx, 1), peer_mbar)
            if tidx < 64:
                _st_async(sCnt[tidx], _cluster_addr(sCntPeer.iterator + tidx, 1), peer_mbar)
    else:
        if tidx == 0:
            cute.arch.mbarrier_arrive_and_expect_tx(mbar, (2 * _E + 64) * 4)
        cute.arch.mbarrier_wait(mbar, 0)
        if tidx < _E:
            col = tidx
            # C_b = (P0 + P2) + (P1 + P3)
            mWsC[b, col] = _add(_add(sPeer[col, 0], sP[col, 0]), _add(sPeer[col, 1], sP[col, 1]))
            if tidx < 64:
                mWsCnt[b, tidx] = sCnt[tidx] + sCntPeer[tidx]


@cute.kernel
def _tail_kernel(
    mWsC: cute.Tensor,
    mWsCnt: cute.Tensor,
    mIn5: cute.Tensor,
    mV27: cute.Tensor,
    mV33: cute.Tensor,
    mV34: cute.Tensor,
):
    """Cross-cluster tail: 1024 threads, thread (y, col) owns torch's last-CTA chain T_y[col]."""
    tidx, _, _ = cute.arch.thread_idx()
    lane = tidx % 32
    warp = tidx // 32
    col = tidx % _E
    y = tidx // _E

    smem = cutlass.utils.SmemAllocator()
    sT = smem.allocate_tensor(Float32, cute.make_layout((_E, 4)), byte_alignment=16)
    sCntP = smem.allocate_tensor(Int32, cute.make_layout((_E, 4)), byte_alignment=16)
    sX = smem.allocate_tensor(Float32, cute.make_layout(_E), byte_alignment=16)
    sTot = smem.allocate_tensor(Int32, cute.make_layout(8), byte_alignment=16)

    n = mIn5[0]
    cute.arch.griddepcontrol_wait()

    w = col // 4
    sh = (col % 4) * 8
    cv = [mWsC[y + 4 * kk, col] for kk in range(16)]
    cw = [mWsCnt[16 * y + kk, w] for kk in range(16)]
    # T_y = C[y] + C[y + 4] + ... + C[y + 60], sequential (torch's last-CTA loop).
    a = cv[0]
    for kk in cutlass.range_constexpr(1, 16):
        a = _add(a, cv[kk])
    sT[col, y] = a
    # Routing counts: bytes of the packed words, b in [16 y, 16 y + 16).
    cnt = Int32(0)
    for kk in cutlass.range_constexpr(16):
        cnt = cnt + ((cw[kk] >> sh) & Int32(0xFF))
    sCntP[col, y] = cnt
    cute.arch.sync_threads()
    if tidx < _E:
        v22 = _add(_add(sT[col, 0], sT[col, 2]), _add(sT[col, 1], sT[col, 3]))
        c = sCntP[col, 0] + sCntP[col, 1] + sCntP[col, 2] + sCntP[col, 3]
        tot = c
        for off in cutlass.range_constexpr(5):
            tot = tot + cute.arch.shuffle_sync_bfly(tot, 16 >> off)
        if lane == 0:
            sTot[warp] = tot
        cute.arch.barrier(barrier_id=1, number_of_threads=_E)
        total = sTot[0]
        for k in cutlass.range_constexpr(1, 8):
            total = total + sTot[k]
        v24 = Float32(total)
        if v24 < Float32(_EPS12):
            v24 = Float32(_EPS12)
        v27 = _mul(_div(Float32(c), v24), Float32(256.0))
        mV27[col] = v27
        sX[col] = _mul(v27, v22)
        cute.arch.barrier(barrier_id=1, number_of_threads=_E)
        if warp == 0:
            s0 = _add(_add(_add(sX[4 * lane], sX[4 * lane + 1]), sX[4 * lane + 2]), sX[4 * lane + 3])
            s1 = _add(
                _add(_add(sX[128 + 4 * lane], sX[128 + 4 * lane + 1]), sX[128 + 4 * lane + 2]),
                sX[128 + 4 * lane + 3],
            )
            u = _add(s0, s1)
            for off in cutlass.range_constexpr(5):
                u = _add(u, cute.arch.shuffle_sync_bfly(u, 16 >> off))
            if lane == 0:
                if n < Int64(1):
                    n = Int64(1)
                v31 = _div(Float32(1.0), Float32(n))
                v33 = _mul(_mul(v31, Float32(1.0)), Float32(_MILLI))
                mV33[0] = v33
                mV34[0] = _mul(u, v33)


@cute.jit
def _launch(
    mIn0: cute.Tensor,
    mIn1: cute.Tensor,
    mIn2: cute.Tensor,
    mIn3w: cute.Tensor,
    mIn4: cute.Tensor,
    mIn5: cute.Tensor,
    mV6: cute.Tensor,
    mV7: cute.Tensor,
    mV9: cute.Tensor,
    mV12: cute.Tensor,
    mV15: cute.Tensor,
    mV17: cute.Tensor,
    mV20: cute.Tensor,
    mV27: cute.Tensor,
    mV33: cute.Tensor,
    mV34: cute.Tensor,
    mWsC: cute.Tensor,
    mWsCnt: cute.Tensor,
    stream,
):
    _kernel(
        mIn0, mIn1, mIn2, mIn3w, mIn4, mV6, mV7, mV9, mV12, mV15, mV17, mV20, mWsC, mWsCnt
    ).launch(grid=[_NCTA, 1, 1], block=[32 * _WARPS, 1, 1], cluster=[2, 1, 1], stream=stream)
    _tail_kernel(mWsC, mWsCnt, mIn5, mV27, mV33, mV34).launch(
        grid=[1, 1, 1], block=[4 * _E, 1, 1], stream=stream, use_pdl=True
    )


def _fake(dtype, shape, align=16):
    stride = []
    acc = 1
    for s in reversed(shape):
        stride.append(acc)
        acc *= s
    return cute.runtime.make_fake_tensor(
        dtype, tuple(shape), stride=tuple(reversed(stride)), assumed_align=align
    )


_COMPILED = None
def _get_compiled():
    global _COMPILED
    if _COMPILED is None:
        _COMPILED = cute.compile(
            _launch,
            _fake(Float32, (_T, _E)),
            _fake(Int64, (_T, _K)),
            _fake(Uint8, (_T,)),
            _fake(Int32, (_T, _E // 4)),
            _fake(Uint8, (_T,)),
            _fake(Int64, (1,), 8),
            _fake(Float32, (_T, _E)),
            _fake(Float32, (_T, _K)),
            _fake(Float32, (_T,)),
            _fake(Float32, (_T, _K)),
            _fake(Float32, (_T,)),
            _fake(Float32, (_T,)),
            _fake(Uint8, (_T,)),
            _fake(Float32, (_E,)),
            _fake(Float32, (1,), 4),
            _fake(Float32, (1,), 4),
            _fake(Float32, (_NB, _E)),
            _fake(Int32, (_NB, _E // 4)),
            cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
            options="--enable-tvm-ffi",
        )
    return _COMPILED


def _alloc(dev):
    f32 = torch.float32
    return (
        torch.empty((_T, _E), dtype=f32, device=dev),  # v6
        torch.empty((_T, _K), dtype=f32, device=dev),  # v7
        torch.empty((_T, 1), dtype=f32, device=dev),  # v9
        torch.empty((_T, _K), dtype=f32, device=dev),  # v12
        torch.empty((_T, 1), dtype=f32, device=dev),  # v15
        torch.empty((_T, 1), dtype=f32, device=dev),  # v16 (v17 = expand)
        torch.empty((_T, 1), dtype=torch.bool, device=dev),  # v20
        torch.empty((_E,), dtype=f32, device=dev),  # v27
        torch.empty((), dtype=f32, device=dev),  # v33
        torch.empty((), dtype=f32, device=dev),  # v34
        torch.empty((_NB, _E), dtype=f32, device=dev),  # C_b workspace
        torch.empty((_NB, _E // 4), dtype=torch.int32, device=dev),  # packed counts workspace
    )


def _dense(t):
    """The kernels assume contiguous inputs with 16-byte aligned bases (vector loads)."""
    if t.is_contiguous() and t.data_ptr() % 16 == 0:
        return t
    return t.clone(memory_format=torch.contiguous_format)


def run(in0, in1, in2, in3, in4, in5):
    in0, in1, in2, in3, in4, in5 = (_dense(t) for t in (in0, in1, in2, in3, in4, in5))
    # Every element of every buffer below is written by the kernels, so skip the NaN fill that
    # torch.use_deterministic_algorithms(True) adds to torch.empty (one fill kernel per buffer).
    det = torch.utils.deterministic
    fill = det.fill_uninitialized_memory
    det.fill_uninitialized_memory = False
    try:
        v6, v7, v9, v12, v15, v16, v20, v27, v33, v34, ws_c, ws_cnt = _alloc(in0.device)
    finally:
        det.fill_uninitialized_memory = fill
    _get_compiled()(
        in0,
        in1,
        in2.view(torch.uint8).view(_T),
        in3.view(torch.uint8).view(torch.int32),
        in4.view(torch.uint8),
        in5.view(1),
        v6,
        v7,
        v9.view(_T),
        v12,
        v15.view(_T),
        v16.view(_T),
        v20.view(torch.uint8).view(_T),
        v27,
        v33.view(1),
        v34.view(1),
        ws_c,
        ws_cnt,
    )
    v17 = v16.expand(_T, _E)
    v35 = torch.ops.aten.alias.default(v6)
    return (v6, v7, v9, v12, v15, v17, v20, v27, v33, v34, v35)
