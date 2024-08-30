import math
from typing import Tuple

import torch
from torch._inductor.comm_analysis import (
    baseLat,
    get_gpu_type,
    hwLat,
    llMaxBws,
    NCCL_ALGO,
    NCCL_COLL,
    NCCL_HW,
    NCCL_PROTO,
)


def estimate_nccl_collective_lat_and_bw(
    coll: NCCL_COLL,
    intraHw: NCCL_HW,
    interHw: NCCL_HW,
    nccl_proto: NCCL_PROTO,
    nccl_algo: NCCL_ALGO,
    group_size: int,
    num_gpus_per_node: int,
) -> Tuple[float, float]:
    """
    Estimates the latency and bandwidth of an NCCL collective operation.
    Args:
        coll (NCCL_COLL): The type of collective operation to perform. Can be one of:
            * `ALL_REDUCE` (0)
            * `ALL_GATHER` (1)
            * `REDUCE_SCATTER` (2)
        intraHw (NCCL_HW): The hardware to use for intra-node communication. Can be one of:
            * `NVLINK` (0)
            * `PCI` (1)
            * `NET` (2)
        interHw (NCCL_HW): The hardware to use for inter-node communication. Can be one of:
            * `NVLINK` (0)
            * `PCI` (1)
            * `NET` (2)
        nccl_proto (NCCL_PROTO): The protocol to use for NCCL communication. Can be one of:
            * `LL` (0) - Low-latency
            * `LL128` (1) - Low-latency 128-byte
            * `SIMPLE` (2)
        nccl_algo (NCCL_ALGO): The algorithm to use for NCCL communication. Can be one of:
            * `TREE` (0)
            * `RING` (1)
        group_size (int): The size of the group participating in the collective operation.
        num_gpus_per_node (int): The number of GPUs per node participating in the collective operation.
    Returns:
        A tuple containing the estimated latency and bandwidth of the collective operation,
        in nanoseconds and GB/nanoseconds, respectively.
    """

    nNodes = math.ceil(group_size / num_gpus_per_node)
    nRanks = group_size  # this is total # of gpus globally that participate in this collective op

    if nRanks <= 1:
        return (0, 0)

    # =============== bandwidth computation ===============
    # First compute bandwidth in GB/s; then at the end, convert it to GB/ns

    bwIntra = torch._inductor.config.intra_node_bw
    bwInter = torch._inductor.config.inter_node_bw

    compCapIndex = get_gpu_type()
    index2 = nNodes - 1 if nNodes <= 2 else 2
    # LL: for single node, we look at GPU type; for multi-node, we look at CPU type
    index1 = compCapIndex if nNodes == 1 else 0
    llMaxBw = llMaxBws[index1][index2]

    # NOTE: each step of ring algorithm is synchronized,
    # and is bottlenecked by the slowest link which is the inter-node interconnect.
    # hence when nNodes >= 2, bw is inter-node bandwidth.
    # NOTE: the original code in https://github.com/NVIDIA/nccl/blob/master/src/graph/tuning.cc
    # have this as `if nNodes <= 2` which seems wrong. Corrected it here.
    bw = bwIntra if nNodes == 1 else bwInter
    nChannels = 2  # Assume # channels is 2
    busBw = nChannels * bw

    # Various model refinements
    busBw = min(
        llMaxBw,
        busBw
        * (1.0 / 4.0 if (nNodes > 1 or coll == NCCL_COLL.ALL_REDUCE) else 1.0 / 3.0),
    )

    if coll == NCCL_COLL.ALL_REDUCE:
        nsteps = 2 * (nRanks - 1)
    elif coll in (NCCL_COLL.REDUCE_SCATTER, NCCL_COLL.ALL_GATHER):
        nsteps = nRanks - 1

    # Convert bus BW to algorithm BW (tensor bytes / algoBW = actual execution time)
    ratio = (1.0 * nRanks) / nsteps  # type: ignore[possibly-undefined]
    bandwidth = busBw * ratio
    # Convert GB/s to GB/ns
    bandwidth_GB_per_ns = bandwidth / 1e9

    # =============== latency computation ===============

    if coll == NCCL_COLL.ALL_REDUCE:
        if nNodes > 1:
            nInterSteps = 2 * nNodes
        else:
            nInterSteps = 0
    elif coll in (NCCL_COLL.REDUCE_SCATTER, NCCL_COLL.ALL_GATHER):
        nInterSteps = nNodes - 1

    # First compute latency in us; then at the end, convert it to ns
    latency = baseLat[nccl_algo][nccl_proto]
    intraLat = hwLat[intraHw][nccl_algo][nccl_proto]
    interLat = hwLat[interHw][nccl_algo][nccl_proto]

    # Inter-node rings still have to launch nsteps * net overhead.
    netOverhead = 0.0
    if nNodes > 1:
        netOverhead = 1.0  # getNetOverhead(comm);
    intraLat = max(intraLat, netOverhead)
    latency += (nsteps - nInterSteps) * intraLat + nInterSteps * interLat  # type: ignore[possibly-undefined]
    # Convert us to ns
    latency_ns = latency * 1e3
    return (latency_ns, bandwidth_GB_per_ns)
