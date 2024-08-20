from typing import Tuple

from comm_analysis import (
    estimate_nccl_collective_lat_and_bw,
    NCCL_ALGO,
    NCCL_COLL,
    NCCL_HW,
    NCCL_PROTO,
)

# FIXME @sanketpurandare: Update the new permalink
# Based on :https://github.com/pytorch/pytorch/blob/44a773c12159a40afb5e31bd233984258774c1b7/torch/_inductor/comm_analysis.py


def get_collective_latency_bandwidth(
    coll: NCCL_COLL,
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
        group_size (int): The size of the group participating in the collective operation.
        num_gpus_per_node (int): The number of GPUs per node participating in the collective operation.
    Returns:
        A tuple containing the estimated latency and bandwidth of the collective operation,
        in miliseconds (ms) and Bytes/milisecondss (B/ms), respectively.
    """
    intraHw = NCCL_HW.NVLINK
    interHw = NCCL_HW.NET
    nccl_proto = NCCL_PROTO.LL
    nccl_algo = NCCL_ALGO.RING
    latency_ns, bandwidth_GB_ns = estimate_nccl_collective_lat_and_bw(
        coll,
        intraHw,
        interHw,
        nccl_proto,
        nccl_algo,
        group_size,
        num_gpus_per_node,
    )
    latency_ms = latency_ns / 1e6
    bandwidth_B_ms = bandwidth_GB_ns * ((1024**3) * 1e6)
    return (latency_ms, bandwidth_B_ms)


if __name__ == "__main__":
    tensor_size = 200 * (1024**2)  # 20 MB Tensor
    world_size = 64
    num_gpus_per_node = 8

    all_gather_latency, all_gather_bw = get_collective_latency_bandwidth(
        NCCL_COLL.ALL_GATHER, world_size, num_gpus_per_node
    )
    all_gather_time = all_gather_latency + (tensor_size / all_gather_bw)
    print(
        f"All Gather time for Tensor Size {tensor_size} B is {all_gather_time:.3f} ms"
    )
