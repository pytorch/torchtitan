# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""ROCm GPU_MAX_HW_QUEUES sizing helper for graph_trainer.

Estimates the number of concurrently active GPU streams from the training config
and recommends a ``GPU_MAX_HW_QUEUES`` value (the next power of two). Run as a
module to print the value; see the README for usage.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from torchtitan.config import JobConfig


def _stream_lanes(
    *,
    dp_shard_active: bool,
    is_moe: bool,
    ep: int,
    tp: int,
    cp: int,
    fsdp_ag_rs_overlap: bool,
    cudagraph: bool,
) -> list[str]:
    """Return labels for each independent GPU stream the compiled step creates."""
    lanes = ["compute", "all_reduce"]
    if dp_shard_active:
        if fsdp_ag_rs_overlap:
            # reassign_collective_pgs_pass gives dense AG and RS dedicated PGs.
            lanes += ["dense_all_gather", "reduce_scatter"]
        else:
            lanes.append("fsdp_comm")
    if is_moe and ep > 1:
        # Expert AG and a2a never overlap, so they share one stream.
        lanes.append("expert_comm")
    if tp > 1:
        lanes.append("tp_all_gather")
    if cp > 1:
        lanes.append("cp_all_gather")
    # cudagraph_pass is skipped when fsdp_ag_rs_overlap rewrites the graph.
    if cudagraph and not fsdp_ag_rs_overlap:
        lanes.append("cudagraph_capture")
    return lanes


def _next_pow2(n: int) -> int:
    return 1 << (n - 1).bit_length()


def recommend_gpu_max_hw_queues(config: JobConfig) -> tuple[int, list[str]]:
    """Estimate the GPU stream count and recommend a GPU_MAX_HW_QUEUES value.

    Returns ``(queues, lanes)`` where ``queues`` is the next power of two at or
    above the estimated number of concurrently active GPU streams, and ``lanes``
    is the list of stream labels used for the estimate. Pure function: reads the
    config only, sets no environment variables.
    """
    from torchtitan.models.common.moe import MoE

    p = config.parallelism
    dp_shard = p.data_parallel_shard_degree
    dp_shard_active = dp_shard == -1 or dp_shard > 1
    is_moe = (
        config.model_spec is not None
        and next(config.model_spec.model.traverse(MoE.Config), None) is not None
    )
    compile_cfg = config.compile
    cudagraph = getattr(compile_cfg, "enable_passes", False) and (
        "cudagraph_pass" not in getattr(compile_cfg, "disable_passes", [])
    )
    fsdp_ag_rs_overlap = getattr(compile_cfg, "enable_fsdp_ag_rs_overlap", False)

    lanes = _stream_lanes(
        dp_shard_active=dp_shard_active,
        is_moe=is_moe,
        ep=p.expert_parallel_degree,
        tp=p.tensor_parallel_degree,
        cp=p.context_parallel_degree,
        fsdp_ag_rs_overlap=fsdp_ag_rs_overlap,
        cudagraph=cudagraph,
    )
    return _next_pow2(len(lanes)), lanes


def main() -> None:
    from torchtitan.config import ConfigManager

    config = ConfigManager().parse_args()
    queues, lanes = recommend_gpu_max_hw_queues(config)

    import sys

    print(
        f"# estimated {len(lanes)} concurrent GPU streams: {', '.join(lanes)}",
        file=sys.stderr,
    )
    if "GPU_MAX_HW_QUEUES" in os.environ:
        print(
            f"# note: GPU_MAX_HW_QUEUES already set to "
            f"{os.environ['GPU_MAX_HW_QUEUES']} in your environment",
            file=sys.stderr,
        )
    print(f"export GPU_MAX_HW_QUEUES={queues}")


if __name__ == "__main__":
    main()
