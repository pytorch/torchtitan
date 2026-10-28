# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Force a collective-order mismatch on two or more CUDA GPUs.

  torchrun --standalone --nproc-per-node=4 scripts/repro_collective_order.py
  torchrun --standalone --nproc-per-node=4 scripts/repro_collective_order.py --separate-group

After matching warmup, rank 0 submits A2A then AG; peers submit AG then A2A.
Shared groups should time out. A dedicated group removes that aliasing, but
reversed cross-communicator order is not a supported NCCL ordering contract.
This is an intentional ordering-fault test, not a natural FSDP prefetch repro.
Run each case separately with an external deadline; a timeout can abort CUDA.
"""

import argparse
import os
from datetime import timedelta

import torch
import torch.distributed as dist


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--separate-group", action="store_true")
    args = parser.parse_args()
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    dist.init_process_group("nccl", timeout=timedelta(seconds=30))
    rank, world = dist.get_rank(), dist.get_world_size()
    elements = 1048576
    assert world >= 2 and elements % world == 0
    cp_group = dist.group.WORLD
    fsdp_group = cp_group
    if args.separate_group:
        fsdp_group = dist.new_group(timeout=timedelta(seconds=30))
    print(
        f"rank={rank} shared={cp_group is fsdp_group} torch={torch.__version__} "
        f"nccl={torch.cuda.nccl.version()} gpu={torch.cuda.get_device_name()} "
        f"implicit={os.getenv('NCCL_LAUNCH_ORDER_IMPLICIT', 'unset')}",
        flush=True,
    )
    ag_input = torch.full((elements,), float(rank), device="cuda")
    ag_output = torch.empty(elements * world, device="cuda")
    ranks = torch.arange(world, dtype=torch.float32, device="cuda")
    a2a_input = (rank * world + ranks).repeat_interleave(elements // world)
    a2a_output = torch.empty_like(a2a_input)
    expected_ag = ranks.repeat_interleave(elements)
    expected_a2a = (ranks * world + rank).repeat_interleave(elements // world)
    ag_stream, cp_stream = torch.cuda.Stream(), torch.cuda.Stream()
    torch.cuda.synchronize()
    for warmup in (True, False):
        order = ("ag", "a2a") if warmup or rank != 0 else ("a2a", "ag")
        works = []
        for operation in order:
            print(f"rank={rank} warmup={warmup} submit={operation}", flush=True)
            with torch.cuda.stream(ag_stream if operation == "ag" else cp_stream):
                if operation == "ag":
                    work = dist.all_gather_into_tensor(
                        ag_output, ag_input, group=fsdp_group, async_op=True
                    )
                else:
                    work = dist.all_to_all_single(
                        a2a_output, a2a_input, group=cp_group, async_op=True
                    )
                works.append(work)
            print(f"rank={rank} warmup={warmup} submitted={operation}", flush=True)
        # Keep both Work objects and buffers alive; submit both before waiting.
        for work in works:
            work.wait()
        torch.cuda.synchronize()
        torch.testing.assert_close(ag_output, expected_ag, rtol=0, atol=0)
        torch.testing.assert_close(a2a_output, expected_a2a, rtol=0, atol=0)
        print(f"rank={rank} {'WARMED' if warmup else 'PASS'}", flush=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
