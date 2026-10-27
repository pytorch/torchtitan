# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
# This source code is licensed under the BSD-style license found in LICENSE.

"""Two CPU ranks must recover their own stream from one real DCP cursor."""

import hashlib
import json
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
from torchtitan.last_words.rng import RNG_STATE, RNGCheckpointState
from torchtitan.last_words.validation import validate_rng_cursor


def sample_digest():
    sample = dict(
        python=[random.random(), random.gauss(0, 1)],
        numpy=np.random.standard_normal(8).tolist(),
        torch=torch.rand(8).tolist(),
    )
    return hashlib.sha256(json.dumps(sample, sort_keys=True).encode()).hexdigest()


def main():
    dist.init_process_group("gloo")
    try:
        rank = dist.get_rank()
        output = Path(sys.argv[1])
        if rank == 0:
            output.mkdir(exist_ok=False)
        dist.barrier()
        random.seed(501 + rank)
        np.random.seed(601 + rank)
        torch.manual_seed(701 + rank)
        state = RNGCheckpointState(torch.device("cpu"), rank)
        state.capture()
        expected = sample_digest()
        checkpoint = output / "cursor"
        dcp.save({RNG_STATE: state}, checkpoint_id=checkpoint)
        for _ in range(3):
            sample_digest()
        state.capture()
        dcp.load({RNG_STATE: state}, checkpoint_id=checkpoint)
        state.restore()
        actual = sample_digest()
        assert actual == expected, f"Rank {rank} recovered another RNG stream"
        (output / f"rank-{rank}.json").write_text(
            json.dumps(
                dict(
                    rank=rank,
                    expected=expected,
                    actual=actual,
                    equal=True,
                    pid=os.getpid(),
                )
            )
        )
        dist.barrier()
        if rank == 0:
            validate_rng_cursor(checkpoint, dict(rng_schema=1, rng_shards=2), 2)
            records = [
                json.loads((output / f"rank-{index}.json").read_text())
                for index in range(2)
            ]
            assert records[0]["actual"] != records[1]["actual"]
            (output / "result.json").write_text(
                json.dumps(
                    dict(
                        world_size=2,
                        all_streams_restored=True,
                        independent_streams=True,
                        ranks=records,
                    ),
                    indent=2,
                )
            )
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
