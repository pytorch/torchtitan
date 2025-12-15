# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import os

from torchtitan.distributed import ParallelDims, utils as dist_utils
from torchtitan.train import main, Trainer


class FTTrainer(Trainer):
    def init_distributed(self) -> ParallelDims:
        job_config = self.job_config

        # determine the global ranks when fault tolerance is enabled
        global_ranks = []
        ft_config = job_config.fault_tolerance
        if ft_config.enable:
            group_size = ft_config.group_size
            replica_id = ft_config.replica_id
            first_rank = replica_id * group_size
            last_rank = first_rank + group_size - 1
            global_ranks = list(range(first_rank, last_rank + 1))

        # init distributed and build meshes
        dist_utils.init_distributed(
            job_config.comm,
            enable_cpu_backend=job_config.training.enable_cpu_offload,
            base_folder=job_config.job.dump_folder,
            ranks=global_ranks,
        )

        world_size = int(os.environ["WORLD_SIZE"])
        parallelism_config = job_config.parallelism

        return ParallelDims(
            dp_shard=parallelism_config.data_parallel_shard_degree,
            dp_replicate=parallelism_config.data_parallel_replicate_degree,
            cp=parallelism_config.context_parallel_degree,
            tp=parallelism_config.tensor_parallel_degree,
            pp=parallelism_config.pipeline_parallel_degree,
            ep=parallelism_config.expert_parallel_degree,
            etp=parallelism_config.expert_tensor_parallel_degree,
            world_size=world_size,
        )


if __name__ == "__main__":
    main(FTTrainer)
