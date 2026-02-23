# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import os

from torchtitan.distributed import ParallelDims, utils as dist_utils
from torchtitan.trainer import Trainer

from .parallel_dims import TorchCommsParallelDims


class TorchCommsTrainer(Trainer):
    parallel_dims: TorchCommsParallelDims

    def init_distributed(self) -> ParallelDims:
        config = self.config
        dist_utils.init_distributed(
            config.comm,
            enable_cpu_backend=config.training.enable_cpu_offload,
            base_folder=config.dump_folder,
        )

        world_size = int(os.environ["WORLD_SIZE"])
        parallelism_config = config.parallelism

        return TorchCommsParallelDims(
            dp_shard=parallelism_config.data_parallel_shard_degree,
            dp_replicate=parallelism_config.data_parallel_replicate_degree,
            cp=parallelism_config.context_parallel_degree,
            tp=parallelism_config.tensor_parallel_degree,
            pp=parallelism_config.pipeline_parallel_degree,
            ep=parallelism_config.expert_parallel_degree,
            etp=parallelism_config.expert_tensor_parallel_degree,
            world_size=world_size,
        )

    def close(self) -> None:
        # Call finalize on all comms after training and before destroying process group.
        if hasattr(self, "parallel_dims"):
            for comm in self.parallel_dims.comms:
                comm.finalize()
        super().close()
