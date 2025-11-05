# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from torchtitan.distributed import ParallelDims
from torchtitan.train import main, Trainer

from .parallel_dims import TorchCommsParallelDims


class TorchCommsTrainer(Trainer):
    parallel_dims: TorchCommsParallelDims

    def _create_parallel_dims(self, parallelism_config, world_size) -> ParallelDims:
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
        if hasattr(trainer, "parallel_dims"):
            for comm in trainer.parallel_dims.comms:
                comm.finalize()
        super().close()


if __name__ == "__main__":
    main(TorchCommsTrainer)
