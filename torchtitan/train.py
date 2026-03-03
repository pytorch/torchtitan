# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import os

import torch
import torch.nn.functional as F

from torchtitan.config import ConfigManager
from torchtitan.tools.logging import init_logger, logger
from torchtitan.trainer import Trainer


def main() -> None:
    """Main entry point for training."""
    init_logger()

    import torchtitan

    logger.info(
        "torchtitan version: %s (0.0.0 means __version__ is not defined correctly).",
        torchtitan.__version__,
    )

    config_manager = ConfigManager()
    config = config_manager.parse_args()
    trainer: Trainer | None = None

    try:
        # TODO(local_tensor): Remove this special case once LocalTensor supports
        # init_weights() and foreach_allgather. In local tensor mode, skip
        # training/checkpointing as the # model is not fully initialized
        # pyrefly: ignore [missing-attribute]
        if config.comm.mode == "local_tensor":
            logger.info("Local tensor mode enabled - skipping training execution")
            return

        # pyrefly: ignore [missing-attribute]
        trainer = config.build()

        # On PT ROCm 7.1 nightly wheels, we are seeing HIPBLAS_STATUS_INVALID_VALUE
        # from hipblasLtMatmulAlgoGetHeuristic errors. Add a warmup as a workaround.
        if torch.version.hip is not None and torch.cuda.is_available():
            try:
                warm_x = torch.randn(1, 1, 8, device="cuda", dtype=torch.bfloat16)
                warm_w = torch.randn(8, 8, device="cuda", dtype=torch.bfloat16)
                _ = F.linear(warm_x, warm_w, None)
                torch.cuda.synchronize()
                logger.info("ROCm warmup linear completed")
            except RuntimeError as error:
                logger.warning("ROCm warmup linear failed: %s", error)

        # pyrefly: ignore [missing-attribute]
        if config.checkpoint.create_seed_checkpoint:
            assert (
                int(os.environ["WORLD_SIZE"]) == 1
            ), "Must create seed checkpoint using a single device, to disable sharding."
            assert (
                # pyrefly: ignore [missing-attribute]
                config.checkpoint.enable
            ), "Must enable checkpointing when creating a seed checkpoint."
            trainer.checkpointer.save(curr_step=0, last_step=True)
            logger.info("Created seed checkpoint")
        else:
            trainer.train()
    except Exception:
        if trainer:
            trainer.close()
        raise
    else:
        trainer.close()
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()
        logger.info("Process group destroyed")


if __name__ == "__main__":
    main()
