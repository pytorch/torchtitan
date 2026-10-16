# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import logging
from typing import cast

import torch

from torchtitan.config import ConfigLoader
from torchtitan.observability import structured_logger as sl
from torchtitan.observability.logging import init_logger
from torchtitan.trainer import Trainer


logger = logging.getLogger(__name__)


def _require_seed_checkpoint_launch(config: Trainer.Config) -> None:
    """Reject a seed checkpoint that would shard initialization across ranks.

    Distributed init has already chosen the process-group world size: ``NGPU``
    in fake mode, and the launched process count under ``torchrun``. Pure fake
    does not set ``WORLD_SIZE``.
    """
    if not torch.distributed.is_initialized():
        raise ValueError(
            "Must create a seed checkpoint on an initialized single-device "
            "process group."
        )
    world_size = torch.distributed.get_world_size()
    if world_size != 1:
        raise ValueError(
            "Must create a seed checkpoint using a single device, to disable "
            f"sharding. Got world size {world_size}."
        )
    if config.checkpointer is None:
        raise ValueError(
            "Must configure checkpointer when creating a seed checkpoint."
        )


def main() -> None:
    """Main entry point for training."""
    init_logger()

    import torchtitan

    logger.info(
        "torchtitan version: %s (0.0.0 means __version__ is not defined correctly).",
        torchtitan.__version__,
    )

    config = cast(Trainer.Config, ConfigLoader().load())

    # NOTE: internal meta tooling relies on source="training".
    sl.init_structured_logger(
        source="training",
        output_dir=config.dump_folder,
        enable=config.debug.enable_structured_logging,
    )
    sl.log_trace_instant("structured_logger_started")

    trainer: Trainer | None = None

    try:
        trainer = config.build()

        if config.create_seed_checkpoint:
            _require_seed_checkpoint_launch(config)
            trainer.engine.save_checkpoint(last_step=True)
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
            with sl.log_trace_span("torch_distributed_teardown"):
                torch.distributed.destroy_process_group()
        logger.info("Process group destroyed")


if __name__ == "__main__":
    main()
