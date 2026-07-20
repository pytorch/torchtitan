# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import datetime
import os
import sys

import torch

from torchtitan.config import ConfigManager
from torchtitan.observability import structured_logger as sl
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

    # Give each run its own dump folder so it never resumes from / collides with a
    # previous run's checkpoints. Skip when --dump_folder is set explicitly (then
    # the user's exact path is honored, e.g. to resume a specific run).
    dump_folder_overridden = any(
        a == "--dump_folder" or a.startswith("--dump_folder=") for a in sys.argv[1:]
    )
    if not dump_folder_overridden:
        run_name = config_manager.config_name or "run"
        timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        # pyrefly: ignore [missing-attribute]
        config.dump_folder = os.path.join(config.dump_folder, f"{run_name}-{timestamp}")
    # pyrefly: ignore [missing-attribute]
    logger.info(f"Dumping run artifacts to: {config.dump_folder}")

    # NOTE: internal meta tooling relies on source="training".
    sl.init_structured_logger(
        source="training",
        # pyrefly: ignore [missing-attribute]
        output_dir=config.dump_folder,
        # pyrefly: ignore [missing-attribute]
        enable=config.debug.enable_structured_logging,
    )
    sl.log_trace_instant("structured_logger_started")

    trainer: Trainer | None = None

    try:
        # TODO(local_tensor): Remove this special case once LocalTensor supports
        # init_states() and foreach_allgather. In local tensor mode, skip
        # training/checkpointing as the # model is not fully initialized
        if config.comm.mode == "local_tensor":  # pyrefly: ignore [missing-attribute]
            logger.info("Local tensor mode enabled - skipping training execution")
            return

        trainer = config.build()  # pyrefly: ignore [missing-attribute]

        if (
            config.checkpoint.create_seed_checkpoint  # pyrefly: ignore[missing-attribute]
        ):
            assert (
                int(os.environ["WORLD_SIZE"]) == 1
            ), "Must create seed checkpoint using a single device, to disable sharding."
            assert (
                config.checkpoint.enable  # pyrefly: ignore [missing-attribute]
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
            with sl.log_trace_span("torch_distributed_teardown"):
                torch.distributed.destroy_process_group()
        logger.info("Process group destroyed")


if __name__ == "__main__":
    main()
