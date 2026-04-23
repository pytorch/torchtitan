# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
import asyncio
import importlib
import os
import pickle
import threading
import sys
import time
from datetime import timedelta
from logging import getLogger
from typing import Any, Generator, Iterable, Optional
import torch
import torchtitan.components.ft as ft
import torchtitan.protocols.train_spec as train_spec_module
from monarch._rust_bindings.monarch_hyperactor.proc_mesh import ProcMesh as HyProcMesh
from monarch.actor_mesh import Actor, current_rank, endpoint
from monarch.proc_mesh import proc_mesh, ProcMesh
from monarch_meta._monarch_meta import hyperactor_meta
from torchtitan.config_manager import ConfigManager, JobConfig
from torchtitan.tools.logging import init_logger, logger
from .train import Trainer

def pretend_you_are_torchrun(global_rank):
    """
    Eventually, Monarch should handle all of this, but it's necessary for now because the job is
    not running torchrun. Also there are already better ways to avoid hardcoding this, but
    it's a demo and we'll live for now.
    """
    # task_id = int(os.environ["TW_TASK_ID"])
    # global_rank = task_id * 8 + (global_rank % 8)
    world_size = int(os.environ["WORLD_SIZE"])
    lr = min(world_size, global_rank % 8)
    local_world_size = min(world_size, 8)
    env = {
        # "MASTER_ADDR": get_master_addr(),
        # "MASTER_PORT": str(20101),
        "RANK": str(global_rank),
        "LOCAL_RANK": str(lr),
        "LOCAL_WORLD_SIZE": str(local_world_size),

        "GROUP_RANK": str(0),
        "GROUP_WORLD_SIZE": str(1),

        "ROLE_RANK": str(global_rank),
        "ROLE_WORLD_SIZE": str(world_size),
        "ROLE_NAME": "rank",

        # Note that WORLD_SIZE is already set.
    }
    os.environ.update(env)


class TrainerActorWrapper(Actor):
    def __init__(self, job_config: JobConfig):
        self.job_config = job_config
        self.rank = current_rank().rank
        pretend_you_are_torchrun(self.rank)

    @endpoint
    def train(self):
        print("Starting training")
        pretend_you_are_torchrun(self.rank)
        config = self.job_config
        trainer: Optional[Trainer] = None

        try:
            trainer = Trainer(config)
            # trainer = self.trainer
            tid = threading.get_native_id()
            logger.error(f"AHMAD tid in train: {tid=}")
            trainer.train()

            if config.checkpoint.create_seed_checkpoint:
                assert (
                    int(os.environ["WORLD_SIZE"]) == 1
                ), "Must create seed checkpoint using a single device, to disable sharding."
                assert (
                    config.checkpoint.enable_checkpoint
                ), "Must enable checkpointing when creating a seed checkpoint."
                trainer.checkpointer.save(curr_step=0, force=True)
                logger.info("Created seed checkpoint")
            else:
                trainer.train()
        finally:
            if trainer:
                trainer.close()

            if torch.distributed.is_initialized():
                torch.distributed.destroy_process_group()
                logger.info("Process group destroyed.")
        print("Done training")

async def async_main(job_config: JobConfig):
    torch.use_deterministic_algorithms(True)
    world_size = int(os.environ["WORLD_SIZE"])
    local_proc_mesh = await proc_mesh(
        gpus=world_size,
        env={
            "MASTER_ADDR": "localhost",
            "MASTER_PORT": "12359",
        },
    )
    print(job_config)
    trainer_actor = await local_proc_mesh.spawn(
        "trainer_actor", TrainerActorWrapper, job_config
    )
    await trainer_actor.train.call()


if __name__ == "__main__":
    init_logger()
    config_manager = ConfigManager()
    config = config_manager.parse_args()
    asyncio.run(async_main(config))
    sys.exit(0)
