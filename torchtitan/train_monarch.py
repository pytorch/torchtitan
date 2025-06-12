# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
import asyncio
import importlib
import os
import pickle
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


class TrainerActorWrapper(Actor):
    def __init__(self, job_config: JobConfig):
        self.job_config = job_config
        self.rank = current_rank().rank
        os.environ["RANK"] = str(self.rank)
        os.environ["ROLE_RANK"] = str(self.rank)
        os.environ["LOCAL_RANK"] = str(self.rank % 8)
        world_size = int(os.environ["WORLD_SIZE"])
        os.environ["LOCAL_WORLD_SIZE"] = str(world_size % 8)
        self.trainer = Trainer(self.job_config)

    @endpoint
    def train(self):
        self.trainer.train()
        print("hello world")

async def async_main(job_config: JobConfig):
    torch.use_deterministic_algorithms(True)
    world_size = int(os.environ["WORLD_SIZE"])
    # world_size = 2
    local_proc_mesh = await proc_mesh(
        gpus=world_size,
        env={
            "MASTER_ADDR": "localhost",
            "MASTER_PORT": "12356",
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
