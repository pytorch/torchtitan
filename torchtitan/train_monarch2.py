# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
import asyncio
import socket
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
from monarch.allocator import RemoteAllocator, StaticRemoteAllocInitializer, RemoteAllocInitializer
from monarch._rust_bindings.hyperactor_extension.alloc import (
    AllocConstraints,
    AllocSpec,
)
from .train import Trainer

def pretend_you_are_torchrun(global_rank):
    """
    Eventually, Monarch should handle all of this, but it's necessary for now because the job is
    not running torchrun. Also there are already better ways to avoid hardcoding this, but
    it's a demo and we'll live for now.
    """
    # task_id = int(os.environ["TW_TASK_ID"])
    # global_rank = task_id * 8 + (global_rank % 8)
    task_id = int(os.environ["SLURM_NODEID"])
    local_world_size = int(os.environ["LOCAL_WORLD_SIZE"])
    num_hosts = int(os.environ["NUM_HOSTS"])

    # Ahmad commented out this because this is only needed for
    # multiple controllers.
    #global_rank = task_id * local_world_size + global_rank

    world_size = num_hosts * local_world_size
    local_rank = min(world_size, global_rank % local_world_size)

    group_rank = global_rank // local_world_size
    group_world_size = (world_size + local_world_size - 1) // local_world_size

    env = {
        # "MASTER_ADDR": get_master_addr(),
        # "MASTER_PORT": str(20101),
        "RANK": str(global_rank),
        "LOCAL_RANK": str(local_rank),


        # Note that local_world_size is already set.

        "GROUP_RANK": str(group_rank),
        "GROUP_WORLD_SIZE": str(group_world_size),

        "ROLE_RANK": str(global_rank),
        "ROLE_WORLD_SIZE": str(world_size),
        "ROLE_NAME": "rank",

        "WORLD_SIZE": str(world_size),
    }
    print(f" AHMAD: {global_rank=} {env}")
    os.environ.update(env)
    if global_rank == 0:
        print(f" AHMAD: {global_rank=} {os.environ}")


class TrainerActorWrapper(Actor):
    def __init__(self, job_config: JobConfig, env_to_merge={}):
        self.job_config = job_config
        self.rank = current_rank().rank
        hostname = socket.gethostname()
        print(f" ===> AHMAD: {self.rank} {hostname=} {current_rank()=}")
        pretend_you_are_torchrun(self.rank)
        os.environ.update(env_to_merge)

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
            logger.error(f"AHMAD tid in train: {self.rank=} {tid=}")
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

class MyAllocInitializer(RemoteAllocInitializer):
    def __init__(self, l):
        super().__init__()
        self.l = l
    async def initialize_alloc(self) -> list[str]:
        return self.l

async def async_main(job_config: JobConfig):
    torch.use_deterministic_algorithms(True)
    local_world_size = int(os.environ["LOCAL_WORLD_SIZE"])
    num_hosts = int(os.environ["NUM_HOSTS"])
    slurm_job_nodes = os.environ["SLURM_JOB_NODES"]
    world_size = local_world_size * num_hosts

    node_ips = slurm_job_nodes.split(" ")
    all_nodes = [f"tcp!{n}:26600" for n in node_ips]
    proc_mesh = None
    env_to_merge = {}
    if len(all_nodes):
        hosts = num_hosts
        gpus = local_world_size
        spec = AllocSpec(AllocConstraints(), host=hosts, gpu=gpus)
        print(f"AHMAD: {all_nodes=}\n{node_ips=}")
        allocator = RemoteAllocator(world_id="test_remote_allocator", initializer=MyAllocInitializer(all_nodes))
        alloc = await allocator.allocate(spec)
        proc_mesh = await ProcMesh.from_alloc(alloc)
        env_to_merge["MASTER_ADDR"] = str(node_ips[0])
        env_to_merge["MASTER_PORT"] = str(12347)
    else:
        proc_mesh = await proc_mesh(
            gpus=local_world_size,
            env={
                "MASTER_ADDR": master_addr,
                "MASTER_PORT": master_port,
            },
        )

    print(job_config)
    trainer_actor = await proc_mesh.spawn(
        "trainer_actor", TrainerActorWrapper, job_config, env_to_merge
    )
    await trainer_actor.train.call()


if __name__ == "__main__":
    init_logger()
    config_manager = ConfigManager()
    config = config_manager.parse_args()
    asyncio.run(async_main(config))
    sys.exit(0)
