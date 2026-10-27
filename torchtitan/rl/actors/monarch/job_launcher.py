# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Run ``torchtitan.rl.train`` on the hosts of a Monarch batch job.

Allocate the hosts with a Monarch job (e.g. ``monarch.job.SlurmJob`` or
``PBSJob``), dump it, and run this module as the job's client with the usual
``train`` arguments::

    TORCHTITAN_MONARCH_JOB=/path/to/job.pkl python3 -m \\
        torchtitan.rl.actors.monarch.job_launcher \\
        --module my_rl_configs --config rl_grpo_qwen3_0_6b_varlen

The trainer takes the first hosts and the generators the rest (see
``plan_host_meshes``).
"""

# Apply process env defaults before anything imports torch, as train.py does.
from torchtitan.rl._env import apply_env_defaults

apply_env_defaults()

import logging
import os

from monarch.job import job_load

from torchtitan.rl import train
from torchtitan.rl.actors.monarch.provisioning import (
    compute_generator_world_size,
    compute_trainer_world_size,
    plan_host_meshes,
)
from torchtitan.rl.actors.monarch.system import MonarchActorSystem
from torchtitan.rl.controller.controller import Controller
from torchtitan.tools.utils import device_module

logger = logging.getLogger(__name__)

# Path to a pickled Monarch job (``JobTrait.dump``) with one ``hosts`` mesh.
MONARCH_JOB_ENV = "TORCHTITAN_MONARCH_JOB"


def build_job_actor_system(config: Controller.Config) -> MonarchActorSystem:
    """A ``MonarchActorSystem`` on the hosts of the job named by ``TORCHTITAN_MONARCH_JOB``.

    Must run before the controller is built: loading the job sets the transport
    that procs on other hosts use to reach this one.
    """
    job_path = os.environ.get(MONARCH_JOB_ENV)
    if not job_path:
        raise ValueError(f"set {MONARCH_JOB_ENV} to the path of a dumped Monarch job")
    if not isinstance(config.actor_system, MonarchActorSystem.Config):
        raise ValueError(
            f"a Monarch job needs actor_system=MonarchActorSystem.Config(), got "
            f"{type(config.actor_system).__qualname__}"
        )
    trainer_world_size = compute_trainer_world_size(config.trainer.parallelism)
    per_generator_world_size = compute_generator_world_size(
        config.generator.parallelism
    )
    host_meshes = plan_host_meshes(
        job_load(job_path).state(cached_path=None).hosts,
        trainer_world_size=trainer_world_size,
        per_generator_world_size=per_generator_world_size,
        num_generators=config.num_generators,
        gpus_per_node=device_module.device_count(),
    )
    if host_meshes.generators[0] is not host_meshes.trainer:  # not one shared host
        p = config.trainer.parallelism
        logger.info(
            f"Mesh split: {len(host_meshes.trainer)} trainer node(s) "
            f"({trainer_world_size} GPUs, "
            f"dp_replicate={p.data_parallel_replicate_degree}, "
            f"dp_shard={p.data_parallel_shard_degree}, "
            f"tp={p.tensor_parallel_degree}), "
            f"{len({id(m) for m in host_meshes.generators})} generator node(s) "
            f"({config.num_generators * per_generator_world_size} GPUs, "
            f"{config.num_generators} generator(s) of "
            f"{per_generator_world_size} GPU(s))"
        )
    return config.actor_system.build(host_meshes=host_meshes)


if __name__ == "__main__":
    train.run(build_job_actor_system)
