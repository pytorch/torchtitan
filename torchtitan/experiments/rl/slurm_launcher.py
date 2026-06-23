# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""SLURM entry point for the RL controller.

Use this module's ``__main__`` instead of ``train.py`` when you want Monarch's
``SlurmJob`` to submit the worker allocation:

  python -m torchtitan.experiments.rl.slurm_launcher \\
      --module rl --config rl_grpo_qwen3_14b

Two modes are supported:

- External controller (default): submit a worker allocation and drive training
  from the calling (login-node) process.
- Batch (``RL_SLURM_BATCH=1``): submit one allocation that runs the workers AND
  re-runs this same command inside the allocation as the controller, then exit
  the login-node process. The in-allocation rerun has ``MONARCH_BATCH_JOB=1``
  set by Monarch and reconnects to the workers via the cached ``BatchJob``.

Single-node and MAST runs do not use this module -- they call
``train.run(config, host_meshes)`` directly with their own host meshes (or
``None`` for ``this_host()``).
"""

import asyncio
import logging
import os
import shlex
import sys

from monarch.job import job_load, SlurmJob

from torchtitan.config import ConfigManager
from torchtitan.experiments.rl.train import (
    _compute_generator_world_size,
    _compute_trainer_world_size,
    HostMeshes,
    run,
)
from torchtitan.experiments.rl.trainer import RLTrainer


logger = logging.getLogger(__name__)


def _generator_mesh_names(num_generators: int) -> list[str]:
    """Host-mesh names for the generators.

    Single source for the names so the side that *creates* the meshes and the
    side that *reads them back* off the JobState cannot drift.
    """
    return [f"generator_{i}" for i in range(num_generators)]


def _build_slurm_job(
    trainer_world_size: int,
    per_generator_world_size: int,
    num_generators: int,
) -> tuple[SlurmJob, int]:
    """Build the ``SlurmJob`` spec from ``RL_SLURM_*`` env; return ``(job,
    gpus_per_node)``.

    Trainer and generators land on disjoint nodes: the trainer gets
    ``trainer_world_size`` GPUs and each of ``num_generators`` generators gets
    ``per_generator_world_size`` GPUs, every role occupying whole nodes (its
    world size must be divisible by ``RL_SLURM_GPUS_PER_NODE``).

    Env vars:
      RL_SLURM_PARTITION    (required)
      RL_SLURM_GPUS_PER_NODE (optional, default 8)
      RL_SLURM_TIME         (optional, HH:MM:SS)
      RL_SLURM_QOS          (optional, e.g. h100_dev)
      RL_SLURM_ACCOUNT      (optional, e.g. pytorch)
      RL_SLURM_CPUS_PER_TASK (optional; partition default if unset, often too small)
      RL_SLURM_MEM          (optional; partition default if unset, often too small)
    """
    partition = os.environ.get("RL_SLURM_PARTITION")
    if not partition:
        raise ValueError(
            "the SLURM launcher requires RL_SLURM_PARTITION to be set"
        )
    gpus_per_node = int(os.environ.get("RL_SLURM_GPUS_PER_NODE", "8"))
    time_limit = os.environ.get("RL_SLURM_TIME")  # HH:MM:SS, optional
    qos = os.environ.get("RL_SLURM_QOS")
    account = os.environ.get("RL_SLURM_ACCOUNT")
    cpus_per_task_env = os.environ.get("RL_SLURM_CPUS_PER_TASK")
    cpus_per_task = int(cpus_per_task_env) if cpus_per_task_env else None
    mem = os.environ.get("RL_SLURM_MEM")

    if trainer_world_size % gpus_per_node != 0:
        raise ValueError(
            f"trainer_world_size ({trainer_world_size}) must be divisible "
            f"by RL_SLURM_GPUS_PER_NODE ({gpus_per_node})"
        )
    if per_generator_world_size % gpus_per_node != 0:
        raise ValueError(
            f"per_generator_world_size ({per_generator_world_size}) must be "
            f"divisible by RL_SLURM_GPUS_PER_NODE ({gpus_per_node})"
        )
    trainer_nodes = trainer_world_size // gpus_per_node
    generator_nodes = per_generator_world_size // gpus_per_node
    meshes = {"trainer": trainer_nodes}
    for name in _generator_mesh_names(num_generators):
        meshes[name] = generator_nodes

    # SlurmJob takes partition/time/gpus_per_node directly; anything else
    # (qos, account, ...) goes through slurm_args, templated as `#SBATCH <arg>`
    # lines into the generated sbatch script.
    extra_slurm_args: list[str] = []
    if qos:
        extra_slurm_args.append(f"--qos={qos}")
    if account:
        extra_slurm_args.append(f"--account={account}")
    logger.info(
        "Building SlurmJob: trainer=%d node(s), %d generator(s) x %d node(s), "
        "gpus_per_node=%d, partition=%s, qos=%s, account=%s, cpus_per_task=%s, "
        "mem=%s",
        trainer_nodes,
        num_generators,
        generator_nodes,
        gpus_per_node,
        partition,
        qos or "(default)",
        account or "(default)",
        cpus_per_task if cpus_per_task is not None else "(default)",
        mem or "(default)",
    )

    job = SlurmJob(
        meshes=meshes,
        gpus_per_node=gpus_per_node,
        partition=partition,
        time_limit=time_limit,
        cpus_per_task=cpus_per_task,
        mem=mem,
        job_name="torchtitan_rl",
        slurm_args=extra_slurm_args,
        # Workers must use the same Python the controller runs.
        python_exe=sys.executable,
        # Don't let SlurmJob fall through to its share_node() codepath (triggered
        # when exclusive=False AND partition is set), which would query
        # clusterscope for cpus_per_task / mem and overwrite the values above.
        exclusive=True,
    )
    return job, gpus_per_node


def _host_meshes_from_state(
    state, num_generators: int, gpus_per_node: int
) -> HostMeshes:
    """Read the trainer/generator host meshes off a JobState, by name."""
    return HostMeshes(
        trainer=state.trainer,
        generators=[getattr(state, n) for n in _generator_mesh_names(num_generators)],
        gpus_per_node=gpus_per_node,
    )


def _self_command() -> str:
    """The shell command that re-runs the SLURM entry point as the in-allocation
    client: same module and args, same interpreter."""
    return shlex.join(
        [
            sys.executable,
            "-m",
            "torchtitan.experiments.rl.slurm_launcher",
            *sys.argv[1:],
        ]
    )


def maybe_submit_batch_job(
    trainer_world_size: int,
    per_generator_world_size: int,
    num_generators: int,
) -> bool:
    """Login-node batch launch.

    With ``RL_SLURM_BATCH=1``, submit one allocation that runs the workers AND
    re-runs this exact command as the in-allocation controller
    (``apply(client_script=...)``), then return True so the caller exits without
    training. The in-allocation rerun has ``MONARCH_BATCH_JOB=1`` set and
    reconnects through :func:`acquire_host_meshes`.

    Returns False when batch mode is not requested, or when already running as
    the in-allocation client.
    """
    if os.environ.get("RL_SLURM_BATCH") != "1":
        return False
    if os.environ.get("MONARCH_BATCH_JOB") == "1":
        return False  # already the in-allocation client -- don't resubmit
    job, _ = _build_slurm_job(
        trainer_world_size, per_generator_world_size, num_generators
    )
    job.apply(client_script=_self_command())
    logger.info(
        "Submitted batch allocation %s; the controller runs inside it. Logs under %s",
        job._slurm_job_id,
        job._log_dir,
    )
    return True


def acquire_host_meshes(
    trainer_world_size: int,
    per_generator_world_size: int,
    num_generators: int,
) -> HostMeshes:
    """Acquire the trainer/generator host meshes from SLURM.

    - In-allocation batch client (``MONARCH_BATCH_JOB=1``): reconnect to the
      workers the runner started via the cached job; no new submission and no
      spec rebuild.
    - External controller: submit a worker allocation and drive training from
      this process.
    """
    if os.environ.get("MONARCH_BATCH_JOB") == "1":
        # Reconnect to the allocation the submit step cached.
        state = job_load().state()
        gpus_per_node = int(os.environ.get("RL_SLURM_GPUS_PER_NODE", "8"))
        return _host_meshes_from_state(state, num_generators, gpus_per_node)
    job, gpus_per_node = _build_slurm_job(
        trainer_world_size, per_generator_world_size, num_generators
    )
    job.apply()
    return _host_meshes_from_state(job.state(), num_generators, gpus_per_node)


def main() -> None:
    config = ConfigManager().parse_args()
    assert isinstance(config, RLTrainer.Config)
    tws = _compute_trainer_world_size(config.trainer.parallelism)
    gws = _compute_generator_world_size(config.generator.parallelism)
    # Login-node batch submitter exits as soon as the sbatch is in.
    if maybe_submit_batch_job(tws, gws, config.num_generators):
        return
    host_meshes = acquire_host_meshes(tws, gws, config.num_generators)
    asyncio.run(
        run(
            config,
            trainer_world_size=tws,
            per_generator_world_size=gws,
            host_meshes=host_meshes,
        )
    )


if __name__ == "__main__":
    main()
