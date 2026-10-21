# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import contextlib
import logging
import os
import shlex
import sys
from dataclasses import dataclass, field

from monarch.actor import HostMesh

from torchtitan.rl.actors.monarch.launcher.base import (
    LaunchPlan,
    Layout,
    MonarchLauncher,
)

logger = logging.getLogger(__name__)

# Absolute path of this job's saved SlurmJob state, exported to the job so the
# controller can load it and attach to the workers.
_JOB_STATE_ENV = "TORCHTITAN_RL_SLURM_JOB_STATE"
# sbatch flags that SlurmJob or this launcher already set, from the layout or other
# config fields (gpus_per_node, exclusive, log_dir, ...). Setting one again in
# slurm_args would conflict with that value, e.g. a --nodes that doesn't match the
# layout, so Config rejects them for both long and short forms.
_RESERVED_SBATCH_FLAGS = {
    "--nodes",
    "-N",
    "--ntasks-per-node",
    "--gpus-per-node",
    "--job-name",
    "-J",
    "--output",
    "-o",
    "--error",
    "-e",
    "--exclusive",
    "--chdir",
    "-D",
    "--export",
    "--parsable",
    "--quiet",
    "-Q",
}


class SlurmLauncher(MonarchLauncher):
    """Runs the job as one Monarch ``SlurmJob`` in batch mode. The controller
    (``python -m torchtitan.rl.train``) runs as the batch script on SLURM's head
    node, the first node of the allocation.

    SlurmLauncher copies nothing to the nodes; every node uses the submit host's
    paths. Each node runs ``python_exe`` by its full path (by default the Python
    you ran ``submit`` with), from the submit directory, with the submit
    environment (SLURM's default ``--export=ALL``). So the venv, the checkout and
    the recipe's relative paths (model, dataset cache) must resolve to the same
    files on every node, typically on a shared network filesystem such as NFS.
    ``dump_folder`` must be shared between the submit host and the nodes, because
    the job reads the job state that ``submit`` writes there.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(MonarchLauncher.Config):
        gpus_per_node: int = 8
        slurm_args: list[str] = field(default_factory=list)
        """``sbatch`` flags, e.g. ``["--partition=gpu", "--time=1:00:00"]``.
        ``SBATCH_*`` environment variables override them."""
        exclusive: bool = True
        """Reserve whole nodes. Without it, jobs may share a node: add ``--cpus-per-task``
        and ``--mem`` to ``slurm_args``, and the cluster must confine each job to its
        own GPUs (cgroup ``ConstrainDevices``)."""
        log_dir: str | None = None
        """Job stdout and stderr; None means ``<dump_folder>/slurm``."""
        bind_to: str | None = None
        """Address Monarch workers listen on, e.g. ``"::"`` on dual-stack clusters."""
        python_exe: str | None = None
        """None means the submitting interpreter."""

        def __post_init__(self) -> None:
            MonarchLauncher.Config.__post_init__(self)
            if self.gpus_per_node <= 0:
                raise ValueError("launcher.gpus_per_node must be greater than 0")
            # SlurmJob writes each entry as one "#SBATCH <entry>" line, which may
            # hold several options and "--flag value" forms; check every option.
            for token in (t for arg in self.slurm_args for t in shlex.split(arg)):
                flag = token.split("=", 1)[0]
                if not flag.startswith("-"):
                    continue
                if flag in _RESERVED_SBATCH_FLAGS or flag[:2] in _RESERVED_SBATCH_FLAGS:
                    raise ValueError(f"launcher.slurm_args must not set {flag}")

    def gpus_per_node(self) -> int:
        return self.config.gpus_per_node

    def submit(self, plan: LaunchPlan) -> str:
        from monarch.job import SlurmJob

        sbatch_export = plan.env.get("SBATCH_EXPORT", os.environ.get("SBATCH_EXPORT"))
        if (sbatch_export or "ALL").split(",")[0] != "ALL":
            raise ValueError("SlurmLauncher needs SBATCH_EXPORT unset or ALL.")
        os.environ.update(plan.env)

        log_dir = os.path.abspath(
            self.config.log_dir or os.path.join(plan.dump_folder, "slurm")
        )
        # monarch.job.SlurmJob.apply() saves the job state to a file. The
        # controller, running on the job's head node rather than this submit host,
        # loads it with monarch.job.job_load() to attach to the workers
        # (acquire_hosts). So the file must sit on a shared network filesystem at a
        # path exported to the job.
        state_dir = os.path.abspath(
            os.path.join(plan.dump_folder, "launcher", plan.job_name)
        )
        os.makedirs(state_dir, exist_ok=True)
        os.environ[_JOB_STATE_ENV] = os.path.join(state_dir, ".monarch/job_state.pkl")
        python_exe = self.config.python_exe or sys.executable
        job = SlurmJob(
            meshes=dict(plan.layout.mesh_nodes),
            gpus_per_node=self.config.gpus_per_node,
            slurm_args=[f"--chdir={os.getcwd()}", *self.config.slurm_args],
            log_dir=log_dir,
            job_name=plan.job_name,
            python_exe=python_exe,
            exclusive=self.config.exclusive,
            bind_to=self.config.bind_to,
        )
        client_script = shlex.join(
            [python_exe, "-m", "torchtitan.rl.train", *plan.controller_args]
        )
        logger.info(f"Submitting {plan.job_name}: {client_script}")
        with contextlib.chdir(state_dir):
            job.apply(client_script=client_script)
        logger.info(f"SLURM logs: {log_dir}")
        # TODO: use public Monarch APIs for the job id and in_batch_job.
        return str(job._slurm_job_id)

    def in_job(self) -> bool:
        from monarch._src.job._batch_env import in_batch_job

        return in_batch_job()

    async def acquire_hosts(self, layout: Layout) -> dict[str, HostMesh]:
        from monarch.job import job_load

        state = job_load(os.environ[_JOB_STATE_ENV]).state()
        return {name: getattr(state, name) for name in layout.mesh_nodes}
