# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import logging
import os
import shlex
import shutil
import sys
from dataclasses import dataclass, field

from monarch.actor import HostMesh

from torchtitan.rl.actors.monarch.launcher.base import (
    LaunchPlan,
    Layout,
    MonarchLauncher,
)

logger = logging.getLogger(__name__)

# Absolute path of this job's saved PBSJob state, exported to the job so the
# controller can load it and attach to the workers.
_JOB_STATE_ENV = "TORCHTITAN_RL_PBS_JOB_STATE"
# Where monarch.job writes the job state, relative to the apply() cwd.
_MONARCH_JOB_STATE = os.path.join(".monarch", "job_state.pkl")
# qsub options that PBSJob or this launcher already set, from the layout or other
# config fields. Setting one again in pbs_args would conflict with that value.
_RESERVED_QSUB_OPTIONS = {"-N", "-j", "-o", "-V"}


class PBSLauncher(MonarchLauncher):
    """Runs the job as one Monarch ``PBSJob`` in batch mode. The controller
    (``python -m torchtitan.rl.train``) runs as the batch script on the job's
    first node; ``mpiexec`` starts one Monarch worker on every node.

    As with ``SlurmLauncher``, every node uses the submit host's paths: each node
    runs ``python_exe`` by its full path, from the submit directory, with the
    submit environment (``qsub -V``). So the venv, the checkout, the recipe's
    relative paths and ``dump_folder`` must be on a filesystem shared by the
    submit host and the nodes.
    """

    # PBS runs the batch script on the job's first node, the first host in `PBS_NODEFILE`. `PBSJob`
    # gives the first nodes, in that order, to the first mesh of `Layout.mesh_nodes`, the trainer's:
    # `MonarchLauncher._layout` places it first.
    controller_on_first_trainer_host = True

    @dataclass(kw_only=True, slots=True)
    class Config(MonarchLauncher.Config):
        gpus_per_node: int = 8
        """GPUs on each node; on Intel GPU Max, one per tile (12 on Aurora)."""
        pbs_args: list[str] = field(default_factory=list)
        """``qsub`` options, written as ``#PBS`` directives, e.g.
        ``["-q debug", "-A my_project", "-l walltime=01:00:00"]``."""
        mpiexec_args: list[str] = field(default_factory=list)
        """Extra ``mpiexec`` flags, e.g. ``["--cpu-bind=none"]`` for Cray PALS."""
        log_dir: str | None = None
        """Job stdout and stderr; None means ``<dump_folder>/pbs``."""
        bind_to: str | None = None
        """Address Monarch workers listen on, e.g. ``"::"`` on dual-stack clusters."""
        python_exe: str | None = None
        """None means the submitting interpreter."""

        def __post_init__(self) -> None:
            MonarchLauncher.Config.__post_init__(self)
            if self.gpus_per_node <= 0:
                raise ValueError("launcher.gpus_per_node must be greater than 0")
            for arg in self.pbs_args:
                tokens = shlex.split(arg)
                if not arg.startswith("-"):
                    # PBSJob drops entries that don't start with "-".
                    raise ValueError(
                        f"launcher.pbs_args entries must be qsub options, got {arg!r}"
                    )
                # One "#PBS <entry>" line may hold several options; check each.
                for token in tokens:
                    if token[:2] in _RESERVED_QSUB_OPTIONS or "select=" in token:
                        raise ValueError(f"launcher.pbs_args must not set {arg!r}")

    def gpus_per_node(self) -> int:
        return self.config.gpus_per_node

    def submit(self, plan: LaunchPlan) -> str:
        from monarch.job import PBSJob

        os.environ.update(plan.env)
        # Node-local JIT caches, as in SlurmLauncher: nodes compiling into one
        # shared cache on a network filesystem can hang or crash.
        cache_root = f"/tmp/torchtitan-{os.getuid()}"
        os.environ.setdefault("TRITON_CACHE_DIR", f"{cache_root}/triton")
        os.environ.setdefault("VLLM_CACHE_ROOT", f"{cache_root}/vllm")
        os.environ.setdefault("FLASHINFER_WORKSPACE_BASE", f"{cache_root}/flashinfer")
        state_dir = os.path.abspath(
            os.path.join(plan.dump_folder, "launcher", plan.job_name)
        )
        os.makedirs(state_dir, exist_ok=True)
        log_dir = os.path.abspath(
            self.config.log_dir or os.path.join(plan.dump_folder, "pbs")
        )
        state_path = os.path.join(state_dir, "job_state.pkl")
        os.environ[_JOB_STATE_ENV] = state_path
        python_exe = self.config.python_exe or sys.executable
        job = PBSJob(
            meshes=dict(plan.layout.mesh_nodes),
            python_exe=python_exe,
            # -V: PBS doesn't pass the submit environment (plan.env) on its own.
            pbs_args=["-V", *self.config.pbs_args],
            job_name=plan.job_name,
            log_dir=log_dir,
            mpiexec_args=self.config.mpiexec_args,
            bind_to=self.config.bind_to,
        )
        client_script = shlex.join(
            [python_exe, "-m", "torchtitan.rl.train", *plan.controller_args]
        )
        logger.info(f"Submitting {plan.job_name}: {client_script}")
        # PBS has no --chdir: the job runs in the directory qsub ran from, so
        # apply() runs here for the recipe's relative paths. It writes the job
        # state to ./.monarch/, which the next submit from here overwrites, so
        # keep this job's copy next to its other launcher files.
        # TODO(meta-pytorch/monarch#5414): once PBSJob takes a workdir, apply()
        # from state_dir (as SlurmLauncher does) and drop this copy.
        job.apply(client_script=client_script)
        shutil.copyfile(_MONARCH_JOB_STATE, state_path)
        logger.info(f"PBS logs: {log_dir}")
        # TODO: use a public Monarch API for the job id.
        return str(job._pbs_job_id)

    async def acquire_hosts(self, layout: Layout) -> dict[str, HostMesh]:
        from monarch.job import job_load

        state = job_load(os.environ[_JOB_STATE_ENV]).state()
        return {name: getattr(state, name) for name in layout.mesh_nodes}
