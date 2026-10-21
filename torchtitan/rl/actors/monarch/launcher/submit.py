# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Submit an RL recipe that sets ``config.actor_system.launcher`` to its scheduler.

    python -m torchtitan.rl.actors.monarch.launcher.submit [--dry-run] \
        --module MODULE --config CONFIG [other torchtitan.rl.train options]

The job runs ``python -m torchtitan.rl.train`` with the same arguments.
``--dry-run`` prints the plan without submitting.
"""

from __future__ import annotations

import sys

from torchtitan.config import ConfigLoader
from torchtitan.observability.logging import init_logger
from torchtitan.rl.actors.monarch.launcher.base import LaunchPlan, MonarchLauncher
from torchtitan.rl.actors.monarch.system import MonarchActorSystem
from torchtitan.rl.controller.controller import Controller


def _split_args(argv: list[str]) -> tuple[bool, list[str]]:
    """Return whether ``--dry-run`` is set, and the other arguments verbatim."""
    forwarded = [arg for arg in argv if arg != "--dry-run"]
    return len(forwarded) != len(argv), forwarded


def _build_launcher(config: Controller.Config) -> MonarchLauncher:
    actor_system = config.actor_system
    if (
        not isinstance(actor_system, MonarchActorSystem.Config)
        or actor_system.launcher is None
    ):
        raise ValueError(
            "The recipe does not set config.actor_system.launcher; run it with "
            "python -m torchtitan.rl.train instead."
        )
    return actor_system.launcher.build()


def _launcher_settings(launcher: MonarchLauncher) -> dict:
    # Env names only: values can hold secrets.
    settings = launcher.config.to_dict()
    settings["env"] = sorted(settings["env"])
    return settings


def _format_plan(launcher: MonarchLauncher, plan: LaunchPlan) -> str:
    lines = [
        f"=== {type(launcher).__name__}: {plan.job_name} ===",
        f"dump folder: {plan.dump_folder}",
        f"controller args: {' '.join(plan.controller_args)}",
        f"env: {', '.join(sorted(plan.env))}",
    ]
    for mesh, num_nodes in plan.layout.mesh_nodes.items():
        roles = [r for r, m in plan.layout.role_mesh.items() if m == mesh]
        lines.append(f"mesh {mesh}: {num_nodes} node(s), roles {', '.join(roles)}")
    lines += [f"launcher.{k}: {v}" for k, v in _launcher_settings(launcher).items()]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> str | None:
    """Return the job handle, or None on ``--dry-run``."""
    argv = sys.argv[1:] if argv is None else argv
    init_logger()
    dry_run, controller_args = _split_args(argv)
    config = ConfigLoader().load(controller_args)
    if not isinstance(config, Controller.Config):
        raise ValueError(f"Expected a Controller.Config, got {type(config).__name__}.")
    # Plan from the config the job will run, including the launcher settings.
    config.resolve_overrides()
    launcher = _build_launcher(config)
    plan = launcher.plan(config, controller_args)
    print(_format_plan(launcher, plan), flush=True)
    if dry_run:
        return None
    job_handle = launcher.submit(plan)
    print(f"Submitted job {job_handle}", flush=True)
    return job_handle


if __name__ == "__main__":
    main()
