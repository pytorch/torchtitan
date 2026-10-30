# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""The scheduler-neutral launcher interface and the logic every scheduler shares."""

from __future__ import annotations

import abc
import dataclasses
import json
import logging
import os
import re
import uuid
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from monarch.actor import HostMesh

from torchtitan.config import Configurable
from torchtitan.rl.actors.monarch.provisioning import (
    compute_generator_world_size,
    compute_trainer_world_size,
    HostMeshes,
)

if TYPE_CHECKING:
    from torchtitan.rl.controller.controller import Controller

logger = logging.getLogger(__name__)

SUBMIT_COMMAND = "python -m torchtitan.rl.actors.monarch.launcher.submit"

# Carries the layout computed by ``plan`` on the submit host to the job.
LAYOUT_ENV = "TORCHTITAN_RL_LAUNCH_LAYOUT"


def generator_role_names(num_generators: int) -> list[str]:
    """Generator mesh and actor names: ``generator``, or ``generator_<i>`` for several."""
    if num_generators == 1:
        return ["generator"]
    return [f"generator_{i}" for i in range(num_generators)]


@dataclass(frozen=True, slots=True)
class Layout:
    """Where the trainer and generators run."""

    gpus_per_node: int
    mesh_nodes: dict[str, int]
    """Host mesh name -> node count."""
    role_mesh: dict[str, str]
    """Role -> host mesh name; roles may share a mesh."""

    def host_meshes(self, meshes: dict[str, HostMesh]) -> HostMeshes:
        """Map acquired meshes to roles; roles sharing a mesh get the same object.

        Example: 8 GPUs per node, a 32-GPU trainer and two 16-GPU generators.
        Each role spans several nodes, so each gets its own mesh::

            mesh_nodes = {"trainer": 4, "generator_0": 2, "generator_1": 2}
            role_mesh = {
                "trainer": "trainer",
                "generator_0": "generator_0",
                "generator_1": "generator_1",
            }
            # meshes: "trainer" -> 4-host HostMesh, "generator_<i>" -> 2-host HostMesh
            HostMeshes(
                trainer=meshes["trainer"],
                generators=[meshes["generator_0"], meshes["generator_1"]],
                gpus_per_node=8,
            )

        Example: the same node size, a 4-GPU trainer and three 4-GPU generators
        packed onto two nodes. ``MonarchLauncher`` does not plan this yet (see the
        TODO in ``_layout``), but a ``Layout`` can describe it::

            mesh_nodes = {"node_0": 1, "node_1": 1}
            role_mesh = {
                "trainer": "node_0",
                "generator_0": "node_0",
                "generator_1": "node_1",
                "generator_2": "node_1",
            }
            HostMeshes(
                trainer=meshes["node_0"],
                generators=[meshes["node_0"], meshes["node_1"], meshes["node_1"]],
                gpus_per_node=8,
            )
        """
        num_generators = len(self.role_mesh) - 1
        return HostMeshes(
            trainer=meshes[self.role_mesh["trainer"]],
            generators=[
                meshes[self.role_mesh[role]]
                for role in generator_role_names(num_generators)
            ],
            gpus_per_node=self.gpus_per_node,
        )

    def to_json(self) -> str:
        return json.dumps(dataclasses.asdict(self))

    @classmethod
    def from_env(cls) -> Layout:
        """Read the layout that ``plan`` passed to the job in ``LAYOUT_ENV``."""
        value = os.environ.get(LAYOUT_ENV)
        if value is None:
            raise ValueError(
                f"{LAYOUT_ENV} is not set; submit the job with `{SUBMIT_COMMAND}`."
            )
        return cls(**json.loads(value))


@dataclass(frozen=True, slots=True)
class LaunchPlan:
    """Everything resolved on the submit host."""

    job_name: str
    layout: Layout
    controller_args: list[str]
    """Arguments for ``python -m torchtitan.rl.train`` inside the job."""
    env: dict[str, str]
    """Set on every host of the job."""
    dump_folder: str


class MonarchLauncher(Configurable, abc.ABC):
    """Gets an RL job's hosts from a cluster scheduler as Monarch host meshes.

    Where the controller runs is up to the scheduler; each launcher documents it.
    If any role fails, the controller exits non-zero and the job releases all of
    its hosts; after a clean finish, no worker keeps holding hosts.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(Configurable.Config):
        job_name: str = "torchtitan-rl"
        """Prefix of the unique job name."""

        env: dict[str, str] = field(default_factory=dict)
        """Environment variables set on every host of the job."""

        def __post_init__(self) -> None:
            if not re.fullmatch(r"[A-Za-z0-9_-]+", self.job_name):
                raise ValueError(
                    "launcher.job_name must be non-empty and contain only letters, "
                    f"digits, '_' or '-', got {self.job_name!r}"
                )

    def __init__(self, config: Config) -> None:
        self.config = config

    def plan(self, config: Controller.Config, args: list[str]) -> LaunchPlan:
        layout = self._layout(
            compute_trainer_world_size(config.trainer.parallelism),
            compute_generator_world_size(config.generator.parallelism),
            config.num_generators,
        )
        return LaunchPlan(
            job_name=f"{self.config.job_name}-{uuid.uuid4().hex[:6]}",
            layout=layout,
            controller_args=list(args),
            env={**self.config.env, LAYOUT_ENV: layout.to_json()},
            dump_folder=config.dump_folder,
        )

    def _layout(
        self, trainer_world_size: int, generator_world_size: int, num_generators: int
    ) -> Layout:
        """Each role gets its own nodes."""
        # TODO: pack roles smaller than a node onto shared nodes, e.g. with 8-GPU
        # nodes, a 4-GPU trainer and generator_0 on node 0 and generator_1 and
        # generator_2 on node 1. Layout and spawn_proc_mesh already support roles
        # sharing a mesh; only this planner is missing.
        gpus_per_node = self.gpus_per_node()
        role_world_sizes = {
            "trainer": trainer_world_size,
            **dict.fromkeys(generator_role_names(num_generators), generator_world_size),
        }
        mesh_nodes = {}
        for role, world_size in role_world_sizes.items():
            if world_size < gpus_per_node:
                logger.warning(
                    f"{role} uses {world_size} of a node's {gpus_per_node} GPUs; "
                    "the rest stay idle."
                )
                mesh_nodes[role] = 1
            elif world_size % gpus_per_node == 0:
                mesh_nodes[role] = world_size // gpus_per_node
            else:
                raise ValueError(
                    f"{role} world size ({world_size}) is larger than a node and not "
                    f"a multiple of gpus_per_node ({gpus_per_node})."
                )
        return Layout(
            gpus_per_node=gpus_per_node,
            mesh_nodes=mesh_nodes,
            role_mesh={role: role for role in role_world_sizes},
        )

    @abc.abstractmethod
    def gpus_per_node(self) -> int:
        """GPUs on each allocated node."""

    @abc.abstractmethod
    def submit(self, plan: LaunchPlan) -> str:
        """Submit a job that sets ``plan.env`` on every host and runs
        ``python -m torchtitan.rl.train`` with ``plan.controller_args``; return
        its job handle."""

    @abc.abstractmethod
    def in_job(self) -> bool:
        """Whether this process is the controller of a submitted job."""

    @abc.abstractmethod
    async def acquire_hosts(self, layout: Layout) -> dict[str, HostMesh]:
        """Attach to the job's hosts, keyed by ``layout.mesh_nodes`` names."""

    async def release_hosts(self, meshes: dict[str, HostMesh]) -> None:
        """Called by ``MonarchActorSystem.stop``. No-op for schedulers that end the
        workers when the controller exits."""
