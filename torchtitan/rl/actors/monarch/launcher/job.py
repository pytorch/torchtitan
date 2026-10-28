# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""The controller of a submitted job: ``torchtitan.rl.train`` on the launcher's hosts.

``submit`` makes the scheduler run this module with its own arguments, so the job
loads the same ``MonarchLaunchConfig``. It attaches to the hosts of the layout
``submit`` planned and builds the actor system on them.
"""

# Imported first: it applies process env defaults before anything imports torch.
from torchtitan.rl import train

from torchtitan.rl.actors.monarch.launcher.base import Layout, MonarchLaunchConfig
from torchtitan.rl.actors.monarch.system import MonarchActorSystem


async def build_actor_system(config: MonarchLaunchConfig) -> MonarchActorSystem:
    layout = Layout.from_env()
    meshes = await config.launcher.build().acquire_hosts(layout)
    return config.actor_system.build(host_meshes=layout.host_meshes(meshes))


if __name__ == "__main__":
    train.run(build_actor_system=build_actor_system)
