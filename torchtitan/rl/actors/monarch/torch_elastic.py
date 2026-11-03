# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Torch elastic env setup for RL proc meshes."""

import socket

from monarch.actor import Actor, endpoint, ProcMesh
from monarch.spmd import setup_torch_elastic_env_async
from torch.distributed import TCPStore


async def setup_torch_elastic_env(mesh: ProcMesh) -> None:
    """Set the torch elastic env vars (``RANK``, ``MASTER_PORT``, ...) on every proc of ``mesh``.

    Unlike Monarch's ``setup_torch_elastic_env_async(mesh)``, ``MASTER_PORT`` is never released:
    1. Monarch probes a free port and closes it.
    2. Rank 0's ``init_process_group`` binds that port seconds later.
    3. A socket opened in between (NCCL, gloo, Monarch) can take it: EADDRINUSE.
    Here rank 0 starts the ``TCPStore`` server on port 0 first and keeps it, as torchrun does:
    https://github.com/pytorch/pytorch/blob/31a78370bbe3a37f11054580f268162d0b86fe5b/torch/distributed/elastic/rendezvous/dynamic_rendezvous.py#L1228-L1233

    Args:
        mesh: Proc mesh that will call ``init_process_group(init_method="env://")``.
    """
    rank_0_mesh = mesh.flatten("rank").slice(rank=0)
    store_actor = rank_0_mesh.spawn("_rendezvous_store", _RendezvousStoreActor)
    master_addr, master_port = await store_actor.start.call_one()
    await setup_torch_elastic_env_async(mesh, master_addr, master_port)


class _RendezvousStoreActor(Actor):
    @endpoint
    def start(self) -> tuple[str, int]:
        """Start a ``TCPStore`` server on an OS-picked port and return its address."""
        hostname = socket.gethostname()
        # init_process_group(init_method="env://") on rank 0 opens a multi-tenant
        # TCPStore on MASTER_PORT, which reuses this server instead of binding again.
        self._store = TCPStore(
            host_name=hostname,
            port=0,
            is_master=True,
            multi_tenant=True,
            wait_for_workers=False,
        )
        return hostname, self._store.port
