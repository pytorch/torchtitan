# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass


@dataclass(kw_only=True, slots=True)
class FTFSDPConfig:
    """Fault tolerant FSDP settings.

    Deployment addresses come from the environment so the same recipe runs
    locally and on a cluster:

    - ``FTFSDP_STORE_ADDR``: ``host:port`` of the shared TCPStore.
    - ``TORCHFT_LIGHTHOUSE``: lighthouse address.
    - ``FTFSDP_HOST_INDEX``: index of this host in ``[0, num_hosts)``.
    - ``FTFSDP_HOST_NAME``: unique host name (defaults to the hostname).
    - ``FTFSDP_RUN_ID``: run identifier shared by all hosts.
    """

    num_active_hosts: int = 2
    """Hosts that train concurrently. Remaining hosts are hot spares."""

    num_hosts: int = 3
    """Total hosts including spares."""

    procs_per_host: int = 1
    """Trainer processes per host."""

    snapshot_interval: int = 1
    """Take an in-memory snapshot every N steps."""

    num_local_snapshots: int = 4
    """Pinned CPU snapshot slots per rank. Needs one slot in flight plus enough
    committed slots to cover the lag between local commit and replication."""

    meta_capacity_bytes: int = 4 << 20
    """Bytes reserved per snapshot for pickled CPU state (dataloader, LR
    schedulers, optimizer scalars)."""

    quorum_timeout_seconds: float = 60.0
    """Timeout for one lighthouse quorum request. Requests are retried. Must
    exceed the lighthouse join timeout, otherwise a heartbeating host that
    never joins (e.g. hung in CUDA) blocks every quorum."""

    recovery_timeout_seconds: float = 600.0
    """Timeout for store waits during recovery. Must cover a spare building
    the model."""

    store_op_timeout_seconds: float = 30.0
    """Abandon a store connection when one operation takes longer than this
    and retry on a new connection."""

    heartbeat_interval_seconds: float = 1.0
    """Lighthouse heartbeat period."""

    max_recoveries: int = 100
    """Give up after this many recoveries."""
