# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Host membership for fault tolerant FSDP.

Every trainer process initializes one ``nccl2`` process group over all hosts
(active and spare) with ``enable_reconfigure=True``. A lighthouse quorum
decides which hosts own the ``num_active_hosts`` slots; the process group is
then reconfigured so that slot ``s``, local rank ``l`` becomes global rank
``s * procs_per_host + l``. Survivors keep their slot, so their rank and
their snapshot stay valid across reconfigurations.
"""

import hashlib
import logging
import threading
import time
import uuid
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeout
from dataclasses import asdict, dataclass
from datetime import timedelta

import torch.distributed as dist

logger = logging.getLogger(__name__)

DONE_KEY = "ftfsdp/done"
STARTED_KEY = "ftfsdp/started"
GEN_KEY = "ftfsdp/gen"

# A quorum can form while some processes of a slot holder are still detecting
# the failure. Such a host is only evicted after this many quorum rounds.
EVICT_INCOMPLETE_ROUNDS = 3


class TrainingFinishedError(Exception):
    """Training completed; spares should exit cleanly."""


class EvictedError(Exception):
    """This host lost its slot and must restart as a spare."""


@dataclass(frozen=True)
class MemberInfo:
    """What a process reports to the lighthouse when joining a quorum."""

    host: str
    local_rank: int
    uid: int
    ident: str
    """Unique per process incarnation."""
    handle: str
    slot: int
    gen: int
    round: int = 0
    """Quorum requests this process made in the current recovery."""


@dataclass(frozen=True)
class Assignment:
    """Result of slot assignment for one quorum.

    ``ranks`` is ordered by new global rank and is empty if there were not
    enough complete spare hosts to fill every slot or a new slot's successor
    slot has no survivor.
    """

    gen: int
    ranks: tuple[MemberInfo, ...]
    host_slots: dict[str, int]
    evict: frozenset[str]
    new_hosts: frozenset[str]

    @property
    def initial(self) -> bool:
        return self.gen == 0


def assign_slots(
    members: list[MemberInfo], *, num_slots: int, procs_per_host: int
) -> Assignment:
    """Deterministically assign hosts to slots.

    A host keeps its slot if all of its processes are present, agree on the
    slot, and are at the newest generation. A host holding a slot that fails
    these checks is evicted; a host that only misses processes is evicted
    once some slot holder has made ``EVICT_INCOMPLETE_ROUNDS`` requests, and
    until then no assignment is made. Free slots go to complete spare hosts in host
    name order.
    """
    by_host: dict[str, list[MemberInfo]] = defaultdict(list)
    for m in members:
        by_host[m.host].append(m)
    slotted_gens = [m.gen for m in members if m.slot >= 0]
    max_gen = max(slotted_gens, default=-1)
    # Spares keep counting across recoveries, so only slot holders count.
    max_round = max((m.round for m in members if m.slot >= 0), default=0)

    def complete(ms: list[MemberInfo]) -> bool:
        return sorted(m.local_rank for m in ms) == list(range(procs_per_host))

    kept: dict[str, int] = {}
    evict: set[str] = set()
    candidates: list[str] = []
    waiting = False
    for host, ms in sorted(by_host.items()):
        if all(m.slot < 0 for m in ms):
            if complete(ms):
                candidates.append(host)
            continue
        slots = {m.slot for m in ms}
        gens = {m.gen for m in ms}
        if len(slots) != 1 or gens != {max_gen}:
            evict.add(host)
            continue
        if not complete(ms):
            if max_round < EVICT_INCOMPLETE_ROUNDS:
                waiting = True
            else:
                evict.add(host)
            continue
        (slot,) = slots
        if slot >= num_slots:
            raise ValueError(f"host {host} claims slot {slot} >= {num_slots}")
        kept[host] = slot

    owners: dict[int, str] = {}
    for host, slot in kept.items():
        if slot in owners:
            raise ValueError(f"hosts {owners[slot]} and {host} both claim slot {slot}")
        owners[slot] = host

    free = [s for s in range(num_slots) if s not in owners]
    # A new host restores from the replica held by the next slot. Without a
    # surviving successor the shard is lost, which usually means a survivor
    # has not detected the failure yet, so wait for it.
    unrecoverable = max_gen >= 0 and any(
        (s + 1) % num_slots not in owners for s in free
    )
    if waiting or unrecoverable or len(candidates) < len(free):
        return Assignment(
            gen=max_gen,
            ranks=(),
            host_slots={},
            evict=frozenset(evict),
            new_hosts=frozenset(),
        )
    new_hosts = candidates[: len(free)]
    for slot, host in zip(free, new_hosts):
        owners[slot] = host

    ranks: list[MemberInfo] = []
    for slot in range(num_slots):
        ms = sorted(by_host[owners[slot]], key=lambda m: m.local_rank)
        ranks.extend(ms)
    return Assignment(
        gen=max_gen + 1,
        ranks=tuple(ranks),
        host_slots={host: slot for slot, host in owners.items()},
        evict=frozenset(evict),
        new_hosts=frozenset(new_hosts),
    )


def reconfigure_uuid(run_id: str, gen: int, quorum_id: int) -> int:
    digest = hashlib.sha256(f"{run_id}/{gen}/{quorum_id}".encode()).digest()
    return int.from_bytes(digest[:8], "little") & ((1 << 63) - 1)


class Membership:
    """Joins lighthouse quorums and reconfigures the default process group."""

    def __init__(
        self,
        *,
        store: dist.Store,
        pg_store: dist.Store,
        lighthouse_addr: str,
        host: str,
        host_index: int,
        local_rank: int,
        num_hosts: int,
        num_slots: int,
        procs_per_host: int,
        run_id: str,
        pg_timeout: timedelta,
        quorum_timeout: timedelta,
        spare_quorum_timeout: timedelta,
        heartbeat_interval: float,
    ) -> None:
        from torchft._torchft import LighthouseClient

        if not 0 <= host_index < num_hosts:
            raise ValueError(f"host index {host_index} not in [0, {num_hosts})")
        if not 0 <= local_rank < procs_per_host:
            raise ValueError(f"local rank {local_rank} not in [0, {procs_per_host})")
        self.store = store
        self.host = host
        self.local_rank = local_rank
        self.num_slots = num_slots
        self.procs_per_host = procs_per_host
        self.run_id = run_id
        self.pg_timeout = pg_timeout
        self.quorum_timeout = quorum_timeout
        self.spare_quorum_timeout = spare_quorum_timeout
        self.uid = host_index * procs_per_host + local_rank
        self.replica_id = f"{host}/{local_rank}"
        self.ident = f"{self.replica_id}/{uuid.uuid4().hex[:12]}"
        self.slot = -1
        self.gen = -1

        # TCPStore clients serialize ops, so a PG thread blocked on the store
        # after an abort would stall membership. ``pg_store`` is a separate
        # client.
        dist.init_process_group(
            "nccl2",
            store=dist.PrefixStore("ftfsdp/pg", pg_store),
            rank=self.uid,
            world_size=num_hosts * procs_per_host,
            enable_reconfigure=True,
            timeout=pg_timeout,
        )
        self.client = LighthouseClient(lighthouse_addr, timedelta(seconds=60))
        self._stop = threading.Event()
        self._heartbeat = threading.Thread(
            target=self._heartbeat_loop,
            args=(heartbeat_interval,),
            name="ftfsdp-heartbeat",
            daemon=True,
        )
        self._heartbeat.start()
        self._quorum_pool = ThreadPoolExecutor(1, thread_name_prefix="ftfsdp-quorum")

    def _heartbeat_loop(self, interval: float) -> None:
        while not self._stop.wait(interval):
            try:
                self.client.heartbeat(self.replica_id)
            except Exception as e:
                logger.warning(f"lighthouse heartbeat failed: {e}")

    def close(self) -> None:
        self._stop.set()

    def recovery_pending(self) -> bool:
        """Whether a process of the current generation requested recovery."""
        return self.store.check([_recovery_key(self.gen)])

    def _recovery_requested(self) -> bool:
        """Whether a quorum is wanted now.

        Spares only join once a slot holder of the latest generation asks for
        recovery. A spare that sits in the lighthouse queue would satisfy
        min_replicas the moment the first survivor joins and form a quorum
        without the remaining survivors.
        """
        if self.slot >= 0 or not self.store.check([GEN_KEY]):
            return True
        return self.store.check([_recovery_key(int(self.store.get(GEN_KEY)))])

    def _wait_for_gen(self, gen: int) -> None:
        """Wait until generation ``gen`` is published or its reconfigure has
        timed out.

        A spare left out of a quorum would otherwise see the previous
        generation's recovery key and queue a new request. The lighthouse
        keeps that request after the spare stops waiting on it, so a later
        quorum includes a process that never reconfigures.
        """
        deadline = time.monotonic() + self.pg_timeout.total_seconds()
        while time.monotonic() < deadline and not self.store.check([DONE_KEY]):
            if self.store.check([GEN_KEY]) and int(self.store.get(GEN_KEY)) >= gen:
                return
            time.sleep(0.2)

    def _quorum(self, data: dict, timeout: timedelta):
        """Run one quorum request, giving up early once training is done so
        a waiting spare does not outlive the job."""
        fut = self._quorum_pool.submit(
            self.client.quorum, replica_id=self.replica_id, timeout=timeout, data=data
        )
        while True:
            try:
                return fut.result(timeout=1.0)
            except FutureTimeout:
                pass
            if self.store.check([DONE_KEY]):
                raise TrainingFinishedError()

    def next_assignment(self) -> Assignment:
        """Block until this process is assigned a slot and the process group
        has been reconfigured to the new membership.

        Raises ``TrainingFinishedError`` once training is done and ``EvictedError`` if
        this host lost its slot.
        """
        if self.slot >= 0:
            self.store.set(_recovery_key(self.gen), "1")
        rounds = 0
        while True:
            if self.store.check([DONE_KEY]):
                raise TrainingFinishedError()
            if not self._recovery_requested():
                time.sleep(0.2)
                continue
            info = MemberInfo(
                host=self.host,
                local_rank=self.local_rank,
                uid=self.uid,
                ident=self.ident,
                handle=dist._get_reconfigure_handle(),
                slot=self.slot,
                gen=self.gen,
                round=rounds,
            )
            rounds += 1
            # The lighthouse keeps a request after the client gives up on it,
            # so a quorum can include a process that never sees the result.
            # Spares therefore wait in a single long request.
            timeout = (
                self.quorum_timeout if self.slot >= 0 else self.spare_quorum_timeout
            )
            try:
                quorum = self._quorum(asdict(info), timeout)
            except TrainingFinishedError:
                raise
            except Exception as e:
                logger.info(f"quorum not formed, retrying: {e}")
                continue
            members = [MemberInfo(**p.data) for p in quorum.participants]
            assignment = assign_slots(
                members,
                num_slots=self.num_slots,
                procs_per_host=self.procs_per_host,
            )
            if self.local_rank == 0:
                procs = Counter(m.host for m in members)
                incomplete = sorted(
                    h for h, n in procs.items() if n != self.procs_per_host
                )
                logger.info(
                    f"quorum {quorum.quorum_id}: {len(members)} procs on "
                    f"{len(procs)} hosts, incomplete={incomplete}, "
                    f"assigned={bool(assignment.ranks)} gen {assignment.gen}, "
                    f"new_hosts={sorted(assignment.new_hosts)} "
                    f"evict={sorted(assignment.evict)}"
                )
            if self.host in assignment.evict:
                raise EvictedError(
                    f"host {self.host} evicted at quorum {quorum.quorum_id}"
                )
            if self.host not in assignment.host_slots:
                if assignment.ranks:
                    self._wait_for_gen(assignment.gen)
                continue
            reconf_id = reconfigure_uuid(self.run_id, assignment.gen, quorum.quorum_id)
            start = time.perf_counter()
            try:
                dist._reconfigure(
                    reconf_id,
                    [m.handle for m in assignment.ranks],
                    timeout=self.pg_timeout,
                ).wait()
            except Exception as e:
                logger.warning(f"reconfigure failed, rejoining quorum: {e}")
                continue
            self.slot = assignment.host_slots[self.host]
            self.gen = assignment.gen
            if dist.get_rank() == 0:
                self.store.set(GEN_KEY, str(self.gen))
            logger.info(
                f"gen {self.gen}: slot {self.slot} rank {dist.get_rank()}/"
                f"{dist.get_world_size()} new_hosts={sorted(assignment.new_hosts)} "
                f"evict={sorted(assignment.evict)} "
                f"reconfigure {time.perf_counter() - start:.2f}s"
            )
            return assignment


def _recovery_key(gen: int) -> str:
    return f"ftfsdp/recover/{gen}"
