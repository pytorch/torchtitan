# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
# This source code is licensed under the BSD-style license found in LICENSE.

"""Standard-library-only single-writer protocol shared with the job supervisor."""

import json
import os
import time
import uuid
from pathlib import Path

PROTOCOL_VERSION = 2


def read_record(path):
    with Path(path).open(encoding="utf-8") as stream:
        return json.load(stream)


def write_record(path, value, *, durable=True):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("x", encoding="utf-8") as stream:
            json.dump(value, stream, sort_keys=True)
            stream.flush()
            if durable:
                os.fsync(stream.fileno())
        os.replace(temporary, path)
        if durable:
            descriptor = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
            try:
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
    finally:
        temporary.unlink(missing_ok=True)


def process_identity(pid):
    try:
        fields = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
        return None if fields[0] == "Z" else fields[19]
    except (FileNotFoundError, ProcessLookupError):
        return None


class SaveElection:
    """One authority; a failed writer is fenced by terminating its whole process tree."""

    def __init__(
        self,
        root,
        run_id,
        *,
        num_replicas,
        world_size,
        heartbeat_timeout,
        timeout,
        validate,
        startup_timeout=None,
    ):
        self.root = Path(root)
        self.run_id = run_id
        self.control = self.root / "last_words/control" / run_id
        self.control.mkdir(parents=True, exist_ok=True)
        self.num_replicas = num_replicas
        self.world_size = world_size
        self.heartbeat_timeout = heartbeat_timeout
        self.startup_timeout = (
            heartbeat_timeout if startup_timeout is None else startup_timeout
        )
        self.timeout = timeout
        self.validate = validate
        self.started = time.monotonic()
        self.fault_at = None
        self.writer = None
        self.writer_deadline = None
        self.round = 0
        self.excluded = set()
        self.signature = None
        self.finished = False
        self.identity = {"protocol_version": PROTOCOL_VERSION, "run_id": run_id}
        write_record(
            self.control / "supervisor.json",
            dict(
                self.identity, pid=os.getpid(), start_time=process_identity(os.getpid())
            ),
        )
        write_record(
            self.control / "supervisor-heartbeat.json",
            dict(self.identity, heartbeat=time.time()),
            durable=False,
        )

    def request_stop(self, reason):
        if self.fault_at is None:
            self.fault_at = time.monotonic()
            write_record(self.control / "stop.json", dict(self.identity, reason=reason))

    def _finish(self, *, target=None, error=""):
        write_record(
            self.control / "result.json",
            dict(
                self.identity,
                success=not error,
                target=str(target) if target else "",
                error=error,
                round=self.round,
            ),
        )
        self.finished = True

    def _ready(self, replica):
        records = []
        for rank in range(self.world_size):
            path = self.control / f"ready-{replica}-{rank}.json"
            if not path.is_file():
                return None
            value = read_record(path)
            expected = dict(
                self.identity, replica_id=replica, rank=rank, world_size=self.world_size
            )
            if any(value.get(key) != item for key, item in expected.items()):
                raise ValueError("Ready record does not match this supervised launch")
            records.append(value)
        first = records[0]
        if any(
            value["state"] != first["state"] or value["signature"] != first["signature"]
            for value in records
        ):
            raise ValueError("Ready shards disagree on completed state or signature")
        if self.signature is None:
            self.signature = first["signature"]
        if self.signature != first["signature"]:
            raise ValueError("Replica checkpoint signatures disagree")
        return first

    def tick(self, children, *, terminate):
        if self.finished:
            return
        now = time.monotonic()
        write_record(
            self.control / "supervisor-heartbeat.json",
            dict(self.identity, heartbeat=time.time()),
            durable=False,
        )
        statuses = {}
        for replica in range(self.num_replicas):
            if children[replica].poll() == 0:
                continue
            for rank in range(self.world_size):
                path = self.control / f"replica-{replica}-rank-{rank}.json"
                if path.is_file():
                    value = read_record(path)
                    statuses[replica, rank] = value
                    if value.get("world_size") != self.world_size:
                        raise ValueError(
                            "Replica world size does not match supervisor spec"
                        )
                    if time.time() - value["heartbeat"] > self.heartbeat_timeout:
                        self.request_stop(
                            f"Missing heartbeat replica={replica} rank={rank}"
                        )
                elif now - self.started > self.startup_timeout:
                    self.request_stop(f"Missing worker replica={replica} rank={rank}")
        if any(self.control.glob("request-*.json")) or any(
            child.poll() not in (None, 0) for child in children
        ):
            self.request_stop("Worker requested stop or exited")
        if self.fault_at is None:
            return
        if now - self.fault_at >= self.timeout:
            self._finish(error="Single-writer fault deadline expired")
            return
        if self.writer is not None:
            target = self.root / "last_words" / self.run_id / f"replica-{self.writer}"
            publication = target / "publication.json"
            acknowledged = all(
                (target / f"ack-{rank}.json").is_file()
                for rank in range(self.world_size)
            )
            if publication.is_file() and acknowledged:
                result = read_record(publication)
                if result.get("success"):
                    self.validate(target)
                    self._finish(target=target)
                    return
            assert self.writer_deadline is not None
            failed = (
                children[self.writer].poll() is not None or now >= self.writer_deadline
            )
            if failed:
                # No timeout-based lease transfer: every old shard must be dead.
                terminate([children[self.writer]])
                if (
                    publication.is_file()
                    and all(
                        (target / f"ack-{rank}.json").is_file()
                        for rank in range(self.world_size)
                    )
                    and read_record(publication).get("success")
                ):
                    self.validate(target)
                    self._finish(target=target)
                    return
                self.excluded.add(self.writer)
                self.writer = None
            else:
                return
        now = time.monotonic()
        if now - self.fault_at >= self.timeout:
            self._finish(
                error="Single-writer deadline expired during writer termination"
            )
            return
        candidates = []
        pending = False
        for replica, child in enumerate(children):
            if replica in self.excluded or child.poll() is not None:
                continue
            local = [
                statuses.get((replica, rank), {}) for rank in range(self.world_size)
            ]
            if any(value.get("phase") == "ineligible" for value in local):
                continue
            ready = self._ready(replica)
            if ready is None:
                pending = True
            else:
                state = ready["state"]
                candidates.append(
                    (state["last_words_num_updates"], state["step"], -replica, ready)
                )
        # The preparation window limits how long an already ready replica waits
        # for peers. With no candidate, a live replica may still be unwinding a
        # failed collective; only the original global fault deadline bounds it.
        if pending and (not candidates or now - self.fault_at < self.heartbeat_timeout):
            return
        if not candidates:
            self._finish(error="No complete live replica is ready to save")
            return
        ready = max(candidates, key=lambda item: item[:3])[3]
        self.writer = ready["replica_id"]
        remaining = self.timeout - (now - self.fault_at)
        self.writer_deadline = now + (
            remaining / 2 if len(candidates) > 1 else remaining
        )
        self.round += 1
        write_record(
            self.control / "grant.json",
            dict(
                self.identity,
                round=self.round,
                replica_id=self.writer,
                state=ready["state"],
                signature=ready["signature"],
            ),
        )
