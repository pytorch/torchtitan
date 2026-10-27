# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
# This source code is licensed under the BSD-style license found in LICENSE.

import json
import threading
from pathlib import Path

try:
    from torchft_last_words.client import (
        OwnerClient,
        ServiceClient as ServiceClient,
        SessionClient as SessionClient,
        WIRE_VERSION as WIRE_VERSION,
        WorkerTransport as WorkerTransport,
    )
except ImportError as error:
    raise ImportError(
        "Last-words service requires the paired TorchFT split wheel providing "
        "torchft_last_words.client; install the reviewed wheel from the delivery manifest."
    ) from error

if WIRE_VERSION != 1:
    raise ImportError(
        f"Unsupported TorchFT last-words wire version {WIRE_VERSION}; expected 1"
    )

from .file_backend import PROTOCOL_VERSION, write_record


__all__ = [
    "ServiceClient",
    "SessionClient",
    "WorkerTransport",
    "ServiceElection",
    "WIRE_VERSION",
]


class ServiceElection(OwnerClient):
    """Launcher adapter: process fencing and validation, never writer selection."""

    def __init__(
        self,
        root,
        run_id,
        *,
        address,
        num_replicas,
        world_size,
        heartbeat_timeout,
        startup_timeout,
        timeout,
        validate,
        elastic=None,
    ):
        self.root, self.run_id = Path(root), run_id
        self.address = address
        self.control = self.root / "last_words/control" / run_id
        self.control.mkdir(parents=True, exist_ok=False)
        super().__init__(
            run_id,
            address=address,
            num_replicas=num_replicas,
            world_size=world_size,
            heartbeat_timeout=heartbeat_timeout,
            startup_timeout=startup_timeout,
            timeout=timeout,
            elastic=elastic,
        )
        self.validate = validate
        self.finished = False
        self.success = False
        self._audit_revision = None
        self._audit()

    def _audit(self):
        snapshot = self.snapshot
        self.finished = snapshot["result"] is not None
        self.success = self.finished and snapshot["result"]["success"]
        if snapshot["revision"] == self._audit_revision:
            return
        self._audit_revision = snapshot["revision"]
        # No tokens are included in snapshots or audit records.
        write_record(
            self.control / f"coordinator-{snapshot['revision']:06d}.json", snapshot
        )
        identity = dict(
            protocol_version=PROTOCOL_VERSION,
            run_id=self.run_id,
            coordinator_instance=self.session.instance,
        )
        if snapshot["stop"]:
            write_record(
                self.control / "stop.json", dict(identity, reason=snapshot["reason"])
            )
        if snapshot["grant"]:
            write_record(
                self.control / "grant.json", dict(identity, **snapshot["grant"])
            )
        event_result = snapshot.get("event_result")
        if event_result is not None:
            write_record(
                self.control / f"event-{snapshot['event_id']:06d}-result.json",
                dict(identity, **event_result),
            )
        if self.finished:
            result = dict(snapshot["result"])
            writer = result.get("replica_id")
            result["target"] = (
                str(self.target(snapshot["grant"]))
                if self.success and writer is not None
                else ""
            )
            write_record(self.control / "result.json", dict(identity, **result))

    def request_stop(self, reason):
        self.snapshot = self.session.call("stop", reason)
        self._audit()

    def statuses(self):
        return [
            shard
            for report in self.snapshot["reports"].values()
            for shard in report["shards"]
            if shard
        ]

    def target(self, grant):
        root = self.root / "last_words" / self.run_id
        if "event_id" in grant:
            root = root / "events" / f"event-{grant['event_id']:06d}"
        return root / f"replica-{grant['replica_id']}"

    def admit_external(self, replica, *, generation, token, fenced):
        """Register an externally started generation after external fencing proof."""
        super().admit_external(
            replica, generation=generation, token=token, fenced=fenced
        )
        self._audit()

    def _published(self, target):
        publication = target / "publication.json"
        return (
            publication.is_file()
            and json.loads(publication.read_text())["success"]
            and all(
                (target / f"ack-{rank}.json").is_file()
                for rank in range(self.world_size)
            )
        )

    def _validate(self, target, grant):
        manifest = json.loads((target / "manifest.json").read_text())
        if manifest.get("run_id") != self.run_id or any(
            manifest.get(key) != grant[key]
            for key in ("replica_id", "state", "signature")
        ):
            raise ValueError("Candidate does not match active service grant")
        if any(
            manifest.get(key) != grant[key]
            for key in ("event_id", "membership_epoch")
            if key in grant
        ):
            raise ValueError("Candidate belongs to another elastic event or membership")
        binding = json.loads((target / "coordinator-binding.json").read_text())
        expected = dict(
            instance=self.session.instance,
            run_id=self.run_id,
            round=grant["round"],
            replica_id=grant["replica_id"],
        )
        expected.update(
            {
                key: grant[key]
                for key in ("event_id", "membership_epoch")
                if key in grant
            }
        )
        if binding != expected:
            raise ValueError("Candidate has no matching Coordinator authorization")
        stopped, errors = threading.Event(), []

        def pulse():
            while not stopped.wait(self.heartbeat_interval):
                try:
                    snapshot = self.session.call("owner_tick", {})
                    if (
                        snapshot["result"] is not None
                        and not snapshot["result"]["success"]
                    ):
                        raise RuntimeError(snapshot["result"]["error"])
                except Exception as error:
                    errors.append(error)
                    return

        thread = threading.Thread(
            target=pulse, name="last-words-validation-heartbeat", daemon=True
        )
        thread.start()
        try:
            self.validate(target)
        finally:
            stopped.set()
            thread.join(timeout=5)
        if thread.is_alive():
            raise RuntimeError(
                "Validation heartbeat did not finish within its network budget"
            )
        if errors:
            raise RuntimeError("Coordinator lost during validation") from errors[0]

    def tick(self, children, *, terminate):
        self.snapshot = self.session.call(
            "owner_tick",
            {
                str(i): child.poll()
                for i, child in enumerate(children)
                if child is not None and child.poll() is not None
            },
        )
        self._audit()
        if self.finished:
            return
        grant = self.snapshot["grant"]
        if grant is None or self.snapshot["phase"] == "RESUMING":
            return
        writer = grant["replica_id"]
        target = self.target(grant)
        if self.snapshot["phase"] == "FENCING":
            terminate(
                [children[writer]]
            )  # Must raise if any owned descendant remains alive.
            if not self._published(target):
                self.snapshot = self.session.call(
                    "fenced", True, round=grant["round"], replica=writer
                )
                self._audit()
                return
        if self._published(target):
            try:
                self._validate(target, grant)
            except Exception as error:
                self.snapshot = self.session.call(
                    "fail", f"checkpoint validation failed: {error}"
                )
            else:
                self.snapshot = self.session.call(
                    "validated", round=grant["round"], replica=writer
                )
            self._audit()
