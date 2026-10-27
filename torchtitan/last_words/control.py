# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
# This source code is licensed under the BSD-style license found in LICENSE.

"""CPU-side stop requests and replica-local safe-boundary coordination."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, fields, is_dataclass
from functools import partial
from pathlib import Path
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable

from .file_backend import process_identity, PROTOCOL_VERSION

logger = logging.getLogger(__name__)


@dataclass(kw_only=True, slots=True)
class LastWordsConfig:
    enable: bool = False
    coordination_backend: str = "service"
    run_id: str = ""
    """Unique launch ID shared by every replica; change it on restart."""
    timeout_seconds: int = 300
    heartbeat_timeout_seconds: int = 60
    resume_path: str = ""
    """A published candidate directory; empty selects the latest compatible candidate."""

    def validate(self) -> None:
        if self.coordination_backend not in {"file", "service"}:
            raise ValueError("coordination_backend must be file or service")
        if not self.enable:
            return
        if (
            not self.run_id
            or Path(self.run_id).name != self.run_id
            or self.run_id in {".", ".."}
        ):
            raise ValueError(
                "last_words.run_id must be a unique nonempty path component shared by all replicas"
            )
        if self.timeout_seconds <= 0 or self.heartbeat_timeout_seconds <= 0:
            raise ValueError("Last-words timeouts must be positive")


def _sync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def sync_checkpoint_tree(path: Path, *, root: Path) -> None:
    """Persist copied checkpoint data and directory entries before publication."""
    path, root = path.resolve(), root.resolve()
    if not path.is_relative_to(root):
        raise ValueError("Checkpoint durability root must contain the checkpoint")
    if not path.is_dir():
        raise FileNotFoundError(f"Checkpoint directory is missing: {path}")

    def failed_walk(error):
        raise error

    for directory, _, files in os.walk(path, topdown=False, onerror=failed_walk):
        for name in files:
            with (Path(directory) / name).open("rb") as stream:
                os.fsync(stream.fileno())
        _sync_directory(Path(directory))
    parent = path.parent
    while parent.is_relative_to(root):
        _sync_directory(parent)
        if parent == root:
            break
        parent = parent.parent


def atomic_json(
    path: Path, value: Any, *, durable: bool = False, sync_file: bool = True
) -> None:
    missing = []
    parent = path.parent
    while durable and not parent.exists():
        missing.append(parent)
        parent = parent.parent
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("x", encoding="utf-8") as stream:
            json.dump(value, stream, sort_keys=True)
            stream.flush()
            if sync_file or durable:
                os.fsync(stream.fileno())
        os.replace(temporary, path)
        if durable:
            _sync_directory(path.parent)
            for directory in missing:
                _sync_directory(directory.parent)
    finally:
        temporary.unlink(missing_ok=True)


def read_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as stream:
        return json.load(stream)


def config_signature(config: Any) -> str:
    def normalize(value: Any) -> Any:
        if is_dataclass(value) and not isinstance(value, type):
            return {
                field.name: normalize(getattr(value, field.name))
                for field in fields(value)
            }
        if isinstance(value, dict):
            return {str(key): normalize(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [normalize(item) for item in value]
        if isinstance(value, (str, int, float, bool)) or value is None:
            return value
        if isinstance(value, partial):
            return {
                "function": normalize(value.func),
                "args": normalize(value.args),
                "keywords": normalize(value.keywords),
            }
        if callable(value):
            return f"{value.__module__}.{value.__qualname__}"
        return str(value)

    training = config.training
    if not hasattr(config, "model_spec"):
        # The Engine generation uses model/optim/token batching instead of the
        # v0.3 model_spec/optimizer fields. Keep old signatures byte-compatible
        # and explicitly distinguish the new configuration contract.
        contract = {
            "schema": "training-engine-v2",
            "model": normalize(config.model),
            "parallelism": normalize(config.parallelism),
            "dataloader": normalize(config.dataloader),
            "tokenizer": normalize(config.tokenizer),
            "optim": normalize(config.optim),
            "training": normalize(training),
            "group_size": config.fault_tolerance.group_size,
            "override": normalize(config.override),
            "activation_remat": normalize(config.activation_remat),
            "debug": normalize(config.debug),
            "loss": normalize(config.loss),
        }
        return hashlib.sha256(json.dumps(contract, sort_keys=True).encode()).hexdigest()
    contract = {
        "model": normalize(config.model_spec.model),
        "flavor": config.model_spec.flavor,
        "parallelism": normalize(config.parallelism),
        "dataloader": normalize(config.dataloader),
        "tokenizer": normalize(config.tokenizer),
        "optimizer": normalize(config.optimizer),
        "lr_scheduler": normalize(config.lr_scheduler),
        "training": {
            key: normalize(getattr(training, key))
            for key in ("local_batch_size", "global_batch_size", "seq_len", "steps")
        },
        "group_size": config.fault_tolerance.group_size,
        "override": normalize(config.override),
        "compile": normalize(config.compile),
    }
    return hashlib.sha256(json.dumps(contract, sort_keys=True).encode()).hexdigest()


class LastWordsExitError(RuntimeError):
    """The job must restart after an emergency save or failed dump."""


class LastWordsCoordinator:
    def __init__(
        self,
        config: LastWordsConfig,
        *,
        folder: str,
        replica_id: int,
        num_replicas: int,
        rank: int,
        world_size: int,
        elastic_enabled: bool = False,
        on_stop: Callable[[], None] | None = None,
    ):
        config.validate()
        self.config = config
        self.replica_id = replica_id
        self.num_replicas = num_replicas
        self.rank = rank
        self.world_size = world_size
        self.elastic_enabled = elastic_enabled
        self._on_stop = on_stop
        self.generation = (
            int(os.environ.get("LAST_WORDS_REPLICA_GENERATION", "0"))
            if elastic_enabled
            else 0
        )
        initial_slots = os.environ.get("LAST_WORDS_INITIAL_REPLICAS", "")
        self.initial_replicas = (
            json.loads(initial_slots)
            if elastic_enabled and initial_slots
            else list(range(num_replicas))
        )
        if (
            self.generation < 0
            or not isinstance(self.initial_replicas, list)
            or not self.initial_replicas
            or any(
                type(replica) is not int or not 0 <= replica < num_replicas
                for replica in self.initial_replicas
            )
            or len(set(self.initial_replicas)) != len(self.initial_replicas)
        ):
            raise ValueError("Invalid elastic replica generation or initial slots")
        self._released_event = -1
        self._recovery_notice = 0
        self.root = Path(folder) / "last_words" / "control" / config.run_id
        if "://" in folder:
            raise ValueError(
                "Last words requires a shared POSIX filesystem checkpoint folder"
            )
        self.root.mkdir(parents=True, exist_ok=True)
        self._own = self.root / f"replica-{replica_id}-rank-{rank}.json"
        if self._own.exists():
            raise ValueError(
                "last_words.run_id was already used; use a fresh launch ID"
            )
        self._lock = threading.Lock()
        self._status: dict[str, Any] = {
            "phase": "initializing",
            "replica_id": replica_id,
            "rank": rank,
        }
        self._status["world_size"] = world_size
        self._status["timeout_seconds"] = config.timeout_seconds
        self._status["heartbeat_timeout_seconds"] = config.heartbeat_timeout_seconds
        self._coordinator = None
        backend = os.environ.get(
            "LAST_WORDS_COORDINATION_BACKEND", config.coordination_backend
        )
        if backend != config.coordination_backend:
            raise ValueError("Launcher and worker coordination backends disagree")
        if backend == "service":
            from .client import WorkerTransport

            self._coordinator = WorkerTransport(config.run_id, replica_id, rank)
        if elastic_enabled and (
            self._coordinator is None or not self._coordinator.elastic_enabled
        ):
            raise ValueError(
                "Elastic last words requires a matching elastic service registration"
            )
        self._supervised = os.environ.get("LAST_WORDS_SUPERVISED_RUN") == config.run_id
        if self._supervised and self._coordinator is None:
            supervisor = read_json(self.root / "supervisor.json")
            if (
                supervisor.get("protocol_version") != PROTOCOL_VERSION
                or supervisor.get("run_id") != config.run_id
                or process_identity(supervisor["pid"]) != supervisor["start_time"]
            ):
                raise ValueError(
                    "Last words requires a live supervisor for this launch"
                )
        self._durability_executor = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="last-words-durability"
        )
        self._closed = threading.Event()
        self._requested = threading.Event()
        self._error: BaseException | None = None
        self._started = time.monotonic()
        self._request_time: float | None = None
        self._monitor_progress = self._started
        self._thread = threading.Thread(
            target=self._monitor, name="last-words-control", daemon=True
        )
        self._watchdog_thread = threading.Thread(
            target=self._watchdog, name="last-words-deadline", daemon=True
        )
        self._watchdog_thread.start()
        try:
            self._write_status()
        except BaseException:
            self._closed.set()
            raise
        self._thread.start()

    def _write_status(self):
        with self._lock:
            status = dict(self._status, heartbeat=time.time())
        if self._coordinator is None:
            atomic_json(self._own, status, sync_file=False)

    def publish(self, phase: str, *, state: dict[str, Any] | None = None) -> None:
        with self._lock:
            self._status["phase"] = phase
            if phase != "quorum_ready":
                self._status.pop("quorum", None)
            if state is not None:
                self._status["state"] = dict(state)
        self._write_status()
        logger.debug(
            "Last words replica=%d rank=%d phase=%s state=%s",
            self.replica_id,
            self.rank,
            phase,
            state,
        )

    def request_recovery(self, reason: str) -> None:
        if not self.elastic_enabled:
            return self.request(reason)
        with self._lock:
            self._recovery_notice += 1
            self._status["recoverable"] = True
        atomic_json(
            self.root / f"recovery-{self.replica_id}-{self.rank}.json",
            {"reason": reason, "time": time.time()},
        )

    def training_boundary(
        self, *, quorum_id: int, members: list[int], state: dict, torchft_state: dict
    ) -> bool:
        """Wait for all native quorum observations; True requests safe saving.

        Called on the training thread after synchronous healing, before reading
        another batch or mutating persistent optimizer state.
        """
        if not self.elastic_enabled or self._coordinator is None:
            raise RuntimeError("Elastic training boundary requires the elastic service")
        observation = dict(
            id=quorum_id,
            members=members,
            state=dict(state),
            torchft=dict(torchft_state),
        )
        with self._lock:
            self._status.update(
                phase="quorum_ready", state=dict(state), quorum=observation
            )
            status = dict(self._status)
        # The health heartbeat remains asynchronous. The pre-update gate must
        # not wait for its cadence to advertise a native quorum boundary.
        snapshot = self._coordinator.report(status, self._requested.is_set())
        deadline = time.monotonic() + self.config.timeout_seconds
        while time.monotonic() < deadline:
            if snapshot.get("stop"):
                self._arm_deadline(snapshot.get("event_id"))
            if self.requested:
                return True
            if (
                snapshot is not None
                and snapshot["phase"] == "RUNNING"
                and snapshot.get("observed_quorum") == observation
            ):
                return False
            time.sleep(0.01)
            snapshot = self._coordinator.poll()
        self.request("Training boundary confirmation timed out")
        raise LastWordsExitError("Training boundary confirmation timed out")

    def checkpoint_target(self, replica_id: int) -> Path:
        root = self.root.parents[1] / self.config.run_id
        if self.elastic_enabled:
            assert self._coordinator is not None
            snapshot = self._coordinator.get()
            if snapshot is None or not snapshot.get("event_id"):
                raise LastWordsExitError("No active save event")
            root = root / "events" / f"event-{snapshot['event_id']:06d}"
        return root / f"replica-{replica_id}"

    def resume_baseline(self, state: dict) -> None:
        if not self.elastic_enabled or self._coordinator is None:
            raise LastWordsExitError("This save mode cannot resume in place")
        snapshot = self._coordinator.get()
        if (
            snapshot is None
            or snapshot.get("event_kind") != "baseline"
            or not snapshot.get("event_result")
        ):
            raise LastWordsExitError("No validated nonterminal baseline")
        with self._lock:
            self._status.update(phase="completed", state=dict(state), recoverable=False)
            self._status.pop("quorum", None)
        self._coordinator.resume_ack()
        deadline = time.monotonic() + self.config.timeout_seconds
        while time.monotonic() < deadline:
            snapshot = self._coordinator.get()
            if snapshot is not None and snapshot["result"] is not None:
                raise LastWordsExitError(snapshot["result"]["error"])
            if snapshot is not None and snapshot["phase"] == "RUNNING":
                with self._lock:
                    self._released_event = snapshot["event_id"]
                    self._request_time = None
                    self._requested.clear()
                return
            time.sleep(0.05)
        raise LastWordsExitError("Baseline resume acknowledgement timed out")

    @property
    def can_resume_baseline(self) -> bool:
        if not self.elastic_enabled or self._coordinator is None:
            return False
        snapshot = self._coordinator.get()
        return bool(
            snapshot is not None
            and snapshot.get("event_kind") == "baseline"
            and snapshot.get("event_result")
            and snapshot["result"] is None
        )

    @property
    def terminal_stop_requested(self) -> bool:
        # Baseline stops must allow native healing and resume in place. Only
        # an explicit terminal request/event may interrupt an elastic quorum.
        if not self.requested:
            return False
        if not self.elastic_enabled:
            return True
        with self._lock:
            local_terminal = self._status.get("terminal_requested", False)
        snapshot = self._coordinator.get() if self._coordinator is not None else None
        return bool(
            local_terminal
            or (
                snapshot
                and snapshot.get("stop")
                and snapshot.get("event_kind") == "terminal"
            )
        )

    def request(self, reason: str) -> None:
        if self.elastic_enabled:
            with self._lock:
                self._status["terminal_requested"] = True
        self._arm_deadline()
        # This fault-only file preserves the original exception for diagnosis.
        # Coordinator never reads it to authorize or select a writer.
        atomic_json(
            self.root
            / (
                f"request-{self.replica_id}-{self.rank}-generation-{self.generation}.json"
                if self.elastic_enabled
                else f"request-{self.replica_id}-{self.rank}.json"
            ),
            {
                "reporting_replica": self.replica_id,
                "reporting_rank": self.rank,
                "reason": reason,
                "time": time.time(),
            },
        )

    def _arm_deadline(self, event_id: int | None = None) -> None:
        callback = None
        with self._lock:
            if (
                self.elastic_enabled
                and event_id is not None
                and event_id <= self._released_event
            ):
                return
            if self._request_time is None:
                self._request_time = time.monotonic()
            if not self._requested.is_set():
                self._requested.set()
                callback = getattr(self, "_on_stop", None)
        if callback is not None:
            callback()

    @property
    def requested(self) -> bool:
        if self._error is not None:
            channel = (
                "Coordinator control channel"
                if self._coordinator is not None
                else "shared control storage"
            )
            raise LastWordsExitError(f"Last-words {channel} failed") from self._error
        if not self._supervised and any(self.root.glob("request-*.json")):
            self._arm_deadline()
        return self._requested.is_set()

    def _watchdog(self):
        # This thread never touches shared storage. Even a blocked heartbeat
        # write/read must not prevent the local process from meeting its limit.
        while not self._closed.wait(0.1):
            now = time.monotonic()
            with self._lock:
                requested_at = self._request_time
                progress = self._monitor_progress
            if (
                requested_at is not None
                and now - requested_at > self.config.timeout_seconds
            ):
                os._exit(75)
            if (
                now - progress
                > self.config.heartbeat_timeout_seconds + self.config.timeout_seconds
            ):
                os._exit(75)

    def _monitor(self):
        try:
            while not self._closed.wait(0.5):
                with self._lock:
                    self._monitor_progress = time.monotonic()
                if self._coordinator is not None:
                    with self._lock:
                        status = dict(self._status)
                        recovery_notice = self._recovery_notice
                    snapshot = self._coordinator.report(
                        status, self._requested.is_set()
                    )
                    if status.get("recoverable") and snapshot["phase"] != "RUNNING":
                        with self._lock:
                            if recovery_notice == self._recovery_notice:
                                self._status["recoverable"] = False
                    if snapshot["stop"]:
                        self._arm_deadline(snapshot.get("event_id"))
                    if (
                        snapshot["result"] is not None
                        and not snapshot["result"]["success"]
                    ):
                        raise LastWordsExitError(snapshot["result"]["error"])
                    continue
                self._write_status()
                if self._supervised:
                    supervisor = read_json(self.root / "supervisor.json")
                    if (
                        supervisor.get("protocol_version") != PROTOCOL_VERSION
                        or supervisor.get("run_id") != self.config.run_id
                        or process_identity(supervisor["pid"])
                        != supervisor["start_time"]
                    ):
                        raise LastWordsExitError(
                            "Single-writer supervisor identity is no longer alive"
                        )
                    heartbeat = self.root / "supervisor-heartbeat.json"
                    if (
                        time.time() - read_json(heartbeat)["heartbeat"]
                        > self.config.heartbeat_timeout_seconds
                    ):
                        raise LastWordsExitError(
                            "Single-writer supervisor heartbeat expired"
                        )
                    if (self.root / "stop.json").is_file():
                        self._arm_deadline()
                    continue
                if self.requested:
                    continue
                if (
                    time.monotonic() - self._started
                    < self.config.heartbeat_timeout_seconds
                ):
                    continue
                for replica in range(self.num_replicas):
                    for rank in range(self.world_size):
                        peer = self.root / f"replica-{replica}-rank-{rank}.json"
                        if (
                            not peer.is_file()
                            or time.time() - read_json(peer)["heartbeat"]
                            > self.config.heartbeat_timeout_seconds
                        ):
                            self.request(
                                f"missing heartbeat replica={replica} rank={rank}"
                            )
        except BaseException as error:
            self._error = error
            self._arm_deadline()
            logger.exception("Last-words control monitor failed")

    def wait_frozen(self, state: dict[str, Any]) -> None:
        self.publish("frozen", state=state)
        deadline = time.monotonic() + self.config.timeout_seconds
        while time.monotonic() < deadline:
            if self._coordinator is not None:
                snapshot = self._coordinator.get()
                report = (
                    snapshot["reports"].get(str(self.replica_id)) if snapshot else None
                )
                statuses = report["shards"] if report else []
                if any(
                    status and status.get("phase") == "ineligible"
                    for status in statuses
                ):
                    raise LastWordsExitError("Replica contains an unsafe rank")
                if len(statuses) == self.world_size and all(
                    status and status.get("phase") == "frozen" for status in statuses
                ):
                    if any(status.get("state") != state for status in statuses):
                        raise LastWordsExitError(
                            "Replica ranks stopped at different completed versions"
                        )
                    return
                _ = self.requested
                time.sleep(0.05)
                continue
            statuses = []
            for rank in range(self.world_size):
                path = self.root / f"replica-{self.replica_id}-rank-{rank}.json"
                if path.is_file():
                    statuses.append(read_json(path))
            if any(status["phase"] == "ineligible" for status in statuses):
                raise LastWordsExitError(
                    "Replica contains an unsafe rank; cannot save its shards"
                )
            if len(statuses) == self.world_size and all(
                status["phase"] == "frozen" for status in statuses
            ):
                if any(status["state"] != state for status in statuses):
                    raise LastWordsExitError(
                        "Replica ranks stopped at different completed versions"
                    )
                return
            time.sleep(0.05)
        raise LastWordsExitError("Timed out collecting complete replica shards")

    def close(self) -> None:
        self._closed.set()
        self._thread.join(timeout=2)
        self._watchdog_thread.join(timeout=2)
        self._durability_executor.shutdown(wait=False)

    def await_save_authorization(self, state, signature):
        if self._coordinator is not None:
            self._coordinator.set_ready(state, signature)
            while True:
                result = self._save_result()
                if result is not None:
                    return False, result
                snapshot = self._coordinator.get()
                grant = snapshot["grant"] if snapshot else None
                if (
                    snapshot is not None
                    and grant
                    and snapshot["phase"] == "SAVING"
                    and grant["replica_id"] == self.replica_id
                ):
                    if grant["state"] != state or grant["signature"] != signature:
                        raise LastWordsExitError(
                            "Native grant does not match frozen state"
                        )
                    self._coordinator_grant = dict(
                        instance=snapshot["instance"],
                        run_id=self.config.run_id,
                        round=grant["round"],
                        replica_id=self.replica_id,
                    )
                    self._coordinator_grant.update(
                        {
                            key: grant[key]
                            for key in ("event_id", "membership_epoch")
                            if key in grant
                        }
                    )
                    return True, self.checkpoint_target(self.replica_id)
                _ = self.requested
                time.sleep(0.05)
        if not self._supervised:
            raise LastWordsExitError(
                "Emergency saving requires the single-writer job supervisor"
            )
        atomic_json(
            self.root / f"ready-{self.replica_id}-{self.rank}.json",
            {
                "protocol_version": PROTOCOL_VERSION,
                "run_id": self.config.run_id,
                "replica_id": self.replica_id,
                "rank": self.rank,
                "world_size": self.world_size,
                "state": state,
                "signature": signature,
            },
            durable=True,
        )
        while True:
            result = self._save_result()
            if result is not None:
                return False, result
            grant_path = self.root / "grant.json"
            if grant_path.is_file():
                grant = read_json(grant_path)
                if (
                    grant.get("protocol_version") != PROTOCOL_VERSION
                    or grant.get("run_id") != self.config.run_id
                ):
                    raise LastWordsExitError(
                        "Writer grant belongs to another protocol or launch"
                    )
                if grant["replica_id"] == self.replica_id:
                    if grant["state"] != state or grant["signature"] != signature:
                        raise LastWordsExitError(
                            "Writer grant does not match frozen state"
                        )
                    return (
                        True,
                        self.root.parents[1]
                        / self.config.run_id
                        / f"replica-{self.replica_id}",
                    )
            _ = self.requested
            time.sleep(0.05)

    def _save_result(self):
        if self._coordinator is not None:
            snapshot = self._coordinator.get()
            result = snapshot["result"] if snapshot else None
            if self.elastic_enabled and snapshot is not None and result is None:
                event_result = snapshot.get("event_result")
                if (
                    event_result is not None
                    and event_result["event_id"] == snapshot["event_id"]
                ):
                    result = dict(
                        event_result, replica_id=event_result["grant"]["replica_id"]
                    )
            if result is None:
                return None
            if not result["success"]:
                raise LastWordsExitError(result["error"])
            if result["replica_id"] is None:
                raise LastWordsExitError("Expected an emergency save result")
            return self.checkpoint_target(result["replica_id"])
        path = self.root / "result.json"
        if not path.is_file():
            return None
        result = read_json(path)
        if (
            result.get("protocol_version") != PROTOCOL_VERSION
            or result.get("run_id") != self.config.run_id
        ):
            raise LastWordsExitError(
                "Save result belongs to another protocol or launch"
            )
        if not result["success"]:
            raise LastWordsExitError(result["error"])
        target = Path(result["target"]).resolve()
        expected = self.root.parents[1] / self.config.run_id
        if target.parent != expected.resolve():
            raise LastWordsExitError(
                "Writer result escaped this launch checkpoint directory"
            )
        return target

    def wait_save_result(self):
        while True:
            result = self._save_result()
            if result is not None:
                return result
            _ = self.requested
            time.sleep(0.05)

    def write_authorization_binding(self, target: Path) -> None:
        if self._coordinator is not None and self.rank == 0:
            atomic_json(
                target / "coordinator-binding.json",
                self._coordinator_grant,
                durable=True,
            )

    def wait_publication(self, target: Path) -> None:
        deadline = time.monotonic() + self.config.timeout_seconds
        while time.monotonic() < deadline:
            result_path = target / "publication.json"
            if result_path.is_file():
                result = read_json(result_path)
                if not result["success"]:
                    raise LastWordsExitError(result["error"])
                # Each rank acknowledges the result before torchrun may kill peers.
                atomic_json(target / f"ack-{self.rank}.json", {"published": True})
                if all(
                    (target / f"ack-{rank}.json").is_file()
                    for rank in range(self.world_size)
                ):
                    return
            time.sleep(0.05)
        raise LastWordsExitError(
            "Timed out waiting for emergency checkpoint publication"
        )
