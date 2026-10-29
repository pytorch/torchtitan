#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
# This source code is licensed under the BSD-style license found in LICENSE.

"""Supervise training modes and preview dependency-aware checkpoint retention."""

import argparse
import contextlib
import fcntl
import json
import os
import shutil
import signal
import subprocess
import sys
import threading
import time
import uuid
from pathlib import Path

from torchtitan.last_words import file_backend

PROTOCOL = vars(file_backend)


def contained(root, path):
    resolved = path.resolve()
    if resolved == root or not resolved.is_relative_to(root) or path.is_symlink():
        raise ValueError(f"Refusing checkpoint path outside its root: {path}")
    return resolved


def validate_storage_ranges(path):
    from torchtitan.last_words.validation import validate_storage_ranges as validate

    validate(path)


def recovery_preflight(path, manifest, num_replicas):
    from torchtitan.last_words.validation import recovery_preflight as validate

    validate(path, manifest, num_replicas)


def retention_plan(root, *, keep=2, active_seconds=120, num_replicas=2):
    from torch.distributed.checkpoint.api import CheckpointException

    if keep < 2 or num_replicas < 2:
        raise ValueError("Retain at least two generations for at least two replicas")
    root = root.resolve()
    candidates, protected, notes = [], set(), []
    now = time.time()
    for path in (root / "last_words/control").glob("*/replica-*-rank-*.json"):
        if now - path.stat().st_mtime < active_seconds:
            protected.add(contained(root, root / "last_words" / path.parent.name))
            for cursor_run in root.glob(f"ft-replicat-*/{path.parent.name}"):
                protected.add(contained(root, cursor_run))
    manifests = list((root / "last_words").glob("*/replica-*/manifest.json"))
    manifests.extend((root / "last_words").glob("*/events/event-*/replica-*/manifest.json"))
    for manifest_path in manifests:
        manifest = json.loads(manifest_path.read_text())
        path = contained(root, manifest_path.parent)
        if any(entry.is_symlink() for entry in path.rglob("*")):
            raise ValueError(f"Checkpoint contains symlinks: {path}")
        try:
            recovery_preflight(path, manifest, num_replicas)
        except (Exception, CheckpointException) as error:
            notes.append(f"Unusable candidate preserved for diagnosis: {path}: {error}")
            protected.add(path)
            continue
        candidates.append(
            (
                ("emergency", manifest["signature"]),
                manifest["state"]["last_words_num_updates"],
                path,
            )
        )
    periodic_bindings = {}
    for binding_path in root.glob("step-*/last_words_binding.json"):
        binding = json.loads(binding_path.read_text())
        path = contained(root, binding_path.parent)
        periodic_bindings[path] = binding
        if root / "last_words" / binding["run_id"] in protected:
            protected.add(path)
        try:
            from torchtitan.last_words.validation import periodic_recovery_preflight

            state, _ = periodic_recovery_preflight(
                root,
                int(path.name.removeprefix("step-")),
                signature=binding["signature"],
                num_replicas=num_replicas,
                elastic_enabled="torchft" in binding,
            )
            candidates.append(
                (
                    ("periodic", binding["signature"]),
                    state["last_words_num_updates"],
                    path,
                )
            )
        except (Exception, CheckpointException) as error:
            notes.append(f"Unusable periodic checkpoint preserved for diagnosis: {path}: {error}")
            protected.add(path)
    signatures = {signature for signature, _, _ in candidates}
    for signature in signatures:
        same = [
            (updates, path) for candidate_signature, updates, path in candidates if signature == candidate_signature
        ]
        generations = sorted({updates for updates, _ in same}, reverse=True)[:keep]
        protected.update(path for updates, path in same if updates in generations)
    for path, binding in periodic_bindings.items():
        if path not in protected:
            continue
        for anchor in binding.get("anchors", {}).values():
            protected.add(contained(root, root / anchor))
        # Preserve each retained periodic checkpoint's entire cursor branch.
        for cursor_run in root.glob(f"ft-replicat-*/{binding['run_id']}"):
            protected.add(contained(root, cursor_run))
    deletes = []
    for _, _, path in candidates:
        if any(path.is_relative_to(anchor) or anchor.is_relative_to(path) for anchor in protected):
            continue
        contained(root, path)
        # Unexpected symlinks make cleanup fail closed, including nested entries.
        if any(entry.is_symlink() for entry in path.rglob("*")):
            raise ValueError(f"Checkpoint contains symlinks: {path}")
        deletes.append(str(path))
    return {
        "root": str(root),
        "keep_generations": keep,
        "delete": sorted(deletes),
        "protected": sorted(map(str, protected)),
        "notes": notes,
    }


def running_group_members(group):
    members = []
    # glob checks /proc/PID/stat existence before yielding it; on Python 3.9
    # an exiting task can raise ESRCH outside the guarded read below.
    for directory in Path("/proc").iterdir():
        if not directory.name.isdigit():
            continue
        status = directory / "stat"
        try:
            fields = status.read_text().rsplit(")", 1)[1].split()
        except (FileNotFoundError, ProcessLookupError):
            continue
        if int(fields[2]) == group and fields[0] != "Z":
            members.append(int(status.parent.name))
    return members


def process_table():
    entries = {}
    # Keep the disappearing-task race inside the guarded read, not glob's
    # implicit exists() check while advancing its generator.
    for directory in Path("/proc").iterdir():
        if not directory.name.isdigit():
            continue
        status = directory / "stat"
        try:
            fields = status.read_text().rsplit(")", 1)[1].split()
        except (FileNotFoundError, ProcessLookupError):
            continue
        entries[int(status.parent.name)] = (int(fields[1]), int(fields[2]), fields[19])
    return entries


def descendant_workers(known, entries):
    owned = {pid for pid, token in known.items() if pid in entries and entries[pid][2] == token}
    while True:
        expanded = owned | {pid for pid, (parent, _, _) in entries.items() if parent in owned}
        if expanded == owned:
            return {pid: entries[pid][2] for pid in owned}
        owned = expanded


OWNED_LAUNCH = r"""
import ctypes, json, os, signal, subprocess, sys, time
from pathlib import Path
# Keep an observable parent until all descendants have exited, including
# workers which detach their sessions and outlive the original launcher.
libc = ctypes.CDLL(None, use_errno=True)
if libc.prctl(36, 1, 0, 0, 0) != 0:  # PR_SET_CHILD_SUBREAPER
    raise OSError(ctypes.get_errno(), 'Cannot establish owned child subreaper')
signal.signal(signal.SIGTERM, lambda *_: None)
descriptor = int(sys.argv[1])
gate = os.environ.get('LAST_WORDS_EXTERNAL_GATE')
if gate:
    # Only the external caller creates this anchor. The owner attaches it
    # before releasing the gate, so untracked workers cannot start training.
    owner = os.environ['LAST_WORDS_EXTERNAL_OWNER_PID']
    owner_token = os.environ['LAST_WORDS_EXTERNAL_OWNER_TOKEN']
    deadline = float(os.environ['LAST_WORDS_EXTERNAL_DEADLINE'])
    while True:
        try:
            token = Path('/proc/' + owner + '/stat').read_text().rsplit(')', 1)[1].split()[19]
        except FileNotFoundError:
            sys.exit(75)
        if token != owner_token or time.monotonic() >= deadline:
            sys.exit(75)
        if Path(gate).is_file():
            own_token = Path('/proc/self/stat').read_text().rsplit(')', 1)[1].split()[19]
            if json.loads(Path(gate).read_text()) != {'pid':os.getpid(), 'start_token':own_token}:
                raise RuntimeError('External attachment does not match this process')
            break
        time.sleep(.02)
target = subprocess.Popen(sys.argv[2:])
code = target.wait()
receipt = os.environ.get('LAST_WORDS_EXTERNAL_RECEIPT')
if receipt:
    token = Path('/proc/self/stat').read_text().rsplit(')', 1)[1].split()[19]
    temporary = Path(receipt + '.tmp')
    with temporary.open('x') as stream:
        os.chmod(temporary, 0o600)
        json.dump({'pid':os.getpid(), 'start_token':token, 'return_code':code}, stream)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, receipt)
try:
    os.write(descriptor, (str(code) + '\n').encode())
except BrokenPipeError:
    if not receipt:
        raise
os.close(descriptor)
while True:
    try:
        os.waitpid(-1, 0)
    except ChildProcessError:
        break
os._exit(code if code >= 0 else 128 - code)
"""


class OwnedChild:
    """Expose the launched command's exit while retaining its cleanup anchor."""

    def __init__(self, process, descriptor):
        self.process, self.descriptor = process, descriptor
        self.pid = process.pid
        self.command_code = None
        self.buffer = b""
        os.set_blocking(descriptor, False)

    def poll(self):
        if self.descriptor is not None:
            try:
                data = os.read(self.descriptor, 64)
                self.buffer += data
                if b"\n" in self.buffer:
                    self.command_code = int(self.buffer.split(b"\n", 1)[0])
                if not data or self.command_code is not None:
                    os.close(self.descriptor)
                    self.descriptor = None
            except BlockingIOError:
                pass
        code = self.process.poll()
        return self.command_code if self.command_code is not None else code

    @property
    def returncode(self):
        return self.poll()

    def wait(self, timeout=None):
        deadline = None if timeout is None else time.monotonic() + timeout
        while (code := self.poll()) is None:
            if deadline is not None and time.monotonic() >= deadline:
                raise subprocess.TimeoutExpired(self.process.args, timeout)
            time.sleep(0.01)
        return code


def start_child(command, **kwargs):
    read_fd, write_fd = os.pipe()
    try:
        process = subprocess.Popen(
            [sys.executable, "-c", OWNED_LAUNCH, str(write_fd), *command],
            **kwargs,
            start_new_session=True,
            pass_fds=(write_fd,),
        )
    except BaseException:
        os.close(read_fd)
        raise
    finally:
        os.close(write_fd)
    child = OwnedChild(process, read_fd)
    # Read only this pid: a full /proc sweep could outlast a short launcher.
    token = Path(f"/proc/{child.pid}/stat").read_text().rsplit(")", 1)[1].split()[19]
    track_child(child, token)
    return child


def track_child(child, token):
    child.owned_workers = {child.pid: token}
    child.tracking_stop = threading.Event()
    child.tracking_lock = threading.Lock()

    def track():
        while not child.tracking_stop.is_set():
            entries = process_table()
            with child.tracking_lock:
                child.owned_workers.update(descendant_workers(child.owned_workers, entries))
            child.tracking_stop.wait(0.05)

    child.tracking_thread = threading.Thread(target=track, daemon=True)
    child.tracking_thread.start()


class ExternalChild:
    """Observe an externally created subreaper; never create a replacement."""

    def __init__(self, pid, token, receipt):
        self.pid, self.token, self.receipt = pid, token, Path(receipt)
        track_child(self, token)

    def poll(self):
        if self.receipt.is_file():
            value = json.loads(self.receipt.read_text())
            if (
                value.get("pid") != self.pid
                or value.get("start_token") != self.token
                or type(value.get("return_code")) is not int
            ):
                raise RuntimeError("External exit receipt has another process identity")
            return value["return_code"]
        entry = process_table().get(self.pid)
        if entry is None or entry[2] != self.token:
            return 75  # Missing exit evidence must never become success.
        try:
            fields = Path(f"/proc/{self.pid}/stat").read_text().rsplit(")", 1)[1].split()
        except FileNotFoundError:
            return 75
        return 75 if fields[0] == "Z" else None

    returncode = property(lambda self: self.poll())

    def wait(self, timeout=None):
        deadline = None if timeout is None else time.monotonic() + timeout
        while (code := self.poll()) is None:
            if deadline is not None and time.monotonic() >= deadline:
                raise subprocess.TimeoutExpired(str(self.pid), timeout)
            time.sleep(0.01)
        return code


def owned_worker_pids(child):
    if not hasattr(child, "owned_workers"):
        return []
    entries = process_table()
    with child.tracking_lock:
        child.owned_workers.update(descendant_workers(child.owned_workers, entries))
        return [pid for pid, token in child.owned_workers.items() if pid in entries and entries[pid][2] == token]


def stop_children(children, *, term_grace_seconds=5, kill_grace_seconds=2):
    def signal_owned(child, signum):
        for pid in owned_worker_pids(child):
            with contextlib.suppress(ProcessLookupError):
                os.kill(pid, signum)
        if not hasattr(child, "owned_workers"):
            with contextlib.suppress(ProcessLookupError):
                os.killpg(child.pid, signum)

    def alive(child):
        if not hasattr(child, "owned_workers") and running_group_members(child.pid):
            return True
        for pid in owned_worker_pids(child):
            try:
                if Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[0] != "Z":
                    return True
            except FileNotFoundError:
                continue
        return False

    for child in children:
        signal_owned(child, signal.SIGTERM)
    deadline = time.monotonic() + term_grace_seconds
    while time.monotonic() < deadline:
        live_children = []
        for child in children:
            child.poll()
            if alive(child):
                live_children.append(child)
        if not live_children:
            break
        time.sleep(0.1)
    for child in children:
        # Freeze the still-observable subtree before KILL so it cannot fork a
        # new untracked writer between the final scan and parent termination.
        if hasattr(child, "owned_workers"):
            frozen = set()
            for _ in range(32):
                current = set(owned_worker_pids(child))
                if current <= frozen:
                    break
                for pid in current - frozen:
                    with contextlib.suppress(ProcessLookupError):
                        os.kill(pid, signal.SIGSTOP)
                frozen.update(current)
            else:
                raise RuntimeError("Owned descendants did not stabilize; refusing to restart")
        signal_owned(child, signal.SIGKILL)
        child.wait(timeout=kill_grace_seconds)
    deadline = time.monotonic() + kill_grace_seconds
    while any(alive(child) for child in children):
        if time.monotonic() > deadline:
            raise RuntimeError("Owned worker group is still alive; refusing to restart")
        time.sleep(0.1)
    for child in children:
        if hasattr(child, "tracking_stop"):
            child.tracking_stop.set()
            child.tracking_thread.join(timeout=1)
        if isinstance(child, OwnedChild):
            child.process.wait(timeout=kill_grace_seconds)
            child.poll()  # Close the result pipe even if the anchor was killed.


GUARDIAN = r"""
import contextlib, json, os, select, signal, sys, time
parent = int(sys.argv[1])
deadline = float(sys.argv[2])
parent_token = sys.argv[3]
buffer = b""
tracked = {}
def descendants_now():
    # Include children created by a Popen that has not returned from exec/chdir.
    parents = {}
    for pid in os.listdir('/proc'):
        if not pid.isdecimal():
            continue
        try:
            with open('/proc/' + pid + '/stat') as status:
                fields = status.read().rsplit(')', 1)[1].split()
            parents[int(pid)] = (int(fields[1]), fields[19])
        except OSError:
            continue
    descendants = {parent} if parent in parents and parents[parent][1] == parent_token else set()
    descendants.update(pid for pid, token in tracked.items() if pid in parents and parents[pid][1] == token)
    while True:
        expanded = descendants | {pid for pid, (ppid, _) in parents.items() if ppid in descendants}
        if expanded == descendants:
            break
        descendants = expanded
    tracked.update({pid: parents[pid][1] for pid in descendants - {parent, os.getpid()} if pid in parents})
    return descendants
def terminate():
    descendants = descendants_now()
    for pid in descendants - {parent, os.getpid()}:
        with contextlib.suppress(ProcessLookupError):
            os.kill(pid, signal.SIGKILL)
    if parent in descendants:
        with contextlib.suppress(ProcessLookupError):
            os.kill(parent, signal.SIGKILL)
while True:
    descendants_now()
    readable, _, _ = select.select([sys.stdin], [], [], max(0, min(.1, deadline - time.monotonic())))
    if readable:
        chunk = os.read(0, 65536)
        if not chunk:
            terminate()
            break
        buffer += chunk
        while b"\n" in buffer:
            line, buffer = buffer.split(b"\n", 1)
            value = json.loads(line)
            if value.get("stop"):
                sys.exit(0)
            deadline = value["deadline"]
            tracked.update({int(pid): token for pid, token in value["workers"].items()})
    if time.monotonic() >= deadline:
        terminate()
        break
"""


@contextlib.contextmanager
def deadline_guard(seconds):
    guardian = subprocess.Popen(
        [
            sys.executable,
            "-c",
            GUARDIAN,
            str(os.getpid()),
            str(time.monotonic() + seconds),
            process_table()[os.getpid()][2],
        ],
        stdin=subprocess.PIPE,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        text=True,
        start_new_session=True,
    )

    def arm(deadline, children):
        workers = {}
        for child in children:
            with child.tracking_lock:
                workers.update(child.owned_workers)
        guardian.stdin.write(json.dumps({"deadline": deadline, "workers": workers}) + "\n")
        guardian.stdin.flush()

    try:
        yield arm
    finally:
        with contextlib.suppress(BrokenPipeError):
            guardian.stdin.write('{"stop": true}\n')
            guardian.stdin.flush()
        guardian.stdin.close()
        guardian.wait(timeout=2)


def supervise(spec, state_dir):
    startup = float(spec.get("startup_timeout_seconds", 60))
    if startup <= 0:
        raise ValueError("Startup timeout must be positive")
    with deadline_guard(startup) as guard:
        return supervised_attempts(spec, state_dir, guard)


def supervised_attempts(spec, state_dir, guard):
    mode = spec.get("mode", "last_words")
    if mode not in {"elastic", "last_words", "elastic_last_words"}:
        raise ValueError("mode must be elastic, last_words or elastic_last_words")
    last_words_enabled = mode != "elastic"
    elastic_enabled = mode != "last_words"
    backend = spec.get("coordination_backend", "service")
    if backend not in {"file", "service"}:
        raise ValueError("Unknown coordination backend")
    commands = spec["commands"]
    initial = spec.get("initial_replicas", list(range(len(commands))))
    if (
        not isinstance(initial, list)
        or not initial
        or len(set(initial)) != len(initial)
        or any(type(replica) is not int or not 0 <= replica < len(commands) for replica in initial)
        or (not elastic_enabled and initial != list(range(len(commands))))
    ):
        raise ValueError("Invalid initial logical replica slots")
    if mode == "elastic_last_words" and backend != "service":
        raise ValueError("Elastic last words requires the service backend")
    envs = spec.get("envs", [{} for _ in commands])
    if len(envs) != len(commands) or any(not isinstance(env, dict) for env in envs):
        raise ValueError("Provide one environment mapping per launch command")
    if not commands or any(not isinstance(command, list) or not command for command in commands):
        raise ValueError("Job commands must be nonempty argument lists")
    if last_words_enabled and any("{run_id}" not in " ".join(command) for command in commands):
        raise ValueError("Every replica launch must consume the shared {run_id}")
    if last_words_enabled and any("{resume_path}" not in " ".join(command) for command in commands):
        raise ValueError("Every replica launch must consume {resume_path}; an empty value enables normal selection")
    attempts = int(spec.get("max_attempts", 1 if elastic_enabled else 3))
    reserve = int(spec.get("required_free_bytes", 0))
    if not 1 <= attempts <= 3 or (last_words_enabled and reserve <= 0):
        raise ValueError("Use one to three launch attempts and a positive emergency-space reserve")
    state_dir.mkdir(parents=True, exist_ok=True)
    root = Path(spec["checkpoint_root"]).resolve()
    root.mkdir(parents=True, exist_ok=True)
    with (root / ".last_words_job.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        for attempt in range(1, attempts + 1):
            run_id = f"launch-{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}-{uuid.uuid4().hex}"
            # Recovery selection stays inside the trainer, including signature checks.
            resume_path = spec.get("resume_path", "") if attempt == 1 else ""
            children, logs = [None] * len(commands), []
            coordinator = None
            election = None
            began = time.monotonic()
            startup_deadline = began + float(spec.get("startup_timeout_seconds", 60))
            deadline = float(spec.get("max_attempt_seconds", 43200))
            last_progress, last_update, forced_reason = began, -1, ""

            def validate_candidate(target, run_id=run_id):
                manifest = json.loads((target / "manifest.json").read_text())
                if manifest.get("run_id") != run_id:
                    raise ValueError("Published candidate belongs to another launch")
                if backend == "file":
                    grant = json.loads((root / "last_words/control" / run_id / "grant.json").read_text())
                    if any(manifest[key] != grant[key] for key in ("replica_id", "state", "signature")):
                        raise ValueError("Published candidate does not match writer authorization")
                recovery_preflight(target, manifest, len(commands))

            if last_words_enabled and backend == "service":
                # Preflight the extension before creating any training process.
                from torchtitan.last_words.client import ServiceElection

                module = {"ServiceElection": ServiceElection}
                address_file = state_dir / f"coordinator-{run_id}-address.json"
                coordinator_log = (state_dir / f"{run_id}-coordinator.log").open("w")
                logs.append(coordinator_log)
                coordinator = start_child(
                    [
                        sys.executable,
                        "-m",
                        "torchft_last_words",
                        "--address-file",
                        str(address_file),
                    ],
                    stdout=coordinator_log,
                    stderr=subprocess.STDOUT,
                    env=os.environ.copy(),
                )
                guard(startup_deadline, [coordinator])
                try:
                    import urllib.request

                    while True:
                        if coordinator.poll() is not None:
                            raise RuntimeError("Coordinator failed before training launch")
                        try:
                            if not address_file.is_file():
                                raise OSError("Coordinator has not advertised its endpoint")
                            address = json.loads(address_file.read_text())["address"]
                            with urllib.request.urlopen(address + "/status", timeout=1):
                                break
                        except OSError as error:
                            if time.monotonic() >= startup_deadline:
                                raise RuntimeError("Coordinator startup timed out") from error
                            time.sleep(0.1)
                    election = module["ServiceElection"](
                        root,
                        run_id,
                        address=address,
                        num_replicas=len(commands),
                        world_size=int(spec.get("replica_world_size", 2)),
                        heartbeat_timeout=float(spec.get("heartbeat_timeout_seconds", 60)),
                        startup_timeout=float(spec.get("startup_timeout_seconds", 60)),
                        timeout=float(spec.get("fault_timeout_seconds", 300)),
                        validate=validate_candidate,
                        elastic=(
                            dict(
                                active_replicas=initial,
                                recovery_seconds=float(spec.get("elastic_recovery_seconds", 180)),
                            )
                            if elastic_enabled
                            else None
                        ),
                    )
                except BaseException:
                    stop_children([coordinator])
                    for log in logs:
                        log.close()
                    raise
            elif last_words_enabled:
                election = PROTOCOL["SaveElection"](
                    root,
                    run_id,
                    num_replicas=len(commands),
                    world_size=int(spec.get("replica_world_size", 2)),
                    heartbeat_timeout=float(spec.get("heartbeat_timeout_seconds", 60)),
                    startup_timeout=float(spec.get("startup_timeout_seconds", 60)),
                    timeout=float(spec.get("fault_timeout_seconds", 300)),
                    validate=validate_candidate,
                )

            def workers(children=children):
                return [child for child in children if child is not None]

            def owned(coordinator=coordinator, workers=workers):
                return workers() + ([coordinator] if coordinator is not None else [])

            admissions = None

            # Extra time lets normal TERM/KILL cleanup finish before the hard guard.
            hard_slack = 8
            guard(
                min(startup_deadline, began + deadline),
                owned(),
            )
            try:
                if last_words_enabled and elastic_enabled:
                    from torchtitan.last_words import external as external_module

                    external = vars(external_module)
                    admissions = external["ExternalAdmissions"](
                        state_dir / f"external-{run_id}",
                        election,
                        children,
                        fence=stop_children,
                        attach=ExternalChild,
                        anchor_code=OWNED_LAUNCH,
                        register=lambda limit=began + deadline, owned=owned: guard(
                            min(time.monotonic() + 60, limit), owned()
                        ),
                        startup_seconds=float(spec.get("startup_timeout_seconds", 60)),
                    )
                if last_words_enabled and shutil.disk_usage(root).free < reserve:
                    raise OSError("Insufficient free space for configured emergency candidate reserve")
                for replica, template in enumerate(commands):
                    if replica not in initial:
                        continue
                    command = [
                        argument.replace("{run_id}", run_id).replace("{resume_path}", resume_path)
                        for argument in template
                    ]
                    log = (state_dir / f"{run_id}-replica-{replica}.log").open("w")
                    logs.append(log)
                    environment = dict(os.environ, **envs[replica])
                    if last_words_enabled:
                        environment.update(LAST_WORDS_SUPERVISED_RUN=run_id)
                        if backend == "service":
                            environment.update(election.environment(replica))
                    else:
                        for key in (
                            "LAST_WORDS_SUPERVISED_RUN",
                            "LAST_WORDS_COORDINATOR_ADDRESS",
                            "LAST_WORDS_COORDINATOR_INSTANCE",
                            "LAST_WORDS_REPLICA_TOKEN",
                        ):
                            environment.pop(key, None)
                    children[replica] = start_child(
                        command,
                        stdout=log,
                        stderr=subprocess.STDOUT,
                        cwd=spec.get("cwd"),
                        env=environment,
                    )
                    guard(
                        min(
                            startup_deadline,
                            began + deadline,
                        ),
                        owned(),
                    )
                failure_at = None
                terminal_saved_at = None
                while any(child.poll() is None for child in workers()):
                    try:
                        if coordinator is not None and coordinator.poll() is not None:
                            raise RuntimeError("Coordinator exited; save session cannot be resumed")
                        if election is not None:
                            election.tick(children, terminate=stop_children)
                        if admissions is not None:
                            admissions.tick()
                    except Exception as error:
                        forced_reason = f"coordination_failed: {error}"
                        stop_children(workers())
                        break
                    # Validation can import torch and inspect checkpoint storage.
                    # Its elapsed time must not consume the terminal exit grace.
                    now = time.monotonic()
                    if (
                        election is not None
                        and election.finished
                        and (
                            election.success
                            if backend == "service"
                            else json.loads((election.control / "result.json").read_text())["success"]
                        )
                    ):
                        if terminal_saved_at is None:
                            terminal_saved_at = now
                        # Publication has passed the full candidate validator.
                        # A native RPC can outlive its main training thread;
                        # validated terminal saves require a whole-job restart.
                        # Nonterminal elastic baselines never set finished.
                        if now - terminal_saved_at >= 5:
                            stop_children(workers())
                            break
                    if (
                        election is not None
                        and election.finished
                        and not (
                            election.success
                            if backend == "service"
                            else json.loads((election.control / "result.json").read_text())["success"]
                        )
                    ):
                        forced_reason = "emergency_save_failed"
                        stop_children(workers())
                        break
                    if (
                        not elastic_enabled
                        and any(child.poll() not in (None, 0) for child in workers())
                        and failure_at is None
                    ):
                        failure_at = now
                    grace = float(spec.get("failure_grace_seconds", 420))
                    progress_limit = last_progress + float(spec.get("no_progress_seconds", 600))
                    if last_update < 0:
                        progress_limit = max(progress_limit, startup_deadline)
                    if not last_words_enabled:
                        progress_limit = began + deadline
                    elif elastic_enabled:
                        progress_limit = max(
                            progress_limit,
                            election.snapshot.get("operation_deadline") or 0,
                        )
                    hard_limit = min(began + deadline, progress_limit)
                    if failure_at is not None:
                        hard_limit = min(hard_limit, failure_at + grace)
                    guard(hard_limit + hard_slack, owned())
                    updates = []
                    if election is None:
                        statuses = []
                    elif backend == "service":
                        statuses = election.statuses()
                    else:
                        statuses = [
                            json.loads(status.read_text())
                            for status in (root / "last_words/control" / run_id).glob("replica-*-rank-*.json")
                        ]
                    for status in statuses:
                        value = status.get("state") or {}
                        updates.append(value.get("last_words_num_updates", -1))
                    current_update = max(updates, default=-1)
                    if current_update > last_update:
                        last_update, last_progress = current_update, now
                        progress_limit = max(
                            progress_limit,
                            now + float(spec.get("no_progress_seconds", 600)),
                        )
                    stalled = last_words_enabled and now > progress_limit
                    if now - began > deadline or stalled or (failure_at is not None and now - failure_at > grace):
                        forced_reason = "no_training_progress" if stalled else "external_deadline"
                        stop_children(workers())
                        break
                    time.sleep(0.5)
                # All workers can exit between two polls after Coordinator loss.
                # Finalize diagnostics/late publication before intentional shutdown.
                coordinator_code = coordinator.poll() if coordinator is not None else None
                if last_words_enabled and backend == "service" and not forced_reason:
                    try:
                        if coordinator_code is not None:
                            raise RuntimeError(f"Coordinator exited with code {coordinator_code}")
                        election.tick(children, terminate=stop_children)
                        if election.finished and not election.success:
                            forced_reason = "emergency_save_failed"
                    except Exception as error:
                        forced_reason = f"coordination_failed: {error}"
                codes = [child.returncode if child is not None else None for child in children]
                completed = all(code == 0 for code in codes)
                if elastic_enabled:
                    members = election.snapshot["members"] if election is not None else initial
                    completed = (
                        bool(members) and all(codes[replica] == 0 for replica in members)
                        if election is not None
                        else any(code == 0 for code in codes)
                    )
                    if election is not None and election.snapshot["phase"] != "RUNNING":
                        completed = False
                stop_children(workers())
                if coordinator is not None:
                    stop_children([coordinator])
                    coordinator = None
                children = []
                guard(time.monotonic() + 60, children)
                result = {
                    "run_id": run_id,
                    "attempt": attempt,
                    "return_codes": codes,
                    "seconds": time.monotonic() - began,
                    "forced_reason": forced_reason,
                    "coordinator_return_code": coordinator_code,
                    "mode": mode,
                    "success": not forced_reason and completed,
                }
                (state_dir / f"{run_id}.json").write_text(json.dumps(result, indent=2))
                print(json.dumps(result), flush=True)
                if result["success"]:
                    return 0
            finally:
                if admissions is not None:
                    admissions.close()
                stop_children(owned())
                for log in logs:
                    log.close()
        print(
            json.dumps({"alert": "restart_limit_reached", "attempts": attempts}),
            flush=True,
        )
        return 1


def main():
    def interrupted(signum, frame):
        raise SystemExit(128 + signum)

    signal.signal(signal.SIGTERM, interrupted)
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="action", required=True)
    cleanup = sub.add_parser("cleanup")
    cleanup.add_argument("--checkpoint-root", type=Path, required=True)
    cleanup.add_argument("--keep", type=int, default=2)
    cleanup.add_argument("--num-replicas", type=int, required=True)
    cleanup.add_argument("--apply", action="store_true")
    run = sub.add_parser("run")
    run.add_argument("--spec", type=Path, required=True)
    run.add_argument("--state-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.action == "run":
        raise SystemExit(supervise(json.loads(args.spec.read_text()), args.state_dir))
    if args.keep < 2:
        raise ValueError("Retain at least two recoverable generations")
    root = args.checkpoint_root.resolve()
    with (root / ".last_words_job.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        plan = retention_plan(root, keep=args.keep, num_replicas=args.num_replicas)
        print(json.dumps(plan, indent=2))
        if args.apply:
            for entry in plan["delete"]:
                shutil.rmtree(contained(root, Path(entry)))


if __name__ == "__main__":
    main()
