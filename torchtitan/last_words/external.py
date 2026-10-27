#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
# This source code is licensed under the BSD-style license found in LICENSE.

"""Explicit admission/attachment only; process creation belongs to the caller."""

import argparse
import json
import os
import secrets
import time
from pathlib import Path


def identity(pid):
    fields = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
    if fields[0] == "Z":
        raise ValueError("Process is a zombie")
    return {"pid": pid, "start_token": fields[19]}


def write_private(path, value):
    if path.is_symlink() or path.parent.is_symlink():
        raise ValueError("External control paths must not be symlinks")
    temporary = path.with_name(path.name + ".tmp-" + secrets.token_hex(8))
    with temporary.open("x") as stream:
        os.chmod(temporary, 0o600)
        json.dump(value, stream)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


class ExternalAdmissions:
    """Serialize operator requests through the existing service owner session."""

    def __init__(
        self,
        root,
        election,
        children,
        *,
        fence,
        attach,
        anchor_code,
        register,
        startup_seconds,
    ):
        self.root, self.election, self.children = Path(root), election, children
        self.fence, self.attach, self.anchor_code, self.register = (
            fence,
            attach,
            anchor_code,
            register,
        )
        self.startup_seconds = startup_seconds
        self.pending = {}
        self.root.mkdir(mode=0o700, exist_ok=False)
        self.owner = identity(os.getpid())
        self.owner.update(run_id=election.run_id, active=True)
        write_private(self.root / "owner.json", self.owner)

    def tick(self):
        for path in sorted(self.root.glob("request-*.json")):
            response = self.root / path.name.replace("request-", "response-", 1)
            if response.exists():
                continue
            try:
                if path.is_symlink() or path.stat().st_size > 16384:
                    raise ValueError("Invalid external request file")
                result = dict(success=True, **self.apply(json.loads(path.read_text())))
            except (ValueError, RuntimeError, OSError, KeyError, TypeError) as error:
                result = dict(success=False, error=str(error))
            write_private(response, result)

    def apply(self, request):
        if request["run_id"] != self.election.run_id:
            raise ValueError("Request belongs to another run")
        replica, generation = request["replica"], request["generation"]
        if (
            type(replica) is not int
            or not 0 <= replica < len(self.children)
            or type(generation) is not int
        ):
            raise ValueError("Invalid external replica identity")
        if request["op"] == "admit":
            snapshot = self.election.snapshot
            if snapshot["phase"] != "RUNNING" or snapshot["result"] is not None:
                raise ValueError("Admission closed during save/recovery")
            if generation != snapshot["generations"][replica] + 1:
                raise ValueError("Stale external generation")
            old = self.children[replica]
            if old is not None and old.poll() is None:
                raise ValueError("Old replica is still active")
            if old is not None:
                self.fence([old])
            token = secrets.token_hex(32)
            self.election.admit_external(
                replica, generation=generation, token=token, fenced=True
            )
            self.children[replica] = None  # Never replay the old generation's exit.
            prefix = self.root / f"replica-{replica}-generation-{generation}"
            environment = self.election.environment(replica)
            environment.update(
                LAST_WORDS_SUPERVISED_RUN=self.election.run_id,
                LAST_WORDS_EXTERNAL_GATE=str(prefix) + "-gate.json",
                LAST_WORDS_EXTERNAL_RECEIPT=str(prefix) + "-exit.json",
                LAST_WORDS_EXTERNAL_OWNER_PID=str(self.owner["pid"]),
                LAST_WORDS_EXTERNAL_OWNER_TOKEN=self.owner["start_token"],
                LAST_WORDS_EXTERNAL_DEADLINE=str(
                    time.monotonic() + self.startup_seconds
                ),
            )
            self.pending[replica] = dict(generation=generation, environment=environment)
            return dict(replica=replica, generation=generation, environment=environment)
        if request["op"] != "attach":
            raise ValueError("Unknown external admission operation")
        if (
            self.election.snapshot["phase"] != "RUNNING"
            or self.election.snapshot["result"] is not None
        ):
            raise ValueError("Attachment closed during save/recovery")
        pending = self.pending.get(replica)
        if pending is None or pending["generation"] != generation:
            raise ValueError("No matching external admission")
        if self.children[replica] is not None:
            raise ValueError("External generation is already attached")
        pid, token = request["pid"], request["start_token"]
        if type(pid) is not int or identity(pid) != {"pid": pid, "start_token": token}:
            raise ValueError("External PID identity changed")
        process = Path(f"/proc/{pid}")
        if process.stat().st_uid != os.getuid() or os.getpgid(pid) != pid:
            raise ValueError(
                "External anchor must belong to this uid and lead its process group"
            )
        arguments = process.joinpath("cmdline").read_bytes().split(b"\0")
        if len(arguments) < 4 or arguments[1:3] != [b"-c", self.anchor_code.encode()]:
            raise ValueError("External process is not the owned subreaper anchor")
        actual = {
            item.split(b"=", 1)[0]: item.split(b"=", 1)[1]
            for item in process.joinpath("environ").read_bytes().split(b"\0")
            if b"=" in item
        }
        if any(
            actual.get(key.encode()) != value.encode()
            for key, value in pending["environment"].items()
        ):
            raise ValueError(
                "External environment does not match its admitted generation"
            )
        environment = pending["environment"]
        gate = Path(environment["LAST_WORDS_EXTERNAL_GATE"])
        if gate.exists() or time.monotonic() >= float(
            environment["LAST_WORDS_EXTERNAL_DEADLINE"]
        ):
            raise ValueError("External admission expired or gate already released")
        child = self.attach(pid, token, environment["LAST_WORDS_EXTERNAL_RECEIPT"])
        self.children[replica] = child
        self.register()  # Register guardian ownership before allowing training.
        write_private(gate, dict(pid=pid, start_token=token))
        del self.pending[replica]
        return dict(replica=replica, generation=generation, pid=pid)

    def close(self):
        write_private(self.root / "owner.json", dict(self.owner, active=False))


def request_external(root, request, *, timeout=30):
    root = Path(root).resolve(strict=True)
    owner = json.loads((root / "owner.json").read_text())
    if (
        not owner["active"]
        or identity(owner["pid"])["start_token"] != owner["start_token"]
    ):
        raise ValueError("External admission owner is no longer active")
    identifier = secrets.token_hex(16)
    write_private(
        root / f"request-{identifier}.json", dict(request, run_id=owner["run_id"])
    )
    response = root / f"response-{identifier}.json"
    deadline = time.monotonic() + timeout
    while not response.exists():
        if time.monotonic() >= deadline:
            raise TimeoutError(
                "Admission response not observed; inspect the same request before retrying: "
                + identifier
            )
        time.sleep(0.05)
    result = json.loads(response.read_text())
    if not result["success"]:
        raise ValueError(result["error"])
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("admit", "attach"))
    parser.add_argument("--control-dir", type=Path, required=True)
    parser.add_argument("--replica", type=int, required=True)
    parser.add_argument("--generation", type=int, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Private response file; may contain credentials",
    )
    parser.add_argument("--pid", type=int)
    parser.add_argument("--start-token")
    args = parser.parse_args()
    request = dict(op=args.action, replica=args.replica, generation=args.generation)
    if args.action == "attach":
        if args.pid is None or args.start_token is None:
            parser.error("attach requires --pid and --start-token")
        request.update(pid=args.pid, start_token=args.start_token)
    write_private(args.output, request_external(args.control_dir, request))


if __name__ == "__main__":
    main()
