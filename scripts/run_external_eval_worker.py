#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import json
import logging
import os
import subprocess
import sys
import time
from pathlib import Path

logger: logging.Logger = logging.getLogger("external_eval_worker")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run queued TorchTitan external evaluations."
    )
    parser.add_argument("--request-root", required=True, type=Path)
    parser.add_argument("--poll-interval-seconds", type=float, default=30.0)
    return parser.parse_args()


def pending_requests(request_root: Path) -> list[Path]:
    requests = []
    for request_path in request_root.glob("step-*/eval_request.json"):
        if not request_path.with_name("eval_result.json").exists():
            requests.append(request_path)
    return sorted(
        requests,
        key=lambda path: int(path.parent.name.split("-", 2)[1]),
    )


def run_request(request_path: Path) -> None:
    with request_path.open() as f:
        request = json.load(f)

    env = os.environ.copy()
    env.update(request.get("env", {}))
    log_path = request_path.with_name("launch.log")
    started_at = time.time()
    logger.info("Running eval request %s -> %s", request_path.parent.name, log_path)
    with log_path.open("a") as log_file:
        result = subprocess.run(
            request["command"],
            stdout=log_file,
            stderr=subprocess.STDOUT,
            env=env,
            check=False,
        )
    logger.info(
        "Finished eval request %s with return code %s in %.1fs",
        request_path.parent.name,
        result.returncode,
        time.time() - started_at,
    )

    result_path = request_path.with_name("eval_result.json")
    temporary_result_path = result_path.with_suffix(".json.tmp")
    with temporary_result_path.open("w") as f:
        json.dump(
            {
                "return_code": result.returncode,
                "started_at": started_at,
                "finished_at": time.time(),
            },
            f,
            indent=2,
        )
    temporary_result_path.replace(result_path)


def main() -> None:
    logging.basicConfig(
        stream=sys.stdout,
        level=logging.INFO,
        format="[external_eval_worker] %(asctime)s %(levelname)s %(message)s",
    )
    args = parse_args()
    args.request_root.mkdir(parents=True, exist_ok=True)
    logger.info(
        "Polling %s every %.1fs for eval requests.",
        args.request_root,
        args.poll_interval_seconds,
    )

    idle_polls = 0
    while True:
        requests = pending_requests(args.request_root)
        if requests:
            idle_polls = 0
            run_request(requests[0])
            continue
        if (args.request_root / "_TRAINING_COMPLETE").exists():
            logger.info("Training complete and no pending requests left; exiting.")
            return
        # Heartbeat roughly every 10 polls so an idle worker is distinguishable
        # from a worker that is watching the wrong directory.
        if idle_polls % 10 == 0:
            logger.info(
                "No pending eval requests under %s (%d dirs present).",
                args.request_root,
                sum(1 for _ in args.request_root.glob("step-*")),
            )
        idle_polls += 1
        time.sleep(args.poll_interval_seconds)


if __name__ == "__main__":
    main()
