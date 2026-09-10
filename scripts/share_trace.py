#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import getpass
import logging
import os
import subprocess
import sys
import urllib.parse
import uuid


PERFETTO_OPEN_TRACE_URL = "https://www.internalfb.com/intern/perfetto/open_trace/"
PERFETTO_UI_ROOT_URL_META_INSIGHTS = "https://www.internalfb.com/intern/metainsights"
MEMORY_SNAPSHOT_ROOT_URL = "https://www.internalfb.com/pytorch_memory_visualizer"
MANIFOLD_BUCKET = "perfetto_internal_traces"
MANIFOLD_TRACE_DIR = "tree/shared_trace"
DEFAULT_TTL_SEC = 28 * 24 * 60 * 60


def upload_trace_file(local_path: str, ttl_sec: int) -> str | None:
    file_name = os.path.basename(local_path)
    trace_path = "/".join(
        [MANIFOLD_TRACE_DIR, f"{getpass.getuser()}_{uuid.uuid4()}_{file_name}"]
    )
    manifold_path = f"{MANIFOLD_BUCKET}/{trace_path}"
    result = subprocess.run(
        [
            "manifold",
            "put",
            local_path,
            manifold_path,
            "--ttl",
            str(ttl_sec),
            "--userData",
            "false",
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        logging.error("Upload failed:\n%s", result.stderr)
        return None

    logging.info("Upload trace successfully.")
    return trace_path


def get_perfetto_ui_url(trace_path: str, use_meta_insights: bool) -> str:
    manifold_path = f"{MANIFOLD_BUCKET}/{trace_path}"
    if use_meta_insights:
        return (
            f"{PERFETTO_UI_ROOT_URL_META_INSIGHTS}#!/?url="
            "https://interncache-all.fbcdn.net/manifold/"
            f"{urllib.parse.quote_plus(manifold_path)}"
        )
    query = urllib.parse.urlencode({"manifold_path": manifold_path})
    return f"{PERFETTO_OPEN_TRACE_URL}?{query}"


def get_memory_snapshot_url(trace_path: str) -> str:
    return f"{MEMORY_SNAPSHOT_ROOT_URL}/{MANIFOLD_BUCKET}/{trace_path}"


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("local_path", help="Trace file or directory to upload.")
    parser.add_argument(
        "-mi",
        "--meta-insights",
        action="store_true",
        help="Open execution traces with Meta Insights.",
    )
    parser.add_argument(
        "--is-memory-snapshot",
        action="store_true",
        help="Open the uploaded file with the PyTorch memory visualizer.",
    )
    parser.add_argument(
        "-t",
        "--ttl",
        type=int,
        default=DEFAULT_TTL_SEC,
        help="Manifold object TTL in seconds.",
    )
    return parser.parse_args(argv[1:])


def get_upload_paths(local_path: str, is_memory_snapshot: bool) -> list[str]:
    if not os.path.isdir(local_path):
        return [local_path]

    suffix = ".pickle" if is_memory_snapshot else "trace.json"
    return [
        os.path.join(local_path, filename)
        for filename in sorted(os.listdir(local_path))
        if suffix in filename
    ]


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    logging.basicConfig(format="%(levelname)s: %(message)s", level=logging.INFO)

    if not os.path.exists(args.local_path):
        logging.error("The trace path does not exist: %s", args.local_path)
        return 1

    paths = get_upload_paths(args.local_path, args.is_memory_snapshot)
    if not paths:
        logging.error("No uploadable files found in %s", args.local_path)
        return 1

    upload_failed = False
    for path in paths:
        logging.info("Uploading %s", path)
        trace_path = upload_trace_file(path, args.ttl)
        if trace_path is None:
            upload_failed = True
            continue

        print(f"Manifold path:\n{MANIFOLD_BUCKET}/{trace_path}")
        if args.is_memory_snapshot:
            print(f"Memory snapshot:\n{get_memory_snapshot_url(trace_path)}")
        else:
            print(
                "Perfetto UI:\n"
                f"{get_perfetto_ui_url(trace_path, args.meta_insights)}"
            )

    return int(upload_failed)


if __name__ == "__main__":
    sys.exit(main(sys.argv))
