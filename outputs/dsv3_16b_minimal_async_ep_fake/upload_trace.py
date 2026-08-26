#!/usr/bin/env fbpython
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
import urllib
import urllib.parse
import uuid
PERFETTO_OPEN_TRACE_URL = "https://www.internalfb.com/intern/perfetto/open_trace/"
PERFETTO_UI_ROOT_URL_META_INSIGHTS = "https://www.internalfb.com/intern/metainsights"
MANIFOLD_BUCKET = "perfetto_internal_traces"
MANIFOLD_TRACE_DIR = "tree/shared_trace"
DEFAULT_TTL_SEC = 28 * 24 * 60 * 60


def upload_trace_file(
    local_path: str, overwrite: bool = False, ttl_sec: int = DEFAULT_TTL_SEC
) -> str | None:
    """Upload a trace file to Manifold.

    Args:
        ttl_sec: Manifold object TTL in seconds (defaults to DEFAULT_TTL_SEC).

    Returns:
        The trace path within the bucket (e.g. "tree/shared_trace/..."),
        or None on failure.
    """
    file_name = os.path.basename(local_path)
    trace_path = "/".join(
        [MANIFOLD_TRACE_DIR, f"{getpass.getuser()}_{str(uuid.uuid4())}_{file_name}"]
    )
    manifold_path = MANIFOLD_BUCKET + "/" + trace_path
    cmd = [
        "manifold",
        "put",
        local_path,
        manifold_path,
        "--ttl",
        str(ttl_sec),
        "--userData",
        "false",
    ]
    ret = subprocess.run(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True
    )
    if ret.returncode == 0:
        logging.info("Upload trace successfully.")
        return trace_path
    else:
        logging.error(f"Upload failed, error info: \n{ret.stderr}")
        return None


def get_perfetto_ui_url(trace_path: str, use_meta_insights: bool = False) -> str:
    """Generate a Perfetto UI URL for a trace stored in Manifold.

    Args:
        trace_path: The trace path within the bucket (e.g. "tree/shared_trace/...")
        use_meta_insights: Whether to use Meta Insights URL instead of standard Perfetto

    Returns:
        URL to view the trace in Perfetto UI
    """
    manifold_path = MANIFOLD_BUCKET + "/" + trace_path
    if use_meta_insights:
        return (
            PERFETTO_UI_ROOT_URL_META_INSIGHTS
            + "#!/?url=https://interncache-all.fbcdn.net/manifold/"
            + urllib.parse.quote_plus(manifold_path)
        )
    return (
        PERFETTO_OPEN_TRACE_URL
        + "?"
        + urllib.parse.urlencode({"manifold_path": manifold_path})
    )


def print_perfetto_ui_urls(trace_path: str, use_meta_insights: bool = False) -> None:
    url = get_perfetto_ui_url(trace_path, use_meta_insights)
    print(f"Perfetto UI:\n{url}")


def print_memory_snapshot_url(trace_path: str) -> None:
    url = (
        "https://www.internalfb.com/pytorch_memory_visualizer/"
        + MANIFOLD_BUCKET
        + "/"
        + trace_path
    )
    print(f"The memory snapshot is accessible at:\n{url}")


def main(argv) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "local_path", help="The local path for the trace file to upload."
    )
    parser.add_argument(
        "-mi",
        "--meta-insights",
        action="store_true",
        help="Share using Meta Insights plugin-enhanced URL in place of standard hosted Perfetto",
    )
    parser.add_argument(
        "--is-memory-snapshot",
        action="store_true",
        default=False,
        help="Treat the file as a memory snapshot HTML and print a direct manifold URL instead of a Perfetto UI URL.",
    )
    parser.add_argument(
        "-t",
        "--ttl",
        type=int,
        default=DEFAULT_TTL_SEC,
        help="Manifold object TTL in seconds (default: 28 days).",
    )
    args = parser.parse_args(argv)
    logging.basicConfig(
        format="%(levelname)s: %(message)s",
        level=logging.INFO,
        handlers=[
            logging.StreamHandler(sys.stdout),
        ],
    )

    if not os.path.exists(args.local_path):
        logging.error(f"The trace file doesn't exist: {args.local_path}")
        return 1

    paths = []
    if os.path.isdir(args.local_path):
        for filename in os.listdir(args.local_path):
            # TODO: add support for other file extensions
            if "trace.json" in filename:
                x = os.path.join(args.local_path, filename)
                logging.info(f"will upload {x}")
                paths.append(x)
    else:
        logging.info(f"will upload {args.local_path}")
        paths.append(args.local_path)

    for path in paths:
        trace_path = upload_trace_file(path, ttl_sec=args.ttl)
        if trace_path:
            logging.info(f"Uploading {path} successfully:")
            print(f"Manifold path:\n{MANIFOLD_BUCKET}/{trace_path}")
            if args.is_memory_snapshot:
                print_memory_snapshot_url(trace_path)
            else:
                print_perfetto_ui_urls(trace_path, args.meta_insights)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
