# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Convert AI2's TMax dataset into Harbor task directories, offline.

AI2 publishes TMax as ``data/*.parquet`` (one row per task: ``ground_truth`` is
the task id and ``env_config`` holds the public Docker Hub ``image``, e.g.
``hamishi740/swerl-tmax-v3:<12 hex>``) and a
``task-data.tar.gz`` with one directory per task: ``instruction.md``,
``setup.sh`` (already baked into the image) and ``tests/test.sh``, which embeds
its pytest file and writes ``/logs/verifier/reward.txt``.

This writes the layout Verifiers' Harbor taskset reads, so the recipe can train
on TMax without a hosted copy of the data. Per task it keeps AI2's image, AI2's
instruction and AI2's own tests, and declares the image in ``task.toml``::

    <out>/<task_id>/task.toml                  [environment] docker_image = <image>, workdir
    <out>/<task_id>/instruction.md             copied unchanged
    <out>/<task_id>/tests/**                   copied unchanged (test.sh included)

No Dockerfile or ``environment/`` directory is written: Verifiers pulls the image
as declared, uploads ``tests/`` to ``/tests`` and runs ``bash /tests/test.sh``.

``workdir`` is ``/home/user``: AI2's instructions work there (96 percent name it) and
every ``setup.sh`` creates it. It has to be set, because the sandbox otherwise starts
its shell with ``cd /testbed``, which does not exist in these images, and fails to
provision.

``--out`` is the dataset directory Verifiers looks for, ``~/.cache/harbor/<name>``
where ``<name>`` is the dataset id with ``/`` and ``@`` replaced by ``_``
(``org/name@ref`` -> ``org_name_ref``). Set the training dataset id to match.

Example::

    python -m torchtitan.rl.examples.verifiers.terminal_bench.prepare_tmax \\
        --parquet tmax/data --task-data tmax/task-data.tar.gz \\
        --out ~/.cache/harbor/local_tmax_v1 --limit 500
"""

import argparse
import glob
import os
import shutil
import tarfile
import tempfile
from dataclasses import dataclass, field


DEFAULT_WORKDIR = "/home/user"


@dataclass
class ConversionSummary:
    written: list[str] = field(default_factory=list)
    skipped: dict[str, str] = field(default_factory=dict)  # task id -> reason


def read_task_rows(parquet: str) -> list[tuple[str, str]]:
    """``(task_id, image)`` for every row of a parquet file or a folder of them."""
    import pyarrow.parquet as pq

    paths = (
        sorted(glob.glob(os.path.join(parquet, "*.parquet")))
        if os.path.isdir(parquet)
        else [parquet]
    )
    if not paths:
        raise FileNotFoundError(f"no parquet files under {parquet}")
    rows = []
    for path in paths:
        for row in pq.read_table(
            path, columns=["ground_truth", "env_config"]
        ).to_pylist():
            env_config = row["env_config"] or {}
            task_id = row["ground_truth"] or env_config.get("task_id")
            rows.append((task_id, env_config.get("image") or ""))
    return rows


def extract_task_data(task_data: str, scratch: str) -> str:
    """Root of the per-task directories: ``task_data`` itself, or the tarball unpacked."""
    if os.path.isdir(task_data):
        return task_data
    with tarfile.open(task_data, "r:gz") as archive:
        archive.extractall(scratch, filter="data")
    return scratch


def find_task_dir(root: str, task_id: str) -> str | None:
    """``<root>/<task_id>`` or ``<root>/<one level>/<task_id>``."""
    direct = os.path.join(root, task_id)
    if os.path.isdir(direct):
        return direct
    for entry in sorted(os.listdir(root)):
        nested = os.path.join(root, entry, task_id)
        if os.path.isdir(nested):
            return nested
    return None


def task_toml(task_id: str, image: str, cpus: int, memory_mb: int, workdir: str) -> str:
    # Harbor requires the task name to be exactly ``org/name``.
    return (
        'schema_version = "1.1"\n\n'
        "[task]\n"
        f'name = "allenai-tmax/{task_id}"\n\n'
        "[environment]\n"
        f'docker_image = "{image}"\n'
        f'workdir = "{workdir}"\n'
        f"cpus = {cpus}\n"
        f"memory_mb = {memory_mb}\n"
    )


def convert_task(
    task_dir: str,
    task_id: str,
    image: str,
    out_root: str,
    *,
    cpus: int,
    memory_mb: int,
    workdir: str,
) -> str | None:
    """Write one Harbor task directory. Returns the reason it was skipped, or None."""
    if not image:
        return "no image in env_config"
    instruction = os.path.join(task_dir, "instruction.md")
    tests = os.path.join(task_dir, "tests")
    if not os.path.isfile(instruction):
        return "no instruction.md"
    if not os.path.isfile(os.path.join(tests, "test.sh")):
        return "no tests/test.sh"

    out = os.path.join(out_root, task_id)
    os.makedirs(out)
    with open(os.path.join(out, "task.toml"), "w") as f:
        f.write(task_toml(task_id, image, cpus, memory_mb, workdir))
    shutil.copy(instruction, os.path.join(out, "instruction.md"))
    shutil.copytree(tests, os.path.join(out, "tests"))
    return None


def read_id_list(path: str | None) -> set[str]:
    if path is None:
        return set()
    with open(path) as f:
        return {line.strip() for line in f if line.strip()}


def convert_dataset(
    parquet: str,
    task_data: str,
    out: str,
    *,
    limit: int | None = None,
    include: set[str] | None = None,
    exclude: set[str] = frozenset(),
    cpus: int = 1,
    memory_mb: int = 2048,
    workdir: str = DEFAULT_WORKDIR,
) -> ConversionSummary:
    """Convert the selected tasks; ``include`` empty or None means every row."""
    summary = ConversionSummary()
    os.makedirs(out, exist_ok=True)
    with tempfile.TemporaryDirectory() as scratch:
        root = extract_task_data(task_data, scratch)
        for task_id, image in read_task_rows(parquet):
            if include and task_id not in include:
                continue
            if task_id in exclude:
                continue
            if limit is not None and len(summary.written) >= limit:
                break
            if os.path.exists(os.path.join(out, task_id)):
                summary.skipped[task_id] = "already converted"
                continue
            task_dir = find_task_dir(root, task_id)
            reason = (
                "task directory not in task-data"
                if task_dir is None
                else convert_task(
                    task_dir,
                    task_id,
                    image,
                    out,
                    cpus=cpus,
                    memory_mb=memory_mb,
                    workdir=workdir,
                )
            )
            if reason is None:
                summary.written.append(task_id)
            else:
                summary.skipped[task_id] = reason
    return summary


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--parquet", required=True, help="parquet file or folder")
    parser.add_argument(
        "--task-data", required=True, help="task-data.tar.gz or its extracted folder"
    )
    parser.add_argument(
        "--out", required=True, help="Harbor dataset directory to write"
    )
    parser.add_argument("--limit", type=int, help="convert at most this many tasks")
    parser.add_argument("--include-file", help="only these task ids, one per line")
    parser.add_argument("--exclude-file", help="never these task ids, one per line")
    parser.add_argument("--cpus", type=int, default=1)
    parser.add_argument("--memory-mb", type=int, default=2048)
    parser.add_argument(
        "--workdir",
        default=DEFAULT_WORKDIR,
        help="directory that exists in every image",
    )
    args = parser.parse_args(argv)

    summary = convert_dataset(
        args.parquet,
        args.task_data,
        os.path.expanduser(args.out),
        limit=args.limit,
        include=read_id_list(args.include_file),
        exclude=read_id_list(args.exclude_file),
        cpus=args.cpus,
        memory_mb=args.memory_mb,
        workdir=args.workdir,
    )
    print(f"wrote {len(summary.written)} tasks to {args.out}")
    for task_id, reason in list(summary.skipped.items())[:20]:
        print(f"skipped {task_id}: {reason}")
    if len(summary.skipped) > 20:
        print(f"... and {len(summary.skipped) - 20} more skipped")


if __name__ == "__main__":
    main()
