#!/usr/bin/env python3

# Copyright (c) 2026, trainstation team
# The following code is copied from https://github.com/open-lm-engine/lm-engine

# **************************************************
# Copyright (c) 2026, Mayank Mishra
# **************************************************

"""Merge Megatron ``.bin``/``.idx`` datasets.

Port of lm-engine's ``tools/data/merge_data.py`` without its Ray and remote
staging paths. Either merge ``--input-prefixes`` into ``--output-prefix``, or
merge every subdirectory of ``--input-directory`` into
``<output-prefix>/<subdir>/<group>``, split into groups of at most
``--max-size`` GB.

Example:
    python scripts/data/megatron_merge.py \\
        --input-prefixes data/megatron/a_text data/megatron/b_text \\
        --output-prefix data/megatron/merged_text
"""

from __future__ import annotations

import json
import os
from argparse import ArgumentParser, Namespace

from torchtitan.components.data.megatron.bin import get_bin_path
from torchtitan.components.data.megatron.indexed_dataset import (
    get_idx_path,
    MMapIndexedDataset,
    MMapIndexedDatasetBuilder,
)

from tqdm import tqdm


def merge_files(input_prefixes: list[str], output_prefix: str) -> None:
    builder = MMapIndexedDatasetBuilder(
        get_bin_path(output_prefix),
        dtype=MMapIndexedDataset(input_prefixes[0]).index.dtype,
    )
    for input_prefix in input_prefixes:
        builder.add_index(input_prefix)
    builder.finalize(get_idx_path(output_prefix))


def get_groups_by_sizes(
    path: str, max_size: int | None = None
) -> list[tuple[str, list[list[str]]]]:
    """Per subdirectory of ``path`` (relative), the groups of dataset prefixes
    to merge; ``max_size`` (GB) splits a subdirectory into several groups."""
    groups = []
    for root, _, names in os.walk(path):
        prefixes = sorted(
            os.path.join(root, name)[: -len(".bin")]
            for name in names
            if name.endswith(".bin")
        )
        if not prefixes:
            continue

        subdir_groups = []
        if max_size is None:
            subdir_groups.append(prefixes)
        else:
            max_size_bytes = max_size * 1024**3
            current_group, current_size = [], 0
            for index, prefix in enumerate(prefixes):
                current_group.append(prefix)
                current_size += os.path.getsize(f"{prefix}.bin")
                if current_size > max_size_bytes or index == len(prefixes) - 1:
                    subdir_groups.append(current_group)
                    current_group, current_size = [], 0

        groups.append((os.path.relpath(root, path), subdir_groups))
    return groups


def get_args() -> Namespace:
    parser = ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--input-prefixes", type=str, nargs="+", help="Dataset prefixes to merge"
    )
    parser.add_argument(
        "--input-directory", type=str, help="Directory of datasets to merge per subdir"
    )
    parser.add_argument(
        "--output-prefix", type=str, required=True, help="Output path without suffix"
    )
    parser.add_argument("--max-size", type=int, help="Max merged group size in GB")
    parser.add_argument(
        "--specific-group", type=int, help="Merge only this group index per subdir"
    )
    args = parser.parse_args()

    if (args.input_prefixes is None) == (args.input_directory is None):
        parser.error("give exactly one of --input-prefixes and --input-directory")
    if args.input_prefixes is not None and args.max_size is not None:
        parser.error("--max-size only applies to --input-directory")
    return args


def main() -> None:
    args = get_args()

    if args.input_prefixes is not None:
        os.makedirs(os.path.dirname(os.path.abspath(args.output_prefix)), exist_ok=True)
        merge_files(args.input_prefixes, args.output_prefix)
        return

    jobs = []
    for subdir, subdir_groups in get_groups_by_sizes(
        args.input_directory, args.max_size
    ):
        for group_id, group in enumerate(subdir_groups):
            if args.specific_group is not None and group_id != args.specific_group:
                continue
            output_dir = os.path.join(args.output_prefix, subdir)
            os.makedirs(output_dir, exist_ok=True)
            with open(os.path.join(output_dir, f"file_map-{group_id}.json"), "w") as f:
                json.dump(
                    {"subdir": subdir, "grp_id": group_id, "group": group}, f, indent=4
                )
            jobs.append((group, os.path.join(output_dir, str(group_id))))

    for group, output_prefix in tqdm(jobs, desc="Merging files"):
        merge_files(group, output_prefix)


if __name__ == "__main__":
    main()
