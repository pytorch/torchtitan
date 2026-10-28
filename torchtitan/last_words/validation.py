# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
# This source code is licensed under the BSD-style license found in LICENSE.

"""Read-only filesystem checks shared by recovery and checkpoint retention."""

import json
from pathlib import Path


def contained(root, path):
    root, path = Path(root).resolve(), Path(path)
    resolved = path.resolve()
    if resolved == root or not resolved.is_relative_to(root) or path.is_symlink():
        raise ValueError(f"Refusing checkpoint path outside its root: {path}")
    return resolved


def validate_storage_ranges(path):
    import torch.distributed.checkpoint as dcp

    path = Path(path).resolve()
    metadata = dcp.FileSystemReader(path).read_metadata()
    if metadata.storage_data is None:
        raise ValueError("Checkpoint has no filesystem storage map")
    for entry in metadata.storage_data.values():
        shard = contained(path, path / entry.relative_path)
        if (
            entry.offset < 0
            or entry.length < 0
            or shard.stat().st_size < entry.offset + entry.length
        ):
            raise ValueError(f"Checkpoint shard is truncated: {shard}")


def checkpoint_state(path):
    import torch.distributed.checkpoint as dcp

    validate_storage_ranges(path)
    states = {"train_state": {"step": 0, "last_words_num_updates": 0}}
    dcp.load(states, checkpoint_id=path, no_dist=True)
    return states["train_state"]


def checkpoint_torchft_state(path):
    import torch.distributed.checkpoint as dcp

    validate_storage_ranges(path)
    states = {"torchft": {"step": 0, "batches_committed": 0}}
    dcp.load(states, checkpoint_id=path, no_dist=True)
    return states["torchft"]


def validate_rng_cursor(path, marker, num_shards):
    import torch.distributed.checkpoint as dcp

    if (
        type(num_shards) is not int
        or num_shards < 1
        or marker.get("rng_schema") != 1
        or marker.get("rng_shards") != num_shards
    ):
        raise ValueError("Cursor RNG schema or shard count differs")
    keys = dcp.FileSystemReader(path).read_metadata().state_dict_metadata
    # Reject incomplete layouts before DCP planning wraps the missing-key error.
    for rank in range(num_shards):
        for field in ("schema", "device_type", "python", "cpu", "numpy"):
            prefix = f"last_words_rng.rank_{rank}.{field}"
            if not any(key == prefix or key.startswith(prefix + ".") for key in keys):
                raise ValueError(f"Cursor has no complete RNG stream for shard {rank}")
    headers = {
        "last_words_rng": {
            f"rank_{rank}": {"schema": 0, "device_type": ""}
            for rank in range(num_shards)
        }
    }
    dcp.load(headers, checkpoint_id=path, no_dist=True)
    for rank in range(num_shards):
        header = headers["last_words_rng"][f"rank_{rank}"]
        if (
            type(header["schema"]) is not int
            or header["schema"] != 1
            or not isinstance(header["device_type"], str)
            or not header["device_type"]
        ):
            raise ValueError(
                f"Cursor RNG schema or device type differs for shard {rank}"
            )
        if header["device_type"] != "cpu":
            name = f"last_words_rng.rank_{rank}.accelerator"
            metadata = keys.get(name)
            import torch
            from torch.distributed.checkpoint.metadata import TensorStorageMetadata

            if (
                not isinstance(metadata, TensorStorageMetadata)
                or metadata.properties.dtype != torch.uint8
                or len(metadata.size) != 1
                or metadata.size[0] <= 0
            ):
                raise ValueError(
                    f"Cursor has no valid accelerator RNG stream for shard {rank}"
                )


def recovery_preflight(path, manifest, num_replicas):
    path = Path(path).resolve()
    expected = {
        key: manifest["state"][key] for key in ("step", "last_words_num_updates")
    }
    if checkpoint_state(path / "model") != expected:
        raise ValueError("Model progress differs from manifest")
    if "torchft" in manifest:
        progress = manifest["torchft"]
        if (
            not isinstance(progress, dict)
            or set(progress) != {"step", "batches_committed"}
            or any(type(value) is not int or value < 0 for value in progress.values())
            or type(manifest.get("event_id")) is not int
            or manifest["event_id"] <= 0
            or type(manifest.get("membership_epoch")) is not int
            or manifest["membership_epoch"] < 0
        ):
            raise ValueError("Invalid elastic checkpoint progress/identity")
        if checkpoint_torchft_state(path / "model") != progress:
            raise ValueError("TorchFT progress differs from manifest")
    if set(manifest["cursors"]) != {str(replica) for replica in range(num_replicas)}:
        raise ValueError("Candidate does not contain every replica cursor")
    for entry in manifest["cursors"].values():
        source = contained(path, path / entry["path"])
        validate_storage_ranges(source)
        marker = json.loads((source / "last_words_cursor.json").read_text())
        if "rng_schema" in manifest:
            if (
                manifest["rng_schema"] != 1
                or type(manifest.get("rng_shards")) is not int
                or manifest["rng_shards"] <= 0
            ):
                raise ValueError("Invalid checkpoint RNG recovery schema")
            validate_rng_cursor(source, marker, manifest["rng_shards"])
        if (
            marker["signature"] != manifest["signature"]
            or marker["state"] != entry["state"]
            or any(entry["state"][key] > value for key, value in expected.items())
        ):
            raise ValueError("Candidate has no compatible safe replica cursor")
