# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
# This source code is licensed under the BSD-style license found in LICENSE.

"""Read-only filesystem checks shared by recovery and checkpoint retention."""

import json
import logging
from pathlib import Path

from .schema import validate_metadata_coverage, validate_state_schema

logger = logging.getLogger(__name__)


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
    validate_metadata_coverage(metadata)
    for entry in metadata.storage_data.values():
        shard = contained(path, path / entry.relative_path)
        if entry.offset < 0 or entry.length < 0 or shard.stat().st_size < entry.offset + entry.length:
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
    headers = {"last_words_rng": {f"rank_{rank}": {"schema": 0, "device_type": ""} for rank in range(num_shards)}}
    dcp.load(headers, checkpoint_id=path, no_dist=True)
    for rank in range(num_shards):
        header = headers["last_words_rng"][f"rank_{rank}"]
        if (
            type(header["schema"]) is not int
            or header["schema"] != 1
            or not isinstance(header["device_type"], str)
            or not header["device_type"]
        ):
            raise ValueError(f"Cursor RNG schema or device type differs for shard {rank}")
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
                raise ValueError(f"Cursor has no valid accelerator RNG stream for shard {rank}")


def recovery_preflight(path, manifest, num_replicas, *, expected_model=None, expected_cursors=None):
    path = Path(path).resolve()
    validate_state_schema(path / "model", manifest.get("recovery_schema"), kind="model", expected=expected_model)
    expected = {key: manifest["state"][key] for key in ("step", "last_words_num_updates")}
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
        validate_state_schema(
            source,
            marker.get("recovery_schema"),
            kind="cursor",
            expected=(expected_cursors or {}).get(entry["path"]),
        )
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


def _read_json(path):
    return json.loads(Path(path).read_text())


def cursor_candidates(root, replica, state, *, signature, run_id, anchors, elastic_enabled, num_shards):
    from torch.distributed.checkpoint.api import CheckpointException

    source_root = Path(root) / f"ft-replicat-{replica}"
    available = []
    branch = source_root / run_id
    index = branch / "last_words_cursor_index.json"
    markers = []
    if index.is_file():
        try:
            value = _read_json(index)
            if value["signature"] != signature:
                raise ValueError("Cursor index has another signature")
            eligible = [
                (entry["last_words_num_updates"], entry["step"], name)
                for name, entry in value["entries"].items()
                if entry["last_words_num_updates"] <= state["last_words_num_updates"] and entry["step"] <= state["step"]
            ]
            for _, _, name in sorted(eligible, reverse=True):
                if Path(name).name != name or not name.startswith("step-"):
                    raise ValueError("Invalid indexed cursor path")
                candidate = branch / name
                validate_storage_ranges(candidate)
                marker = candidate / "last_words_cursor.json"
                cursor = _read_json(marker)
                if cursor["signature"] != signature or cursor["state"] != value["entries"][name]:
                    raise ValueError("Cursor index disagrees with published marker")
                markers = [marker]
                break
        except (Exception, CheckpointException) as error:
            logger.warning("Cursor index unavailable; scanning published cursors: %s", error)
    if not markers:
        markers = list(branch.glob("step-*/last_words_cursor.json"))
    if elastic_enabled:
        markers.extend(branch.glob("generation-*/step-*/last_words_cursor.json"))
        markers.extend(branch.glob("initial/step-*/last_words_cursor.json"))
    if replica in anchors:
        markers.append(anchors[replica] / "last_words_cursor.json")
    for marker in markers:
        if not marker.is_file():
            continue
        try:
            contained(Path(root), marker.parent)
            validate_storage_ranges(marker.parent)
            cursor = _read_json(marker)
            validate_state_schema(marker.parent, cursor.get("recovery_schema"), kind="cursor")
            if num_shards is not None:
                validate_rng_cursor(marker.parent, cursor, num_shards)
        except (Exception, CheckpointException) as error:
            logger.warning("Ignoring incomplete cursor %s: %s", marker, error)
            continue
        if (
            cursor["signature"] == signature
            and cursor["state"]["last_words_num_updates"] <= state["last_words_num_updates"]
            and (cursor["state"]["step"] <= state["step"])
            and (marker.parent / ".metadata").is_file()
        ):
            available.append(
                (cursor["state"]["last_words_num_updates"], cursor["state"]["step"], marker.parent, cursor)
            )
    return available


def periodic_recovery_preflight(
    root, step, *, signature, num_replicas, elastic_enabled, num_shards=None, expected_model=None, expected_cursors=None
):
    """Validate a periodic model and its cursor chain without a live manager."""
    root = Path(root)
    path = contained(root, root / f"step-{step}")
    if not (path / ".metadata").is_file():
        raise ValueError("Periodic checkpoint metadata is missing")
    validate_storage_ranges(path)
    if step == 0:
        return {}, {}
    binding_path = path / "last_words_binding.json"
    if not binding_path.is_file():
        raise ValueError("Periodic checkpoint has no bound cursor recovery chain")
    binding = _read_json(binding_path)
    validate_state_schema(path, binding.get("recovery_schema"), kind="model", expected=expected_model)
    if binding["signature"] != signature:
        raise ValueError("Periodic checkpoint configuration differs")
    if (
        binding.get("rng_schema") != 1
        or type(binding.get("rng_shards")) is not int
        or binding["rng_shards"] <= 0
        or (num_shards is not None and binding["rng_shards"] != num_shards)
    ):
        raise ValueError("Periodic checkpoint has no matching RNG recovery chain")
    num_shards = binding["rng_shards"]
    state = checkpoint_state(path)
    if state["step"] != step or any(binding["state"][key] != value for key, value in state.items()):
        raise ValueError("Periodic checkpoint binding progress differs")
    if elastic_enabled and checkpoint_torchft_state(path) != binding.get("torchft"):
        raise ValueError("Periodic checkpoint has no matching TorchFT progress")
    anchors = {int(replica): contained(root, root / anchor) for replica, anchor in binding["anchors"].items()}
    selected = {}
    for replica in range(num_replicas):
        available = cursor_candidates(
            root,
            replica,
            state,
            signature=signature,
            run_id=binding["run_id"],
            anchors=anchors,
            elastic_enabled=elastic_enabled,
            num_shards=num_shards,
        )
        if not available:
            raise ValueError("Periodic model checkpoint has no compatible safe replica cursor")
        _, _, cursor_path, _ = max(available, key=lambda entry: (entry[0], entry[1], str(entry[2])))
        selected[replica] = cursor_path
        if expected_cursors and replica in expected_cursors:
            marker = _read_json(cursor_path / "last_words_cursor.json")
            validate_state_schema(
                cursor_path, marker.get("recovery_schema"), kind="cursor", expected=expected_cursors[replica]
            )
    return state, selected
