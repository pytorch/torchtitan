# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
# This source code is licensed under the BSD-style license found in LICENSE.

"""Check DCP metadata against the application's required recovery state."""

from math import prod


def state_schema(states, *, kind):
    """Describe live DCP state without saving or loading tensor values."""
    from torch.distributed.checkpoint import DefaultSavePlanner
    from torch.distributed.checkpoint.stateful import Stateful

    values = {key: value.state_dict() if isinstance(value, Stateful) else value for key, value in states.items()}
    planner = DefaultSavePlanner()
    planner.set_up_planner(values, is_coordinator=True)
    fields = {}
    for item in planner.create_local_plan().items:
        field = {"kind": "object"}
        if item.tensor_data is not None:
            field = {
                "kind": "tensor",
                "shape": list(item.tensor_data.size),
                "dtype": str(item.tensor_data.properties.dtype),
            }
        previous = fields.setdefault(item.index.fqn, field)
        if previous != field:
            raise ValueError("Inconsistent local tensor recovery schema")
    groups = {}
    reserved = {"optimizer", "lr_scheduler", "train_state", "ema", "torchft"}
    if kind == "cursor":
        reserved = {"dataloader", "last_words_rng"}
    elif kind != "model":
        raise ValueError("Unknown recovery state schema kind")
    for name in fields:
        prefix = name.split(".", 1)[0]
        group = prefix if prefix in reserved else "model"
        groups.setdefault(group, []).append(name)
    return {"version": 1, "kind": kind, "fields": fields, "groups": groups}


def _validate_description(schema, *, kind):
    if not isinstance(schema, dict) or schema.get("version") != 1 or schema.get("kind") != kind:
        raise ValueError("Checkpoint has no compatible required-state schema")
    fields, groups = schema.get("fields"), schema.get("groups")
    if not isinstance(fields, dict) or not isinstance(groups, dict):
        raise ValueError("Invalid required-state schema layout")
    required = {"model", "optimizer", "lr_scheduler", "train_state"}
    if kind == "cursor":
        required = {"dataloader", "last_words_rng"}
    if not required <= groups.keys():
        raise ValueError(f"Checkpoint schema lacks required state: {sorted(required - groups.keys())}")
    covered = []
    for group, names in groups.items():
        if not isinstance(group, str) or not isinstance(names, list) or not names:
            raise ValueError("Checkpoint schema has an empty state group")
        for name in names:
            if not isinstance(name, str) or name not in fields:
                raise ValueError("Checkpoint state group references a missing field")
            if group != "model" and name != group and not name.startswith(group + "."):
                raise ValueError("Checkpoint state group and field name disagree")
        covered.extend(names)
    if len(covered) != len(set(covered)) or set(covered) != fields.keys():
        raise ValueError("Checkpoint state groups do not partition required fields")
    for name, field in fields.items():
        if not isinstance(field, dict) or field.get("kind") not in {"tensor", "object"}:
            raise ValueError(f"Invalid recovery field schema: {name}")
        if field["kind"] == "tensor" and (
            not isinstance(field.get("shape"), list)
            or any(type(size) is not int or size < 0 for size in field["shape"])
            or not isinstance(field.get("dtype"), str)
        ):
            raise ValueError(f"Invalid tensor recovery schema: {name}")
    if kind == "model" and not any(fields[name]["kind"] == "tensor" for name in groups["model"]):
        raise ValueError("Checkpoint model schema has no tensor state")
    return fields


def validate_state_schema(path, schema, *, kind, expected=None):
    """Check published fields, then the independently derived live schema."""
    import torch.distributed.checkpoint as dcp
    from torch.distributed.checkpoint.metadata import BytesStorageMetadata, TensorStorageMetadata

    required = dict(_validate_description(schema, kind=kind))
    if expected is not None:
        # The candidate's own declaration cannot relax the live application's
        # model, optimizer, EMA, counters or current logical-replica cursor.
        for name, field in _validate_description(expected, kind=kind).items():
            if name in required and required[name] != field:
                raise ValueError(f"Published and live recovery schemas differ: {name}")
            required[name] = field
    metadata = dcp.FileSystemReader(path).read_metadata().state_dict_metadata
    for name, field in required.items():
        stored = metadata.get(name)
        if field["kind"] == "tensor":
            if (
                not isinstance(stored, TensorStorageMetadata)
                or list(stored.size) != field["shape"]
                or str(stored.properties.dtype) != field["dtype"]
            ):
                raise ValueError(f"Checkpoint has no matching required tensor: {name}")
        elif not isinstance(stored, BytesStorageMetadata):
            raise ValueError(f"Checkpoint has no required serialized state: {name}")


def validate_metadata_coverage(metadata):
    """Reject metadata with missing storage or incomplete/overlapping chunks."""
    from torch.distributed.checkpoint.metadata import BytesStorageMetadata, TensorStorageMetadata

    storage = {(item.fqn, tuple(item.offset) if item.offset is not None else None) for item in metadata.storage_data}
    for name, entry in metadata.state_dict_metadata.items():
        if isinstance(entry, BytesStorageMetadata):
            if (name, None) not in storage:
                raise ValueError(f"Checkpoint has no storage for state: {name}")
            continue
        if not isinstance(entry, TensorStorageMetadata):
            raise ValueError(f"Unsupported checkpoint metadata: {name}")
        volume, chunks = 0, []
        for chunk in entry.chunks:
            offsets, sizes = tuple(chunk.offsets), tuple(chunk.sizes)
            if (
                len(offsets) != len(entry.size)
                or len(sizes) != len(entry.size)
                or any(
                    offset < 0 or size < 0 or offset + size > full
                    for offset, size, full in zip(offsets, sizes, entry.size, strict=True)
                )
            ):
                raise ValueError(f"Checkpoint has invalid tensor chunks: {name}")
            current = prod(sizes)
            if current:
                if (name, offsets) not in storage:
                    raise ValueError(f"Checkpoint tensor chunk has no storage: {name}")
                for old_offsets, old_sizes in chunks:
                    if all(
                        a < b + y and b < a + x
                        for a, x, b, y in zip(offsets, sizes, old_offsets, old_sizes, strict=True)
                    ):
                        raise ValueError(f"Checkpoint tensor chunks overlap: {name}")
                chunks.append((offsets, sizes))
            volume += current
        if volume != prod(entry.size):
            raise ValueError(f"Checkpoint tensor chunks are incomplete: {name}")
