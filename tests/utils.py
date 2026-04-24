# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import hashlib
import json

import torch
from torch.distributed.tensor import DTensor


def _hash_model_impl(
    model: torch.nn.Module,
    algo: str,
    per_tensor: bool,
    include_weights: bool,
    include_gradients: bool,
) -> str:
    """Internal implementation for hashing model parameters, buffers, and/or gradients."""

    # Only compute hash on rank 0 in distributed settings.
    if torch.distributed.is_initialized() and torch.distributed.get_rank() != 0:
        return ""

    h = hashlib.new(algo)
    hashes: dict[str, str] = {}

    def hash_named_tensor(name: str, obj) -> None:
        if isinstance(obj, torch.Tensor):
            if isinstance(obj, DTensor):
                t = obj.to_local().cpu().contiguous()
            else:
                t = obj.cpu().contiguous()

            # NOTE: data.numpy().tobytes() is the fastest way to convert a
            # tensor to a bytestream. See benchmark results at
            # https://github.com/pytorch/pytorch/issues/108565#issuecomment-3067330004
            raw_bytes = t.numpy().tobytes()
            if per_tensor:
                tensor_hash = hashlib.new(algo)
                tensor_hash.update(raw_bytes)
                hashes[name] = tensor_hash.hexdigest()
            else:
                h.update(name.encode("utf-8"))
                h.update(raw_bytes)

    with torch.no_grad():
        for name, param in model.named_parameters():
            if include_weights:
                hash_named_tensor(name, param)
            if include_gradients and param.grad is not None:
                hash_named_tensor(f"{name}.grad", param.grad)
        if include_weights:
            for name, buffer in model.named_buffers():
                hash_named_tensor(name, buffer)

    if per_tensor:
        return json.dumps(hashes, sort_keys=True)
    return h.hexdigest()


def hash_model(
    model: torch.nn.Module,
    algo: str = "sha256",
    per_tensor: bool = False,
) -> str:
    """Computes a hash of model parameters and buffers.

    Useful for verifying deterministic training by comparing model states
    across runs. Handles DTensor by calling to_local() before hashing.
    For distributed training, only rank 0 performs the hashing.

    Args:
        model: The model to hash.
        algo: The hash algorithm to use (default: "sha256").
        per_tensor: If True, returns a JSON-encoded dictionary mapping each tensor
            name to its hex hash. If False, returns a single hash of all tensors.

    Returns:
        A hex string hash, or a JSON-encoded per-tensor hash dictionary.
        Empty string for non-rank0 processes in distributed settings.
    """
    return _hash_model_impl(
        model,
        algo=algo,
        per_tensor=per_tensor,
        include_weights=True,
        include_gradients=False,
    )


def hash_gradient(
    model: torch.nn.Module,
    algo: str = "sha256",
    per_tensor: bool = False,
) -> str:
    """Computes a hash of model parameter gradients.

    Useful for verifying deterministic training by comparing gradient states
    across runs. Handles DTensor by calling to_local() before hashing.
    For distributed training, only rank 0 performs the hashing.
    Parameters without gradients are skipped.

    Args:
        model: The model to hash gradients for.
        algo: The hash algorithm to use (default: "sha256").
        per_tensor: If True, returns a JSON-encoded dictionary mapping each gradient
            name to its hex hash. If False, returns a single hash of all gradients.

    Returns:
        A hex string hash, or a JSON-encoded per-tensor hash dictionary.
        Empty string for non-rank0 processes in distributed settings.
    """
    return _hash_model_impl(
        model,
        algo=algo,
        per_tensor=per_tensor,
        include_weights=False,
        include_gradients=True,
    )
