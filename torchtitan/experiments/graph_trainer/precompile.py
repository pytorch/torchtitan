# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import pickle
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import torch
import torch.utils._pytree as pytree
from torch._dynamo.aot_compile_types import BundledAOTAutogradSerializableCallable

from torchtitan.experiments.graph_trainer.storage import StorageAdapter
from torchtitan.tools.logging import logger


@dataclass
class PrecompiledArtifact:
    serialized_fn: bytes
    params_spec: list[str]
    buffers_spec: list[str]
    in_spec: Any
    out_spec: Any
    metadata: dict[str, Any] = field(default_factory=dict)


def precompile_save(
    model: torch.nn.Module,
    compiled_fn: BundledAOTAutogradSerializableCallable,
    storage: StorageAdapter,
    artifact_key: str,
    in_spec: Any,
    out_spec: Any,
    metadata: dict[str, Any] | None = None,
) -> str:
    """
    Serialize a compiled function and save it via the storage adapter.

    Returns the path/URI of the saved artifact.
    """
    serialized_fn = BundledAOTAutogradSerializableCallable.serialize_compile_artifacts(
        compiled_fn
    )

    params_spec = [name for name, _ in model.named_parameters()]
    buffers_spec = [name for name, _ in model.named_buffers()]

    artifact = PrecompiledArtifact(
        serialized_fn=serialized_fn,
        params_spec=params_spec,
        buffers_spec=buffers_spec,
        in_spec=in_spec,
        out_spec=out_spec,
        metadata=metadata or {},
    )

    data = pickle.dumps(artifact)
    path = storage.save(artifact_key, data)
    logger.info(
        f"Precompile artifact saved: key={artifact_key}, "
        f"params={len(params_spec)}, buffers={len(buffers_spec)}, "
        f"size={len(data)} bytes, path={path}"
    )
    return path


def precompile_load(
    model: torch.nn.Module,
    storage: StorageAdapter,
    artifact_key: str,
) -> Callable:
    """
    Load a precompiled artifact and return a wrapper function that
    binds model parameters/buffers (same calling convention as
    joint_graph_builder's wrapper_fn).
    """
    data = storage.load(artifact_key)
    artifact: PrecompiledArtifact = pickle.loads(data)

    logger.info(
        f"Precompile artifact loaded: key={artifact_key}, "
        f"params={len(artifact.params_spec)}, "
        f"buffers={len(artifact.buffers_spec)}, "
        f"metadata={artifact.metadata}"
    )

    out_spec = artifact.out_spec
    serialized_fn_bytes = artifact.serialized_fn
    compiled_fn_holder: list = []

    def wrapper_fn(args, kwargs):
        # Defer deserialization to first call so that Triton kernels
        # are loaded on the correct CUDA device (which is guaranteed
        # to be set by the time the first forward runs).
        if not compiled_fn_holder:
            logger.info(
                f"Deserializing compiled fn on device " f"{torch.cuda.current_device()}"
            )
            compiled_fn_holder.append(
                BundledAOTAutogradSerializableCallable.deserialize_compile_artifacts(
                    serialized_fn_bytes
                )
            )
        compiled_fn = compiled_fn_holder[0]

        # Build the flat input list: params + buffers + user args.
        # This mirrors the calling convention in joint_graph_builder's
        # wrapper_fn (graph_utils.py).
        inputs = [
            *model.parameters(),
            *model.buffers(),
            *args,
        ]
        # The deserialized fn returns flat outputs. We need to
        # unflatten them using the saved out_spec to match the
        # original model output structure.
        flat_outputs = compiled_fn(*inputs, **kwargs)
        if out_spec is not None:
            return pytree.tree_unflatten(flat_outputs, out_spec)
        return flat_outputs

    return wrapper_fn
