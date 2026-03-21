# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import hashlib
import pickle
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from torchtitan.distributed import ParallelDims
    from torchtitan.experiments.graph_trainer.configs import GraphTrainerCompileConfig

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
    # in_spec is preserved for debugging and artifact inspection;
    # it is not used during precompile_load since the deserialized
    # callable already knows its input format.
    in_spec: Any
    out_spec: Any
    metadata: dict[str, Any] = field(default_factory=dict)
    config_fingerprint: str = ""


def compute_config_fingerprint(
    model: torch.nn.Module,
    compile_config: GraphTrainerCompileConfig,
    parallel_dims: ParallelDims,
) -> str:
    """
    Compute a fingerprint that captures everything affecting the compiled output:
    model parameter/buffer shapes and dtypes, parallelism dimensions, and
    compile configuration. Returns the first 16 chars of a SHA-256 hex digest.

    Note: pass lists are sorted before hashing so that equivalent configs
    listed in different orders produce the same fingerprint. If pass ordering
    ever becomes semantically significant (non-commutative passes), this
    should be revisited to hash in order instead.
    """
    h = hashlib.sha256()

    for name, param in model.named_parameters():
        h.update(f"param:{name}:{list(param.shape)}:{param.dtype}\n".encode())
    for name, buf in model.named_buffers():
        h.update(f"buffer:{name}:{list(buf.shape)}:{buf.dtype}\n".encode())

    for dim_name in (
        "world_size",
        "dp_replicate",
        "dp_shard",
        "cp",
        "tp",
        "pp",
        "ep",
        "etp",
    ):
        h.update(f"parallel:{dim_name}:{getattr(parallel_dims, dim_name)}\n".encode())

    h.update(f"compile:mode:{compile_config.mode}\n".encode())
    h.update(f"compile:backend:{compile_config.backend}\n".encode())
    h.update(f"compile:passes:{sorted(compile_config.passes)}\n".encode())
    h.update(f"compile:joint_passes:{sorted(compile_config.joint_passes)}\n".encode())

    return h.hexdigest()[:16]


def precompile_save(
    model: torch.nn.Module,
    compiled_fn: BundledAOTAutogradSerializableCallable,
    storage: StorageAdapter,
    artifact_key: str,
    in_spec: Any,
    out_spec: Any,
    metadata: dict[str, Any] | None = None,
    config_fingerprint: str = "",
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
        config_fingerprint=config_fingerprint,
    )

    data = pickle.dumps(artifact)
    path = storage.save(artifact_key, data)
    logger.info(
        f"Precompile artifact saved: key={artifact_key}, "
        f"params={len(params_spec)}, buffers={len(buffers_spec)}, "
        f"size={len(data)} bytes, fingerprint={config_fingerprint}, "
        f"path={path}"
    )
    return path


def precompile_load(
    model: torch.nn.Module,
    storage: StorageAdapter,
    artifact_key: str,
    expected_fingerprint: str,
) -> Callable:
    """
    Load a precompiled artifact and return a wrapper function that
    binds model parameters/buffers (same calling convention as
    joint_graph_builder's wrapper_fn).
    """
    data = storage.load(artifact_key)
    artifact: PrecompiledArtifact = pickle.loads(data)

    current_params = [name for name, _ in model.named_parameters()]
    current_buffers = [name for name, _ in model.named_buffers()]
    if current_params != artifact.params_spec:
        raise ValueError(
            f"Parameter mismatch between saved artifact and current model. "
            f"Saved: {artifact.params_spec}, Current: {current_params}"
        )
    if current_buffers != artifact.buffers_spec:
        raise ValueError(
            f"Buffer mismatch between saved artifact and current model. "
            f"Saved: {artifact.buffers_spec}, Current: {current_buffers}"
        )

    if expected_fingerprint and artifact.config_fingerprint:
        if artifact.config_fingerprint != expected_fingerprint:
            raise ValueError(
                f"Config fingerprint mismatch: the precompiled artifact was "
                f"saved with a different model/parallelism/compile configuration. "
                f"Artifact fingerprint: {artifact.config_fingerprint}, "
                f"current fingerprint: {expected_fingerprint}. "
                f"Delete the stale artifact and re-run with precompile to "
                f"generate a fresh one."
            )
    elif expected_fingerprint and not artifact.config_fingerprint:
        logger.warning(
            "Precompiled artifact has no config fingerprint (legacy artifact). "
            "Skipping fingerprint validation. Re-save the artifact to enable "
            "fingerprint checks."
        )

    logger.info(
        f"Precompile artifact loaded: key={artifact_key}, "
        f"params={len(artifact.params_spec)}, "
        f"buffers={len(artifact.buffers_spec)}, "
        f"fingerprint={artifact.config_fingerprint}, "
        f"metadata={artifact.metadata}"
    )

    out_spec = artifact.out_spec
    serialized_fn_bytes = artifact.serialized_fn
    compiled_fn_holder: list[Callable] = []

    def wrapper_fn(args, kwargs):
        # Defer deserialization to first call so that Triton kernels
        # are loaded on the correct CUDA device (which is guaranteed
        # to be set by the time the first forward runs).
        if not compiled_fn_holder:
            logger.info(
                f"Deserializing compiled fn on device {torch.cuda.current_device()}"
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
