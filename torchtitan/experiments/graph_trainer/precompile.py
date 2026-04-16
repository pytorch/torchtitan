# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import dataclasses
import hashlib
import os
import pickle
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, NewType, TYPE_CHECKING

if TYPE_CHECKING:
    from torchtitan.distributed import ParallelDims
    from torchtitan.experiments.graph_trainer.configs import GraphTrainerCompileConfig

import torch
import torch.utils._pytree as pytree
from torch._dynamo.aot_compile_types import BundledAOTAutogradSerializableCallable

from torchtitan.experiments.graph_trainer.make_fx_tracer import (
    SubclassLayout,
    TracedResult,
)
from torchtitan.experiments.graph_trainer.storage import StorageAdapter
from torchtitan.tools.logging import logger

ConfigFingerprint = NewType("ConfigFingerprint", str)

# Single artifact key — there is exactly one compiled artifact per
# precompile_artifact_dir, so no key-based dispatch is needed.
_ARTIFACT_KEY = "default"


@dataclass
class PrecompiledArtifact:
    serialized_fn: bytes
    params_spec: tuple[str, ...]
    buffers_spec: tuple[str, ...]
    out_spec: pytree.TreeSpec | None
    metadata: dict[str, Any] = field(default_factory=dict)
    config_fingerprint: ConfigFingerprint = ConfigFingerprint("")


def compute_config_fingerprint(
    model: torch.nn.Module,
    compile_config: GraphTrainerCompileConfig,
    parallel_dims: ParallelDims,
) -> ConfigFingerprint:
    """
    Compute a fingerprint that captures everything affecting the compiled output:
    model parameter/buffer shapes and dtypes, parallelism dimensions, and
    compile configuration. Returns the first 16 chars of a SHA-256 hex digest.
    """
    h = hashlib.sha256()

    for name, param in model.named_parameters():
        h.update(f"param:{name}:{list(param.shape)}:{param.dtype}\n".encode())
    for name, buf in model.named_buffers():
        h.update(f"buffer:{name}:{list(buf.shape)}:{buf.dtype}\n".encode())

    for f in dataclasses.fields(parallel_dims):
        if not f.name.startswith("_"):
            h.update(f"parallel:{f.name}:{getattr(parallel_dims, f.name)}\n".encode())

    h.update(f"compile:mode:{compile_config.mode}\n".encode())
    h.update(f"compile:backend:{compile_config.backend}\n".encode())
    h.update(f"compile:passes:{list(compile_config.passes)}\n".encode())
    h.update(f"compile:joint_passes:{list(compile_config.joint_passes)}\n".encode())

    # Include PyTorch version since compiled artifacts (AOT graphs,
    # Triton kernels) are not guaranteed to be compatible across
    # different PyTorch versions.
    h.update(f"torch_version:{torch.__version__}\n".encode())

    # Compiled Triton kernels are architecture-specific (e.g. SM80 vs
    # SM90), so artifacts saved on one GPU type may not work on another.
    # Include the GPU capability to catch cross-machine mismatches.
    if torch.cuda.is_available():
        capability = torch.cuda.get_device_capability()
        h.update(f"cuda_capability:{capability}\n".encode())

    return ConfigFingerprint(h.hexdigest()[:16])


def _register_coor_ops() -> None:
    """Register CooR custom ops required for deserialization.

    CooR-compiled artifacts reference custom ops (e.g.
    device_mesh._runtime_compute_coordinate_on_dim) that are lazily
    registered. The ops module uses @torch.library.custom_op with
    DeviceMesh, which requires DeviceMesh to be registered as an
    opaque type first. Must be called before deserializing any
    CooR-compiled artifact.
    """
    from torch.distributed.device_mesh import _register_distributed_opaque_types

    _register_distributed_opaque_types()
    from torch.distributed._ops import device_mesh as _dm_ops  # noqa: F401


def _unwrap_serializable(
    compiled_fn: Any,
) -> BundledAOTAutogradSerializableCallable:
    """
    Extract the BundledAOTAutogradSerializableCallable from compiled_fn.
    PyTorch's aot_compile_joint_with_descriptors wraps the serializable
    callable in a plain function via functools.wraps, so we walk the
    __wrapped__ chain until we find the serializable callable.
    """
    current = compiled_fn
    while current is not None:
        if isinstance(current, BundledAOTAutogradSerializableCallable):
            return current
        current = getattr(current, "__wrapped__", None)
    raise TypeError(
        "precompile_save requires the compiled function to be a "
        "BundledAOTAutogradSerializableCallable, but got "
        f"{type(compiled_fn).__name__}. Ensure your compiler pass "
        "pipeline produces serializable output (e.g. by including "
        "'full_inductor_compilation' in --compile.passes)."
    )


def precompile_save(
    model: torch.nn.Module,
    compiled_fn: BundledAOTAutogradSerializableCallable,
    storage: StorageAdapter,
    out_spec: pytree.TreeSpec | None,
    metadata: dict[str, Any] | None = None,
    config_fingerprint: ConfigFingerprint | None = None,
) -> str:
    """
    Serialize a compiled function and save it via the storage adapter.

    Returns the path/URI of the saved artifact.
    """
    compiled_fn = _unwrap_serializable(compiled_fn)

    serialized_fn = BundledAOTAutogradSerializableCallable.serialize_compile_artifacts(
        compiled_fn
    )

    params_spec = tuple(name for name, _ in model.named_parameters())
    buffers_spec = tuple(name for name, _ in model.named_buffers())

    artifact = PrecompiledArtifact(
        serialized_fn=serialized_fn,
        params_spec=params_spec,
        buffers_spec=buffers_spec,
        out_spec=out_spec,
        metadata=metadata or {},
        config_fingerprint=config_fingerprint or ConfigFingerprint(""),
    )

    data = pickle.dumps(artifact)
    path = storage.save(_ARTIFACT_KEY, data)
    logger.info(
        f"Precompile artifact saved: "
        f"params={len(params_spec)}, buffers={len(buffers_spec)}, "
        f"size={len(data)} bytes, fingerprint={config_fingerprint}, "
        f"path={path}"
    )
    return path


def precompile_load(
    model: torch.nn.Module,
    storage: StorageAdapter,
    expected_fingerprint: ConfigFingerprint,
) -> Callable:
    """
    Load a precompiled artifact and return a wrapper function that
    binds model parameters/buffers (same calling convention as
    joint_graph_builder's wrapper_fn).
    """
    data = storage.load(_ARTIFACT_KEY)
    # SAFETY: pickle.loads executes arbitrary code during deserialization.
    # This is acceptable here because storage backends are assumed to be
    # trusted (local disk or controlled shared filesystem).
    artifact: PrecompiledArtifact = pickle.loads(data)

    current_params = tuple(name for name, _ in model.named_parameters())
    current_buffers = tuple(name for name, _ in model.named_buffers())
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

    _validate_config_fingerprint(artifact.config_fingerprint, expected_fingerprint)

    logger.info(
        f"Precompile artifact loaded: "
        f"params={len(artifact.params_spec)}, "
        f"buffers={len(artifact.buffers_spec)}, "
        f"fingerprint={artifact.config_fingerprint}, "
        f"metadata={artifact.metadata}"
    )

    out_spec = artifact.out_spec
    serialized_fn_bytes = artifact.serialized_fn
    compiled_fn: Callable | None = None

    def wrapper_fn(args, kwargs):
        nonlocal compiled_fn
        # Defer deserialization to first call so that Triton kernels
        # are loaded on the correct CUDA device (which is guaranteed
        # to be set by the time the first forward runs).
        # NOTE: not thread-safe — assumes single-threaded forward calls.
        if compiled_fn is None:
            logger.info(
                f"Deserializing compiled fn on device {torch.cuda.current_device()}"
            )

            _register_coor_ops()

            compiled_fn = (
                BundledAOTAutogradSerializableCallable.deserialize_compile_artifacts(
                    serialized_fn_bytes
                )
            )

        # Build the flat input tuple: params + buffers + user args.
        # This mirrors the calling convention in joint_graph_builder's
        # wrapper_fn (graph_utils.py).
        inputs = (
            *model.parameters(),
            *model.buffers(),
            *args,
        )
        # The deserialized fn returns flat outputs. We need to
        # unflatten them using the saved out_spec to match the
        # original model output structure. See also graph_utils.py:wrapper_fn
        # which does NOT unflatten because the live-compiled fn already
        # handles it via unflattened_compiled_fn.
        flat_outputs = compiled_fn(*inputs, **kwargs)
        if out_spec is not None:
            return pytree.tree_unflatten(flat_outputs, out_spec)
        return flat_outputs

    return wrapper_fn


def _validate_config_fingerprint(
    artifact_fingerprint: ConfigFingerprint,
    expected_fingerprint: ConfigFingerprint,
) -> None:
    """Validate that artifact and current config fingerprints match.

    Raises ValueError on mismatch unless TORCHTITAN_SKIP_FINGERPRINT_CHECK=1
    is set in the environment, in which case a warning is emitted instead.
    No-ops when either fingerprint is empty (legacy artifact / missing config).
    """
    if not expected_fingerprint or not artifact_fingerprint:
        if expected_fingerprint and not artifact_fingerprint:
            logger.warning(
                "Precompiled artifact has no config fingerprint (legacy artifact). "
                "Skipping fingerprint validation. Re-save the artifact to enable "
                "fingerprint checks."
            )
        return

    if artifact_fingerprint == expected_fingerprint:
        return

    if os.environ.get("TORCHTITAN_SKIP_FINGERPRINT_CHECK", "") == "1":
        logger.warning(
            "Config fingerprint mismatch IGNORED due to "
            "TORCHTITAN_SKIP_FINGERPRINT_CHECK=1. "
            f"Artifact: {artifact_fingerprint}, "
            f"current: {expected_fingerprint}."
        )
        return

    raise ValueError(
        f"Config fingerprint mismatch: the precompiled artifact was "
        f"saved with a different configuration. "
        f"Artifact fingerprint: {artifact_fingerprint}, "
        f"current fingerprint: {expected_fingerprint}. "
        f"Delete the stale artifact and re-run precompile to "
        f"generate a fresh one. Set TORCHTITAN_SKIP_FINGERPRINT_CHECK=1 "
        f"to bypass this check."
    )


_FX_TRACE_ARTIFACT_KEY = "fx_trace_default"


@dataclass
class PrecompiledFxTraceArtifact:
    """Serialized form of a TracedResult for aot_fx_trace precompilation.

    Stores the traced FX graph (as a GraphPickler-serialized
    GraphModule) alongside the TracedResult metadata needed to unwrap
    DTensor inputs and rewrap outputs at runtime. Compiled Triton
    kernels (AOTCompiledArtifact nodes from regional_inductor) are
    baked into the serialized graph at precompile time — no Inductor
    recompilation is needed at load time.
    """

    serialized_gm: bytes
    state_fqns: list[str]
    num_flat_inputs: int
    input_subclass_layouts: dict[int, SubclassLayout]
    num_flat_outputs: int
    output_subclass_layouts: dict[int, SubclassLayout]
    output_spec: pytree.TreeSpec
    tensor_input_indices: list[int]
    config_fingerprint: ConfigFingerprint = ConfigFingerprint("")

    @classmethod
    def from_traced_result(
        cls,
        traced_result: TracedResult,
        config_fingerprint: ConfigFingerprint | None = None,
    ) -> "PrecompiledFxTraceArtifact":
        """Create an artifact from a TracedResult by serializing its GraphModule.

        Uses GraphPickler (not plain pickle) to preserve SymInt
        expressions in the graph. Plain pickle evaluates SymInts to
        concrete trace-time values, baking in rank-specific constants
        (e.g. the embedding vocab offset from
        _runtime_compute_coordinate_on_dim).
        """
        from torch.fx._graph_pickler import GraphPickler, Options

        from torchtitan.experiments.graph_trainer.passes import (
            _node_metadata_key_filter_distributed,
        )

        serialized_gm = GraphPickler.dumps(
            traced_result.gm,
            Options(
                ops_filter=None,
                node_metadata_key_filter=_node_metadata_key_filter_distributed,
            ),
        )

        return cls(
            serialized_gm=serialized_gm,
            state_fqns=traced_result.state_fqns,
            num_flat_inputs=traced_result.num_flat_inputs,
            input_subclass_layouts=traced_result.input_subclass_layouts,
            num_flat_outputs=traced_result.num_flat_outputs,
            output_subclass_layouts=traced_result.output_subclass_layouts,
            output_spec=traced_result.output_spec,
            tensor_input_indices=traced_result.tensor_input_indices,
            config_fingerprint=config_fingerprint or ConfigFingerprint(""),
        )

    def to_traced_result(self) -> TracedResult:
        """Deserialize back into a TracedResult.

        Registers CooR custom ops, then deserializes the GraphModule
        via GraphPickler under a FakeTensorMode (needed so that
        placeholder metadata contains FakeTensors for downstream
        passes like regional_inductor).
        """
        _register_coor_ops()

        from torch._subclasses import FakeTensorMode
        from torch.fx._graph_pickler import GraphPickler

        fake_mode = FakeTensorMode(
            allow_non_fake_inputs=True,
            shape_env=torch.fx.experimental.symbolic_shapes.ShapeEnv(),
        )
        gm = GraphPickler.loads(self.serialized_gm, fake_mode)
        gm.recompile()

        return TracedResult(
            gm=gm,
            example_inputs=(),
            state_fqns=self.state_fqns,
            num_flat_inputs=self.num_flat_inputs,
            input_subclass_layouts=self.input_subclass_layouts,
            num_flat_outputs=self.num_flat_outputs,
            output_subclass_layouts=self.output_subclass_layouts,
            output_spec=self.output_spec,
            tensor_input_indices=self.tensor_input_indices,
        )


def precompile_fx_trace_save(
    traced_result: TracedResult,
    storage: StorageAdapter,
    config_fingerprint: ConfigFingerprint | None = None,
) -> str:
    """Serialize a traced and compiled FX graph artifact and save it.

    The GraphModule should have graph passes (cleanup, annotation,
    regional_inductor) already applied so compiled Triton kernels are
    baked into the artifact as AOTCompiledArtifact nodes.
    """
    artifact = PrecompiledFxTraceArtifact.from_traced_result(
        traced_result, config_fingerprint
    )

    data = pickle.dumps(artifact)
    path = storage.save(_FX_TRACE_ARTIFACT_KEY, data)
    logger.info(
        f"FxTrace precompile artifact saved: "
        f"state_fqns={len(artifact.state_fqns)}, "
        f"num_flat_inputs={artifact.num_flat_inputs}, "
        f"size={len(data)} bytes, fingerprint={config_fingerprint}, "
        f"path={path}"
    )
    return path


def precompile_fx_trace_load(
    storage: StorageAdapter,
    expected_fingerprint: ConfigFingerprint,
) -> TracedResult:
    """Load a precompiled aot_fx_trace artifact.

    Returns a TracedResult with the deserialized GraphModule and
    metadata. The caller uses this with run_traced_train_step to
    execute the graph (same path as non-precompiled aot_fx_trace).

    DeviceMesh objects are graph inputs (placeholders), not baked-in
    constants, so ProcessGroup names are resolved at runtime from
    the caller-provided DeviceMesh — no post-deserialization PG
    remapping is needed.
    """
    data = storage.load(_FX_TRACE_ARTIFACT_KEY)
    artifact: PrecompiledFxTraceArtifact = pickle.loads(data)

    _validate_config_fingerprint(artifact.config_fingerprint, expected_fingerprint)

    logger.info(
        f"FxTrace precompile artifact loaded: "
        f"state_fqns={len(artifact.state_fqns)}, "
        f"num_flat_inputs={artifact.num_flat_inputs}, "
        f"fingerprint={artifact.config_fingerprint}"
    )

    return artifact.to_traced_result()
