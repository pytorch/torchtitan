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
from dataclasses import dataclass
from typing import NewType, TYPE_CHECKING

if TYPE_CHECKING:
    from torchtitan.distributed import ParallelDims
    from torchtitan.experiments.graph_trainer.configs import GraphTrainerCompileConfig

import torch
import torch.utils._pytree as pytree

from torchtitan.experiments.graph_trainer.make_fx_tracer import (
    SubclassLayout,
    TracedResult,
)
from torchtitan.experiments.graph_trainer.storage import StorageAdapter
from torchtitan.tools.logging import logger

ConfigFingerprint = NewType("ConfigFingerprint", str)


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

    h.update(f"torch_version:{torch.__version__}\n".encode())

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
    user_inputs_spec: pytree.TreeSpec
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

        from torchtitan.experiments.graph_trainer.inductor_passes import (
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
            user_inputs_spec=traced_result.user_inputs_spec,
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
            num_flat_inputs=self.num_flat_inputs,
            input_subclass_layouts=self.input_subclass_layouts,
            user_inputs_spec=self.user_inputs_spec,
            tensor_input_indices=self.tensor_input_indices,
            num_flat_outputs=self.num_flat_outputs,
            output_subclass_layouts=self.output_subclass_layouts,
            output_spec=self.output_spec,
            state_fqns=self.state_fqns,
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
    metadata. The caller uses this with run_traced(..., module=model) to
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
