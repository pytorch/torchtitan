# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import dataclasses
import enum
import functools
import hashlib
import json
import os
import pickle
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, NewType, TYPE_CHECKING

if TYPE_CHECKING:
    from torchtitan.components.loss import BaseLoss
    from torchtitan.distributed import ParallelDims
    from torchtitan.experiments.graph_trainer.configs import GraphTrainerCompileConfig
    from torchtitan.experiments.graph_trainer.graph_pp.graph_builder import (
        GraphTrainerStageGraphs,
    )
    from torchtitan.protocols.model import BaseModel

import torch
import torch.utils._pytree as pytree
from torch.distributed.device_mesh import DeviceMesh
from torch.fx.experimental.symbolic_shapes import ShapeEnv

from torchtitan.experiments.graph_trainer.make_fx_tracer import (
    _unwrap_subclasses,
    extract_train_state,
    SubclassLayout,
    TracedResult,
)
from torchtitan.experiments.graph_trainer.storage import StorageAdapter
from torchtitan.tools.logging import logger

ConfigFingerprint = NewType("ConfigFingerprint", str)


def _qualified_name(value: object) -> str:
    value_type = value if isinstance(value, type) else type(value)
    return f"{value_type.__module__}.{value_type.__qualname__}"


def _canonical_config_value(value: Any) -> Any:
    """Convert model configuration to deterministic JSON-compatible values."""
    if dataclasses.is_dataclass(value):
        return {
            "type": _qualified_name(value),
            "fields": {
                field.name: _canonical_config_value(getattr(value, field.name))
                for field in dataclasses.fields(value)
                if not field.name.startswith("_")
            },
        }
    if isinstance(value, enum.Enum):
        return {"enum": _qualified_name(value), "name": value.name}
    if isinstance(value, functools.partial):
        return {
            "partial": _canonical_config_value(value.func),
            "args": _canonical_config_value(value.args),
            "keywords": _canonical_config_value(value.keywords or {}),
        }
    if callable(value):
        module = getattr(value, "__module__", None)
        qualname = getattr(value, "__qualname__", None)
        if module is None or qualname is None:
            raise TypeError(f"Cannot fingerprint callable config value {value!r}")
        return {"callable": f"{module}.{qualname}"}
    if isinstance(value, Mapping):
        items = [
            (_canonical_config_value(key), _canonical_config_value(item))
            for key, item in value.items()
        ]
        items.sort(key=lambda pair: json.dumps(pair[0], sort_keys=True))
        return {"mapping": items}
    if isinstance(value, (list, tuple)):
        return {
            "sequence_type": type(value).__name__,
            "items": [_canonical_config_value(item) for item in value],
        }
    if isinstance(value, (set, frozenset)):
        items = [_canonical_config_value(item) for item in value]
        items.sort(key=lambda item: json.dumps(item, sort_keys=True))
        return {"set": items}
    if isinstance(value, (str, int, float, bool, type(None))):
        return value
    if isinstance(value, (torch.dtype, torch.device)):
        return {"type": _qualified_name(value), "value": str(value)}
    raise TypeError(
        "Cannot fingerprint model config value of type "
        f"{_qualified_name(value)}: {value!r}"
    )


def flatten_runtime_inputs(
    module: torch.nn.Module,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    precompile_meshes: list[DeviceMesh] | None = None,
) -> tuple[Any, ...]:
    """Flatten live model state, meshes, and call inputs for a precompiled graph."""
    model_state, optim_state = extract_train_state(module)
    state_flat, _ = pytree.tree_flatten({"model": model_state, "optim": optim_state})
    user_inputs_flat, _ = pytree.tree_flatten((args, kwargs))
    flat_inputs, _ = _unwrap_subclasses(
        [*state_flat, *(precompile_meshes or []), *user_inputs_flat]
    )
    return tuple(flat_inputs)


def get_spmd_precompile_meshes(parallel_dims: ParallelDims) -> list[DeviceMesh]:
    """
    Return SPMD meshes that must be registered as runtime graph inputs.

    Pre-registering meshes allows PG lookups for collectives in forward code (ambient mesh)
    to appear in graph as custom op results (indexing input meshes), rather than
    opaque objects with no source, matching graph structure from legacy DTensor path.
    """
    candidates = [
        parallel_dims.spmd_dense_mesh(),
        parallel_dims.spmd_sparse_mesh(),
        parallel_dims.get_optional_mesh("pp"),
    ]
    meshes: list[DeviceMesh] = []
    for mesh in candidates:
        if mesh is not None and all(mesh is not other for other in meshes):
            meshes.append(mesh)
    return meshes


def compute_config_fingerprint(
    model: torch.nn.Module,
    compile_config: GraphTrainerCompileConfig,
    parallel_dims: ParallelDims,
    *,
    loss_config: BaseLoss.Config | None = None,
    model_config: BaseModel.Config | None = None,
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

    compile_fields = dataclasses.asdict(compile_config)
    compile_fields.pop("debug_graph_passes", None)
    compile_fields.pop("precompile_artifact_dir", None)
    h.update(
        b"compile:"
        + json.dumps(compile_fields, sort_keys=True, separators=(",", ":")).encode()
        + b"\n"
    )
    if loss_config is not None:
        loss_type = type(loss_config)
        h.update(
            f"loss_type:{loss_type.__module__}.{loss_type.__qualname__}\n".encode()
        )
        h.update(
            b"loss_config:"
            + json.dumps(
                dataclasses.asdict(loss_config),
                sort_keys=True,
                separators=(",", ":"),
            ).encode()
            + b"\n"
        )
    if model_config is not None:
        h.update(
            b"model_config:"
            + json.dumps(
                _canonical_config_value(model_config),
                sort_keys=True,
                separators=(",", ":"),
            ).encode()
            + b"\n"
        )
    h.update(
        "torch:deterministic_algorithms:"
        f"{torch.are_deterministic_algorithms_enabled()}\n".encode()
    )
    h.update(f"torch_version:{torch.__version__}\n".encode())

    if torch.cuda.is_available():
        capability = torch.cuda.get_device_capability()
        h.update(f"cuda_capability:{capability}\n".encode())

    return ConfigFingerprint(h.hexdigest()[:16])


def _register_coor_ops() -> None:
    """Register CooR custom ops required for tracing and deserialization.

    CooR-compiled artifacts reference custom ops (e.g.
    device_mesh._runtime_compute_coordinate_on_dim) that are lazily
    registered. The ops module uses @torch.library.custom_op with
    DeviceMesh, which requires DeviceMesh to be registered as an
    opaque type first. Must be called before tracing or deserializing a
    CooR-compiled artifact.
    """
    from torch.distributed.device_mesh import _register_distributed_opaque_types

    _register_distributed_opaque_types()
    from torch.distributed._ops import device_mesh as _dm_ops  # noqa: F401
    from torch.distributed.tensor import _collective_utils  # noqa: F401


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
_GRAPH_PP_STAGE_ARTIFACT_KEY = "graph_pp_stage_default"


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
    # user_inputs_spec is intentionally omitted: it can contain
    # FlexAttention _MaskModWrapper objects that are not picklable,
    # and the mask_mod is already compiled into standalone Inductor
    # HOPs (AOTCompiledArtifact) baked into serialized_gm. The spec
    # is only used for optional runtime validation in run_traced().
    config_fingerprint: ConfigFingerprint = ConfigFingerprint("")
    # Retained separately because user_inputs_spec is not serialized.
    num_optimizer_state_inputs: int = 0
    num_runtime_mesh_inputs: int = 0

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
        if traced_result.graph_state.mappings:
            raise ValueError(
                "Precompiled FX artifacts do not yet support trainer-owned "
                "gradient state"
            )

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
            config_fingerprint=config_fingerprint or ConfigFingerprint(""),
            num_optimizer_state_inputs=traced_result.num_optimizer_state_inputs,
            num_runtime_mesh_inputs=traced_result.num_runtime_mesh_inputs,
        )

    def to_traced_result(self, example_inputs: tuple[Any, ...]) -> TracedResult:
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
            shape_env=ShapeEnv(),
        )
        gm = GraphPickler.loads(self.serialized_gm, fake_mode)
        gm.recompile()

        # Provide a minimal dummy spec since user_inputs_spec is not
        # serialized (see comment on the dataclass field above).
        dummy_spec = pytree.tree_flatten(((), {}))[1]

        return TracedResult(
            gm=gm,
            example_inputs=example_inputs,
            num_flat_inputs=self.num_flat_inputs,
            input_subclass_layouts=self.input_subclass_layouts,
            user_inputs_spec=dummy_spec,
            tensor_input_indices=self.tensor_input_indices,
            num_flat_outputs=self.num_flat_outputs,
            output_subclass_layouts=self.output_subclass_layouts,
            output_spec=self.output_spec,
            state_fqns=self.state_fqns,
            num_optimizer_state_inputs=self.num_optimizer_state_inputs,
            num_runtime_mesh_inputs=self.num_runtime_mesh_inputs,
        )


@dataclass
class PrecompiledGraphPPStageArtifact:
    """Serialized GraphPP callables and metadata for one PP=1 stage."""

    serialized_modules: dict[str, bytes | None]
    meta: Any
    state_fqns: list[str]
    num_runtime_mesh_inputs: int
    config_fingerprint: ConfigFingerprint

    @classmethod
    def from_stage_graphs(
        cls,
        stage_graphs: GraphTrainerStageGraphs,
        *,
        state_fqns: list[str],
        num_runtime_mesh_inputs: int,
        config_fingerprint: ConfigFingerprint,
    ) -> "PrecompiledGraphPPStageArtifact":
        from torch.fx._graph_pickler import GraphPickler, Options

        from torchtitan.experiments.graph_trainer.inductor_passes import (
            _node_metadata_key_filter_distributed,
        )

        options = Options(
            ops_filter=None,
            node_metadata_key_filter=_node_metadata_key_filter_distributed,
        )
        return cls(
            serialized_modules={
                name: None if gm is None else GraphPickler.dumps(gm, options)
                for name, gm in (
                    (field.name, getattr(stage_graphs.modules, field.name))
                    for field in dataclasses.fields(stage_graphs.modules)
                )
            },
            meta=stage_graphs.meta,
            state_fqns=state_fqns,
            num_runtime_mesh_inputs=num_runtime_mesh_inputs,
            config_fingerprint=config_fingerprint,
        )

    def to_stage_graphs(
        self,
        *,
        runtime_meshes: list[DeviceMesh],
    ) -> GraphTrainerStageGraphs:
        from torch._subclasses import FakeTensorMode
        from torch.fx._graph_pickler import GraphPickler

        from torchtitan.experiments.graph_trainer.graph_pp.graph_builder import (
            _StageGraphModules,
            GraphTrainerStageGraphs,
        )

        if len(runtime_meshes) != self.num_runtime_mesh_inputs:
            raise ValueError(
                "GraphPP precompile runtime mesh count mismatch: "
                f"expected {self.num_runtime_mesh_inputs}, got {len(runtime_meshes)}"
            )
        _register_coor_ops()
        fake_mode = FakeTensorMode(
            allow_non_fake_inputs=True,
            shape_env=ShapeEnv(),
        )
        modules = {}
        for name, serialized_gm in self.serialized_modules.items():
            if serialized_gm is None:
                modules[name] = None
                continue
            gm = GraphPickler.loads(serialized_gm, fake_mode)
            gm.recompile()
            modules[name] = gm
        return GraphTrainerStageGraphs(
            modules=_StageGraphModules(**modules),
            meta=self.meta,
            compiled=True,
            runtime_meshes=tuple(runtime_meshes),
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
    example_inputs: tuple[Any, ...],
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

    return artifact.to_traced_result(example_inputs)


def precompile_graph_pp_stage_save(
    stage_graphs: GraphTrainerStageGraphs,
    storage: StorageAdapter,
    *,
    state_fqns: list[str],
    num_runtime_mesh_inputs: int,
    config_fingerprint: ConfigFingerprint,
) -> str:
    """Serialize one compiled PP=1 GraphPP stage."""
    artifact = PrecompiledGraphPPStageArtifact.from_stage_graphs(
        stage_graphs,
        state_fqns=state_fqns,
        num_runtime_mesh_inputs=num_runtime_mesh_inputs,
        config_fingerprint=config_fingerprint,
    )
    data = pickle.dumps(artifact)
    path = storage.save(_GRAPH_PP_STAGE_ARTIFACT_KEY, data)
    logger.info(
        "GraphPP stage precompile artifact saved: "
        f"state_fqns={len(state_fqns)}, size={len(data)} bytes, path={path}"
    )
    return path


def precompile_graph_pp_stage_load(
    storage: StorageAdapter,
    *,
    expected_fingerprint: ConfigFingerprint,
    expected_state_fqns: list[str],
    runtime_meshes: list[DeviceMesh],
) -> GraphTrainerStageGraphs:
    """Load and bind one compiled PP=1 GraphPP stage."""
    data = storage.load(_GRAPH_PP_STAGE_ARTIFACT_KEY)
    artifact: PrecompiledGraphPPStageArtifact = pickle.loads(data)
    _validate_config_fingerprint(
        artifact.config_fingerprint,
        expected_fingerprint,
    )
    if artifact.state_fqns != expected_state_fqns:
        raise ValueError(
            "GraphPP precompile model state differs from the runtime model: "
            f"artifact={artifact.state_fqns}, runtime={expected_state_fqns}"
        )
    logger.info(
        "GraphPP stage precompile artifact loaded: "
        f"state_fqns={len(artifact.state_fqns)}, "
        f"fingerprint={artifact.config_fingerprint}"
    )
    return artifact.to_stage_graphs(runtime_meshes=runtime_meshes)
