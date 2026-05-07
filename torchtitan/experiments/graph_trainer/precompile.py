# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import dataclasses
import hashlib
import operator
import os
import pickle
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, NewType, TYPE_CHECKING

if TYPE_CHECKING:
    from torch.distributed.device_mesh import DeviceMesh

    from torchtitan.distributed import ParallelDims
    from torchtitan.experiments.graph_trainer.configs import GraphTrainerCompileConfig

import torch
import torch.distributed as dist
import torch.utils._pytree as pytree
from torch._dynamo.aot_compile_types import BundledAOTAutogradSerializableCallable

from torchtitan.experiments.graph_trainer.make_fx_tracer import (
    ProcessGroupInputSpec,
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
    from torch.distributed.tensor import _collective_utils as _dtensor_ops  # noqa: F401


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
        if callable(getattr(current, "serialize", None)):
            return BundledAOTAutogradSerializableCallable(current)
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
            *kwargs.values(),
        )
        # The deserialized fn returns flat outputs. We need to
        # unflatten them using the saved out_spec to match the
        # original model output structure. See also graph_utils.py:wrapper_fn
        # which does NOT unflatten because the live-compiled fn already
        # handles it via unflattened_compiled_fn.
        flat_outputs = compiled_fn(*inputs)
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


def _get_graph_attr(module: torch.nn.Module, target: str) -> Any:
    attr: Any = module
    for atom in target.split("."):
        attr = getattr(attr, atom)
    return attr


def _safe_placeholder_suffix(axis: str) -> str:
    return "".join(ch if ch.isalnum() or ch == "_" else "_" for ch in axis)


def _process_group_axis_from_mesh(mesh: Any, mesh_axis: int) -> str:
    mesh_axis_names = getattr(mesh, "mesh_dim_names", None)
    if mesh_axis_names is None:
        raise ValueError(
            "Cannot hoist ProcessGroup input for an unnamed DeviceMesh axis. "
            "Graph trainer precompile expects named mesh axes."
        )
    axis = mesh_axis_names[mesh_axis]
    if axis is None:
        raise ValueError(
            "Cannot hoist ProcessGroup input for a DeviceMesh axis with name None."
        )
    return axis


def _process_group_axis_from_parallel_dims(
    process_group: dist.ProcessGroup,
    parallel_dims: ParallelDims | None,
) -> str:
    if parallel_dims is None:
        raise ValueError(
            "Cannot infer the mesh axis for a captured ProcessGroup without "
            "parallel_dims."
        )

    process_group_name = getattr(process_group, "group_name", None)
    for axis, mesh in parallel_dims.get_all_one_dimensional_meshes().items():
        candidate = mesh.get_group()
        if candidate is process_group:
            return axis
        if (
            process_group_name is not None
            and getattr(candidate, "group_name", None) == process_group_name
        ):
            return axis

    raise ValueError(
        "Captured ProcessGroup does not match any one-dimensional mesh axis "
        "in parallel_dims."
    )


def _is_mesh_get_process_group_node(node: torch.fx.Node) -> bool:
    return (
        node.op == "call_function"
        and node.target == torch.ops._dtensor.mesh_get_process_group.default
    )


def _process_group_from_mesh_get_process_group_node(
    gm: torch.fx.GraphModule,
    node: torch.fx.Node,
) -> dist.ProcessGroup:
    process_group = node.meta.get("val")
    if isinstance(process_group, dist.ProcessGroup):
        return process_group

    mesh_arg, mesh_axis = node.args
    if (
        isinstance(mesh_arg, torch.fx.Node)
        and mesh_arg.op == "get_attr"
        and isinstance(mesh_arg.target, str)
        and isinstance(mesh_axis, int)
    ):
        mesh = _get_graph_attr(gm, mesh_arg.target)
        return mesh.get_group(mesh_axis)

    raise ValueError("Could not recover ProcessGroup from mesh_get_process_group node.")


def _axis_from_mesh_get_process_group_node(
    gm: torch.fx.GraphModule,
    node: torch.fx.Node,
    parallel_dims: ParallelDims | None,
) -> str:
    mesh_arg, mesh_axis = node.args
    if (
        isinstance(mesh_arg, torch.fx.Node)
        and mesh_arg.op == "get_attr"
        and isinstance(mesh_arg.target, str)
        and isinstance(mesh_axis, int)
    ):
        return _process_group_axis_from_mesh(
            _get_graph_attr(gm, mesh_arg.target), mesh_axis
        )

    process_group = _process_group_from_mesh_get_process_group_node(gm, node)
    return _process_group_axis_from_parallel_dims(process_group, parallel_dims)


def _first_non_placeholder(graph: torch.fx.Graph) -> torch.fx.Node | None:
    for node in graph.nodes:
        if node.op != "placeholder":
            return node
    return None


def _last_placeholder(graph: torch.fx.Graph) -> torch.fx.Node | None:
    last_placeholder = None
    for node in graph.nodes:
        if node.op != "placeholder":
            break
        last_placeholder = node
    return last_placeholder


def _is_all_to_all_single_node(node: torch.fx.Node) -> bool:
    return (
        node.op == "call_function"
        and node.target == torch.ops._c10d_functional.all_to_all_single.default
    )


def _validate_permute_tensor_split_sizes(split_sizes: Any) -> int | None:
    if not (
        isinstance(split_sizes, (list, tuple))
        and split_sizes
        and all(isinstance(size, int) for size in split_sizes)
    ):
        return None

    nonzero_sizes = [size for size in split_sizes if size != 0]
    if len(nonzero_sizes) != 1:
        return None
    return nonzero_sizes[0]


def _runtime_coordinate_source_for_axis(
    axis: str,
    parallel_dims: ParallelDims | None,
) -> tuple[DeviceMesh, int]:
    if parallel_dims is None:
        raise ValueError(
            f"Cannot derive runtime coordinate source for mesh axis {axis!r} "
            "without parallel_dims."
        )

    if axis == "cp":
        # CP split sizes need the coordinate in the full dataloading mesh.
        # The 1D CP mesh seen during a one-rank trace only describes the
        # tracing rank's local slice.
        global_meshes = getattr(parallel_dims, "_global_meshes", {}) or {}
        global_mesh = global_meshes.get("dataloading")
        mesh_axis_names = getattr(global_mesh, "mesh_dim_names", None)
        if (
            global_mesh is not None
            and mesh_axis_names is not None
            and axis in mesh_axis_names
        ):
            return global_mesh, mesh_axis_names.index(axis)

    mesh = parallel_dims.get_mesh(axis)
    return mesh, 0


def _full_mesh_tensor_for_runtime_coordinate(mesh: DeviceMesh) -> torch.Tensor:
    rank_map = torch.tensor(mesh._rank_map.tolist(), device="cpu", dtype=torch.int)
    return mesh._layout.remap_to_tensor(rank_map)


def hoist_process_group_inputs(
    traced_result: TracedResult,
    *,
    parallel_dims: ParallelDims | None = None,
) -> None:
    """Replace captured ProcessGroups with explicit graph inputs.

    ``compile_on_one_rank`` traces DeviceMesh-to-ProcessGroup resolution via
    ``_dtensor.mesh_get_process_group``. Before serializing or compiling the
    FX graph, hoist those ProcessGroups to placeholders so Inductor and
    GraphPickler see them as opaque inputs rather than captured attributes.
    """
    _register_coor_ops()

    gm = traced_result.gm
    graph = gm.graph
    axis_to_placeholder: dict[str, torch.fx.Node] = {}
    axis_to_coordinate_source: dict[str, tuple[DeviceMesh, int]] = {}
    axis_to_coordinate_node: dict[str, torch.fx.Node] = {}
    process_group_inputs: list[Any] = []
    process_group_input_specs: list[ProcessGroupInputSpec] = []
    attrs_to_delete: set[str] = set()
    last_placeholder = _last_placeholder(graph)
    first_non_placeholder = _first_non_placeholder(graph)

    def get_or_create_placeholder(
        axis: str,
        process_group: dist.ProcessGroup,
    ) -> torch.fx.Node:
        nonlocal last_placeholder
        if axis in axis_to_placeholder:
            return axis_to_placeholder[axis]

        placeholder_name = f"_process_group_{_safe_placeholder_suffix(axis)}"
        if last_placeholder is not None:
            insert_ctx = graph.inserting_after(last_placeholder)
        elif first_non_placeholder is not None:
            insert_ctx = graph.inserting_before(first_non_placeholder)
        else:
            insert_ctx = graph.inserting_before(None)

        with insert_ctx:
            placeholder = graph.placeholder(placeholder_name)
        placeholder.meta["val"] = process_group
        placeholder.meta["mesh_axis"] = axis
        axis_to_placeholder[axis] = placeholder
        last_placeholder = placeholder
        process_group_inputs.append(process_group)
        process_group_input_specs.append(ProcessGroupInputSpec(axis=axis))
        return placeholder

    def record_coordinate_source_for_mesh_get(
        axis: str,
        node: torch.fx.Node,
    ) -> None:
        mesh_arg, mesh_axis = node.args
        if (
            isinstance(mesh_arg, torch.fx.Node)
            and mesh_arg.op == "get_attr"
            and isinstance(mesh_arg.target, str)
            and isinstance(mesh_axis, int)
        ):
            axis_to_coordinate_source.setdefault(
                axis, (_get_graph_attr(gm, mesh_arg.target), mesh_axis)
            )

    def copy_region_meta(dst: torch.fx.Node, src: torch.fx.Node) -> None:
        if "autograd_backward" in src.meta:
            dst.meta["autograd_backward"] = src.meta["autograd_backward"]
        custom = src.meta.get("custom")
        if custom is not None:
            dst.meta["custom"] = custom.copy()

    def get_or_create_coordinate_node(
        axis: str,
        before_node: torch.fx.Node,
    ) -> torch.fx.Node:
        if axis in axis_to_coordinate_node:
            return axis_to_coordinate_node[axis]

        if axis == "cp" and parallel_dims is not None:
            mesh, mesh_axis = _runtime_coordinate_source_for_axis(axis, parallel_dims)
        elif axis in axis_to_coordinate_source:
            mesh, mesh_axis = axis_to_coordinate_source[axis]
        else:
            mesh, mesh_axis = _runtime_coordinate_source_for_axis(axis, parallel_dims)
        full_mesh = _full_mesh_tensor_for_runtime_coordinate(mesh)
        attr_name = f"_runtime_{_safe_placeholder_suffix(axis)}_full_mesh"
        gm.register_buffer(attr_name, full_mesh, persistent=False)

        with graph.inserting_before(before_node):
            full_mesh_node = graph.get_attr(attr_name)
            full_mesh_node.meta["val"] = full_mesh
            copy_region_meta(full_mesh_node, before_node)
            coordinate_node = graph.call_function(
                torch.ops.device_mesh._runtime_compute_coordinate_on_dim.default,
                (full_mesh_node, mesh_axis),
            )
            copy_region_meta(coordinate_node, before_node)
        axis_to_coordinate_node[axis] = coordinate_node
        return coordinate_node

    def runtime_permute_tensor_split_sizes(
        rank_node: torch.fx.Node,
        *,
        kind: str,
        numel: int,
        group_size: int,
        before_node: torch.fx.Node,
    ) -> list[torch.fx.Node]:
        split_sizes: list[torch.fx.Node] = []
        with graph.inserting_before(before_node):
            for split_index in range(group_size):
                if kind == "input":
                    expected_rank = (split_index - 1) % group_size
                else:
                    expected_rank = (split_index + 1) % group_size
                rank_matches = graph.call_function(
                    operator.eq,
                    (rank_node, expected_rank),
                )
                copy_region_meta(rank_matches, before_node)
                split_size = graph.call_function(
                    torch.sym_ite, (rank_matches, numel, 0)
                )
                copy_region_meta(split_size, before_node)
                split_sizes.append(split_size)
        return split_sizes

    for node in list(graph.nodes):
        if _is_mesh_get_process_group_node(node):
            process_group = _process_group_from_mesh_get_process_group_node(gm, node)
            axis = _axis_from_mesh_get_process_group_node(gm, node, parallel_dims)
            record_coordinate_source_for_mesh_get(axis, node)
            placeholder = get_or_create_placeholder(axis, process_group)
            node.replace_all_uses_with(placeholder)
            graph.erase_node(node)
            continue

        if (
            node.op == "get_attr"
            and isinstance(node.target, str)
            and isinstance(_get_graph_attr(gm, node.target), dist.ProcessGroup)
        ):
            process_group = _get_graph_attr(gm, node.target)
            axis = _process_group_axis_from_parallel_dims(process_group, parallel_dims)
            axis_to_coordinate_source.setdefault(
                axis, _runtime_coordinate_source_for_axis(axis, parallel_dims)
            )
            placeholder = get_or_create_placeholder(axis, process_group)
            node.replace_all_uses_with(placeholder)
            graph.erase_node(node)
            attrs_to_delete.add(node.target)

    for node in list(graph.nodes):
        if not _is_all_to_all_single_node(node):
            continue
        if len(node.args) < 4:
            continue

        group_arg = node.args[3]
        axis = (
            group_arg.meta.get("mesh_axis")
            if isinstance(group_arg, torch.fx.Node)
            else None
        )
        if axis != "cp":
            continue

        args = list(node.args)
        rank_node = get_or_create_coordinate_node(axis, node)
        for arg_index, kind in ((1, "output"), (2, "input")):
            split_sizes = args[arg_index]
            numel = _validate_permute_tensor_split_sizes(split_sizes)
            if numel is None:
                raise ValueError(
                    "Cannot rewrite CP all_to_all split sizes with unsupported "
                    f"pattern: {split_sizes!r}"
                )
            args[arg_index] = runtime_permute_tensor_split_sizes(
                rank_node,
                kind=kind,
                numel=numel,
                group_size=len(split_sizes),
                before_node=node,
            )
        node.args = tuple(args)

    if not process_group_inputs:
        return

    graph.eliminate_dead_code()
    graph.lint()
    gm.recompile()

    live_get_attrs = {
        node.target
        for node in graph.nodes
        if node.op == "get_attr" and isinstance(node.target, str)
    }
    for target in attrs_to_delete - live_get_attrs:
        if "." not in target and hasattr(gm, target):
            delattr(gm, target)

    remaining_process_group_attrs = [
        name for name, attr in vars(gm).items() if isinstance(attr, dist.ProcessGroup)
    ]
    if remaining_process_group_attrs:
        raise ValueError(
            "ProcessGroup attrs remained after hoisting: "
            f"{remaining_process_group_attrs}"
        )

    traced_result.example_inputs = (
        *traced_result.example_inputs,
        *process_group_inputs,
    )
    traced_result.process_group_inputs = (
        *traced_result.process_group_inputs,
        *process_group_inputs,
    )
    traced_result.process_group_input_specs = (
        *traced_result.process_group_input_specs,
        *process_group_input_specs,
    )


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
    process_group_input_specs: tuple[ProcessGroupInputSpec, ...] = field(
        default_factory=tuple
    )
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
        _register_coor_ops()

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
            process_group_input_specs=traced_result.process_group_input_specs,
            config_fingerprint=config_fingerprint or ConfigFingerprint(""),
        )

    def to_traced_result(
        self,
        process_group_inputs: tuple[Any, ...] = (),
    ) -> TracedResult:
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

        process_group_input_specs = getattr(self, "process_group_input_specs", ())
        if len(process_group_inputs) != len(process_group_input_specs):
            raise ValueError(
                "Expected "
                f"{len(process_group_input_specs)} ProcessGroup input(s) for "
                f"the precompiled FX trace, got {len(process_group_inputs)}."
            )

        return TracedResult(
            gm=gm,
            example_inputs=process_group_inputs,
            state_fqns=self.state_fqns,
            num_flat_inputs=self.num_flat_inputs,
            input_subclass_layouts=self.input_subclass_layouts,
            num_flat_outputs=self.num_flat_outputs,
            output_subclass_layouts=self.output_subclass_layouts,
            output_spec=self.output_spec,
            tensor_input_indices=self.tensor_input_indices,
            process_group_inputs=process_group_inputs,
            process_group_input_specs=process_group_input_specs,
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
    parallel_dims: ParallelDims | None = None,
) -> TracedResult:
    """Load a precompiled aot_fx_trace artifact.

    Returns a TracedResult with the deserialized GraphModule and
    metadata. The caller uses this with run_traced_train_step to
    execute the graph (same path as non-precompiled aot_fx_trace).

    ProcessGroups are explicit graph inputs. The artifact records their
    mesh axes, and load resolves those axes to the live runtime
    ProcessGroups from the caller-provided ParallelDims.
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

    process_group_inputs: tuple[Any, ...] = ()
    process_group_input_specs = getattr(artifact, "process_group_input_specs", ())
    if process_group_input_specs:
        if parallel_dims is None:
            raise ValueError(
                "Loading an artifact with hoisted ProcessGroup inputs requires "
                "parallel_dims to provide runtime process groups."
            )
        process_group_inputs = tuple(
            parallel_dims.get_mesh(spec.axis).get_group()
            for spec in process_group_input_specs
        )

    return artifact.to_traced_result(process_group_inputs)
