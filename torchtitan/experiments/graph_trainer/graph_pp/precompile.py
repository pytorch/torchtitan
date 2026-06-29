# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""GraphPP precompile serialization.

GraphPP precompile saves one rank-local bundle per pipeline-parallel rank.
Each bundle holds the stage-local graph callables (the partitioned
forward/backward/FSDP/dI-dW FX graphs) plus the StageGraphMeta
calling-convention record for every stage that PP rank owns.

This mirrors the non-PP ``PrecompiledFxTraceArtifact``: load is
deserialize-and-run, with no Inductor reinvocation at load and no placeholder
``meta["val"]`` needed at load. With ``--compile.enable`` the saved graphs have
regional/full Inductor kernels baked in (the runner executes them directly);
without it the saved graphs are the uncompiled partitioned FX graphs, which the
runner also executes directly -- the saved compiled state is recorded per stage
and restored on load. Each FX graph is serialized with ``GraphPickler`` (not
plain pickle) so SymInt expressions and DeviceMesh graph inputs survive instead
of being baked into rank-specific constants.

What precompile removes per rank: make_fx tracing, partitioning, FSDP/dI-dW
splitting, the metadata-preserving GraphTrainer passes, and Inductor -- all
captured in the saved compiled graphs.

Schedules that fuse a forward and a backward through ``OVERLAP_F_B``
(currently only ``DualPipeV``) build a multiplexed callable from the
*uncompiled* fw/full_bw graphs, which the compiled bundles do not carry, so
precompile is not yet supported for them. ``Interleaved1F1B`` and
``ZBVZeroBubble`` (FULL_BACKWARD / dI-dW, no multiplex) are supported.
"""

from __future__ import annotations

import dataclasses
import hashlib
import pickle
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, TYPE_CHECKING

import torch

from torchtitan.experiments.graph_trainer.precompile import (
    _register_coor_ops,
    _validate_config_fingerprint,
    ConfigFingerprint,
)
from torchtitan.experiments.graph_trainer.storage import StorageAdapter
from torchtitan.tools.logging import logger

if TYPE_CHECKING:
    import torch.fx as fx

    from torchtitan.config import (
        Configurable,
        DebugConfig,
        ParallelismConfig,
        TrainingConfig,
    )
    from torchtitan.distributed import ParallelDims
    from torchtitan.experiments.graph_trainer.configs import GraphTrainerCompileConfig
    from torchtitan.experiments.graph_trainer.graph_pp.runner import (
        GraphCallables,
        GraphPipelineStage,
        StageGraphMeta,
        StageTraceSpec,
    )

# Callable slots serialized per stage. fw and full_bw are required (to_graph_callables
# raises if either is missing); the rest are optional and simply absent from the dict
# when not built. The bundle keys callables by name, so this order has no meaning.
_CALLABLE_SLOTS = ("fw", "full_bw", "bw_di", "bw_dw", "unshard", "reduce_grad")


def graph_pp_rank_artifact_key(pp_rank: int) -> str:
    """Flat storage key for one pipeline-parallel rank's GraphPP bundle.

    Keys are flat opaque strings with no path separators (see
    ``StorageAdapter``). Each torchrun rank loads the bundle for its own PP
    coordinate; CooR makes the saved graphs agnostic to the dp/tp/cp/ep
    coordinate, so one bundle per PP rank is sufficient for all ranks sharing
    that PP coordinate.
    """
    return f"graph_pp_pp{pp_rank}"


def ensure_schedule_precompilable(schedule_name: str) -> None:
    """Raise if the schedule needs OVERLAP_F_B multiplexing (unsupported).

    DualPipeV builds a multiplexed forward+backward callable from the
    *uncompiled* fw/full_bw graphs at runtime, which the compiled precompile
    bundle does not carry. Interleaved1F1B and ZBVZeroBubble do not multiplex
    and are supported. The runtime would also fail (it cannot multiplex
    already-compiled graphs); this guard makes the reason explicit and is
    applied on both the save and load paths.
    """
    from torch.distributed.pipelining.schedules import get_schedule_class

    try:
        from torch.distributed.pipelining.schedules import ScheduleDualPipeV
    except ImportError:
        ScheduleDualPipeV = None

    if ScheduleDualPipeV is not None and (
        get_schedule_class(schedule_name) is ScheduleDualPipeV
    ):
        raise NotImplementedError(
            "GraphPP precompile does not yet support OVERLAP_F_B schedules "
            f"(got '{schedule_name}'). Use Interleaved1F1B or ZBVZeroBubble, or "
            "run without --compile.precompile_artifact_dir."
        )


def make_distributed_objects_deepcopy_safe() -> None:
    """Make runtime distributed singletons share (not copy) on deepcopy.

    Under compile-on-one-rank the traced stage graph carries TorchBind
    ProcessGroup and DeviceMesh constants. GraphPP's partition step deepcopies
    the joint GraphModule, but a TorchBind ProcessGroup has no
    __getstate__/__deepcopy__ and raises. These are runtime singletons that must
    be shared, not deep-copied, so register an identity deepcopy for them. Only
    the save path needs this (load installs prebuilt callables without
    partitioning).
    """
    import copy

    from torch.distributed.device_mesh import DeviceMesh

    copy._deepcopy_dispatch[torch.ScriptObject] = copy._deepcopy_atomic
    copy._deepcopy_dispatch[DeviceMesh] = copy._deepcopy_atomic
    # _deepcopy_dispatch is keyed by exact type, so also give the ScriptObject
    # base class an identity __deepcopy__ to cover any TorchBind subclass.
    torch.ScriptObject.__deepcopy__ = lambda self, memo=None: self


def compute_graph_pp_fingerprint(
    model_parts: list[torch.nn.Module],
    compile_config: "GraphTrainerCompileConfig",
    parallel_dims: "ParallelDims",
    *,
    schedule_name: str,
    parallelism: "ParallelismConfig",
    training: "TrainingConfig",
    loss_config: "Configurable.Config",
    debug_config: "DebugConfig",
) -> ConfigFingerprint:
    """Fingerprint everything that affects one PP rank's saved stage graphs.

    Hashes the per-part parameter/buffer shapes and dtypes, the parallelism
    dimensions, every compile setting that flows into the GraphPP pass pipeline
    or Inductor (so a config that changes the baked graphs forces a clear
    mismatch instead of silently loading stale numerics), the pipeline schedule
    name, the torch version, and the CUDA capability. The schedule name matters
    because it decides stage assignment and OVERLAP_F_B pairs;
    ``enable_async_tensor_parallel`` (adds a graph pass) and
    ``fsdp_reshard_after_forward`` (feeds the default memory policy) live on the
    parallelism config (not on ``ParallelDims``) and are hashed too. The
    microbatch shape (``pipeline_parallel_microbatch_size`` x ``seq_len``) and
    the resolved global batch size are hashed because they are baked into the
    traced graphs (static shapes) and the ``global_valid_tokens`` loss-divisor
    constant; without them a stale-shape or wrong-loss-scale artifact would load
    silently. ``mixed_precision_param``/``mixed_precision_reduce`` are hashed
    because they set the FSDP all-gather/reduce-scatter dtypes baked into the
    ``unshard``/``reduce_grad`` callables (and they do not change the stored
    param dtype, so the param-shape loop above does not cover them). The loss
    config type and fields (e.g. ``num_chunks``) are hashed because the last
    stage bakes the chunked-loss structure (static chunk shapes and an unrolled
    chunk count) into its saved graph. The deterministic-mode flags are hashed
    because the save process sets ``use_deterministic_algorithms`` from
    ``debug.deterministic`` before tracing, and the compiled backward captures
    that mode and asserts it at runtime; a save/load mismatch would otherwise hit
    that assert (or run the wrong kernels) instead of a clear fingerprint error.
    Returns the first 16 chars of a SHA-256 hex digest.

    Computed over only the stage submodules a single PP rank owns, so the
    save side (which builds all stages in one process) must fingerprint each
    rank's stages separately and the load side (which only has its own rank's
    stages) produces a matching digest.
    """
    h = hashlib.sha256()

    for part_index, part in enumerate(model_parts):
        for name, param in part.named_parameters(remove_duplicate=False):
            h.update(
                f"part{part_index}:param:{name}:"
                f"{list(param.shape)}:{param.dtype}\n".encode()
            )
        for name, buf in part.named_buffers(remove_duplicate=False):
            h.update(
                f"part{part_index}:buffer:{name}:"
                f"{list(buf.shape)}:{buf.dtype}\n".encode()
            )

    for f in dataclasses.fields(parallel_dims):
        if not f.name.startswith("_"):
            h.update(f"parallel:{f.name}:{getattr(parallel_dims, f.name)}\n".encode())

    # Every compile setting that changes the traced/partitioned/compiled graph.
    for field_name in (
        "mode",
        "backend",
        "inductor_compilation",
        "enable",
        "enable_passes",
        "numerics_changing_optim",
        "memory_policy",
        "cpu_offload_prefetch_n_layers",
        "cpu_offload_defer_n_layers",
        "cpu_offload_budget_gb",
    ):
        h.update(
            f"compile:{field_name}:{getattr(compile_config, field_name)}\n".encode()
        )
    h.update(f"compile:passes:{list(compile_config.passes)}\n".encode())
    h.update(f"compile:disable_passes:{list(compile_config.disable_passes)}\n".encode())
    # Parallelism settings that add a graph pass or change the memory policy but
    # do not live on ParallelDims (so the field loop above misses them).
    h.update(
        "parallelism:enable_async_tensor_parallel:"
        f"{parallelism.enable_async_tensor_parallel}\n".encode()
    )
    h.update(
        "parallelism:fsdp_reshard_after_forward:"
        f"{parallelism.fsdp_reshard_after_forward}\n".encode()
    )
    # Trace-shape and loss-divisor inputs baked into the saved graphs: the
    # microbatch shape and the resolved global batch size (which sets the
    # global_valid_tokens constant). These are not visible in param shapes, so
    # a mismatch would otherwise load stale graphs / wrong loss scaling silently.
    h.update(
        "parallelism:pipeline_parallel_microbatch_size:"
        f"{parallelism.pipeline_parallel_microbatch_size}\n".encode()
    )
    h.update(f"training:seq_len:{training.seq_len}\n".encode())
    h.update(f"training:local_batch_size:{training.local_batch_size}\n".encode())
    h.update(f"training:global_batch_size:{training.global_batch_size}\n".encode())
    # FSDP all-gather/reduce-scatter dtypes baked into unshard/reduce_grad.
    # These do not change the stored param dtype (params stay in their declared
    # dtype; the cast happens at gather time), so the param loop above misses them.
    h.update(
        f"training:mixed_precision_param:{training.mixed_precision_param}\n".encode()
    )
    h.update(
        f"training:mixed_precision_reduce:{training.mixed_precision_reduce}\n".encode()
    )
    # Loss config: the last stage bakes the (chunked) loss structure into its
    # saved graph -- the loss type and fields like num_chunks set static chunk
    # shapes and the unrolled chunk count, none of which appear in param shapes.
    h.update(f"loss:type:{type(loss_config).__qualname__}\n".encode())
    for f in dataclasses.fields(loss_config):
        if not f.name.startswith("_"):
            h.update(f"loss:{f.name}:{getattr(loss_config, f.name)}\n".encode())
    # Deterministic mode is set before tracing and captured by the compiled
    # backward; a save/load mismatch must invalidate the artifact loudly.
    h.update(f"debug:deterministic:{debug_config.deterministic}\n".encode())
    h.update(
        f"debug:deterministic_warn_only:{debug_config.deterministic_warn_only}\n".encode()
    )
    h.update(f"schedule:{schedule_name}\n".encode())
    h.update(f"torch_version:{torch.__version__}\n".encode())

    if torch.cuda.is_available():
        capability = torch.cuda.get_device_capability()
        h.update(f"cuda_capability:{capability}\n".encode())

    return ConfigFingerprint(h.hexdigest()[:16])


def _serialize_gm(gm: "fx.GraphModule") -> bytes:
    """Serialize one extracted GraphPP callable with GraphPickler.

    Uses the same options as the non-PP fx_trace artifact: no op filter (so
    distributed collective ops and the raw flex_attention HOP survive) and the
    distributed node-metadata filter (so opaque DeviceMesh/ProcessGroup values
    in node.meta are stripped). SymInt expressions are preserved instead of
    being baked into rank-specific constants.
    """
    from torch.fx._graph_pickler import GraphPickler, Options

    from torchtitan.experiments.graph_trainer.inductor_passes import (
        _node_metadata_key_filter_distributed,
    )

    # Compile-on-one-rank can bake opaque TorchBind ProcessGroup get_attr
    # constants into the graph (currently only the EP MoE all-to-all path, via
    # c10d.alltoall_). GraphPickler cannot serialize them and they have no
    # group-name accessor to re-resolve from, so reject early with a precise
    # message. Non-EP collectives use functional ops keyed by rank-agnostic
    # group-name strings and serialize fine.
    _reject_opaque_process_group_constants(gm)
    return GraphPickler.dumps(
        gm,
        Options(
            ops_filter=None,
            node_metadata_key_filter=_node_metadata_key_filter_distributed,
        ),
    )


def _is_torchbind_process_group(value: Any) -> bool:
    if not isinstance(value, torch.ScriptObject):
        return False
    try:
        return value._type().qualified_name() == (
            "__torch__.torch.classes.c10d.ProcessGroup"
        )
    except Exception:
        return False


def _reject_opaque_process_group_constants(gm: "fx.GraphModule") -> None:
    """Fail clearly if the graph bakes an opaque TorchBind ProcessGroup constant.

    These appear today only for expert parallelism (the MoE all-to-all uses the
    non-functional ``c10d.alltoall_`` with a ProcessGroup object). The TorchBind
    PG is opaque (no group-name accessor) and rank-specific, so it cannot be
    serialized into a rank-agnostic bundle yet. Making EP precompile work needs
    the EP group re-resolved from the live mesh at load (the EP collective nodes
    carry ``meta['custom']['EP']``) -- tracked as follow-up.
    """
    for node in gm.graph.nodes:
        if node.op != "get_attr":
            continue
        try:
            attr = getattr(gm, node.target)
        except AttributeError:
            continue
        if _is_torchbind_process_group(attr):
            raise NotImplementedError(
                "GraphPP precompile cannot yet serialize the opaque TorchBind "
                "ProcessGroup baked by expert parallelism (MoE all-to-all). Run "
                "the save with --parallelism.expert_parallel_degree 1, or add EP "
                "precompile support (re-resolve the EP group from the live mesh "
                "at load)."
            )


def _deserialize_gm(data: bytes) -> "fx.GraphModule":
    """Deserialize one GraphPP callable under a fresh FakeTensorMode.

    GraphPickler.loads requires a FakeTensorMode with a ShapeEnv to
    reconstruct tensor metadata and SymInt expressions during deserialization;
    a fresh one is used per callable. Placeholder meta["val"] is intentionally
    not restored (load never recompiles). CooR custom ops must be registered
    before GraphPickler.loads.
    """
    _register_coor_ops()

    from torch._subclasses import FakeTensorMode
    from torch.fx._graph_pickler import GraphPickler

    fake_mode = FakeTensorMode(
        allow_non_fake_inputs=True,
        shape_env=torch.fx.experimental.symbolic_shapes.ShapeEnv(),
    )
    gm = GraphPickler.loads(data, fake_mode)
    gm.recompile()
    return gm


@dataclass
class SerializedStageBundle:
    """Serialized form of one stage's GraphPP graph bundle.

    ``serialized_callables`` maps a callable slot name (fw / full_bw /
    bw_di / bw_dw / unshard / reduce_grad) to its GraphPickler bytes; absent
    optional callables are simply missing from the dict. ``graph_meta`` and
    ``trace_spec`` are plain-picklable dataclasses stored verbatim -- the
    fw/bw graphs pair purely by the names recorded in graph_meta, so it must
    round-trip exactly alongside the graphs.

    ``compiled`` records whether Inductor kernels are baked into the saved
    graphs (true when ``--compile.enable``). It is restored on load so the
    downstream per-stage compile is skipped for compiled bundles and remains a
    no-op (gated on ``enable``) for uncompiled ones, matching live training
    either way. Neither case recompiles at load.
    """

    stage_index: int
    is_first: bool
    is_last: bool
    compiled: bool
    serialized_callables: dict[str, bytes]
    graph_meta: "StageGraphMeta"
    trace_spec: "StageTraceSpec"

    @classmethod
    def from_stage(cls, stage: "GraphPipelineStage") -> "SerializedStageBundle":
        if stage.graph_callables is None or stage.graph_meta is None:
            raise ValueError(
                "GraphPP precompile cannot serialize stage "
                f"{stage.stage_index}: graph bundle was not built."
            )

        serialized_callables: dict[str, bytes] = {}
        for slot in _CALLABLE_SLOTS:
            gm = getattr(stage.graph_callables, slot)
            if gm is not None:
                serialized_callables[slot] = _serialize_gm(gm)

        return cls(
            stage_index=stage.stage_index,
            is_first=stage.is_first,
            is_last=stage.is_last,
            compiled=stage._graph_pp_callables_compiled,
            serialized_callables=serialized_callables,
            graph_meta=stage.graph_meta,
            trace_spec=stage.trace_spec,
        )

    def to_graph_callables(self) -> "GraphCallables":
        from torchtitan.experiments.graph_trainer.graph_pp.runner import GraphCallables

        deserialized = {
            slot: _deserialize_gm(data)
            for slot, data in self.serialized_callables.items()
        }
        if "fw" not in deserialized or "full_bw" not in deserialized:
            raise ValueError(
                "GraphPP precompile bundle for stage "
                f"{self.stage_index} is missing required fw/full_bw callables."
            )
        return GraphCallables(
            fw=deserialized["fw"],
            full_bw=deserialized["full_bw"],
            bw_di=deserialized.get("bw_di"),
            bw_dw=deserialized.get("bw_dw"),
            unshard=deserialized.get("unshard"),
            reduce_grad=deserialized.get("reduce_grad"),
        )


@dataclass
class SerializedRankBundle:
    """All GraphPP stage bundles owned by a single pipeline-parallel rank."""

    pp_rank: int
    schedule_name: str
    stages: list[SerializedStageBundle]
    config_fingerprint: ConfigFingerprint = field(
        default_factory=lambda: ConfigFingerprint("")
    )

    @classmethod
    def from_stages(
        cls,
        stages: list["GraphPipelineStage"],
        *,
        pp_rank: int,
        schedule_name: str,
        config_fingerprint: ConfigFingerprint | None = None,
    ) -> "SerializedRankBundle":
        return cls(
            pp_rank=pp_rank,
            schedule_name=schedule_name,
            stages=[SerializedStageBundle.from_stage(stage) for stage in stages],
            config_fingerprint=config_fingerprint or ConfigFingerprint(""),
        )


def save_graph_pp_rank_bundle(
    stages: list["GraphPipelineStage"],
    storage: StorageAdapter,
    *,
    pp_rank: int,
    schedule_name: str,
    config_fingerprint: ConfigFingerprint | None = None,
) -> str:
    """Serialize and persist one PP rank's compiled stage graph bundles."""
    bundle = SerializedRankBundle.from_stages(
        stages,
        pp_rank=pp_rank,
        schedule_name=schedule_name,
        config_fingerprint=config_fingerprint,
    )
    data = pickle.dumps(bundle)
    path = storage.save(graph_pp_rank_artifact_key(pp_rank), data)
    logger.info(
        "GraphPP precompile bundle saved: pp_rank=%s, stages=%s, "
        "size=%s bytes, fingerprint=%s, path=%s",
        pp_rank,
        [s.stage_index for s in bundle.stages],
        len(data),
        config_fingerprint,
        path,
    )
    return path


def load_graph_pp_rank_bundle(
    storage: StorageAdapter,
    *,
    pp_rank: int,
    expected_fingerprint: ConfigFingerprint,
) -> SerializedRankBundle:
    """Load one PP rank's GraphPP bundle and validate its fingerprint."""
    data = storage.load(graph_pp_rank_artifact_key(pp_rank))
    bundle: SerializedRankBundle = pickle.loads(data)
    _validate_config_fingerprint(bundle.config_fingerprint, expected_fingerprint)
    logger.info(
        "GraphPP precompile bundle loaded: pp_rank=%s, stages=%s, fingerprint=%s",
        bundle.pp_rank,
        [s.stage_index for s in bundle.stages],
        bundle.config_fingerprint,
    )
    return bundle


def build_graph_pp_stage_loader(
    bundle: SerializedRankBundle,
) -> Callable[["GraphPipelineStage"], None]:
    """Build a stage_bundle_loader closure for build_graph_pp_graph_bundles.

    The returned callable deserializes and installs the loaded callables and
    metadata onto a stage and restores its compiled flag, so the shared
    downstream path skips tracing. For compiled bundles (Inductor kernels baked
    in) the per-stage compile is skipped; for uncompiled bundles it stays a
    no-op (gated on ``enable``). Neither recompiles at load.
    """
    from torchtitan.experiments.graph_trainer.graph_pp.runner import (
        _annotate_graph_pp_callables,
        GraphPipelineStage,
    )

    stage_bundles = {s.stage_index: s for s in bundle.stages}

    def loader(stage: GraphPipelineStage) -> None:
        serialized = stage_bundles.get(stage.stage_index)
        if serialized is None:
            raise ValueError(
                "GraphPP precompile bundle for pp_rank "
                f"{bundle.pp_rank} has no entry for stage {stage.stage_index}. "
                "The artifact was saved with a different stage assignment; "
                "delete it and re-run precompile."
            )
        callables = serialized.to_graph_callables()
        _annotate_graph_pp_callables(callables, stage_index=stage.stage_index)
        stage.graph_callables = callables
        stage.graph_meta = serialized.graph_meta
        stage.trace_spec = serialized.trace_spec
        stage._graph_pp_callables_compiled = serialized.compiled

    return loader
