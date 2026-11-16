# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Data parallelism with FlexShard in place of FSDP2's ``fully_shard``.

With ``parallelism.fsdp_backend=FlexShardConfig()``, ``apply_fsdp_to_decoder``
gives FlexShard (meta-pytorch/flex_shard) one ``fsdp2_compatible`` bucket for
each group it would pass to ``fully_shard``, with the same placements, mixed
precision and reshard policy. The buckets' collectives then match FSDP2's byte
for byte, so training from the same parameters is bitwise identical.

With expert parallelism, ``fully_shard`` gives an MoE block one group per mesh,
and the block gets one bucket per group: its routed experts on ``edp_shard``,
the rest on ``dp_shard``. EP has already split each routed-expert parameter over
``ep``, and torchtitan declares that split as the parameter's outer layout.

FlexShard parameters and grads are plain local shards, with no DTensor.
``init_param`` initializes each parameter as one device would, while each of
FSDP2's DTensor shards draws its own random values. ``flex_shard_grad_norm_mesh``
gives the mesh a grad's norm is reduced over. Two more differences change only
memory and communication. FSDP2 skips the embedding's backward all-gather
(``set_unshard_in_backward``), which FlexShard can't yet, so it gathers the
embedding again in backward, with the same values. And an MoE block's expert
bucket hooks the routed experts, not the block, so FlexShard reshards them as
soon as the routed experts finish forward or backward, while FSDP2 keeps the
block's expert group unsharded until the whole block finishes.
``flex_shard_partial_backwards`` keeps ``lm_head``'s bucket unsharded across
``ChunkedLossWrapper``'s chunks, as ``fsdp_partial_backwards`` keeps FSDP2's
group. FlexShard is imported only when it is selected.

FlexShard's state dicts hold plain local shards too, so each declares where its
shards sit in the full tensors (DCP's ``CheckpointableTensor``): the model's,
through ``declare_state_dict_layouts``, and the optimizer's, through
``declare_optimizer_state_layouts``. DCP then saves the same checkpoints as for
FSDP2's DTensors, and either backend loads the other's.
``load_hf_into_local_shards`` loads a Hugging Face checkpoint into those local
shards, each rank reading only its part of each Hugging Face tensor.
"""

import math
import sys
import weakref
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from importlib.util import find_spec
from typing import Any, cast, TYPE_CHECKING

import torch
import torch.distributed.checkpoint as dcp
import torch.nn as nn
from torch.distributed.device_mesh import DeviceMesh

if TYPE_CHECKING:
    from torchtitan.models.base.model import BaseModel
    from torchtitan.models.base.state_dict_adapter import BaseStateDictAdapter
    from torchtitan.training_engine import TrainingEngine

__all__ = [
    "declare_optimizer_state_layouts",
    "declare_state_dict_layouts",
    "flex_shard_grad_norm_mesh",
    "flex_shard_partial_backwards",
    "has_flex_shard_bucket",
    "has_flex_shard_params",
    "init_param",
    "load_hf_into_local_shards",
    "record_buckets",
    "record_grad_norm_mesh",
    "record_param_layouts",
    "reshard_flex_shard",
    "validate_flex_shard_training",
]

# Where each FlexShard parameter's local shard sits in the full parameter, by
# module and parameter name, for init_param. to_empty() replaces FlexShard's
# parameters but keeps the modules.
# TODO: Read each parameter's layout from flex_shard instead, once its
# parameters keep their layouts through to_empty().
_PARAM_LAYOUTS: weakref.WeakKeyDictionary[
    nn.Module, dict[str, Any]
] = weakref.WeakKeyDictionary()

# The mesh to reduce a grad norm over for the parameters of the FlexShard buckets
# on a mesh, when it isn't that mesh, for flex_shard_grad_norm_mesh. Keyed by the
# mesh the buckets hold, which keeps the entry alive as long as the model.
_GRAD_NORM_MESHES: weakref.WeakKeyDictionary[
    DeviceMesh, DeviceMesh
] = weakref.WeakKeyDictionary()

# For flex_shard_partial_backwards, reshard_flex_shard and
# declare_optimizer_state_layouts: the models FlexShard sharded, and for the
# first, the bucket of each module whose FSDP2 group is one bucket, e.g.
# lm_head's, which also holds the final norm. A bucket holds its model, which
# holds the module, so the bucket is referenced weakly to let the entry go.
_FLEX_SHARD_MODELS: weakref.WeakSet[nn.Module] = weakref.WeakSet()
_MODULE_BUCKETS: weakref.WeakKeyDictionary[
    nn.Module, weakref.ref[Any]
] = weakref.WeakKeyDictionary()


def record_param_layouts(model: nn.Module) -> None:
    """Record where each of ``model``'s FlexShard parameters sits in the full
    parameter, for ``init_param``. Call it after ``flex_shard(model)``.

    flex_shard composes each layout with the outer layout the parameter
    declared before ``flex_shard(model)``, such as expert parallelism's split of
    the routed experts.
    """
    # pyrefly: ignore [missing-import]
    from flex_shard import get_flex_shard_global_layouts

    for fqn, layout in get_flex_shard_global_layouts(model).items():
        module_fqn, _, name = fqn.rpartition(".")
        _PARAM_LAYOUTS.setdefault(model.get_submodule(module_fqn), {})[name] = layout


def init_param(
    module: nn.Module,
    name: str,
    param: torch.Tensor,
    init_fn: Callable[[torch.Tensor], object],
) -> None:
    """Initialize ``module``'s parameter ``name``, ``param``, with ``init_fn``.

    A FlexShard parameter is a plain local shard. ``init_fn`` fills a tensor of
    the full parameter's shape, and the shard keeps its part (flex_shard's
    ``init_from_full``). With the same RNG state on every rank, the shards are
    one device's initialization, whatever the sharding. Each call briefly holds
    one full parameter: under expert parallelism, a routed-expert parameter with
    all experts.
    """
    layout = _PARAM_LAYOUTS.get(module, {}).get(name)
    if layout is None:
        init_fn(param)
        return
    # pyrefly: ignore [missing-import]
    from flex_shard import init_from_full

    init_from_full(param, layout, init_fn)


def has_flex_shard_params(module: nn.Module) -> bool:
    """Whether FlexShard sharded any of ``module``'s own parameters, which only
    ``init_param`` initializes as one device would."""
    return module in _PARAM_LAYOUTS


def record_grad_norm_mesh(bucket_mesh: DeviceMesh, mesh: DeviceMesh) -> None:
    """Record that ``flex_shard_grad_norm_mesh`` gives the 1D ``mesh`` for the
    parameters of the FlexShard buckets on ``bucket_mesh``.

    For parameters that an outer sharding split over more ranks before FlexShard
    sharded each part over ``bucket_mesh``, e.g. routed experts under expert
    parallelism.
    """
    # Meshes compare by value. An equal mesh that an earlier model recorded
    # would stay the key, and the entry would go when that model does.
    _GRAD_NORM_MESHES.pop(bucket_mesh, None)
    _GRAD_NORM_MESHES[bucket_mesh] = mesh


def flex_shard_grad_norm_mesh(param: torch.Tensor) -> DeviceMesh | None:
    """The mesh ``param``'s grad norm is reduced over if it's a FlexShard
    parameter, else None.

    A FlexShard parameter and its grad are plain local shards over the mesh of
    their bucket, or over the mesh ``record_grad_norm_mesh`` recorded for it.
    """
    # FlexShard parameters exist only once flex_shard is imported, which
    # happens only when it is selected.
    flex_shard = sys.modules.get("flex_shard")
    mesh = None if flex_shard is None else flex_shard.get_mesh(param)
    return None if mesh is None else _GRAD_NORM_MESHES.get(mesh, mesh)


def record_buckets(model: nn.Module, module_buckets: dict[nn.Module, Any]) -> None:
    """Record that FlexShard sharded ``model``, and the bucket storage of each
    module in ``module_buckets``, for ``flex_shard_partial_backwards``,
    ``reshard_flex_shard`` and ``declare_optimizer_state_layouts``. Call it
    after ``flex_shard(model)``."""
    _FLEX_SHARD_MODELS.add(model)
    for module, bucket in module_buckets.items():
        _MODULE_BUCKETS[module] = weakref.ref(bucket)


def has_flex_shard_bucket(module: nn.Module) -> bool:
    """Whether FlexShard sharded ``module``'s FSDP2 group as one bucket, as it
    does ``lm_head``'s, for ``flex_shard_partial_backwards``."""
    return module in _MODULE_BUCKETS


@contextmanager
def flex_shard_partial_backwards(
    module: nn.Module, *, run_backward: bool, outer_backward_follows: bool
) -> Iterator[Callable[[], None]]:
    """``fsdp_partial_backwards`` for a module with a FlexShard bucket, with
    the collectives FSDP2 issues for its group. Yields ``before_last_forward``.

    A call of ``module`` on its own leaves its bucket's forward open, so
    FlexShard runs no post-forward for it and finishes the bucket as soon as
    the call's input grads are ready: with gradient sync, it reduce-scatters
    and reshards; without, it keeps the grads, and reshards only with
    ``reshard_after_backward``. That backward finishes no other bucket, so
    unlike FSDP2, the other buckets need no changes. Gradient sync is on
    outside, since ``validate_flex_shard_training`` rejects deferred gradient
    reduction, so the last call's backward reduce-scatters the grads of all
    calls, as FSDP2's does.

    Without backward, the bucket's forward stays open, as FSDP2 leaves its
    group's; ``post_validate_fsdp_for_accelerator_graph`` closes both.
    """
    if run_backward and not outer_backward_follows:
        # Calls whose inputs don't require grad have no trigger that finishes
        # the bucket. FSDP2 completes each call's grouped forward instead.
        raise ValueError(
            "parallelism.fsdp_backend=FlexShardConfig() does not support "
            "ChunkedLossWrapper with a frozen decoder yet."
        )
    bucket = _MODULE_BUCKETS[module]()
    assert bucket is not None, "FlexShard buckets live as long as their model."
    reshard_after_backward = bucket.reshard_after_backward
    # TODO: Gradient sync is set to True, not restored, because FlexShard has
    # no public getter for it, and validate_flex_shard_training rejects
    # deferred gradient reduction, so it's always on outside. Allowing deferral
    # needs the caller's setting restored, and FlexShard's state of grads kept
    # without sync, to leave the last reduce-scatter to the decoder backward
    # when earlier microbatches kept grads, as FSDP2 does.
    try:
        bucket.set_requires_gradient_sync(False)
        bucket.set_reshard_after_backward(False)
        yield lambda: bucket.set_requires_gradient_sync(True)
    finally:
        bucket.set_requires_gradient_sync(True)
        bucket.set_reshard_after_backward(reshard_after_backward)
        # The last call's backward has resharded the bucket, unless there was
        # no backward.
        bucket.reshard()


def reshard_flex_shard(model: nn.Module) -> None:
    """If FlexShard sharded ``model``, reshard all its buckets and close the
    forwards they left open, with FlexShard's ``reshard()``."""
    if model in _FLEX_SHARD_MODELS:
        cast(Any, model).reshard()


def declare_state_dict_layouts(model: nn.Module) -> None:
    """Make ``model.state_dict()`` declare where each FlexShard parameter's
    local shard sits in the full parameter. Call it after ``flex_shard(model)``.

    Without the layouts, DCP saves and loads each local shard as the whole
    parameter. Every state dict ``CheckpointManager`` saves or loads comes from
    ``model.state_dict()``, including the one it takes again after a load.
    """
    # pyrefly: ignore [missing-import]
    from flex_shard import set_state_dict_global_layouts

    # TODO: Drop once a FlexShard module's state_dict() declares its layouts.
    def hook(module, state_dict, prefix, local_metadata) -> None:
        set_state_dict_global_layouts(
            module,
            {
                key.removeprefix(prefix): value
                for key, value in state_dict.items()
                if key.startswith(prefix)
            },
        )

    model.register_state_dict_post_hook(hook)


def declare_optimizer_state_layouts(
    optimizer: torch.optim.Optimizer, model: nn.Module
) -> None:
    """Make ``optimizer.state_dict()`` declare where each state of a FlexShard
    parameter sits in the full state, if FlexShard sharded ``model``, which
    holds the parameters.

    ``step`` declares no layout, so DCP saves it as replicated, as for FSDP2.
    flex_shard raises ``NotImplementedError`` for optimizers other than Adam
    and AdamW.
    """
    if model in _FLEX_SHARD_MODELS:
        # pyrefly: ignore [missing-import]
        from flex_shard import register_optimizer_checkpoint_hook

        register_optimizer_checkpoint_hook(optimizer, model)


def load_hf_into_local_shards(
    sd_adapter: "BaseStateDictAdapter",
    state_dict: dict[str, Any],
    storage_reader: dcp.StorageReader,
) -> None:
    """Load a Hugging Face checkpoint into ``state_dict`` in place, where
    ``state_dict`` is a model state dict whose local shards declare their
    layouts, as FlexShard's does (``declare_state_dict_layouts``).

    The state dict adapters convert full tensors (and FSDP2's DTensors), so
    they would take each local shard for the whole tensor. Their conversions
    only move rows: each Hugging Face tensor holds rows of one parameter, its
    leading dims re-indexed and its last dim kept whole and in order. For each
    local shard, ``to_hf`` runs on a small probe of the full parameter that
    holds row indices, which says which row of the parameter each Hugging Face
    row holds (``_hf_local_tensors``). Each region of the shard is a box of the
    parameter, which holds only some columns of its rows if the shard splits
    the last dim, as tensor parallelism does for row-wise parallel weights. So
    this rank's part of a Hugging Face tensor is, for each region, the rows
    that hold the region's rows, with the region's columns. That part becomes
    a plain tensor that declares it to DCP as rectangular chunks, so
    ``dcp.load`` reads only those elements. If it comes from one region whose
    rows it holds in order, the tensor is a view of the shard, which the load
    fills in place; otherwise it's a buffer whose rows are then copied into the
    shard. Entries without a layout go through ``to_hf`` and ``from_hf``, as
    for FSDP2.
    """
    # pyrefly: ignore [missing-import]
    from flex_shard import get_global_layout

    hf_state_dict: dict[str, torch.Tensor] = {}
    copies = []
    others = {}
    for key, value in state_dict.items():
        layout = get_global_layout(value)
        if layout is None:
            others[key] = value
            continue
        for hf_key, hf_tensor, hf_copies in _hf_local_tensors(
            sd_adapter, key, value, layout
        ):
            hf_state_dict[hf_key] = hf_tensor
            copies += hf_copies
    # from_hf runs even without entries, for its checks of the model config,
    # e.g. Llama 3's of its rotary embeddings.
    others_hf = sd_adapter.to_hf(others)
    assert not others_hf.keys() & hf_state_dict.keys()
    dcp.load({**hf_state_dict, **others_hf}, storage_reader=storage_reader)
    for dest, index, source in copies:
        dest.index_copy_(0, index, source)
    state_dict.update(sd_adapter.from_hf(others_hf))


def _hf_local_tensors(
    sd_adapter: "BaseStateDictAdapter",
    key: str,
    local: torch.Tensor,
    layout: Any,
) -> list[tuple[str, torch.Tensor, list[tuple[torch.Tensor, ...]]]]:
    """``(hf_key, tensor, copies)`` for each Hugging Face tensor that holds rows
    of ``local``, the local shard of ``key`` with ``layout``.

    ``tensor`` holds this rank's part of the Hugging Face tensor and declares
    it as chunks. ``copies`` is empty if ``tensor`` is a view of ``local``,
    else it lists ``(dest, index, source)``, where ``dest`` is a view of
    ``local`` and ``source`` of ``tensor``, for ``dest.index_copy_`` after the
    load.
    """
    # pyrefly: ignore [missing-import]
    from flex_shard import GlobalLayout, set_global_layout

    shape = layout.global_shape
    regions = list(
        zip(
            layout.global_offsets, layout.local_offsets, layout.local_sizes, strict=True
        )
    )
    # A row runs along the last dim, unless there's no other dim. Then each
    # element is a row. A region holds the same columns of each of its rows:
    # all of them, unless the shard splits the last dim.
    by_row = len(shape) > 1
    row_shape = shape[:-1] if by_row else shape
    row_dims = len(row_shape)
    if by_row:
        # Column c of row r holds 2 * r + c, so a conversion that changes the
        # last dim shows up.
        probe = torch.arange(2 * math.prod(row_shape)).view(*row_shape, 2)
        local_rows = local.view(-1, local.shape[-1])
    else:
        probe = torch.arange(math.prod(shape)).view(shape)
        local_rows = local.view(-1)
    local_row_shape = local.shape[:row_dims]
    hf_probes = sd_adapter.to_hf({key: probe})
    if not hf_probes:
        raise ValueError(
            f"{type(sd_adapter).__name__}.to_hf gives no Hugging Face tensor for "
            f"{key}, so FlexShard can't load it."
        )

    hf_tensors = []
    # How often each region's local rows are loaded.
    hits = torch.zeros(len(regions), local_rows.shape[0], dtype=torch.int64)
    for hf_key, hf_probe in hf_probes.items():
        rows = hf_probe[..., 0] // 2 if by_row else hf_probe
        if by_row and not torch.equal(
            hf_probe, torch.stack([2 * rows, 2 * rows + 1], dim=-1)
        ):
            raise ValueError(
                f"{type(sd_adapter).__name__}.to_hf changes the last dim of {key} "
                f"in {hf_key}, so FlexShard can't load it."
            )
        if rows.ndim != 1:
            raise NotImplementedError(
                f"FlexShard can't load {hf_key}, which holds rows of {key} in "
                "more than one dim."
            )
        # For each region, which of the rows it holds and where each is in
        # local_rows. zip stops at the row dims.
        coords = torch.unravel_index(rows, row_shape)
        blocks = []
        for region, (global_offset, local_offset, size) in enumerate(regions):
            inside = torch.ones_like(rows, dtype=torch.bool)
            index = torch.zeros_like(rows)
            for coord, start, local_start, length, local_length in zip(
                coords, global_offset, local_offset, size, local_row_shape
            ):
                inside &= (coord >= start) & (coord < start + length)
                index = index * local_length + coord - start + local_start
            if not inside.any():
                continue
            # Each run of the rows, with the region's columns, is one chunk.
            zero = torch.zeros(1, dtype=torch.int8)
            change = torch.diff(inside.to(torch.int8), prepend=zero, append=zero)
            starts = (change == 1).nonzero().flatten().tolist()
            ends = (change == -1).nonzero().flatten().tolist()
            index = torch.cat([index[start:end] for start, end in zip(starts, ends)])
            hits[region].index_add_(0, index, torch.ones_like(index))
            blocks.append((starts, ends, index, global_offset, local_offset, size))
        if not blocks:
            continue
        # The runs lie one after another in the tensor, region by region, each
        # with its region's columns from column 0.
        global_offsets, local_offsets, local_sizes, copies = [], [], [], []
        num_rows = 0
        for starts, ends, index, global_offset, local_offset, size in blocks:
            width = size[row_dims:]
            local_columns = tuple(
                slice(start, start + length)
                for start, length in zip(local_offset[row_dims:], width)
            )
            dest = local_rows[(slice(None), *local_columns)]
            copies.append((dest, index, num_rows, width))
            for start, end in zip(starts, ends, strict=True):
                global_offsets.append((start, *global_offset[row_dims:]))
                local_offsets.append((num_rows, *(0 for _ in width)))
                local_sizes.append((end - start, *width))
                num_rows += end - start
        dest, index, _, _ = copies[0]
        first = int(index[0])
        if len(copies) == 1 and torch.equal(
            index, torch.arange(first, first + len(index))
        ):
            tensor, copies = dest[first : first + len(index)], []
        else:
            widest = tuple(map(max, zip(*(width for *_, width in copies))))
            tensor = local_rows.new_empty((num_rows, *widest))
            copies = [
                (
                    dest,
                    index.to(local.device),
                    tensor[
                        (slice(row, row + len(index)), *(slice(0, w) for w in width))
                    ],
                )
                for dest, index, row, width in copies
            ]
        set_global_layout(
            tensor,
            GlobalLayout(
                global_shape=(len(rows), *shape[row_dims:]),
                global_offsets=tuple(global_offsets),
                local_offsets=tuple(local_offsets),
                local_sizes=tuple(local_sizes),
            ),
        )
        hf_tensors.append((hf_key, tensor, copies))
    num_local_rows = sum(math.prod(size[:row_dims]) for *_, size in regions)
    if not (hits <= 1).all() or hits.sum() != num_local_rows:
        raise ValueError(
            f"{type(sd_adapter).__name__}.to_hf doesn't give each row of {key} "
            "exactly once, so FlexShard can't load it."
        )
    return hf_tensors


def validate_flex_shard_training(
    config: "TrainingEngine.Config", model_config: "BaseModel.Config"
) -> None:
    """Reject what FlexShard doesn't support yet, before anything is built.

    ``TrainingEngine.__init__`` calls this for every trainer, with the config
    after recipes and overrides.
    """
    from torchtitan.components.checkpointer import CheckpointManager
    from torchtitan.components.optim.optimizer import DistMuon
    from torchtitan.distributed.accelerator_graph import accelerator_graphs_supported
    from torchtitan.models.base.module import Module
    from torchtitan.models.common.decoder import Decoder
    from torchtitan.models.common.moe import MoE, RoutedExperts
    from torchtitan.models.common.moe.token_dispatcher import AllToAllTokenDispatcher
    from torchtitan.models.llama3.state_dict_adapter import Llama3StateDictAdapter
    from torchtitan.models.qwen3.state_dict_adapter import Qwen3StateDictAdapter
    from torchtitan.quantization.utils import has_quantization
    from torchtitan.trainer import Trainer

    parallelism = config.parallelism
    unsupported = [
        f"parallelism.{name}={degree}"
        for name, degree in (
            (
                "data_parallel_replicate_degree",
                parallelism.data_parallel_replicate_degree,
            ),
            ("tensor_parallel_degree", parallelism.tensor_parallel_degree),
            ("context_parallel_degree", parallelism.context_parallel_degree),
            ("pipeline_parallel_degree", parallelism.pipeline_parallel_degree),
        )
        if degree != 1
    ]
    if parallelism.fsdp_symm_mem_scope is not None:
        unsupported.append(
            f"parallelism.fsdp_symm_mem_scope={parallelism.fsdp_symm_mem_scope!r}"
        )
    if parallelism.fsdp_enable_cpu_offloading:
        unsupported.append("parallelism.fsdp_enable_cpu_offloading=True")
    if parallelism.fsdp_unshard_in_current_stream:
        unsupported.append("parallelism.fsdp_unshard_in_current_stream=True")
    if parallelism.fsdp_defer_gradient_reduction:
        # FlexShard has no equivalent of FSDP2's set_is_last_backward.
        # TODO: When allowing it, change flex_shard_partial_backwards, which
        # sets lm_head's bucket's gradient sync to True instead of restoring it.
        unsupported.append("parallelism.fsdp_defer_gradient_reduction=True")

    # Decoder._apply_fsdp is the only path that passes fsdp_backend on, so a
    # model that overrides it would shard with FSDP2.
    model_cls = model_config._owner
    if getattr(model_cls, "_apply_fsdp", None) is not Decoder._apply_fsdp:
        name = (
            type(model_config).__qualname__ if model_cls is None else model_cls.__name__
        )
        unsupported.append(f"models that override Decoder._apply_fsdp ({name})")
    if isinstance(model_config, Decoder.Config) and model_config.enable_weight_tying:
        # TODO: Parallelization unties tok_embeddings and lm_head, and only
        # init_states ties them again, after FlexShard has captured both.
        # Allow tied weights once _parallelize keeps them tied.
        unsupported.append("enable_weight_tying")
    # FlexShard's Shard placement all-gathers a bucket in one dtype.
    if any(
        module_config.preserve_dtype_params
        for _, module_config, _, _ in model_config.traverse(Module.Config, recurse=True)
    ):
        unsupported.append("preserve_dtype_params")
    # FlexShard stores parameters as plain tensors, which would drop the
    # quantized all-gather that MXFP8 and NVFP4 weights implement for FSDP2.
    if has_quantization(model_config):
        unsupported.append("MXFP8 and NVFP4 weights")
    if parallelism.expert_parallel_degree > 1:
        # Only RoutedExperts with AllToAllTokenDispatcher has been checked
        # against FSDP2 under EP. FlexShard's expert bucket treats every
        # routed-expert parameter as split over ep, but output_postprocess
        # parameters are replicated over ep (models/common/moe/SHARDING.md),
        # so their grads would never be summed over it.
        for _, moe, _, _ in model_config.traverse(MoE.Config):
            routed_experts = moe.routed_experts
            if type(routed_experts) is not RoutedExperts.Config:
                feature = type(routed_experts).__qualname__
            elif (
                type(routed_experts.token_dispatcher)
                is not AllToAllTokenDispatcher.Config
            ):
                feature = type(routed_experts.token_dispatcher).__qualname__
            elif routed_experts.output_postprocess is not None:
                feature = "routed-expert output_postprocess"
            else:
                continue
            entry = f"{feature} with expert parallelism"
            if entry not in unsupported:
                unsupported.append(entry)

    if (
        not config.training.disable_accelerator_graphs
        and accelerator_graphs_supported()
    ):
        unsupported.append(
            "accelerator graphs (set training.disable_accelerator_graphs=True)"
        )
    if config.checkpointer is not None:
        # FlexShard's state dicts hold local shards that declare their layouts
        # to DCP. Each of these would lose the layouts and treat every rank's
        # shard as the whole tensor, without an error.
        checkpointer = config.checkpointer
        if not isinstance(checkpointer, CheckpointManager.Config):
            # TODO: Allow once torch_checkpointing's resharder reads the
            # layouts, not only DTensor placements.
            unsupported.append("checkpointers other than CheckpointManager")
        elif checkpointer.async_mode == "async_with_pinned_mem":
            # TODO: Allow once DCP's process-based async save keeps the
            # layouts, which torch.multiprocessing drops on the way to the save
            # process.
            unsupported.append("checkpointer.async_mode='async_with_pinned_mem'")
        # load_hf_into_local_shards has been checked against FSDP2 for these
        # adapters only.
        adapter_cls = getattr(model_cls, "state_dict_adapter_cls", None)
        if checkpointer.initial_load_in_hf and adapter_cls not in (
            Llama3StateDictAdapter,
            Qwen3StateDictAdapter,
        ):
            name = getattr(adapter_cls, "__name__", "no state dict adapter")
            unsupported.append(f"checkpointer.initial_load_in_hf=True with {name}")
        if checkpointer.initial_load_in_hf_quantized:
            unsupported.append("checkpointer.initial_load_in_hf_quantized=True")
        # The state dict adapters need DTensor parameters.
        if checkpointer.last_save_in_hf:
            unsupported.append("checkpointer.last_save_in_hf=True")
    if config.optim.ema is not None:
        unsupported.append("optim.ema")
    if any(isinstance(o, DistMuon.Config) for o in config.optim.optimizer.optimizers):
        # DistMuon needs DTensor parameters.
        unsupported.append("DistMuon")
    if config.debug.spmd_typechecking:
        unsupported.append("debug.spmd_typechecking")

    # Other trainers shard, reduce or publish parameters in ways that FlexShard
    # doesn't support yet: GraphTrainer shards with SimpleFSDP, TorchFT's
    # cross-replica all-reduce hooks FSDP2's parameter groups, and the RL
    # trainer publishes state_dict(), which holds local shards, as full weights.
    trainer = type(config)._owner
    if trainer is not Trainer:
        name = (
            "None"
            if trainer is None
            else f"{trainer.__module__}.{trainer.__qualname__}"
        )
        unsupported.append(f"trainers other than torchtitan.trainer.Trainer ({name})")
    if unsupported:
        raise ValueError(
            "parallelism.fsdp_backend=FlexShardConfig() does not support "
            f"{', '.join(unsupported)} yet."
        )
    # Checked here because flex_shard is imported only once the model is
    # sharded, after distributed init and the model build.
    if find_spec("flex_shard") is None:
        raise ImportError(
            "parallelism.fsdp_backend=FlexShardConfig() needs flex_shard "
            "(github.com/meta-pytorch/flex_shard), which isn't installed. "
            ".github/workflows/unit_test_gpu.yaml installs the commit torchtitan "
            "is tested with."
        )
