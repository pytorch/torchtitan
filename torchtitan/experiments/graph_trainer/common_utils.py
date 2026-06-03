# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import time
from collections.abc import Callable
from contextlib import contextmanager

import torch
import torch.nn as nn
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.fsdp import DataParallelMeshDims
from torch.distributed.tensor import DTensor, Replicate
from torch.fx.traceback import annotate_fn
from torch.utils._pytree import register_constant, register_pytree_node, tree_map

from torchtitan.config import TORCH_DTYPE_MAP, TrainingConfig
from torchtitan.distributed import ParallelDims
from torchtitan.distributed.context_parallel import apply_cp_to_forward
from torchtitan.experiments.graph_trainer.simple_fsdp import (
    data_parallel,
    MixedPrecisionPolicy,
)
from torchtitan.tools.logging import logger


@contextmanager
def log_timer(label: str):
    start = time.perf_counter()
    yield
    elapsed_s = time.perf_counter() - start
    logger.info("%s took %.3fs", label, elapsed_s)


_MODULE_FQN = "module_fqn"
_NOT_IN_LAYERS = -1


def _is_backward_node(node: torch.fx.Node) -> bool:
    return node.meta.get("autograd_backward", False)


def _get_layer_id(node: torch.fx.Node) -> int:
    """Extract the layer index from the node's module_fqn metadata.

    Nodes under ``layers.<N>`` return ``N``.
    All other nodes (tok_embeddings, norm, output) return ``_NOT_IN_LAYERS``.
    """
    fqn = node.meta.get("custom", {}).get(_MODULE_FQN, "")
    parts = fqn.split(".")
    if parts[0] == "layers" and len(parts) >= 2:
        try:
            return int(parts[1])
        except ValueError:
            pass
    return _NOT_IN_LAYERS


def annotate_module_fqns(model: nn.Module) -> None:
    """Annotate all modules' forward with their fully-qualified names.

    Every named submodule (excluding the root) gets its forward method wrapped
    with ``annotate_fn`` so that FX nodes carry ``module_fqn`` in
    ``node.meta["custom"]``.

    Call once after model construction, before tracing/compilation.
    """
    for fqn, submodule in model.named_modules():
        if fqn:  # skip root module
            submodule.forward = annotate_fn({_MODULE_FQN: fqn})(submodule.forward)


def parallelize_inputs(parallel_dims, args, kwargs):
    if not parallel_dims.tp_enabled:
        return args, kwargs

    def to_dtensor(tensor):
        if isinstance(tensor, torch.Tensor):
            return DTensor.from_local(
                tensor, parallel_dims.get_mesh("tp"), [Replicate()]
            )
        return tensor

    dt_args = tree_map(to_dtensor, args)

    # TODO: When using flex_attention, BlockMask would show up in kwargs,
    # and it's unclear how to convert it to DTensor. If I use to_dtensor,
    # it would fail with Dynamo Error: P2011360347
    # dt_kwargs = tree_map(to_dtensor, kwargs)

    dt_kwargs = kwargs

    return dt_args, dt_kwargs


def register_blockmask_pytree_node():
    from torch.nn.attention.flex_attention import BlockMask

    if BlockMask not in torch.utils._pytree.SUPPORTED_NODES:
        register_pytree_node(
            BlockMask,
            BlockMask._flatten,
            BlockMask._unflatten,
            flatten_with_keys_fn=BlockMask._flatten_with_keys,
            serialized_type_name="torch.nn.attention.flex_attention.BlockMask",
        )


def maybe_register_blockmask_pytree_node() -> None:
    """Register flex-attention pytree helpers if they are missing."""
    from torch.nn.attention.flex_attention import _MaskModWrapper, BlockMask

    if BlockMask not in torch.utils._pytree.SUPPORTED_NODES:
        register_blockmask_pytree_node()
    if _MaskModWrapper not in torch.utils._pytree.SUPPORTED_NODES:
        register_constant(_MaskModWrapper)


def end_with_pass(passes: list[Callable], names: list[str]) -> bool:
    return (
        len(passes) > 0
        and (last_pass_name := getattr(passes[-1], "__name__", None))
        and (last_pass_name in names)
    )


def get_default_transformer_block_buckets(
    n_layers: int,
) -> list[list[str] | str]:
    """Get default transformer block buckets for manual bucketing passes.

    Assumes the standard Decoder layout: tok_embeddings, layers.0..N-1,
    norm, and output (e.g., Llama3, DeepSeekV3, Qwen3).
    """
    return [
        "tok_embeddings",
        *[f"layers.{i}" for i in range(n_layers)],
        ["norm", "lm_head"],
    ]


def get_transformer_block_buckets(model) -> list[list[str] | str]:
    """Get transformer block buckets for manual bucketing passes.

    Works for any model with tok_embeddings, layers (OrderedDict), norm, and output
    attributes (e.g., Llama3, DeepSeekV3).
    """
    # [TODO](ruisizhang123) add EP support for transformer block bucketing
    module_list = [
        model.tok_embeddings,
        [model.norm, model.lm_head],
    ]
    for layer_id, transformer_block in model.layers.items():
        module_list.append(transformer_block)

    def convert_modules_to_fqns(modules, module_to_fqn_mapping):
        """Convert a (possibly nested) list of modules to FQN strings."""
        result = []
        for m in modules:
            if isinstance(m, list):
                if fqn_list := convert_modules_to_fqns(m, module_to_fqn_mapping):
                    result.append(fqn_list)
            else:
                if fqn := module_to_fqn_mapping.get(m):
                    result.append(fqn)
        return result

    module_to_name = {m: n for n, m in model.named_modules()}
    module_fqns = convert_modules_to_fqns(module_list, module_to_name)
    return module_fqns


def apply_cp_to_attention(
    model: nn.Module,
    parallel_dims: ParallelDims,
) -> None:
    """Wrap each layer's inner attention with CP logic."""
    attention_modules = [
        # pyrefly: ignore [missing-attribute]
        block.attention.inner_attention
        for block in model.layers.values()
    ]
    apply_cp_to_forward(attention_modules, parallel_dims.get_mesh("cp"))


def legacy_dp_mesh(parallel_dims: ParallelDims) -> DeviceMesh:
    """Dense DP mesh for the non-full_dtensor path (pre-flattened ``fsdp``)."""
    if parallel_dims.dp_replicate_enabled:
        if parallel_dims.dp_shard_enabled or parallel_dims.cp_enabled:
            names = ["dp_replicate", "fsdp"]
        else:
            names = ["dp_replicate"]
    else:
        names = ["fsdp"]
    return parallel_dims.get_mesh(names)


def legacy_edp_mesh(parallel_dims: ParallelDims) -> DeviceMesh | None:
    """Routed-expert DP mesh for the non-full_dtensor path (``None`` if no EP)."""
    if not parallel_dims.ep_enabled:
        return None
    names = (
        ["dp_replicate", "efsdp"] if parallel_dims.dp_replicate_enabled else ["efsdp"]
    )
    return parallel_dims.get_optional_mesh(names)


def apply_simple_fsdp(
    model: nn.Module,
    dp_mesh: DeviceMesh,
    edp_mesh: DeviceMesh | None = None,
    *,
    training: TrainingConfig,
    ep_degree: int = 1,
    dp_mesh_dims: "DataParallelMeshDims | None" = None,
    edp_mesh_dims: "DataParallelMeshDims | None" = None,
) -> nn.Module:
    """Wrap the model (and any MoE experts) with graph_trainer's simple_fsdp.

    Mirrors core ``apply_fsdp``: ``dp_mesh`` / ``edp_mesh`` are the dense /
    routed-expert DP meshes, and ``dp_mesh_dims`` / ``edp_mesh_dims`` name their
    DP axes under full_dtensor (``None`` on the legacy path, where the meshes
    are pre-flattened). ``edp_mesh`` is ``None`` when EP is disabled.
    """
    mp_policy = MixedPrecisionPolicy(
        param_dtype=TORCH_DTYPE_MAP[training.mixed_precision_param],
        reduce_dtype=TORCH_DTYPE_MAP[training.mixed_precision_reduce],
    )

    # ``edp_mesh`` exists only when EP is enabled, so the experts_shard_dim
    # heuristic (Shard(1) to avoid padding when the FSDP group exceeds
    # num_experts) only applies to EP runs. TODO: an EP==1 MoE run wraps its
    # experts in the dense pass on Shard(0), missing core apply_fsdp's
    # shard_placement_fn Shard(1) optimization; correct but can pad when the
    # dense FSDP degree exceeds num_experts.
    if edp_mesh is not None and hasattr(model, "layers"):
        for _, transformer_block in model.layers.items():
            if not getattr(transformer_block, "moe_enabled", False):
                continue
            assert hasattr(transformer_block, "moe")
            experts_shard_dim = (
                1
                if edp_mesh["efsdp"].size() * ep_degree
                > transformer_block.moe.experts.num_experts
                else 0
            )
            transformer_block.moe.experts = data_parallel(
                transformer_block.moe.experts,
                edp_mesh,
                mp_policy=mp_policy,
                shard_dim=experts_shard_dim,
                dp_mesh_dims=edp_mesh_dims,
            )

    model = data_parallel(
        model,
        dp_mesh,
        mp_policy=mp_policy,
        dp_mesh_dims=dp_mesh_dims,
    )
    logger.info("Applied Data Parallel (simple_fsdp) to the model")
    return model
