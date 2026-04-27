# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from collections.abc import Callable

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed.tensor import DTensor, Replicate
from torch.fx.traceback import annotate_fn
from torch.utils._pytree import register_constant, register_pytree_node, tree_map

from torchtitan.config import ParallelismConfig, TORCH_DTYPE_MAP
from torchtitan.distributed import ParallelDims
from torchtitan.experiments.graph_trainer.simple_fsdp import (
    data_parallel,
    MixedPrecisionPolicy,
)
from torchtitan.tools.logging import logger

_MODULE_FQN = "module_fqn"
_NOT_IN_LAYERS = -1


def _is_backward_node(node: torch.fx.Node) -> bool:
    return node.meta.get("autograd_backward", False)


def _is_recomputed_node(node: torch.fx.Node) -> bool:
    # TODO: Workaround — recomputed nodes (from SAC) should carry
    # autograd_backward=True but remat_using_tags_for_fwd_loss_bwd_graph
    # copies metadata from the original forward node. Fix upstream to
    # tag recomputed nodes with autograd_backward=True.
    return node.name.endswith("_recomputed")


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


# Maps original FSDP group_name -> extra PG group_name
_EXTRA_FSDP_PG_REGISTRY: dict[str, str] = {}


def create_extra_fsdp_pg(parallel_dims: ParallelDims) -> None:
    """Create an extra process group mirroring the FSDP topology.

    This creates a new NCCL process group with the same ranks as each FSDP
    sub-group but a different communicator, giving it a separate CUDA stream.
    """
    if not parallel_dims.fsdp_enabled:
        logger.info("FSDP not enabled, skipping extra PG creation")
        return

    fsdp_mesh = parallel_dims.get_mesh("fsdp")
    fsdp_pg = fsdp_mesh.get_group()
    original_name = fsdp_pg.group_name

    if original_name in _EXTRA_FSDP_PG_REGISTRY:
        logger.info("Extra FSDP PG already created, skipping")
        return

    ranks = dist.get_process_group_ranks(fsdp_pg)
    pg = dist.new_group(
        ranks=ranks, group_desc="fsdp_extra", use_local_synchronization=True
    )
    _EXTRA_FSDP_PG_REGISTRY[original_name] = pg.group_name
    logger.info(
        f"Created extra FSDP PG " f"(original: {original_name}, extra: {pg.group_name})"
    )


def get_extra_fsdp_pg_name(original_pg_name: str) -> str | None:
    """Look up the extra PG name for a given original FSDP PG name."""
    return _EXTRA_FSDP_PG_REGISTRY.get(original_pg_name)


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


def apply_simple_fsdp(
    model: nn.Module,
    *,
    parallel_dims: ParallelDims,
    parallelism: ParallelismConfig,
) -> nn.Module:
    """Wrap the model (and any MoE experts) with graph_trainer's simple_fsdp.

    For MoE-enabled models, ``moe.experts`` submodules are separately wrapped
    on the EDP mesh when expert parallelism is enabled.
    """
    if parallel_dims.dp_replicate_enabled:
        if parallel_dims.dp_shard_enabled or parallel_dims.cp_enabled:
            dp_mesh_dim_names = ["dp_replicate", "fsdp"]
            dp_mode = "hybrid_shard"
        else:
            dp_mesh_dim_names = ["dp_replicate"]
            dp_mode = "replicate"
    else:
        dp_mesh_dim_names = ["fsdp"]
        dp_mode = "fully_shard"

    dp_mesh = parallel_dims.get_mesh(dp_mesh_dim_names)
    mp_policy = MixedPrecisionPolicy(
        param_dtype=TORCH_DTYPE_MAP[parallelism.fsdp_mixed_precision_param],
        reduce_dtype=TORCH_DTYPE_MAP[parallelism.fsdp_mixed_precision_reduce],
    )

    if parallel_dims.ep_enabled and hasattr(model, "layers"):
        edp_mesh_names = (
            ["dp_replicate", "efsdp"]
            if parallel_dims.dp_replicate_enabled
            else ["efsdp"]
        )
        edp_mesh = parallel_dims.get_optional_mesh(edp_mesh_names)
        assert edp_mesh is not None

        for _, transformer_block in model.layers.items():
            if not getattr(transformer_block, "moe_enabled", False):
                continue
            assert hasattr(transformer_block, "moe")
            experts_shard_dim = 0
            if (
                edp_mesh["efsdp"].size() * parallel_dims.ep
                > transformer_block.moe.experts.num_experts
            ):
                experts_shard_dim = 1

            transformer_block.moe.experts = data_parallel(
                transformer_block.moe.experts,
                edp_mesh,
                dp_mode,
                mp_policy=mp_policy,
                shard_dim=experts_shard_dim,
            )

    model = data_parallel(
        model,
        dp_mesh,
        dp_mode,
        mp_policy=mp_policy,
    )
    logger.info(
        "Applied Data Parallel (simple_fsdp) (dp mode=%s) to the model", dp_mode
    )
    return model
