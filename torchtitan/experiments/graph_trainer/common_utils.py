# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from collections.abc import Callable, Generator
from contextlib import contextmanager

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed.tensor import DTensor, Replicate
from torch.fx.traceback import annotate_fn
from torch.utils._pytree import register_constant, register_pytree_node, tree_map

from torchtitan.config import CompileConfig
from torchtitan.distributed import ParallelDims
from torchtitan.tools.logging import logger

_AC_REGION_ID = "ac_region_id"


def annotate_ac_regions(model: nn.Module) -> None:
    """Annotate each transformer block with a unique AC region ID.

    This enables apply_sac_pass to assign different ac_graph_id values
    per block, creating AC region boundaries between transformer blocks.
    """
    layers = model.get_submodule("layers")
    for layer_id, transformer_block in layers.named_children():
        transformer_block.forward = annotate_fn({_AC_REGION_ID: int(layer_id)})(
            transformer_block.forward
        )


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


def get_transformer_block_buckets(model) -> list[list[str] | str]:
    """Get transformer block buckets for manual bucketing passes.

    Works for any model with tok_embeddings, layers (OrderedDict), norm, and output
    attributes (e.g., Llama3, DeepSeekV3).
    """
    # [TODO](ruisizhang123) add EP support for transformer block bucketing
    module_list = [
        model.tok_embeddings,
        [model.norm, model.output],
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


@contextmanager
def annotate_flex_attention_for_regional_inductor() -> Generator[None, None, None]:
    """Annotate FlexAttention.forward so regional_inductor compiles flex attention HOPs.

    Uses the same inductor configs as FlexAttention._compiled_flex_attn
    to ensure bitwise-identical kernels between eager and regional_inductor paths.
    """
    from torchtitan.models.common.attention import FlexAttention

    orig = FlexAttention.forward
    FlexAttention.forward = annotate_fn(
        {"compile_with_inductor": {"inductor_configs": FlexAttention.inductor_configs}}
    )(orig)
    try:
        yield
    finally:
        FlexAttention.forward = orig


def apply_graph_ac(
    compile_config: CompileConfig,
    ac_config: "ActivationCheckpointConfig",
) -> None:
    """Add apply_sac to compile joint passes for graph-based selective AC.

    Must be called only when ac_config.mode != "none". Only "selective" mode
    is supported; other modes raise ValueError.
    """
    if ac_config.mode != "selective":
        raise ValueError(
            f"graph_trainer only supports activation_checkpoint.mode 'selective' or "
            f"'none', got {ac_config.mode!r}. Use 'selective' for graph-based SAC."
        )

    joint_pass_names = getattr(compile_config, "joint_passes", [])
    if "apply_sac" not in joint_pass_names:
        compile_config.joint_passes = list(joint_pass_names) + ["apply_sac"]
        logger.info(
            "activation_checkpoint.mode is 'selective', added apply_sac to "
            "compile.joint_passes"
        )
