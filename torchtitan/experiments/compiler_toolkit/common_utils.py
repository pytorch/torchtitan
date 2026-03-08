# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from collections.abc import Callable
from contextlib import contextmanager

import torch
import torch.distributed as dist
from torch.distributed.tensor import DTensor, Replicate
from torch.utils._pytree import register_pytree_node, tree_map
from torchtitan.config import CompileConfig
from torchtitan.distributed import ParallelDims
from torchtitan.tools.logging import logger


@contextmanager
def disable_compile(compile_config: CompileConfig):
    """Context manager to temporarily disable compilation."""
    original_value = compile_config.enable
    compile_config.enable = False
    try:
        yield
    finally:
        compile_config.enable = original_value


def parallelize_inputs(parallel_dims, args, kwargs):
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


def maybe_disable_eager_ac(
    compile_config: CompileConfig,
    ac_config: "ActivationCheckpointConfig",
) -> None:
    """Disable eager AC when apply_sac graph pass is enabled.

    When apply_sac is used as a joint graph pass, eager activation checkpointing
    must be disabled to avoid double-checkpointing. This must be called before
    the model parallelization step that applies eager AC.
    """
    joint_pass_names = getattr(compile_config, "joint_passes", [])
    if "apply_sac" in joint_pass_names:
        if ac_config.mode != "none":
            logger.info(
                "apply_sac graph pass is enabled, overriding eager AC mode to none"
            )
            ac_config.mode = "none"
