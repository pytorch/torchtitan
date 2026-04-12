"""Shared parallelization utilities for MoE models (EP + FSDP only)."""

from typing import Any

import torch
import torch.nn as nn
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.fsdp import CPUOffloadPolicy, MixedPrecisionPolicy, fully_shard
from torch.distributed.tensor import Shard
from torch.distributed.tensor.parallel import parallelize_module

from src.distributed.expert_parallel import ExpertParallel
from src.models.moe import moe as moe_module
from src.logging import logger


def apply_fsdp(
    model: nn.Module,
    dp_mesh: DeviceMesh,
    param_dtype: torch.dtype,
    reduce_dtype: torch.dtype,
    pp_enabled: bool = False,
    cpu_offload: bool = False,
    reshard_after_forward_policy: str = "default",
    ep_degree: int = 1,
    edp_mesh: DeviceMesh | None = None,
    gradient_divide_factor: int | None = None,
):
    """Apply FSDP2 to the model with MoE-aware expert sharding."""
    mp_policy = MixedPrecisionPolicy(param_dtype=param_dtype, reduce_dtype=reduce_dtype)
    fsdp_config: dict[str, Any] = {"mesh": dp_mesh, "mp_policy": mp_policy}
    if cpu_offload:
        fsdp_config["offload_policy"] = CPUOffloadPolicy()

    match reshard_after_forward_policy:
        case "always":
            reshard_after_forward = True
        case "never":
            reshard_after_forward = False
        case "default":
            reshard_after_forward = not pp_enabled
        case _:
            raise ValueError(
                f"Invalid reshard_after_forward_policy: {reshard_after_forward_policy}."
            )

    if model.tok_embeddings is not None:
        fully_shard(
            model.tok_embeddings,
            **fsdp_config,
            reshard_after_forward=reshard_after_forward,
        )

    for layer_id, transformer_block in model.layers.items():
        if transformer_block.moe_enabled and ep_degree > 1:
            fsdp_mod_ep_config = fsdp_config.copy()
            fsdp_mod_ep_config["mesh"] = edp_mesh

            _experts_shard_placement_fn = None
            assert edp_mesh is not None
            assert hasattr(transformer_block, "moe")
            if (
                edp_mesh["efsdp"].size() * ep_degree
                > transformer_block.moe.experts.num_experts
            ):

                def _experts_shard_placement_fn(param):  # noqa: E731
                    return Shard(1)

            fully_shard(
                transformer_block.moe.experts,
                **fsdp_mod_ep_config,
                reshard_after_forward=reshard_after_forward,
                shard_placement_fn=_experts_shard_placement_fn,
            )

            transformer_block.moe.experts.set_gradient_divide_factor(
                gradient_divide_factor,
            )

        fully_shard(
            transformer_block,
            **fsdp_config,
            reshard_after_forward=reshard_after_forward,
        )

    if model.norm is not None and model.output is not None:
        fully_shard(
            [model.norm, model.output],
            **fsdp_config,
            reshard_after_forward=reshard_after_forward_policy == "always",
        )

    fully_shard(model, **fsdp_config)

    # Set up explicit prefetching when EP is enabled
    if ep_degree == 1:
        return

    transformer_blocks = list(model.layers.values())
    next_transformer_blocks = transformer_blocks[1:] + [None]

    if model.tok_embeddings is not None and len(model.layers) > 0:
        model.tok_embeddings.set_modules_to_forward_prefetch([transformer_blocks[0]])

    for transformer_block, next_transformer_block in zip(
        transformer_blocks, next_transformer_blocks
    ):
        if next_transformer_block is not None:
            if next_transformer_block.moe_enabled:
                transformer_block.set_modules_to_forward_prefetch(
                    [next_transformer_block, next_transformer_block.moe.experts]
                )
            else:
                transformer_block.set_modules_to_forward_prefetch(
                    [next_transformer_block]
                )
        elif model.norm is not None and model.output is not None:
            transformer_block.set_modules_to_forward_prefetch(
                [model.norm, model.output]
            )

    reversed_transformer_blocks = list(reversed(model.layers.values()))
    prev_transformer_blocks = reversed_transformer_blocks[1:] + [None]

    if model.norm is not None and model.output is not None and len(model.layers) > 0:
        model.output.set_modules_to_backward_prefetch([reversed_transformer_blocks[0]])

    for transformer_block, prev_transformer_block in zip(
        reversed_transformer_blocks, prev_transformer_blocks
    ):
        if prev_transformer_block is not None:
            if prev_transformer_block.moe_enabled:
                transformer_block.set_modules_to_backward_prefetch(
                    [prev_transformer_block, prev_transformer_block.moe.experts]
                )
            else:
                transformer_block.set_modules_to_backward_prefetch(
                    [prev_transformer_block]
                )
        elif model.tok_embeddings is not None:
            transformer_block.set_modules_to_backward_prefetch([model.tok_embeddings])


def apply_moe_ep(
    model: nn.Module,
    ep_mesh: DeviceMesh,
):
    """Apply Expert Parallelism to all MoE layers (EP only, no TP)."""
    for transformer_block in model.layers.values():
        if not transformer_block.moe_enabled:
            continue

        experts_plan = ExpertParallel()
        parallelize_module(
            module=transformer_block.moe.experts,
            device_mesh=ep_mesh,
            parallelize_plan=experts_plan,
        )

    logger.info("Applied Expert Parallelism to the model")


def apply_compile(
    model: nn.Module, backend: str = "inductor", ep_enabled: bool = False
):
    """Apply torch.compile to each TransformerBlock."""
    torch._dynamo.config.capture_scalar_outputs = True

    for layer_id, transformer_block in model.layers.named_children():
        if transformer_block.moe_enabled:
            from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
                CheckpointWrapper,
            )

            if isinstance(transformer_block, CheckpointWrapper):
                block = transformer_block._checkpoint_wrapped_module
            else:
                block = transformer_block

            for attr_name, submod in block.named_children():
                if isinstance(submod, moe_module.MoE):
                    moe = submod
                    for moe_attr, moe_submod in moe.named_children():
                        if moe_attr == "experts":
                            continue
                        setattr(
                            moe,
                            moe_attr,
                            torch.compile(moe_submod, backend=backend, fullgraph=True),
                        )
                else:
                    setattr(
                        block,
                        attr_name,
                        torch.compile(submod, backend=backend, fullgraph=True),
                    )
        else:
            transformer_block = torch.compile(
                transformer_block, backend=backend, fullgraph=True
            )

        model.layers.register_module(layer_id, transformer_block)

    already_patched = (
        "_run_experts_grouped_mm_dynamic"
        in moe_module._run_experts_grouped_mm.__qualname__
    )
    if not already_patched:
        moe_module._run_experts_grouped_mm = torch.compile(
            moe_module._run_experts_grouped_mm, backend=backend, fullgraph=True
        )

        if ep_enabled:
            compiled_fn = moe_module._run_experts_grouped_mm

            def _run_experts_grouped_mm_dynamic(
                w1: torch.Tensor,
                w2: torch.Tensor,
                w3: torch.Tensor,
                x: torch.Tensor,
                num_tokens_per_expert: torch.Tensor,
            ) -> torch.Tensor:
                torch._dynamo.mark_dynamic(x, 0)
                return compiled_fn(w1, w2, w3, x, num_tokens_per_expert)

            moe_module._run_experts_grouped_mm = _run_experts_grouped_mm_dynamic

    logger.info("Compiling each TransformerBlock with torch.compile")
