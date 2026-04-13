"""Shared parallelization utilities for MoE models (EP + FSDP only)."""

from typing import Any

import torch
import torch.nn as nn
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.fsdp import CPUOffloadPolicy, MixedPrecisionPolicy, fully_shard
from torch.distributed.tensor import Shard
from torch.distributed.tensor.parallel import parallelize_module

from src.distributed.expert_parallel import ExpertParallel
from src.logging import logger
from src.models.moe import moe as moe_module


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
    """Apply FSDP2 to the model with MoE-aware expert sharding.

    Wraps the model in Fully Sharded Data Parallel so that each parameter is
    sharded across the data-parallel ranks: every rank holds one slice, the
    full tensor is all-gathered for forward/backward compute and
    reduce-scattered back for gradient accumulation. Under mixed precision,
    parameters are gathered in ``param_dtype`` (typically bf16) while
    gradients are reduced in ``reduce_dtype`` (typically fp32 to avoid
    small-value underflow when summing across many ranks).

    High-level steps:

    1. **Build the shared config** — mesh, mixed-precision policy, and
        optional CPU offload — then resolve ``reshard_after_forward`` from
        the caller's policy string. Resharding frees the gathered tensor
        after forward (saves memory, costs a re-gather in backward); keeping
        it uses more memory but avoids the re-gather. The ``"default"``
        policy disables resharding under pipeline parallelism, where the
        same parameters may be reused shortly.

    2. **Shard the token embeddings** as their own FSDP unit (if present;
        absent on non-first pipeline stages).

    3. **Shard each transformer block**, with MoE-aware expert handling:

        a. If the block has MoE and ``ep_degree > 1``, the experts get
          sharded *first* on the ``edp_mesh`` (not the main ``dp_mesh``),
            because EP has already claimed some ranks and experts use the
            ``efsdp`` sub-mesh for what's left. Sharding experts first
            matters: FSDP skips already-sharded parameters when the outer
            block wrap runs, so the experts stay on the right mesh.

        b. Handle the "too many efsdp×ep ranks for ``num_experts``" edge
            case by sharding expert weights along dim 1 (``hidden_dim``)
            instead of dim 0 (``num_experts``) via ``shard_placement_fn``.

        c. Set an explicit ``gradient_divide_factor`` on the experts. The
            default FSDP reduce divides by mesh size, but EP ranks have
            already aggregated their contributions via all-to-all, so a
            custom factor is needed to recover the correct global mean.

        d. Then shard the whole block on ``dp_mesh`` — this sweeps up
            attention, norms, and shared experts, leaving the already-sharded
            experts alone.

    4. **Shard the final norm and output head** as one grouped FSDP unit
        (the output projection is the largest single parameter in the model;
        grouping with the tiny norm keeps the all-gather efficient). This
        group uses a stricter ``reshard_after_forward`` policy — only
        resharding if explicitly asked — because there's no subsequent
        forward to amortize a re-gather against.

    5. **Wrap the whole model as a root FSDP unit**, which catches any
        remaining root-level parameters and gives FSDP a top-level handle
        for coordinating state_dict, final-loss backward hooks, etc.

    6. **Explicit forward/backward prefetching (EP only).** Under EP,
        experts are a separate FSDP unit on a different mesh, and FSDP's
        automatic prefetch heuristics don't always pick the right modules.
        This step manually wires up "while computing block i, start the
        all-gather for block i+1 and its experts" (and the symmetric thing
        for backward), so expert all-gather latency hides behind compute.
        Boundary cases connect embeddings ↔ first block and last block ↔
        norm/output in both directions. Without EP this is skipped and
        FSDP's defaults handle prefetching.

    Args:
        model: The model to wrap. Must expose ``tok_embeddings``,
            ``layers`` (an ``nn.ModuleDict`` of transformer blocks),
            ``norm``, and ``output``. MoE blocks must expose
            ``block.moe.experts``.
        dp_mesh: Mesh for sharding dense parameters — the ``fsdp`` axis,
            which is ``dp_shard * cp``.
        param_dtype: Dtype for parameters during forward compute (e.g.
            ``torch.bfloat16``).
        reduce_dtype: Dtype for gradient reduction (e.g. ``torch.float32``).
        pp_enabled: Whether pipeline parallelism is enabled. Only used to
            resolve the ``"default"`` reshard policy.
        cpu_offload: If ``True``, offload parameter shards to CPU when
            idle. Saves GPU memory at significant PCIe cost; rarely used.
        reshard_after_forward_policy: One of ``"always"``, ``"never"``, or
            ``"default"``. Controls whether gathered parameters are freed
            after forward.
        ep_degree: Degree of expert parallelism. When ``> 1``, triggers
            expert-specific sharding on ``edp_mesh`` and explicit
            prefetching.
        edp_mesh: Mesh for sharding expert parameters under EP — the
            ``efsdp`` sub-mesh. Required when ``ep_degree > 1``.
        gradient_divide_factor: Override for FSDP's default gradient
            reduction scaling on expert parameters. Should be
            ``dp_replicate * dp_shard * cp`` to produce the correct global
            mean after EP's all-to-all has already aggregated gradients.
    """
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
    # ? notice each moe block has its own experts module
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
    torch._dynamo.config.capture_scalar_outputs = True  # ? let the dynamo to handled the scalar outputs, which is/could be used in MoE expert selection.

    for layer_id, transformer_block in model.layers.named_children():
        if transformer_block.moe_enabled:
            from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
                CheckpointWrapper,
            )

            # ? check if the activation checkpoint already in used
            if isinstance(transformer_block, CheckpointWrapper):
                block = transformer_block._checkpoint_wrapped_module  # ? if so, get the inner module to compile, which is the actual transformer block.
            else:
                block = transformer_block

            for (
                attr_name,
                submod,
            ) in block.named_children():  # ? iterate over [attention, ffn, moe, norm]
                if isinstance(submod, moe_module.MoE):  # ? if it is the MoE submodule
                    moe = submod
                    for (
                        moe_attr,
                        moe_submod,
                    ) in moe.named_children():  # ? we dive one layer deeper for [experts, router, reorder, shared experts]
                        if (
                            moe_attr == "experts"
                        ):  # ? we skip the the expert as the shape could be dynamic (the tokens routed to each expert)
                            continue
                        setattr(
                            moe,
                            moe_attr,
                            torch.compile(moe_submod, backend=backend, fullgraph=True),
                        )
                else:  # ? for non moe submodule, directly compile it.
                    setattr(
                        block,
                        attr_name,
                        torch.compile(submod, backend=backend, fullgraph=True),
                    )
        else:  # ? the dense block need no special treatment
            transformer_block = torch.compile(
                transformer_block, backend=backend, fullgraph=True
            )
        model.layers.register_module(
            layer_id, transformer_block
        )  # ? replace the original block with the compiled block

    # ? for ep grouped mm compile
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
                torch._dynamo.mark_dynamic(
                    x, 0
                )  # ? if we explictly tell the compiler the dimension 0 of x is dynamic, which is the token dimension, then the compiler can generate a more efficient code for the grouped mm with dynamic shape.
                return compiled_fn(w1, w2, w3, x, num_tokens_per_expert)

            moe_module._run_experts_grouped_mm = _run_experts_grouped_mm_dynamic

    logger.info("Compiling each TransformerBlock with torch.compile")
