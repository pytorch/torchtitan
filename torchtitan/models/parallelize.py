"""Shared parallelization utilities for MoE models."""

from typing import Any

import torch
import torch.nn as nn
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    CheckpointWrapper,
)
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.fsdp import CPUOffloadPolicy, MixedPrecisionPolicy, fully_shard
from torch.distributed.tensor import Partial, Replicate, Shard
from torch.distributed.tensor.parallel import (
    ColwiseParallel,
    PrepareModuleInputOutput,
    RowwiseParallel,
    parallelize_module,
)

from torchtitan.config.job_config import Compile as CompileConfig
from torchtitan.distributed import NoParallel
from torchtitan.distributed.dual_pipe_v import (
    DualPipeExpertParallel,
)
from torchtitan.distributed.expert_parallel import (
    BaseExpertParallel,
    DeepEPExpertParallel,
    ExpertParallel,
    ExpertTensorParallel,
    ReordererSequenceParallel,
    TensorParallel,
)
from torchtitan.models.moe import moe as moe_module
from torchtitan.tools.logging import logger


def apply_fsdp(
    model: nn.Module,
    dp_mesh: DeviceMesh,
    param_dtype: torch.dtype,
    reduce_dtype: torch.dtype,
    pp_enabled: bool,
    cpu_offload: bool = False,
    reshard_after_forward_policy: str = "default",
    ep_degree: int = 1,
    edp_mesh: DeviceMesh | None = None,
    gradient_divide_factor: int | None = None,
):
    """
    Apply data parallelism (via FSDP2) to the model.

    Args:
        model (nn.Module): The model to apply data parallelism to.
        dp_mesh (DeviceMesh): The device mesh to use for data parallelism.
        param_dtype (torch.dtype): The data type to use for model parameters.
        reduce_dtype (torch.dtype): The data type to use for reduction operations.
        pp_enabled (bool): Whether pipeline parallelism is enabled.
        cpu_offload (bool, optional): Whether to offload model parameters to CPU. Defaults to False.
        reshard_after_forward_policy (str, optional): The policy to use for resharding after forward pass. Defaults to "default".
            Other options: "never", "always".
            - "default" applies default resharding behavior, implementing "smart defaults" for known optimal scenarios.
            - "always" will enable `reshard_after_forward` for all forward passes.
            - "never" will disable `reshard_after_forward` for all forward passes.

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
            # For PP, by default do not reshard after forward to avoid per-microbatch
            # all-gathers, which can be expensive and non-overlapped
            reshard_after_forward = not pp_enabled
        case _:
            raise ValueError(
                f"Invalid reshard_after_forward_policy: {reshard_after_forward_policy}."
            )

    if model.tok_embeddings is not None:
        # pyrefly: ignore [no-matching-overload]
        fully_shard(  # type: ignore
            model.tok_embeddings,  # type: ignore
            **fsdp_config,
            reshard_after_forward=reshard_after_forward,
        )

    # pyrefly: ignore [missing-attribute]
    for layer_id, transformer_block in model.layers.items():  # type: ignore
        # NOTE: When EP is enabled, In an MoE layer, we use the following FSDP wrapping
        # - the router and the shared experts are sharded together with the TransformerBlock
        # - the routed experts are sharded with the remaining edp_mesh
        if transformer_block.moe_enabled and ep_degree > 1:
            fsdp_mod_ep_config = fsdp_config.copy()
            fsdp_mod_ep_config["mesh"] = edp_mesh

            # NOTE: EP alreadys shards the routed experts on dim 0 (num_experts).
            #       When dp_mod_ep * ep > num_experts, FSDP default dim-0 sharding
            #       causes inefficiency, so we choose to do FSDP sharding on dim-1.
            #       Even when EP is not used, we may still want to shard the experts
            #       on non-0 dim. For now it may not be worth the complexity to support
            #       shard_placement_fn on the outer TransformerBlock-level FSDP.
            _experts_shard_placement_fn = None  # type: ignore
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

            # NOTE: # Although the FSDP sharding of experts is done on a mesh of
            #       a different size than other parameters, the gradient division
            #       factor should be consistent with data.
            transformer_block.moe.experts.set_gradient_divide_factor(
                gradient_divide_factor,
            )

        fully_shard(
            transformer_block,
            **fsdp_config,
            reshard_after_forward=reshard_after_forward,
        )

    # As an optimization, do not reshard_after_forward the last layers by default
    # since FSDP would prefetch them immediately after the forward pass
    if model.norm is not None and model.output is not None:
        # pyrefly: ignore [no-matching-overload]
        fully_shard(  # type: ignore
            [model.norm, model.output],  # type: ignore
            **fsdp_config,
            reshard_after_forward=reshard_after_forward_policy == "always",
        )

    fully_shard(model, **fsdp_config)

    # NOTE: set up explicit prefetching when EP is enabled, as D2H syncs
    # in EP could interfere with implicit prefetching in FSDP
    if ep_degree == 1:
        return

    # forward
    # pyrefly: ignore [not-callable]
    transformer_blocks = list(model.layers.values())  # type: ignore
    next_transformer_blocks = transformer_blocks[1:] + [None]

    # pyrefly: ignore [bad-argument-type]
    if model.tok_embeddings is not None and len(model.layers) > 0:  # type: ignore
        # pyrefly: ignore [missing-attribute]
        model.tok_embeddings.set_modules_to_forward_prefetch([transformer_blocks[0]])  # type: ignore

    for transformer_block, next_transformer_block in zip(
        transformer_blocks, next_transformer_blocks
    ):
        if next_transformer_block is not None:
            # pyrefly: ignore [missing-attribute]
            if next_transformer_block.moe_enabled:  # type: ignore
                # pyrefly: ignore [missing-attribute]
                transformer_block.set_modules_to_forward_prefetch(  # type: ignore
                    # pyrefly: ignore [missing-attribute]
                    [next_transformer_block, next_transformer_block.moe.experts]  # type: ignore
                )
            else:
                # pyrefly: ignore [missing-attribute]
                transformer_block.set_modules_to_forward_prefetch(  # type: ignore
                    [next_transformer_block]
                )
        elif model.norm is not None and model.output is not None:
            # pyrefly: ignore [missing-attribute]
            transformer_block.set_modules_to_forward_prefetch(  # type: ignore
                [model.norm, model.output]
            )

    # backward
    # pyrefly: ignore [not-callable]
    reversed_transformer_blocks = list(reversed(model.layers.values()))  # type: ignore
    prev_transformer_blocks = reversed_transformer_blocks[1:] + [None]

    # pyrefly: ignore [bad-argument-type]
    if model.norm is not None and model.output is not None and len(model.layers) > 0:  # type: ignore
        # pyrefly: ignore [missing-attribute]
        model.output.set_modules_to_backward_prefetch([reversed_transformer_blocks[0]])  # type: ignore

    for transformer_block, prev_transformer_block in zip(
        reversed_transformer_blocks, prev_transformer_blocks
    ):
        if prev_transformer_block is not None:
            # pyrefly: ignore [missing-attribute]
            if prev_transformer_block.moe_enabled:  # type: ignore
                # pyrefly: ignore [missing-attribute]
                transformer_block.set_modules_to_backward_prefetch(  # type: ignore
                    # pyrefly: ignore [missing-attribute]
                    [prev_transformer_block, prev_transformer_block.moe.experts]  # type: ignore
                )
            else:
                # pyrefly: ignore [missing-attribute]
                transformer_block.set_modules_to_backward_prefetch(  # type: ignore
                    [prev_transformer_block]
                )
        elif model.tok_embeddings is not None:
            # pyrefly: ignore [missing-attribute]
            transformer_block.set_modules_to_backward_prefetch([model.tok_embeddings])  # type: ignore


def apply_moe_ep_tp(
    model: nn.Module,
    tp_mesh: DeviceMesh | None,
    ep_mesh: DeviceMesh | None,
    etp_mesh: DeviceMesh | None,
    ep_etp_mesh: DeviceMesh | None,
    dual_pipe_v: bool = False,
    use_deepep: bool = False,
):
    assert ep_mesh is not None or tp_mesh is not None

    # pyrefly: ignore [not-callable]
    for transformer_block in model.layers.values():  # type: ignore
        # pyrefly: ignore [missing-attribute]
        if not transformer_block.moe_enabled:  # type: ignore
            continue

        if tp_mesh is not None:
            moe_layer_plan = {
                # input / output sharding on the seqlen dim
                # all-gather for input, reduce-scatter for output
                "moe": PrepareModuleInputOutput(
                    input_layouts=(Shard(1),),
                    desired_input_layouts=(Replicate(),),
                    use_local_input=True,
                    output_layouts=(Partial(),),
                    desired_output_layouts=(Shard(1),),
                ),
                # replicate computation for the router
                "moe.router.gate": NoParallel(),
            }
            if ep_mesh is not None and etp_mesh is None:
                # If TP is borrowed for EP, then split the tokens across TP ranks so that
                # the reorderer, the all-to-all comms, and routed experts computation
                # are effectively running Sequence Parallel (split along the folded bs*slen dim)
                # pyrefly: ignore [no-matching-overload]
                moe_layer_plan.update({"moe.reorderer": ReordererSequenceParallel()})
            # pyrefly: ignore [missing-attribute]
            if transformer_block.moe.shared_experts is not None:  # type: ignore
                # input Replicate, output Partial
                # pyrefly: ignore [no-matching-overload]
                moe_layer_plan.update(
                    {
                        "moe.shared_experts.w1": ColwiseParallel(),
                        "moe.shared_experts.w2": RowwiseParallel(
                            output_layouts=Partial()
                        ),
                        "moe.shared_experts.w3": ColwiseParallel(),
                    }
                )
            parallelize_module(
                # pyrefly: ignore [bad-argument-type]
                module=transformer_block,  # type: ignore
                device_mesh=tp_mesh,
                # pyrefly: ignore [bad-argument-type]
                parallelize_plan=moe_layer_plan,
            )

        experts_mesh, experts_plan = None, None
        if ep_mesh is None:
            assert ep_etp_mesh is None
            experts_mesh = tp_mesh
            # input Replicate, output Partial
            experts_plan = TensorParallel()
        elif tp_mesh is None or etp_mesh is None:
            assert ep_etp_mesh is None
            experts_mesh = ep_mesh
            if use_deepep:
                # pyrefly: ignore [missing-attribute]
                score_before_experts = transformer_block.moe.score_before_experts  # type: ignore

                experts_plan = DeepEPExpertParallel(
                    score_before_experts=score_before_experts,
                )
                logger.info("Applying DeepEP to MoE layer")
            else:
                # input / output sharding on the batch / tokens dim
                experts_plan = ExpertParallel()
        else:
            experts_mesh = ep_etp_mesh
            experts_plan = ExpertTensorParallel()

        if dual_pipe_v and isinstance(experts_plan, BaseExpertParallel):
            experts_plan = DualPipeExpertParallel(experts_plan)

        parallelize_module(
            # pyrefly: ignore [missing-attribute]
            module=transformer_block.moe.experts,  # type: ignore
            device_mesh=experts_mesh,
            parallelize_plan=experts_plan,
        )


def apply_compile(model: nn.Module, compile_config: CompileConfig, ep_enabled: bool):
    """
    Apply torch.compile to each TransformerBlock, which makes compilation efficient due to
    repeated structure. Alternatively one can compile the whole model (after applying DP).
    """
    # NOTE: This flag is needed for torch.compile to avoid graph breaking on dynamic shapes in token-choice MoE
    # but it is experimental.
    torch._dynamo.config.capture_scalar_outputs = True
    # pyrefly: ignore [missing-attribute]
    for layer_id, transformer_block in model.layers.named_children():  # type: ignore
        # pyrefly: ignore[missing-attribute]
        if transformer_block.moe_enabled:
            # If it is a MoE layer, FSDP(GroupedExperts) will cause a graph break
            # So we must weave compile wrappers around those FSDP hooks to
            # prevent AC from falling back the whole graph to eager.
            # TODO: Fix Compile(AC(graph break))

            if isinstance(transformer_block, CheckpointWrapper):
                # TODO: Make CheckpointWrapper a transparent wrapper
                # unwrap so that .named_children() works
                block = transformer_block._checkpoint_wrapped_module
            else:
                block = transformer_block

            for attr_name, submod in block.named_children():
                assert getattr(block, attr_name) == getattr(
                    transformer_block, attr_name
                )

                if isinstance(submod, moe_module.MoE):
                    # avoid graph breaking on the GroupedExperts' FSDP hooks
                    # by wrapping each submod's forward instead of their __call__
                    moe = submod
                    for attr_name, submod in moe.named_children():
                        if attr_name == "experts":
                            # NOTE: We don't compile token dispatch and token combine due to an issue on B200:
                            # https://github.com/pytorch/torchtitan/issues/1940
                            continue
                        setattr(
                            moe,
                            attr_name,
                            torch.compile(
                                submod, backend=compile_config.backend, fullgraph=True
                            ),
                        )
                else:
                    setattr(
                        block,
                        attr_name,
                        torch.compile(
                            submod, backend=compile_config.backend, fullgraph=True
                        ),
                    )

        else:
            # If it's not a MoE layer, there is no FSDP(GroupedExperts)
            # So we can compile the whole block
            transformer_block = torch.compile(
                transformer_block,
                backend=compile_config.backend,
                fullgraph=True,
            )

        # pyrefly: ignore [missing-attribute]
        model.layers.register_module(layer_id, transformer_block)  # type: ignore

    # Patch some globals only once (apply_compile is called multiple times for PP setup)
    already_patched = (
        "_run_experts_grouped_mm_dynamic"
        in moe_module._run_experts_grouped_mm.__qualname__
    )
    if not already_patched:
        moe_module._run_experts_grouped_mm = torch.compile(
            moe_module._run_experts_grouped_mm,
            backend=compile_config.backend,
            fullgraph=True,
        )

        if ep_enabled:
            compiled_fn = moe_module._run_experts_grouped_mm

            # keep function logic in sync with `already_patched` above
            def _run_experts_grouped_mm_dynamic(
                w1: torch.Tensor,
                w2: torch.Tensor,
                w3: torch.Tensor,
                x: torch.Tensor,
                num_tokens_per_expert: torch.Tensor,
            ) -> torch.Tensor:
                # dynamic number of tokens in expert parallel
                torch._dynamo.mark_dynamic(x, 0)
                return compiled_fn(w1, w2, w3, x, num_tokens_per_expert)

            moe_module._run_experts_grouped_mm = _run_experts_grouped_mm_dynamic

    # NOTE: We don't compile for loop code path due to an issue with unbacked symints:
    # https://github.com/pytorch/pytorch/issues/166460

    logger.info("Compiling each TransformerBlock with torch.compile")
