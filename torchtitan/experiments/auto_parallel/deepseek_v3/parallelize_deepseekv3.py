# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import time
import types

import torch

from autoparallel.api import AutoParallel

from torch.distributed.tensor.placement_types import Replicate, Shard
from torchtitan.config import JobConfig
from torchtitan.distributed import ParallelDims

from torchtitan.tools.logging import logger


def _moe_forward(
    x, router, expert_bias, reorderer, score_before_experts, experts, shared_experts
):
    bs, slen, dim = x.shape
    x = x.view(-1, dim)

    # top_scores and selected_experts_indices shape (bs*slen*top_k,)
    # num_tokens_per_expert shape (num_experts,)
    (
        top_scores,
        selected_experts_indices,
        num_tokens_per_expert,
    ) = router(x, expert_bias)
    num_tokens_per_expert_update = num_tokens_per_expert

    # top_scores and token_indices_experts_sorted shape (bs*slen*top_k,)
    # num_tokens_per_expert shape (num_experts,)
    # NOTE: the reason we need to compute num_tokens_per_expert again is:
    #       1st computation in router is to update self.tokens_per_expert
    #       which would be the same across all TP ranks.
    #       2nd computation in reorderer is for the actual routing and experts computation
    #       which would be sharded over TP ranks if expert_tensor_parallel_degree==1.
    #       If tensor_paralllel_degree == expert_tensor_parallel_degree, they agree.
    (
        top_scores_experts_sorted,
        token_indices_experts_sorted,
        num_tokens_per_expert,
    ) = reorderer(top_scores, selected_experts_indices)

    # shape (bs*slen*top_k, dim)
    token_indices_experts_sorted = token_indices_experts_sorted.reshape(-1, 1).expand(
        -1, dim
    )

    # shape (bs*slen*top_k, dim)
    routed_input = torch.gather(x, dim=0, index=token_indices_experts_sorted)

    if score_before_experts:
        routed_input = (
            routed_input.to(torch.float32) * top_scores_experts_sorted.reshape(-1, 1)
        ).to(x.dtype)

    # shape (bs*slen*top_k, dim)
    routed_output = experts(routed_input, num_tokens_per_expert)

    # shared expert
    # Note: we execute the shared expert before scoring the output of the routed expert
    # to "implicitly" overlap the shared expert compute with token combine communication
    if shared_experts is not None:
        out = shared_experts(x)
    else:
        out = torch.zeros_like(x)

    if not score_before_experts:
        routed_output = (
            routed_output.to(torch.float32) * top_scores_experts_sorted.reshape(-1, 1)
        ).to(x.dtype)

    out = out.scatter_add(dim=0, index=token_indices_experts_sorted, src=routed_output)
    out = out.reshape(bs, slen, dim)
    return out, num_tokens_per_expert_update


def moe_forward(self, x: torch.Tensor) -> torch.Tensor:
    out, num_tokens_per_expert = _moe_forward(
        x,
        self.router,
        self.expert_bias,
        self.reorderer,
        self.score_before_experts,
        self.experts,
        self.shared_experts,
    )
    # HOPs don't support buffer mutations, keep this outside
    # tokens_per_expert will be used to update the expert bias for load balancing.
    # and also to count the expert usage
    # TODO: Activation Checkpointing has the side effect of double counting tokens_per_expert --
    #       first in the forward pass, and then in the backward pass. However, this has no
    #       effect on the expert bias update thanks to the torch.sign() operator.
    with torch.no_grad():
        self.tokens_per_expert.add_(num_tokens_per_expert)

    with torch.no_grad():
        self.tokens_per_expert.add_(num_tokens_per_expert)
    return out


def monkey_patch_local_map_moe(model, world_mesh):
    """
    TODO: fix HOPs not restoring the original signature.
    TODO: fix tracing with local shapes so that we can use Shard placements

    Current HOP signature we get:
    """
    from torch.distributed._tensor.experimental import local_map

    # from torchtitan.models.moe import moe
    global _moe_forward
    _moe_forward = local_map(
        _moe_forward,
        out_placements=(
            (Replicate(),),  # (Shard(0),),
            (Replicate(),),
        ),
        in_placements=(
            (Replicate(),),  # (Shard(0),),
            (Replicate(),),
            (Replicate(),),
            (Replicate(),),
            (Replicate(),),
            (Replicate(),),
            (Replicate(),),
            (Replicate(),),
            (Replicate(),),
        ),
        redistribute_inputs=True,
        in_grad_placements=None,
        device_mesh=world_mesh,
    )

    for block in model.layers.children():
        if not block.moe_enabled:
            continue
        block.moe.forward = types.MethodType(moe_forward, block.moe)

    # torch.distributed.breakpoint()
    # moe.forward = moe_forward
    # moe._moe_forward = local_map(
    #     moe._moe_forward,
    #     out_placements=(
    #         (Replicate(),),  # (Shard(0),),
    #         (Replicate(),),
    #     ),
    #     in_placements=(
    #         (Replicate(),),  # (Shard(0),),
    #         (Replicate(),),
    #         (Replicate(),),
    #         (Replicate(),),
    #         (Replicate(),),
    #         (Replicate(),),
    #         (Replicate(),),
    #         (Replicate(),),
    #         (Replicate(),),
    #     ),
    #     redistribute_inputs=True,
    #     in_grad_placements=None,
    #     device_mesh=mesh,
    # )


# Run workflow with:
# CONFIG_FILE="./torchtitan/models/deepseek_v3/train_configs/debug_model.toml" ./run_train.sh --model.name deepseekv3_auto_parallel
def parallelize_deepseekv3(
    model,
    parallel_dims: ParallelDims,
    job_config: JobConfig,
):
    """
    Apply Autoparallel to the model

    NOTE: The passed-in model preferably should be on meta device. Otherwise,
    the model must fit on GPU or CPU memory.
    """
    world_mesh = parallel_dims.world_mesh

    def input_fn():
        global_batch_size = job_config.training.global_batch_size
        if global_batch_size < 0:
            # This global batch size results in 1 gradient accumulation
            # step.
            dp_degree = parallel_dims.dp_replicate * parallel_dims.dp_shard
            global_batch_size = job_config.training.local_batch_size * dp_degree
        return (
            torch.randint(
                0,
                model.model_args.vocab_size,
                (global_batch_size, job_config.training.seq_len),
                device=torch.device("cuda"),
            ),
        )

    # TODO make autop work correctly with different combinations of DP, DP+TP, TP, and support DDP / HSDP
    assert parallel_dims.dp_replicate_enabled is False, "DDP not supported yet"
    assert parallel_dims.cp_enabled is False, "CP not supported yet"
    assert parallel_dims.pp_enabled is False, "PP not supported yet"

    # apply local_map to MoE
    monkey_patch_local_map_moe(model, world_mesh)

    # torch._inductor.config.bucket_all_gathers_fx_bucket_size_determinator = (
    #     lambda bucket_idx: 500 / parallel_dims.tp
    # )
    # torch._inductor.config.bucket_reduce_scatters_fx_bucket_size_determinator = (
    #     lambda bucket_idx: 1000 / parallel_dims.tp
    # )

    # if job_config.experimental.autop_force_bf16:
    #     logger.info("Forcing bf16 on model")
    #     model = model.bfloat16()

    # param_dtype = TORCH_DTYPE_MAP[job_config.training.mixed_precision_param]
    # reduce_dtype = TORCH_DTYPE_MAP[job_config.training.mixed_precision_reduce]
    # mp_policy = MixedPrecisionPolicy(param_dtype=param_dtype, reduce_dtype=reduce_dtype)
    mp_policy = None
    with AutoParallel(
        model,
        input_fn,
        world_mesh,
        mp_policy=mp_policy,
        compile=job_config.compile,
    ) as autop:
        autop.add_parameter_memory_constraint(low=None, high=None)

        possible_input_shardings = {
            # maps relative to mesh dim names used in torchtitan
            "dp_replicate": Shard(0),
            "dp_shard": Shard(0),
            "tp": Replicate(),
        }
        # only used if loss parallel is enabled
        possible_output_shardings = {
            # maps relative to mesh dim names used in torchtitan
            "dp_shard": Shard(0),
            "tp": Shard(2),
        }
        assert all(
            name in possible_input_shardings for name in world_mesh.mesh_dim_names
        ), f"Unsupported mesh dim in world mesh, only {possible_input_shardings.keys()} are supported by AutoParallel"
        x_sharding = tuple(
            possible_input_shardings[name] for name in world_mesh.mesh_dim_names
        )
        out_sharding = x_sharding
        loss_parallel_enabled = (
            parallel_dims.tp_enabled
            and not job_config.parallelism.disable_loss_parallel
        )
        if loss_parallel_enabled:
            out_sharding = tuple(
                possible_output_shardings[name]
                for name in world_mesh.mesh_dim_names
                if name != "dp_replicate"
            )
        autop.add_input_constraints([x_sharding])
        autop.add_output_constraints([out_sharding])
        t0 = time.time()
        sharding_placement = autop.optimize_placement()
        t1 = time.time()
        logger.info(f"AutoParallel took {t1 - t0} seconds")
        parallel_mod = autop.apply_placement(sharding_placement)

    if loss_parallel_enabled:

        # current PyTorch's implementation of loss parallel assumes
        # that the DTensor has a 1d device mesh. This is not true
        # in our case, but we can work around it by adding
        # casting the output to a DTensor on a 1d device mesh.
        # We should just use AutoParallel to do this for us, but
        # it would require putting the loss inside the model as well
        def _return_as_dtensor_for_loss_parallel(module, args, output):
            return torch.distributed.tensor.DTensor.from_local(
                output, world_mesh["tp"], (Shard(2),)
            )

        # not keeping a reference to the hook, don't plan on
        # removing it at any point
        parallel_mod.register_forward_hook(_return_as_dtensor_for_loss_parallel)

    _preserve_moe_attributes(model, parallel_mod)

    return parallel_mod


def _preserve_moe_attributes(original_model, parallel_model):
    """
    Preserve MoE custom attributes from the original model to the parallel model.
    This is only needed for attributes that aren't used in the graph, so they aren't
    lifted as graph inputs and fetched by the pre-graph runtime wrapper.

    `moe_enabled` and `load_balance_coeff` are used later in the optimizer to identify
    this block as a moe block. This should be safe as they are read-only.
    """

    def get_moe_modules(model):
        """Extract all MoE modules from the model."""
        moe_modules = []
        if hasattr(model, "layers"):
            if isinstance(model.layers, torch.nn.ModuleDict):
                # regular torchtitan structure
                blocks = model.layers.values()
            else:
                # autoparallel might change structure
                blocks = (
                    model.layers.children() if hasattr(model.layers, "children") else []
                )

            for block in blocks:
                if (
                    hasattr(block, "moe_enabled")
                    and block.moe_enabled
                    and hasattr(block, "moe")
                ):
                    moe_modules.append(block.moe)
                elif hasattr(block, "moe"):  # fallback for autoparallel
                    moe_modules.append(block.moe)
        return moe_modules

    original_moe_modules = get_moe_modules(original_model)
    parallel_moe_modules = get_moe_modules(parallel_model)

    # Copy custom attributes from original to parallel MoE modules
    # This is fine to do since these attributes are read only
    for orig_moe, par_moe in zip(original_moe_modules, parallel_moe_modules):
        if hasattr(orig_moe, "moe_enabled"):
            par_moe.load_balance_coeff = orig_moe.load_balance_coeff

        # Copy load_balance_coeff
        if hasattr(orig_moe, "load_balance_coeff"):
            par_moe.load_balance_coeff = orig_moe.load_balance_coeff
