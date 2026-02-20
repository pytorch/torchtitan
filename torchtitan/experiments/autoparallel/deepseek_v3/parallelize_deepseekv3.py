# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import time
import types
from collections.abc import Callable
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from autoparallel.api import AutoParallel
from autoparallel.auto_bucketing import configure_inductor_for_autobucketing
from torch.distributed.tensor.placement_types import Replicate, Shard
from torchtitan.config import (
    ActivationCheckpointConfig,
    ParallelismConfig,
    TrainingConfig,
)
from torchtitan.distributed import ParallelDims
from torchtitan.experiments.autoparallel.configs import AutoParallelCompileConfig
from torchtitan.models.common.moe.moe import _run_experts_grouped_mm
from torchtitan.protocols.model_converter import ModelConvertersContainer
from torchtitan.tools.logging import logger


def create_functional_router_forward(
    self: nn.Module,
) -> Callable:  # TokenChoiceTopKRouter
    def functional_router_forward(
        x: torch.Tensor, gate_weight: torch.nn.Parameter, expert_bias: torch.Tensor
    ):
        # scores shape (bs*slen, num_experts)
        scores = F.linear(x, gate_weight)

        # By default, sigmoid or softmax is performed in float32 to avoid loss explosion
        if self.score_func == "sigmoid":
            scores = torch.sigmoid(scores.to(torch.float32))
        elif self.score_func == "softmax":
            scores = F.softmax(scores.to(torch.float32), dim=1)
        else:
            raise NotImplementedError(f"Unknown score function {self.score_func}")

        # top scores shape (bs*slen, top_k)
        # NOTE: The expert_bias is only used for routing. The gating value
        #       top_scores is still derived from the original scores.
        if expert_bias is not None:
            _, selected_experts_indices = torch.topk(
                scores + expert_bias, k=self.top_k, dim=1
            )
            top_scores = scores.gather(dim=1, index=selected_experts_indices)
        else:
            top_scores, selected_experts_indices = torch.topk(
                scores, k=self.top_k, dim=1
            )

        # debug override: balanced round-robin routing
        if self._debug_force_load_balance:
            (
                selected_experts_indices,
                top_scores,
            ) = self._debug_force_load_balance_routing(scores)

        if self.route_norm:
            denominator = top_scores.sum(dim=-1, keepdim=True) + 1e-20
            top_scores = top_scores / denominator
        top_scores = top_scores * self.route_scale

        # group tokens together by expert indices from 0 to num_experts and pass that to experts forward
        num_tokens_per_expert = torch.histc(
            selected_experts_indices.view(-1),
            bins=self.num_experts,
            min=0,
            max=self.num_experts,
        )

        return top_scores, selected_experts_indices, num_tokens_per_expert

    return functional_router_forward


def _moe_forward(
    x: torch.Tensor,
    router_gate_weight: torch.nn.Parameter,
    expert_bias: Optional[torch.Tensor],
    experts_w1: torch.Tensor,
    experts_w3: torch.Tensor,
    experts_w2: torch.Tensor,
    shared_w1_weight: torch.Tensor,
    shared_w3_weight: torch.Tensor,
    shared_w2_weight: torch.Tensor,
    functional_router_forward: Callable,
    reorderer: nn.Module,  # TokenReorderer
    top_k: int,
):
    bs, slen, dim = x.shape
    x = x.view(-1, dim)

    # top_scores and selected_experts_indices shape (bs*slen, top_k)
    # num_tokens_per_expert shape (num_experts,)
    (
        top_scores,
        selected_experts_indices,
        num_tokens_per_expert,
    ) = functional_router_forward(x, router_gate_weight, expert_bias)
    num_tokens_per_expert_update = num_tokens_per_expert

    # top_scores_experts_sorted and token_indices_experts_sorted shape (bs*slen*top_k,)
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
    routed_input = x[token_indices_experts_sorted // top_k]

    # DSv3 score_before_experts is always False
    # if score_before_experts:
    #     routed_input = (
    #         routed_input.to(torch.float32) * top_scores_experts_sorted.reshape(-1, 1)
    #     ).to(x.dtype)

    # shape (bs*slen*top_k, dim)
    # routed_output = experts(routed_input, num_tokens_per_expert)
    routed_output = _run_experts_grouped_mm(
        experts_w1, experts_w2, experts_w3, routed_input, num_tokens_per_expert
    )

    # always has shared expert
    # Note: we execute the shared expert before scoring the output of the routed expert
    # to "implicitly" overlap the shared expert compute with token combine communication
    _h1 = F.linear(x, shared_w1_weight)
    _h3 = F.linear(x, shared_w3_weight)
    out = F.linear(F.silu(_h1) * _h3, shared_w2_weight)

    # Unsort routed outputs
    routed_output_unsorted = torch.zeros(
        (bs * slen * top_k, dim),
        dtype=routed_output.dtype,
        device=routed_output.device,
    )
    routed_output_unsorted[token_indices_experts_sorted] = routed_output
    routed_output_unsorted = routed_output_unsorted.reshape(-1, top_k, dim)
    # DSv3 score_before_experts is False
    # if not self.score_before_experts:
    out_experts = (
        torch.bmm(
            top_scores.reshape(-1, 1, top_k),
            routed_output_unsorted.float(),
        )
        .to(x.dtype)
        .squeeze(1)
    )
    # else:
    #     out_experts = routed_output_unsorted.sum(dim=1)

    # always has shared experts
    # if out is None:
    return (out + out_experts).reshape(bs, slen, dim), num_tokens_per_expert_update


def moe_forward(self, x: torch.Tensor) -> torch.Tensor:
    functional_router_forward = create_functional_router_forward(self.router)
    out, num_tokens_per_expert = _moe_forward(
        x,
        self.router.gate.weight,
        self.expert_bias,
        self.experts.w1,
        self.experts.w3,
        self.experts.w2,
        self.shared_experts.w1.weight,
        self.shared_experts.w3.weight,
        self.shared_experts.w2.weight,
        functional_router_forward,
        self.reorderer,
        self.router.top_k,
    )
    # HOPs don't support buffer mutations, keep this outside
    # tokens_per_expert will be used to update the expert bias for load balancing.
    # and also to count the expert usage
    # TODO: Activation Checkpointing has the side effect of double counting tokens_per_expert --
    #       first in the forward pass, and then in the backward pass. However, this has no
    #       effect on the expert bias update thanks to the torch.sign() operator.
    with torch.no_grad():
        self.tokens_per_expert.add_(num_tokens_per_expert)
    return out


def monkey_patch_checks(moe):
    # causes data-dependent issue, hardcoded into monkey patch
    assert not moe.score_before_experts
    assert moe.router.gate.bias is None
    assert moe.experts.use_grouped_mm
    assert moe.shared_experts is not None
    assert moe.shared_experts.w1.bias is None
    assert moe.shared_experts.w2.bias is None
    assert moe.shared_experts.w3.bias is None
    assert not list(moe.reorderer.parameters())
    assert not list(moe.reorderer.buffers())


def monkey_patch_local_map_moe(model, sparse_mesh):
    """
    TODO: fix HOPs not restoring the original signature.
    TODO: fix tracing with local shapes so that we can use Shard placements

    Current HOP signature we get:
    """
    from torch.distributed._tensor.experimental import local_map

    # from torchtitan.models.common.moe import moe
    global _moe_forward
    _moe_forward = local_map(
        _moe_forward,
        out_placements=(
            (Replicate(),),  # out: torch.Tensor
            (Replicate(),),  # num_tokens_per_expert_update: torch.Tensor
        ),
        in_placements=(
            (Replicate(),),  # x: torch.Tensor,
            (Replicate(),),  # router_gate_weight: torch.nn.Parameter,
            (Replicate(),),  # expert_bias: Optional[torch.Tensor],
            (Replicate(),),  # experts_w1: torch.Tensor,
            (Replicate(),),  # experts_w3: torch.Tensor,
            (Replicate(),),  # experts_w2: torch.Tensor,
            (Replicate(),),  # shared_w1: torch.Tensor,
            (Replicate(),),  # shared_w3: torch.Tensor,
            (Replicate(),),  # shared_w2: torch.Tensor,
            None,  # functional_router_forward: Callable,
            None,  # reorderer: TokenReorderer,
            None,  # top_k
        ),
        redistribute_inputs=True,
        in_grad_placements=None,
        device_mesh=sparse_mesh,
    )

    for block in model.layers.children():
        if not block.moe_enabled:
            continue
        block.moe.forward = types.MethodType(moe_forward, block.moe)
        monkey_patch_checks(block.moe)


# TODO: Autoparallel should transparently wrap the original nn.Module
# but I don't know how to do that.
def set_torchtitan_fields(orig, new):
    assert isinstance(new.layers, torch.nn.ModuleDict)
    for block in new.layers.values():
        block.moe_enabled = hasattr(block, "moe")


def parallelize_deepseekv3(
    model,
    *,
    parallel_dims: ParallelDims,
    training: TrainingConfig,
    model_converters: ModelConvertersContainer.Config,
    parallelism: ParallelismConfig,
    compile_config: AutoParallelCompileConfig,
    ac_config: ActivationCheckpointConfig,
    dump_folder: str,
):
    """
    Apply Autoparallel to the model

    NOTE: The passed-in model preferably should be on meta device. Otherwise,
    the model must fit on GPU or CPU memory.
    """

    # TODO(whc)
    # I do this because otherwise sometimes inductor will skip re-running passes like comms reordering
    torch._inductor.config.force_disable_caches = True
    # this is necessary for working with reordering passes. Just leave it set for all the jobs for now.
    torch._inductor.config.allow_buffer_reuse = False

    # allow configuring inductor comms optimizations from torchtitan commandline
    configure_inductor_for_autobucketing(compile_config.comms_bucket_reorder_strategy)

    sparse_names = ["dp_replicate", "efsdp", "ep", "etp"]
    sparse_names = [
        name
        for name in sparse_names
        if parallel_dims.get_optional_mesh(name) is not None
    ]
    sparse_mesh = parallel_dims.get_mesh(sparse_names)

    def input_fn():
        global_batch_size = training.global_batch_size
        if global_batch_size < 0:
            # This global batch size results in 1 gradient accumulation
            # step.
            dp_degree = parallel_dims.dp_replicate * parallel_dims.dp_shard
            global_batch_size = training.local_batch_size * dp_degree
        return (
            torch.randint(
                0,
                model.config.vocab_size,
                (global_batch_size, training.seq_len),
                device=torch.device("cuda"),
            ),
        )

    # TODO make autop work correctly with different combinations of DP, DP+TP, TP, and support DDP / HSDP
    assert parallel_dims.dp_replicate_enabled is False, "DDP not supported yet"
    assert parallel_dims.cp_enabled is False, "CP not supported yet"
    assert parallel_dims.pp_enabled is False, "PP not supported yet"

    # apply local_map to MoE
    monkey_patch_local_map_moe(model, sparse_mesh)

    # torch._inductor.config.bucket_all_gathers_fx_bucket_size_determinator = (
    #     lambda bucket_idx: 500 / parallel_dims.tp
    # )
    # torch._inductor.config.bucket_reduce_scatters_fx_bucket_size_determinator = (
    #     lambda bucket_idx: 1000 / parallel_dims.tp
    # )

    # param_dtype = TORCH_DTYPE_MAP[job_config.training.mixed_precision_param]
    # reduce_dtype = TORCH_DTYPE_MAP[job_config.training.mixed_precision_reduce]
    # mp_policy = MixedPrecisionPolicy(param_dtype=param_dtype, reduce_dtype=reduce_dtype)
    mp_policy = None
    with AutoParallel(
        model,
        input_fn,
        sparse_mesh,
        mp_policy=mp_policy,
        compile=compile_config,
    ) as autop:
        autop.add_parameter_memory_constraint(low=None, high=None)

        possible_input_shardings = {
            # maps relative to mesh dim names used in torchtitan
            "dp_replicate": Shard(0),
            "efsdp": Shard(0),
            "ep": Shard(0),
            "etp": Replicate(),
        }
        # only used if loss parallel is enabled
        possible_output_shardings = {
            # maps relative to mesh dim names used in torchtitan
            "efsdp": Shard(0),
            "etp": Shard(2),
        }
        assert all(
            name in possible_input_shardings for name in sparse_mesh.mesh_dim_names
        ), f"Unsupported mesh dim in world mesh, only {possible_input_shardings.keys()} are supported by AutoParallel"
        x_sharding = tuple(
            possible_input_shardings[name] for name in sparse_mesh.mesh_dim_names
        )
        out_sharding = x_sharding
        loss_parallel_enabled = (
            parallel_dims.tp_enabled and not parallelism.disable_loss_parallel
        )
        if loss_parallel_enabled:
            out_sharding = tuple(
                possible_output_shardings[name]
                for name in sparse_mesh.mesh_dim_names
                if name != "dp_replicate"
            )
        autop.add_input_constraints([x_sharding])
        autop.add_output_constraints([out_sharding])
        t0 = time.time()
        sharding_placement = autop.optimize_placement()
        t1 = time.time()
        logger.info(f"AutoParallel took {t1 - t0} seconds")
        parallel_mod = autop.apply_placement(sharding_placement)

    set_torchtitan_fields(model, parallel_mod)

    if loss_parallel_enabled:

        # current PyTorch's implementation of loss parallel assumes
        # that the DTensor has a 1d device mesh. This is not true
        # in our case, but we can work around it by adding
        # casting the output to a DTensor on a 1d device mesh.
        # We should just use AutoParallel to do this for us, but
        # it would require putting the loss inside the model as well
        def _return_as_dtensor_for_loss_parallel(module, args, output):
            return torch.distributed.tensor.DTensor.from_local(
                output, sparse_mesh["etp"], (Shard(2),)
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
