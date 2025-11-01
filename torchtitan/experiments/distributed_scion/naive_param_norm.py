# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import math
import warnings

import torch
import torch.distributed as dist
from torch.distributed.tensor import DTensor
from torch.distributed.tensor.placement_types import Replicate

from torchtitan.experiments.distributed_scion import DistributedScion
from torchtitan.experiments.distributed_scion.norm_helper import calculate_norm

from .utils import remove_orig_mod_and_weight_for_p_name


"""
This is the naive version of parameter norm calculation.
It is used to compute norm for other optimizers.
In distributed scion, we will automatically calcuate the norms in distributed mode.
"""


def gather_and_merge(local_stats: dict, dst: int = 0):
    # this is the old-implementation of gather_and_merge, which is used
    # to gather the norms of the parameters
    # it only tested on the "FSDP-only" case.
    world = dist.get_world_size()
    rank = dist.get_rank()
    dtype = torch.bfloat16

    my_keys = list(local_stats.keys())

    if len(my_keys) > 0:
        val_tensor = torch.stack([local_stats[k].to(dtype) for k in my_keys])
    else:
        my_keys = "padding"
        val_tensor = None

    key_bucket = [None] * world if rank == dst else None
    val_bucket = [None] * world if rank == dst else None

    dist.gather_object(my_keys, key_bucket, dst=dst)
    # dist.barrier()
    dist.gather_object(val_tensor, val_bucket, dst=dst)
    dist.barrier()

    merged = {}
    if rank == dst:
        for peer, keys in enumerate(key_bucket):
            if val_bucket[peer] is None:
                continue
            for k, v in zip(keys, val_bucket[peer]):
                if k != "padding":
                    merged[k] = v

    dist.barrier()
    if rank == dst:
        return merged
    else:
        return {}


def compute_grad(p, optimizer=None, **kwargs):
    if isinstance(optimizer, (Scion, DistributedScion)):
        momentum = kwargs.pop("momentum")
        nesterov = kwargs.pop("nesterov")
        g = optimizer.get_momentum_or_grad(
            p,
            momentum,
            nesterov,
            update_buffer=False,
            gather_to_local=optimizer.fsdp_enabled and p.ndim < 3,
            # we do not gather the moe's grads
        )
        if g is None:
            return None
        else:
            g = g.to_local() if isinstance(g, DTensor) else g
            return optimizer.lmo(g, **kwargs)
    elif isinstance(optimizer, (torch.optim.Adam, torch.optim.AdamW)):
        if p.ndim == 3:
            warnings.warn(
                f"Optimizer {optimizer.__class__.__name__} does not support "
                f"gradient computation for 3D tensors for logging."
            )
            return None

        eps = kwargs["eps"]
        weight_decay = kwargs["weight_decay"]
        beta1, beta2 = kwargs["betas"]
        assert weight_decay == 0.0, "Weight decay not supported for grad computation."

        param_optim_state = optimizer.state[p]
        if "step" not in param_optim_state:
            step = 0
        else:
            step = param_optim_state["step"].item()
        if "exp_avg_sq" in param_optim_state and "exp_avg" in param_optim_state:
            bias_correction1 = 1 - beta1**step
            bias_correction2 = 1 - beta2**step
            denom = (
                param_optim_state["exp_avg_sq"].sqrt() / math.sqrt(bias_correction2)
            ) + eps
            step_size = 1 / bias_correction1
            g = step_size * param_optim_state["exp_avg"].div(denom)
        else:
            # TODO(JSC): if we shard the MoE model, we need to remove the following code
            g = p.grad

        if isinstance(g, DTensor):
            g = g.redistribute(placements=[Replicate()] * g.device_mesh.ndim)
        return g
    else:
        raise TypeError(
            f"Optimizer {optimizer.__class__.__name__} does not support "
            f"gradient computation."
        )


def get_parameter_norms(model_parts, optimizers, norms_to_log):
    all_norms = {}
    for i, _ in enumerate(model_parts):
        # NB: assumes correspondences between model parts and optimizers
        optimizer = optimizers[i]
        for group in optimizer.param_groups:
            if isinstance(optimizer, (Scion, DistributedScion)):
                param_kwargs = {
                    "momentum": group["momentum"],
                    "nesterov": group["nesterov"],
                    "eps": group["eps"],
                    "norm_factor": group["norm_factor"],
                    "zeropower_backend": group["backend"],
                    "backend_steps": group["backend_steps"],
                }
            elif isinstance(optimizer, (torch.optim.Adam, torch.optim.AdamW)):
                param_kwargs = {
                    "eps": group["eps"],
                    "betas": group["betas"],
                    "weight_decay": group["weight_decay"],
                }
            else:
                warnings.warn(
                    f"Optimizer {optimizer.__class__.__name__} does not support "
                    f"norm computation."
                )
                continue

            FLAG_NEED_SYNC = False
            moe_norms, fsdp_norms = {}, {}
            for p_name, p in zip(group["param_names"], group["params"]):
                # The module is usually named
                # `track_update_condition_number/model_part_0/layers.0._orig_mod.attention.wo.weight`
                cleaned_p_name = remove_orig_mod_and_weight_for_p_name(p_name)
                g = compute_grad(p, optimizer, **param_kwargs)
                if g is None:
                    continue
                assert not torch.isnan(g).any(), f"There is nan in the grad of {p_name}"

                if p.ndim < 3:
                    p = (
                        p.redistribute(placements=[Replicate()] * p.device_mesh.ndim)
                        if isinstance(p, DTensor)
                        else p
                    )
                else:
                    FLAG_NEED_SYNC = True
                    local_rank = dist.get_rank()
                    world_size = dist.get_world_size()
                    ep_per_rank = math.ceil(p.shape[0] / world_size)
                    # We dont gather the parameters for 3D tensors,
                    # which is [G, D_in, D_out] of GroupedExperts
                    pass
                p = p.to_local() if isinstance(p, DTensor) else p
                g = g.to_local() if isinstance(g, DTensor) else g
                update = -group["lr"] * g

                # ####################################################
                for task, matrix in [("update", update), ("param", p)]:
                    if matrix.ndim == 3:
                        moe_norm_key_template = f"track_{task}_{{norm_name}}/ep_{{actual_ep_idx}}/{cleaned_p_name}"
                        for ep_idx in range(matrix.shape[0]):
                            actual_ep_idx = ep_idx + local_rank * ep_per_rank
                            update_norms = calculate_norm(
                                matrix[ep_idx], norms_to_log, transpose=True
                            )
                            # Template for MoE norm keys
                            moe_norms.update(
                                {
                                    moe_norm_key_template.format(
                                        norm_name=norm_name,
                                        actual_ep_idx=actual_ep_idx,
                                    ): norm_value
                                    for norm_name, norm_value in update_norms.items()
                                }
                            )
                    else:
                        if matrix.ndim > 2:
                            warnings.warn(
                                f"Encountered parameter or update {cleaned_p_name} with "
                                f"shape {p.shape} or {update.shape}, respectively; "
                                f"this may not be an issue, but please ensure its "
                                f"norms are calculated correctly."
                            )

                        transpose = "tok_embeddings" in p_name
                        update_norms = calculate_norm(
                            matrix,
                            norms_to_log,
                            transpose=transpose,
                        )

                        # Template for FSDP norm keys
                        fsdp_norm_key_template = (
                            f"track_{task}_{{norm_name}}/{cleaned_p_name}"
                        )
                        fsdp_norms.update(
                            {
                                fsdp_norm_key_template.format(
                                    norm_name=norm_name
                                ): norm_value
                                for norm_name, norm_value in update_norms.items()
                            }
                        )

            if FLAG_NEED_SYNC:
                # remove the comment below to gather the moe_norms on all ranks
                moe_norms = gather_and_merge(moe_norms)
                pass

            all_norms.update(fsdp_norms)
            all_norms.update(moe_norms)

    return all_norms
