# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import functools
from functools import partial
from typing import Callable, TypeAlias, Optional

import torch

from torchtitan.config import JobConfig
from torchtitan.tools.logging import logger

LossFunction: TypeAlias = Callable[..., torch.Tensor]


def cross_entropy_loss(pred: dict, labels: torch.Tensor) -> torch.Tensor:
    """Common cross-entropy loss function for Transformer models training."""
    logits = pred["logits"]
    loss = torch.nn.functional.cross_entropy(
        logits.flatten(0, 1).float(), labels.flatten(0, 1), ignore_index=-1
    )
    return loss


def build_cross_entropy_loss(job_config: JobConfig):
    loss_fn = cross_entropy_loss
    if job_config.compile.enable and "loss" in job_config.compile.components:
        logger.info("Compiling the loss function with torch.compile")
        loss_fn = torch.compile(loss_fn)
    return loss_fn


def rescale_accumulated_loss(unwrapped_loss_fn, accumulation_steps):
    """Add a mean reduction over `accumulation_steps` to the given
    `unwrapped_loss_fn`.
    """

    @functools.wraps(unwrapped_loss_fn)
    def accumulated_loss_fn(*args, **kwargs):
        loss = unwrapped_loss_fn(*args, **kwargs)
        return loss / accumulation_steps

    return accumulated_loss_fn



# loss function for load balancing for Qwen3Moe
def load_balancing_loss_func(
    pred: dict,
    top_k: int,
    num_experts: int,
) -> torch.Tensor:
    r"""
    This is copied from Huggingface's source code for Qwen3Moe.
    Note that this loss seems to not be exactly what is describer in the technical report.
    It seems to be off by constants. In the paper it seems that they multiply the loss by num_experts.
    Here, the probabilities are summing to 1, which means that each gate per layer will contribute by 1/num_layers.
    The frequencies (tokens_per_expert) are summing to topk, they sum up to 1 for each "slot" in the topk.
    I will keep this implementation, as we used it just fine, and it seems to make sense to me.

    Computes auxiliary load balancing loss as in Switch Transformer - implemented in Pytorch.

    See Switch Transformer (https://huggingface.co/papers/2101.03961) for more details. This function implements the loss
    function presented in equations (4) - (6) of the paper. It aims at penalizing cases where the routing between
    experts is too unbalanced.

    Args:
        gate_logits:
            Logits from the `gate`, should be a tuple of model.config.num_hidden_layers tensors of
            shape [batch_size X sequence_length, num_experts].
        num_experts:
            Number of experts
        top_k:
            The number of experts to route per-token, can be also interpreted as the `top-k` routing
            parameter.
        loss_mask (`torch.Tensor`, *optional*):
            The loss_mask used in forward function
            shape [batch_size X sequence_length] if not None.

    Returns:
        The auxiliary loss.


    How to get top_k and num_experts:
    top_k = model.config.num_experts_per_tok
    num_experts = model.config.num_experts
    """
    gate_logits: tuple[torch.Tensor] = pred["router_logits"]
    loss_mask: Optional[torch.Tensor] = pred.get("loss_mask", None)

    compute_device = gate_logits[0].device
    concatenated_gate_logits = torch.cat([layer_gate.to(compute_device) for layer_gate in gate_logits], dim=0)
    # shape: (num_layers * seqlen, num_experts)

    routing_weights = torch.nn.functional.softmax(concatenated_gate_logits, dim=-1)

    _, selected_experts = torch.topk(routing_weights, top_k, dim=-1)
    # shape: (num_layers * seqlen, top_k)

    expert_mask = torch.nn.functional.one_hot(selected_experts, num_experts)
    # shape: (num_layers * seqlen, top_k, num_experts)

    if loss_mask is None:
        # Compute the percentage of tokens routed to each experts
        tokens_per_expert = torch.mean(expert_mask.float(), dim=0)

        # Compute the average probability of routing to these experts
        router_prob_per_expert = torch.mean(routing_weights, dim=0)
    else:
        batch_size, sequence_length = loss_mask.shape
        num_hidden_layers = concatenated_gate_logits.shape[0] // (batch_size * sequence_length)

        # Compute the mask that masks all padding tokens as 0 with the same shape of expert_mask
        expert_loss_mask = (
            loss_mask[None, :, :, None, None]
            .expand((num_hidden_layers, batch_size, sequence_length, top_k, num_experts))
            .reshape(-1, top_k, num_experts)
            .to(compute_device)
        )

        # Compute the percentage of tokens routed to each experts
        tokens_per_expert = torch.sum(expert_mask.float() * expert_loss_mask, dim=0) / torch.sum(
            expert_loss_mask, dim=0
        )

        # Compute the mask that masks all padding tokens as 0 with the same shape of tokens_per_expert
        router_per_expert_loss_mask = (
            loss_mask[None, :, :, None]
            .expand((num_hidden_layers, batch_size, sequence_length, num_experts))
            .reshape(-1, num_experts)
            .to(compute_device)
        )

        # Compute the average probability of routing to these experts
        router_prob_per_expert = torch.sum(routing_weights * router_per_expert_loss_mask, dim=0) / torch.sum(
            router_per_expert_loss_mask, dim=0
        )

    # sync the router frequencies across all instances in the distributed training
    # TODO: Does this work with PP? Because in PP only certain ranks will have a loss. We should probably allreduce only across dp-cp dimension.
    torch.distributed.all_reduce(tokens_per_expert, op=torch.distributed.ReduceOp.AVG)
    assert isinstance(tokens_per_expert, torch.Tensor), (
        f"tokens_per_expert is not a tensor: {tokens_per_expert}, type: {type(tokens_per_expert)}"
    )

    overall_loss = torch.sum(tokens_per_expert * router_prob_per_expert.unsqueeze(0))
    return overall_loss * num_experts

def sft_with_moe_aux_loss(pred: dict, labels: torch.Tensor, top_k: int, num_experts: int) -> torch.Tensor:
    loss = cross_entropy_loss(pred, labels)
    aux_loss = load_balancing_loss_func(pred, top_k=top_k, num_experts=num_experts)
    final_loss = loss + aux_loss * 0.001
    return final_loss

def build_sft_with_moe_aux_loss(job_config: JobConfig, top_k: int, num_experts: int):
    overall_loss_fn = partial(sft_with_moe_aux_loss, top_k=top_k, num_experts=num_experts)
    if job_config.compile.enable and "loss" in job_config.compile.components:
        logger.info("Compiling the overall loss function with torch.compile")
        overall_loss_fn = torch.compile(overall_loss_fn)
    return overall_loss_fn