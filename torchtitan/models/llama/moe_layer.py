# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Callable, Optional

import torch
import torch.nn.functional as F
from torch import nn
from torchtitan.logging import init_logger, logger
from torchtitan.moe_token_tracker import ExpertTokenTracker


class GroupedExperts(nn.Module):
    """This class implements the grouped experts layer used in Mixture of Experts. Each expert
    is a variant of the Gated Linear Units network. See more details in https://arxiv.org/pdf/2002.05202.

    Args:
        dim_in (int): Input dimension.
        dim_out (int): Output dimension.
        num_experts (int): Number of experts in this grouped experts layer. Default is 1.
        swiglu (bool): Whether to use gated linear unit. Default is True.
        activation (nn.Module): Activation function to use. Default is F.silu.
    """

    def __init__(
        self,
        *,
        dim_in: int,
        dim_out: int,
        num_experts: int = 1,
        swiglu: bool = True,
        activation: Callable = F.silu,
    ):
        super().__init__()
        self.dim_in = dim_in
        self.num_experts = num_experts
        self.gate_proj = nn.Parameter(torch.empty(num_experts, dim_in, dim_out))
        self.down_proj = nn.Parameter(torch.empty(num_experts, dim_out, dim_in))
        if swiglu:
            self.up_proj = nn.Parameter(torch.empty(num_experts, dim_in, dim_out))
            self.act_fn = F.silu
        else:
            self.up_proj = None
            self.act_fn = activation

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): with shape (num_experts, tokens_per_expert, dim_in) for Expert Choice(EC).

        Returns:
            torch.Tensor: with shape (num_experts, tokens_per_expert, dim_in) for Expert Choice(EC).
        """
        # Expert Choice(EC) forward
        # x shape (num_experts, tokens_per_expert, dim_in)
        h = self.act_fn(torch.bmm(x, self.gate_proj))
        if self.up_proj is not None:
            h = h * torch.bmm(x, self.up_proj)
        # out shape (num_experts, tokens_per_expert, dim_out)
        out = torch.bmm(h, self.down_proj)
        return out

    def init_weights(self, init_std: float):
        nn.init.trunc_normal_(self.gate_proj, mean=0.0, std=0.02)
        if self.up_proj is not None:
            nn.init.trunc_normal_(self.up_proj, mean=0.0, std=init_std)
        nn.init.trunc_normal_(self.down_proj, mean=0.0, std=init_std)


class ExpertChoiceTopKRouter(nn.Module):
    """This class implements experts choice routing. Each experts will select it's top K tokens based on
        the router scores. Refer to more details in https://arxiv.org/abs/2202.09368

    Args:
        gate (nn.Module): Gate module to calculate the scores, typically nn.Linear(dim, num_experts).
        dim (int): Dimension of input tokens.
        num_experts (int): Number of experts in each moe layer.
        capacity_factor (float): Capacity factor determines how many tokens each expert can choose.
            expert capacity = (number of tokens * capacity factor) / number of experts.
        use_sigmoid (bool): Whether to use sigmoid or softmax for router scores. Default is False.
    """

    def __init__(
        self,
        *,
        gate: nn.Module,
        dim: int,
        num_experts: int,
        capacity_factor: float,
        use_sigmoid: bool = True,
    ):
        super().__init__()
        self.gate = gate
        self.dim = dim
        self.num_experts = num_experts
        self.capacity_factor = 1.0  # capacity_factor
        self.use_sigmoid = use_sigmoid
        logger.info(
            f"Num Experts: {self.num_experts}, Capacity Factor: {self.capacity_factor}"
        )

    def forward(self, x: torch.Tensor):
        """
        Args:
            x (torch.Tensor): Input tensor with shape ``(bs*slen, dim)``.

        Returns:
            routed_input (torch.Tensor): input tokens grouped together by experts indices with shape
                ``(num_experts*tokens_per_expert, dim)``.
            token_indices (torch.Tensor): token indices for routed_input. Shape ``(num_experts*tokens_per_expert,)``.
        """
        # scores shape (num_experts, bs*slen)
        scores = self.gate(x).transpose(0, 1)
        # By default, we perform sigmoid and softmax in float32 to avoid loss explosion.
        if self.use_sigmoid:
            scores = torch.sigmoid(scores.to(torch.float32)).to(x.dtype)
        else:
            scores = F.softmax(scores.to(torch.float32), dim=0).to(x.dtype)
        tokens_per_expert = int(x.shape[0] * self.capacity_factor / self.num_experts)
        tokens_per_expert += -tokens_per_expert % 8
        # Take the smaller of tokens_per_expert and the number of tokens
        tokens_per_expert = min(tokens_per_expert, x.shape[0])
        logger.info(f"router: tokens_per_expert: {tokens_per_expert}")
        # top_scores shape (num_experts, tokens_per_expert)
        top_scores, selected_token_indices = torch.topk(
            scores, k=tokens_per_expert, dim=1
        )
        # print("top_scores", {top_scores}, top_scores.shape)
        # logger.info(
        #    f"selected tokens: {selected_token_indices} {selected_token_indices.shape}"
        # )
        return top_scores, selected_token_indices

    def init_weights(self, init_std: float):
        nn.init.trunc_normal_(self.gate.weight, mean=0.0, std=init_std)


class MoE(nn.Module):
    """This class implements the moe layer which is Mixture of Experts. Mixture of Experts
    typically consists of a set of expert networks, alongside with a router, which directs input tokens
    to the appropriate experts. See more details in https://arxiv.org/pdf/2407.06204.

    Args:
        experts (nn.Module): experts module.
        router (nn.Module): router module.
        shared_expert (Optional[nn.Module]): shared expert module. Default is None.
    """

    def __init__(
        self,
        *,
        experts: nn.Module,
        router: nn.Module,
        shared_expert: Optional[nn.Module] = None,
        token_tracker: Optional[ExpertTokenTracker] = None,
    ):
        super().__init__()
        self.experts = experts
        self.router = router
        self.shared_expert = shared_expert
        self.token_tracker = token_tracker

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Input tensor with shape ``(bz, slen, dim)``.

        Returns:
            out (torch.Tensor): Output tensor with shape ``(bz, slen, dim)``.
        """
        bz, slen, dim = x.shape

        # routed_input shape (num_experts*tokens_per_expert, dim) for EC
        x = x.reshape(bz * slen, dim)
        top_scores, selected_token_indices = self.router(x)
        self.token_tracker.record_assignments(selected_token_indices)

        num_experts, _ = top_scores.shape

        # token_indices shape (num_experts*tokens_per_expert, dim)
        token_indices = selected_token_indices.reshape(-1, 1).expand(-1, dim)
        print(f"\ntoken_indices, {token_indices[0][0:2]=}, {token_indices.shape=}")

        # routed_input shape (num_experts*tokens_per_expert, dim)
        routed_input = torch.gather(x, dim=0, index=token_indices)
        print(f"routed_input, {routed_input[0][0]=}, {routed_input.shape=}")
        routed_input = routed_input * top_scores.reshape(-1, 1)

        # routed_input shape (num_experts, tokens_per_expert, dim_in)
        routed_input = routed_input.reshape(num_experts, -1, dim)
        print(
            f"routed_input_reshaped, {routed_input[0][0][0:2]=}, {routed_input.shape=}"
        )

        # routed_output shape (num_experts, tokens_per_expert, dim_out)
        routed_output = self.experts(routed_input)
        print(f"routed_output, {routed_output[0][0][0:2]=}, {routed_output.shape=}")
        # routed_output shape (num_experts*tokens_per_expert, dim_out)
        routed_output = routed_output.reshape(-1, dim)
        print(
            f"routed_output_reshaped, {routed_output[0][0:2]=}, {routed_output.shape=}"
        )

        # shared expert
        if self.shared_expert is not None:
            out = self.shared_expert(x.reshape(1, bz * slen, dim)).reshape(
                bz * slen, dim
            )
        else:
            out = torch.zeros_like(x.reshape(bz * slen, dim))

        # add experts output
        # doing in in place might be faster
        out = out.scatter_add(dim=0, index=token_indices, src=routed_output)
        out = out.reshape(bz, slen, dim)
        return out

    def init_weights(self, init_std: float):
        self.experts.init_weights(init_std)
        self.router.init_weights(init_std)
        if self.shared_expert is not None:
            self.shared_expert.init_weights(init_std)
