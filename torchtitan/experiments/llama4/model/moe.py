# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn.functional as F
from torch import nn

from .args import TransformerModelArgs


class GroupedExperts(nn.Module):
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        num_experts: int,
    ):
        super().__init__()
        self.num_experts = num_experts
        self.w1 = nn.Parameter(torch.empty(num_experts, dim, hidden_dim))
        self.w2 = nn.Parameter(torch.empty(num_experts, hidden_dim, dim))
        self.w3 = nn.Parameter(torch.empty(num_experts, dim, hidden_dim))

    def forward(
        self,
        x: torch.Tensor,
        num_local_tokens_per_expert: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if num_local_tokens_per_expert is not None:
            # a tuple of tensors indexed by experts
            # each with shape (tokens_per_expert(varying), dim)
            x = torch.split(
                x,
                split_size_or_sections=num_local_tokens_per_expert.tolist(),
                dim=0,
            )
            out_experts_splits = []
            for expert_idx, x_expert in enumerate(x):
                w1, w2, w3 = (
                    self.w1[expert_idx],
                    self.w2[expert_idx],
                    self.w3[expert_idx],
                )
                h = F.silu(torch.matmul(x_expert, w1))
                h = h * torch.matmul(x_expert, w3)
                h = torch.matmul(h, w2)
                # h shape (tokens_per_expert(varying), dim)
                out_experts_splits.append(h)
            out = torch.cat(out_experts_splits, dim=0)

            # TODO:optimize with GroupedGEMM
            # https://github.com/pytorch/pytorch/pull/150374
            # _gouped_mm requires shapes to be multiple of 8
            # offsets = torch.cumsum(num_local_tokens_per_expert, dim=0, dtype=torch.int32)
            # h = F.silu(torch._grouped_mm(x, self.w1.transpose(-2, -1), offs=offsets, out_dtype=torch.bfloat16))
            # h = h * torch._grouped_mm(x, self.w3.transpose(-2, -1), offs=offsets, out_dtype=torch.bfloat16)
            # out = torch._grouped_mm(h, self.w2.transpose(-2, -1), offs=offsets, out_dtype=torch.bfloat16)
        else:
            # x shape (num_experts, tokens_per_expert, dim)
            h = F.silu(torch.bmm(x, self.w1))
            h = h * torch.bmm(x, self.w3)
            # out shape (num_experts, tokens_per_expert, dim)
            out = torch.bmm(h, self.w2)
        return out

    def init_weights(self, init_std: float):
        nn.init.trunc_normal_(self.w1, mean=0.0, std=0.02)
        nn.init.trunc_normal_(self.w2, mean=0.0, std=init_std)
        nn.init.trunc_normal_(self.w3, mean=0.0, std=init_std)


class TokenChoiceTopKRouter(nn.Module):
    """This class implements token-choice routing. In token-choice top-K routing, each token is
        routed to top K experts based on the router scores.

    Args:
        gate (nn.Module): Gate module to calculate the scores, typically nn.Linear(dim, num_experts).
        dim (int): Dimension of input tokens.
        num_experts (int): Number of experts in each moe layer.
        top_k (int): Number of experts each token will be routed to in token-choice routing.
        use_sigmoid (bool): Whether to use sigmoid or softmax for router scores. Default is False.
    """

    def __init__(
        self,
        dim: int,
        num_experts: int,
        top_k: int,
        use_sigmoid: bool = False,
    ):
        super().__init__()
        self.gate = nn.Linear(dim, num_experts, bias=False)
        self.num_experts = num_experts
        self.top_k = top_k
        self.use_sigmoid = use_sigmoid

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            x (torch.Tensor): Input tensor with shape ``(bs*slen, dim)``.

        Returns:
            routed_input (torch.Tensor):
                Tokens grouped together by experts indices with shape ``(bs*slen*top_k,)``.
            token_indices (torch.Tensor):
                Token indices for routed_input with shape ``(bs*slen*top_k,)``.
            num_local_tokens_per_expert (torch.Tensor):
                Number of tokens assigned to each expert with shape ``(num_experts,)``.
        """
        # scores shape (bs*slen, num_experts)
        scores = self.gate(x)

        # By default, sigmoid or softmax is performed in float32 to avoid loss explosion
        if self.use_sigmoid:
            scores = torch.sigmoid(scores.to(torch.float32)).to(x.dtype)
        else:
            scores = F.softmax(scores.to(torch.float32), dim=1).to(x.dtype)

        # top scores shape (bs*slen, top_k)
        top_scores, selected_experts_indices = torch.topk(scores, k=self.top_k, dim=1)
        # top_scores /= top_scores.sum(dim=-1, keep_dim=True).to(x.dtype)

        # group tokens together by expert indices from 0 to num_experts and pass that to experts forward
        num_local_tokens_per_expert = torch.histc(
            selected_experts_indices.view(-1),
            bins=self.num_experts,
            min=0,
            max=self.num_experts,
        )
        # token_indices_experts_sorted shape (bs*slen*top_k,)
        token_indices_experts_sorted = torch.argsort(
            selected_experts_indices.view(-1), stable=True
        )
        top_scores = top_scores.view(-1)[token_indices_experts_sorted]
        token_indices_experts_sorted = token_indices_experts_sorted // self.top_k

        return top_scores, token_indices_experts_sorted, num_local_tokens_per_expert

    def init_weights(self, init_std: float):
        nn.init.trunc_normal_(self.gate.weight, mean=0.0, std=init_std)


# TODO: implement load balancing auxiliary loss for token-choice routing
class MoE(nn.Module):
    def __init__(self, model_args: TransformerModelArgs):
        super().__init__()
        dim = model_args.dim
        hidden_dim = 4 * model_args.dim
        ffn_dim_multiplier = model_args.ffn_dim_multiplier
        hidden_dim = int(2 * hidden_dim / 3)
        if ffn_dim_multiplier is not None:
            hidden_dim = int(ffn_dim_multiplier * hidden_dim)

        num_experts = model_args.num_experts

        hidden_dim_denom = 1
        if model_args.auto_scale_hidden_dim:
            hidden_dim_denom = model_args.top_k + int(model_args.use_shared_expert)

        if model_args.auto_scale_hidden_dim:
            hidden_dim = int(hidden_dim / hidden_dim_denom)
        hidden_dim += -hidden_dim % model_args.multiple_of

        self.experts = GroupedExperts(
            dim=dim, hidden_dim=hidden_dim, num_experts=num_experts
        )
        self.router = TokenChoiceTopKRouter(
            dim=dim, num_experts=num_experts, top_k=model_args.top_k
        )
        self.shared_expert = (
            GroupedExperts(dim=dim, hidden_dim=hidden_dim, num_experts=1)
            if model_args.use_shared_expert
            else None
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Input tensor with shape ``(bs, slen, dim)``.

        Returns:
            out (torch.Tensor): Output tensor with shape ``(bs, slen, dim)``.
        """
        bs, slen, dim = x.shape
        # top_scores and selected_indices shape (bs*slen*top_k,)
        # num_local_tokens_per_expert shape (num_experts,)
        (
            top_scores,
            token_indices,
            num_local_tokens_per_expert,
        ) = self.router(x.reshape(bs * slen, dim))

        # shape (bs*slen*top_k, dim)
        token_indices = token_indices.reshape(-1, 1).expand(-1, dim)

        # shape (bs*slen*top_k, dim)
        routed_input = torch.gather(
            x.view(-1, dim),
            dim=0,
            index=token_indices,
        )
        routed_input = routed_input * top_scores.reshape(-1, 1)

        # shape (bs*slen*top_k, dim)
        routed_output = self.experts(routed_input, num_local_tokens_per_expert)

        # shared expert
        if self.shared_expert is not None:
            out = self.shared_expert(x.reshape(1, bs * slen, dim)).reshape(
                bs * slen, dim
            )
        else:
            out = torch.zeros_like(x.reshape(bs * slen, dim))

        out = out.scatter_add(dim=0, index=token_indices, src=routed_output)
        out = out.reshape(bs, slen, dim)
        return out

    def init_weights(self, init_std: float):
        self.experts.init_weights(init_std)
        self.router.init_weights(init_std)
        if self.shared_expert is not None:
            self.shared_expert.init_weights(init_std)
