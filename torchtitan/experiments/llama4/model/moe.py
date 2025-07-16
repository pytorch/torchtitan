# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn.functional as F
from torch import nn

from ..infra.expert_parallel import expert_parallel

from .args import TransformerModelArgs


class GroupedExperts(nn.Module):
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        num_experts: int,
        use_grouped_mm: bool,
    ):
        super().__init__()
        self.num_experts = num_experts
        self.w1 = nn.Parameter(torch.empty(num_experts, dim, hidden_dim))
        self.w2 = nn.Parameter(torch.empty(num_experts, hidden_dim, dim))
        self.w3 = nn.Parameter(torch.empty(num_experts, dim, hidden_dim))
        self.use_grouped_mm = use_grouped_mm

    def forward(
        self,
        x: torch.Tensor,
        num_tokens_per_expert: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self.use_grouped_mm:
            return GroupedExperts._run_experts_grouped_mm(
                self.w1, self.w2, self.w3, x, num_tokens_per_expert
            )
        else:
            return GroupedExperts._run_experts_for_loop(
                self.w1, self.w2, self.w3, x, num_tokens_per_expert
            )

    # TODO: keeping this for-loop implementation for comparison
    #       and readability, may remove later
    @expert_parallel
    @staticmethod
    def _run_experts_for_loop(
        w1: torch.Tensor,
        w2: torch.Tensor,
        w3: torch.Tensor,
        x: torch.Tensor,
        num_tokens_per_expert: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if num_tokens_per_expert is not None:
            # NOTE: this would incur a synchronization between device and host
            num_tokens_per_expert = num_tokens_per_expert.tolist()

            # side-effect code due to the usage of generate_permute_indices
            num_padding = x.shape[0] - sum(num_tokens_per_expert)

            # a tuple of tensors indexed by experts
            # each with shape (tokens_per_expert(varying), dim)
            x = torch.split(
                x[: sum(num_tokens_per_expert)],
                split_size_or_sections=num_tokens_per_expert,
                dim=0,
            )
            out_experts_splits = []
            for expert_idx, x_expert in enumerate(x):
                h = F.silu(torch.matmul(x_expert, w1[expert_idx]))
                h = h * torch.matmul(x_expert, w3[expert_idx])
                h = torch.matmul(h, w2[expert_idx])
                # h shape (tokens_per_expert(varying), dim)
                out_experts_splits.append(h)
            out = torch.cat(out_experts_splits, dim=0)

            # side-effect code due to the usage of generate_permute_indices
            out = torch.vstack((out, out.new_zeros((num_padding, out.shape[-1]))))
        else:
            # x shape (num_experts, tokens_per_expert, dim)
            h = F.silu(torch.bmm(x, w1))
            h = h * torch.bmm(x, w3)
            # out shape (num_experts, tokens_per_expert, dim)
            out = torch.bmm(h, w2)

        return out

    @expert_parallel
    @staticmethod
    def _run_experts_grouped_mm(
        w1: torch.Tensor,
        w2: torch.Tensor,
        w3: torch.Tensor,
        x: torch.Tensor,
        num_tokens_per_expert: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if num_tokens_per_expert is not None:
            offsets = torch.cumsum(num_tokens_per_expert, dim=0, dtype=torch.int32)
            # grouped mm between a 2D tensor and a 3D tensor
            assert x.dim() == 2
        else:
            offsets = None
            # fall back to regular bmm between 3D tensors
            assert x.dim() == 3

        h = F.silu(torch._grouped_mm(x.bfloat16(), w1.bfloat16(), offs=offsets))
        h = h * torch._grouped_mm(x.bfloat16(), w3.bfloat16(), offs=offsets)
        out = torch._grouped_mm(h, w2.bfloat16(), offs=offsets).type_as(x)

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
        self, x: torch.Tensor, expert_bias: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            x (torch.Tensor): Input tensor with shape ``(bs*slen, dim)``.

        Returns:
            routed_input (torch.Tensor):
                Tokens grouped together by experts indices with shape ``(bs*slen*top_k,)``.
            token_indices (torch.Tensor):
                Token indices for routed_input with shape ``(bs*slen*top_k,)``.
            num_tokens_per_expert (torch.Tensor):
                Number of tokens assigned to each expert with shape ``(num_experts,)``.
        """
        # scores shape (bs*slen, num_experts)
        scores = self.gate(x)

        # By default, sigmoid or softmax is performed in float32 to avoid loss explosion
        if self.use_sigmoid:
            scores = torch.sigmoid(scores.to(torch.float32))
        else:
            scores = F.softmax(scores.to(torch.float32), dim=1)

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

        # group tokens together by expert indices from 0 to num_experts and pass that to experts forward
        num_tokens_per_expert = torch.histc(
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

        return top_scores, token_indices_experts_sorted, num_tokens_per_expert

    def init_weights(self, init_std: float):
        nn.init.trunc_normal_(self.gate.weight, mean=0.0, std=init_std)


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
            dim=dim,
            hidden_dim=hidden_dim,
            num_experts=num_experts,
            use_grouped_mm=model_args.use_grouped_mm,
        )
        self.router = TokenChoiceTopKRouter(
            dim=dim, num_experts=num_experts, top_k=model_args.top_k
        )
        self.shared_expert = (
            GroupedExperts(
                dim=dim,
                hidden_dim=hidden_dim,
                num_experts=1,
                use_grouped_mm=model_args.use_grouped_mm,
            )
            if model_args.use_shared_expert
            else None
        )

        # define fields for auxiliary-loss-free load balancing (https://arxiv.org/abs/2408.15664)
        # NOTE: tokens_per_expert is accumulated in the model forward pass.
        #       expert_bias is updated outside the model in an optimzer step pre hook
        #       to work with gradient accumulation.
        self.load_balance_coeff = model_args.load_balance_coeff
        if self.load_balance_coeff is not None:
            assert self.load_balance_coeff > 0.0
            self.register_buffer(
                "expert_bias",
                torch.zeros(num_experts, dtype=torch.float32),
            )
            self.register_buffer(
                "tokens_per_expert",
                torch.zeros(num_experts, dtype=torch.float32),
            )
        else:
            self.expert_bias = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Input tensor with shape ``(bs, slen, dim)``.

        Returns:
            out (torch.Tensor): Output tensor with shape ``(bs, slen, dim)``.
        """
        bs, slen, dim = x.shape

        # top_scores and selected_indices shape (bs*slen*top_k,)
        # num_tokens_per_expert shape (num_experts,)
        (
            top_scores,
            token_indices,
            num_tokens_per_expert,
        ) = self.router(x.reshape(bs * slen, dim), self.expert_bias)

        # tokens_per_expert will be used to update the expert bias for load balancing.
        # Prevent extra local tokens accumulation on evaluation or activation recomputation.
        if self.load_balance_coeff is not None and torch.is_grad_enabled():
            with torch.no_grad():
                self.tokens_per_expert.add_(num_tokens_per_expert)

        # shape (bs*slen*top_k, dim)
        token_indices = token_indices.reshape(-1, 1).expand(-1, dim)

        # shape (bs*slen*top_k, dim)
        routed_input = torch.gather(
            x.view(-1, dim),
            dim=0,
            index=token_indices,
        )
        routed_input = (routed_input.to(torch.float32) * top_scores.reshape(-1, 1)).to(
            x.dtype
        )

        # shape (bs*slen*top_k, dim)
        routed_output = self.experts(routed_input, num_tokens_per_expert)

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

    def init_weights(
        self,
        init_std: float,
        buffer_device: torch.device,
    ):
        self.experts.init_weights(init_std)
        self.router.init_weights(init_std)
        if self.shared_expert is not None:
            self.shared_expert.init_weights(init_std)

        if self.load_balance_coeff is not None:
            with torch.device(buffer_device):
                self.expert_bias = torch.zeros(
                    self.experts.num_experts, dtype=torch.float32
                )
                self.tokens_per_expert = torch.zeros(
                    self.experts.num_experts, dtype=torch.float32
                )
