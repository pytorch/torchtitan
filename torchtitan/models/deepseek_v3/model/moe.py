# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn.functional as F
from torch import nn

from .args import DeepSeekV3ModelArgs


class FeedForward(nn.Module):
    """
    FeedForward module

    Args:
        dim (int): Input dimension.
        hidden_dim (int): Hidden dimension of the feedforward layer.
        multiple_of (int): Value to ensure hidden dimension is a multiple of this value.
        ffn_dim_multiplier (float | None): Custom multiplier for hidden dimension. Defaults to None.

    Attributes:
        w1 (Linear): Linear transformation for the first layer.
        w2 (Linear): Linear transformation for the second layer.
        w3 (Linear): Linear transformation for the third layer.

    """

    def __init__(
        self,
        dim: int,
        hidden_dim: int,
    ):
        super().__init__()
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))

    def init_weights(self, init_std: float = 0.02):
        nn.init.trunc_normal_(self.w1.weight, mean=0.0, std=0.02)
        for linear in (self.w2, self.w3):
            nn.init.trunc_normal_(linear.weight, mean=0.0, std=init_std)


# Reference: torchtitan/experiments/llama4/model/
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
        num_local_tokens_per_expert: torch.Tensor | list[int] | None = None,
    ) -> torch.Tensor:
        # TODO: keeping this for loop implementation for comparison
        #       and readability, will remove later
        if not self.use_grouped_mm:
            if num_local_tokens_per_expert is not None:
                # a tuple of tensors indexed by experts
                # each with shape (tokens_per_expert(varying), dim)
                x = torch.split(
                    x,
                    split_size_or_sections=num_local_tokens_per_expert,
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
            else:
                # x shape (num_experts, tokens_per_expert, dim)
                h = F.silu(torch.bmm(x, self.w1))
                h = h * torch.bmm(x, self.w3)
                # out shape (num_experts, tokens_per_expert, dim)
                out = torch.bmm(h, self.w2)

            return out

        # grouped mm implementation
        if num_local_tokens_per_expert is not None:
            # https://github.com/pytorch/pytorch/pull/150374
            # NOTE: torch._gouped_mm requires bf16 dtypes
            #       and shapes to be multiple of 8
            offsets = torch.cumsum(
                num_local_tokens_per_expert, dim=0, dtype=torch.int32
            )
            # grouped mm between a 2D tensor and a 3D tensor
            assert x.dim() == 2
        else:
            offsets = None
            # fall back to regular bmm between 3D tensors
            assert x.dim() == 3

        assert (
            x.dtype == self.w1.dtype == self.w2.dtype == self.w3.dtype == torch.bfloat16
        ), "torch._grouped_mm only supports bf16 dtypes"
        h = F.silu(torch._grouped_mm(x, self.w1, offs=offsets))
        h = h * torch._grouped_mm(x, self.w3, offs=offsets)
        out = torch._grouped_mm(h, self.w2, offs=offsets)

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
        route_sclaing_factor: float = 1.0,
    ):
        super().__init__()

        self.dim = dim
        self.num_experts = num_experts
        self.top_k = top_k
        self.use_sigmoid = use_sigmoid
        self.route_sclaing_factor = route_sclaing_factor
        self.gate = nn.Linear(self.dim, self.num_experts, bias=False)

    def forward(
        self, x: torch.Tensor, expert_bias: torch.Tensor = None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        TODO: We haven't implement the group-based routing (node limit routing),
        and currently EP is not supporting node limit routing yet.

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
            scores = torch.sigmoid(scores.to(torch.float32))
        else:
            scores = F.softmax(scores.to(torch.float32), dim=1)

        # top scores shape (bs*slen, top_k)
        # NOTE: The expert_bias is only used for routing. The gating value
        #       top_scores is still derived from the original scores.
        _, selected_experts_indices = torch.topk(
            scores + expert_bias, k=self.top_k, dim=1
        )
        top_scores = scores.gather(dim=1, index=selected_experts_indices)

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

        top_scores = (
            top_scores * self.route_sclaing_factor
        )  # must multiply the scaling factor
        print("In TokenChoiceTopKRouter, top_scores shape: ", top_scores)
        return top_scores, token_indices_experts_sorted, num_local_tokens_per_expert

    def init_weights(self, init_std: float):
        nn.init.trunc_normal_(self.gate.weight, mean=0.0, std=init_std)


class MoE(nn.Module):
    def __init__(self, model_args: DeepSeekV3ModelArgs):

        super().__init__()
        dim = model_args.dim

        num_experts = model_args.n_routed_experts
        hidden_dim = model_args.moe_inter_dim
        top_k = model_args.n_activated_experts
        route_scaling_factor = model_args.route_scale

        self.use_grouped_mm = model_args.use_grouped_mm
        self.experts = GroupedExperts(
            dim=dim,
            hidden_dim=hidden_dim,
            num_experts=num_experts,
            use_grouped_mm=self.use_grouped_mm,
        )
        self.router = TokenChoiceTopKRouter(
            dim=dim,
            num_experts=num_experts,
            top_k=top_k,
            use_sigmoid=model_args.score_func == "sigmoid",
            route_sclaing_factor=route_scaling_factor,
        )
        self.shared_expert = (
            # Reference: https://huggingface.co/deepseek-ai/DeepSeek-V3-Base/blob/main/modeling_deepseek.py#L517
            GroupedExperts(
                dim=dim,
                hidden_dim=hidden_dim * model_args.n_shared_experts,
                num_experts=1,  # Here needs to be 1 to make it equivalent to the MLP
                use_grouped_mm=self.use_grouped_mm,
            )
            if model_args.n_shared_experts > 0
            else None
        )

        # auxiliary-loss-free load balancing
        self.load_balance_coeff = model_args.load_balance_coeff
        # the fields below are defined even when load_balance_coeff is None
        # to make initialization and checkpointing code simpler
        self.register_buffer(
            "expert_bias",
            torch.zeros(num_experts, dtype=torch.float32),
            persistent=True,
        )
        self.register_buffer(
            "tokens_per_expert",
            torch.zeros(num_experts, dtype=torch.float32),
            persistent=True,
        )

        # NOTE: forward hook, forward pre hook, or backward pre hook
        #       would conflict with activation checkpointing
        if self.load_balance_coeff is not None and self.load_balance_coeff > 0:
            self.register_full_backward_hook(self._update_expert_bias)

    def _update_expert_bias(self, *_):
        expert_bias_delta = self.load_balance_coeff * torch.sign(
            self.tokens_per_expert.mean() - self.tokens_per_expert
        )
        expert_bias_delta = expert_bias_delta - expert_bias_delta.mean()
        self.expert_bias.add_(expert_bias_delta)

        self.tokens_per_expert.zero_()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Input tensor with shape ``(bs, slen, dim)``.

        Returns:
            out (torch.Tensor): Output tensor with shape ``(bs, slen, dim)``.
        """
        print("In MoE input, x shape: ", x)
        bs, slen, dim = x.shape

        # top_scores and selected_indices shape (bs*slen*top_k,)
        # num_local_tokens_per_expert shape (num_experts,)
        (
            top_scores,
            token_indices,
            num_local_tokens_per_expert,
        ) = self.router(x.reshape(bs * slen, dim), self.expert_bias)

        # print(
        #     "In MoE, top_scores shape: ",
        #     top_scores.shape,
        #     "token_indices: ",
        #     token_indices.shape,
        #     "num_local_tokens: ",
        #     num_local_tokens_per_expert.shape,
        # )

        # will be used to update the expert bias for load balancing
        self.tokens_per_expert += num_local_tokens_per_expert

        # shape (bs*slen*top_k, dim)
        token_indices = token_indices.reshape(-1, 1).expand(-1, dim)

        # shape (bs*slen*top_k, dim)
        routed_input = torch.gather(
            x.view(-1, dim),
            dim=0,
            index=token_indices,
        )
        print("Routed input: ", routed_input)

        # TODO: remove this line, this is a temporary test
        routed_input = (routed_input.to(torch.float32) * top_scores.reshape(-1, 1)).to(
            x.dtype
        )

        if self.use_grouped_mm:
            # NOTE: In order to use torch._grouped_mm, we need to make sure
            # the number of tokens each expert gets is a multiple of 16.
            # The following kernel helps achieve this via padding, without
            # incurring synchronization between device and host.
            from torchtitan.experiments.kernels.moe.indices import (
                generate_permute_indices,
            )

            ALIGN_SIZE_M = 16

            with torch.no_grad():
                (
                    permuted_indices,
                    num_local_tokens_per_expert,
                    _,
                ) = generate_permute_indices(
                    num_local_tokens_per_expert,
                    self.experts.num_experts,
                    1,
                    token_indices.shape[0] + self.experts.num_experts * ALIGN_SIZE_M,
                    ALIGN_SIZE_M,
                )

            routed_input = torch.vstack((routed_input, routed_input.new_zeros((dim))))
            input_shape = routed_input.shape
            routed_input = routed_input[permuted_indices, :]
        else:
            # NOTE: this would incur a synchronization between device and host
            num_local_tokens_per_expert = num_local_tokens_per_expert.tolist()
            input_shape, permuted_indices = None, None

        # shape (bs*slen*top_k, dim)
        routed_output = self.experts(
            routed_input, num_local_tokens_per_expert
        )  # torch.Size([16384(bsz), 256])

        routed_output_unpermuted = routed_output.new_empty(input_shape)
        routed_output_unpermuted[permuted_indices, :] = routed_output
        routed_output = routed_output_unpermuted[:-1]

        # TODO: Use this line instead if routed_input*top_scores, need to pad top_scores to be multiple of 16
        # routed_output = (routed_output.to(torch.float32) * top_scores.unsqueeze(-1)).to(
        #     x.dtype
        # )

        # shared expert
        if self.shared_expert is not None:
            out = self.shared_expert(x.reshape(1, bs * slen, dim)).reshape(
                bs * slen, dim
            )  #  torch.Size([16384, 256]) None
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

        with torch.device(buffer_device):
            self.expert_bias = torch.zeros(
                self.experts.num_experts, dtype=torch.float32
            )
            self.tokens_per_expert = torch.zeros(
                self.experts.num_experts, dtype=torch.float32
            )
