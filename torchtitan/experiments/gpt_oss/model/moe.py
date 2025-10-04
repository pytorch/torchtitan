# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch
from torch.distributed.tensor import DTensor
import torch.nn.functional as F
from torch import nn
from torchtitan.models.gpt_oss.infra.expert_parallel import expert_parallel

from .args import GptOssModelArgs

def swiglu(x, alpha: float = 1.702, limit: float = 7.0):
    x_glu, x_linear = x[..., ::2], x[..., 1::2]
    # Clamp the input values
    x_glu = x_glu.clamp(min=None, max=limit)
    x_linear = x_linear.clamp(min=-limit, max=limit)
    out_glu = x_glu * torch.sigmoid(alpha * x_glu)
    # Note we add an extra bias of 1 to the linear layer
    return out_glu * (x_linear + 1)

class GroupedExperts(nn.Module):
    def __init__(
        self,
        dim: int,
        num_experts: int,
        use_grouped_mm: bool,
    ):
        super().__init__()
        self.num_experts = num_experts
        self.use_grouped_mm = use_grouped_mm

        self.mlp1_weight = nn.Parameter(torch.empty((num_experts, dim, dim * 2)))
        self.mlp1_bias = nn.Parameter(torch.empty((num_experts, dim * 2)))
        self.mlp2_weight = nn.Parameter(torch.empty((num_experts, dim, dim)))
        self.mlp2_bias = nn.Parameter(torch.empty((num_experts, dim)))

    def forward(
        self,
        x: torch.Tensor,
        num_tokens_per_expert: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self.use_grouped_mm:
            return GroupedExperts._run_experts_grouped_mm(
                self.mlp1_weight, self.mlp1_bias, self.mlp2_weight, self.mlp2_bias, x, num_tokens_per_expert
            )
        else:
            return GroupedExperts._run_experts_for_loop(
                self.mlp1_weight, self.mlp1_bias, self.mlp2_weight, self.mlp2_bias, x, num_tokens_per_expert
            )

    # TODO: keeping this for-loop implementation for comparison
    #       and readability, may remove later
    @expert_parallel
    @staticmethod
    def _run_experts_for_loop(
        mlp1_weight: torch.Tensor,
        mlp1_bias: torch.Tensor,
        mlp2_weight: torch.Tensor,
        mlp2_bias: torch.Tensor,
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
                h = torch.matmul(x_expert, mlp1_weight[expert_idx]) + mlp1_bias[expert_idx]
                h = swiglu(h)
                h = torch.matmul(h, mlp2_weight[expert_idx]) + mlp2_bias[expert_idx]
                out_experts_splits.append(h)
            out = torch.cat(out_experts_splits, dim=0)

            # side-effect code due to the usage of generate_permute_indices
            out = torch.vstack((out, out.new_zeros((num_padding, out.shape[-1]))))
        else:
            # x shape (num_experts, tokens_per_expert, dim)
            h = torch.bmm(x, mlp1_weight) + mlp1_bias.unsqueeze(1)
            h = swiglu(h)
            out = torch.bmm(h, mlp2_weight) + mlp2_bias.unsqueeze(1)

        return out

    # @expert_parallel  # NOTE: EP currently reduces 20B MFU from 17.8% to 16.5%!
    @staticmethod
    def _run_experts_grouped_mm(
        mlp1_weight: torch.Tensor,
        mlp1_bias: torch.Tensor,
        mlp2_weight: torch.Tensor,
        mlp2_bias: torch.Tensor,
        x: torch.Tensor,
        num_tokens_per_expert: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if num_tokens_per_expert is not None:
            offsets = torch.cumsum(num_tokens_per_expert, dim=0, dtype=torch.int32)
            # grouped mm between a 2D tensor and a 3D tensor
            assert x.dim() == 2
            num_tokens_per_expert_long = num_tokens_per_expert.to(torch.long)
        else:
            offsets = None
            # fall back to regular bmm between 3D tensors
            assert x.dim() == 3

        if isinstance(mlp1_weight, DTensor):
            mlp1_weight, mlp1_bias, mlp2_weight, mlp2_bias = mlp1_weight.to_local(), mlp1_bias.to_local(), mlp2_weight.to_local(), mlp2_bias.to_local()

        h = torch._grouped_mm(x.bfloat16(), mlp1_weight.bfloat16(), offs=offsets)
        if offsets is not None:
            b1 = mlp1_bias.repeat_interleave(num_tokens_per_expert_long, dim=0)
            tail_slack = x.shape[0] - int(offsets[-1])
            if tail_slack:
                b1 = torch.cat([b1, b1.new_zeros((tail_slack, b1.shape[-1]))], dim=0)
            h = h + b1.to(h.dtype)

        h = swiglu(h)
        h = torch._grouped_mm(h, mlp2_weight.bfloat16(), offs=offsets)
        if offsets is not None:
            b2 = mlp2_bias.repeat_interleave(num_tokens_per_expert_long, dim=0)
            tail_slack = x.shape[0] - int(offsets[-1])
            if tail_slack:
                b2 = torch.cat([b2, b2.new_zeros((tail_slack, b2.shape[-1]))], dim=0)
            h = h + b2.to(h.dtype)

        return h

    def init_weights(self, init_std: float):
        nn.init.trunc_normal_(self.mlp1_weight, mean=0.0, std=init_std)
        nn.init.trunc_normal_(self.mlp1_bias, mean=0.0, std=init_std)
        nn.init.trunc_normal_(self.mlp2_weight, mean=0.0, std=init_std)
        nn.init.trunc_normal_(self.mlp2_bias, mean=0.0, std=init_std)

    def extra_repr(self):
        return (f"num_experts={self.num_experts}, "
                f"use_grouped_mm={self.use_grouped_mm}, "
                f"mlp1_weight={tuple(self.mlp1_weight.shape)}, "
                f"mlp1_bias={tuple(self.mlp1_bias.shape)}, "
                f"mlp2_weight={tuple(self.mlp2_weight.shape)}, "
                f"mlp2_bias={tuple(self.mlp2_bias.shape)}")

class TokenChoiceTopKRouter(nn.Module):
    """This class implements token-choice routing. In token-choice top-K routing, each token is
        routed to top K experts based on the router scores.

    Args:
        dim (int): Dimension of the input.
        num_experts (int): Number of experts in each moe layer.
        top_k (int): Number of experts each token will be routed to in token-choice routing.
    """

    def __init__(
        self,
        dim: int,
        num_experts: int,
        top_k: int,
    ):
        super().__init__()

        self.dim = dim
        self.num_experts = num_experts
        self.top_k = top_k
        self.gate = nn.Linear(self.dim, self.num_experts, bias=True)

    def forward(
        self, x: torch.Tensor, expert_bias: torch.Tensor | None = None
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
            num_tokens_per_expert (torch.Tensor):
                Number of tokens assigned to each expert with shape ``(num_experts,)``.
        """
        # scores shape (bs*slen, num_experts)
        router_logits = self.gate(x)

        if expert_bias is not None:
            router_logits = router_logits + expert_bias

        # top scores shape (bs*slen, top_k)
        top_scores, selected_experts_indices = torch.topk(
            router_logits, k=self.top_k, dim=1
        )

        top_scores = F.softmax(top_scores, dim=1)

        # group tokens together by expert indices from 0 to num_experts and pass that to experts forward
        num_tokens_per_expert = torch.histc(
            selected_experts_indices.view(-1),
            bins=self.num_experts,
            min=0,
            max=self.num_experts,
        )

        # Reorder the token indices to match the order of the experts
        # token_indices_experts_sorted shape (bs*slen*top_k,)
        token_indices_experts_sorted = torch.argsort(
            selected_experts_indices.view(-1), stable=True
        )

        # reorder the scores to match the order of the token indices
        top_scores = top_scores.view(-1)[token_indices_experts_sorted]
        token_indices_experts_sorted = token_indices_experts_sorted // self.top_k

        return top_scores, token_indices_experts_sorted, num_tokens_per_expert

    def init_weights(self, init_std: float):
        nn.init.trunc_normal_(self.gate.weight, mean=0.0, std=init_std)


class MoE(nn.Module):
    def __init__(self, model_args: GptOssModelArgs):

        super().__init__()
        dim = model_args.hidden_size

        num_experts = model_args.num_local_experts
        top_k = model_args.num_experts_per_tok

        self.experts = GroupedExperts(
            dim=dim,
            num_experts=num_experts,
            use_grouped_mm=model_args.use_grouped_mm,
        )
        self.router = TokenChoiceTopKRouter(
            dim=dim,
            num_experts=num_experts,
            top_k=top_k,
        )
        self.load_balance_coeff = model_args.load_balance_coeff
        if self.load_balance_coeff is not None:
            assert self.load_balance_coeff > 0.0
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

        # shape (bs*slen*top_k, dim)
        routed_output = self.experts(routed_input, num_tokens_per_expert)

        routed_output = (routed_output.to(torch.float32) * top_scores.unsqueeze(-1)).to(
            x.dtype
        )

        out = torch.zeros_like(x.reshape(bs * slen, dim))

        # Accumulate multiple expert results becase each token can be routed to multiple experts
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
        if self.load_balance_coeff is not None:
            with torch.device(buffer_device):
                self.expert_bias = torch.zeros(
                    self.experts.num_experts, dtype=torch.float32
                )
                self.tokens_per_expert = torch.zeros(
                    self.experts.num_experts, dtype=torch.float32
                )
