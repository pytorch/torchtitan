# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# This code is based on model definition of `deepseek-ai/DeepSeek-V3-Base` on
# Hugging Face Model Hub. Url:
# https://huggingface.co/deepseek-ai/DeepSeek-V3-Base/blob/main/modeling_deepseek.py
# https://huggingface.co/deepseek-ai/DeepSeek-V3-Base/resolve/main/configuration_deepseek.py
#
# It has been modified from its original forms to accommodate naming convention
# and usage patterns of the TorchTitan project.

import math
from typing import Optional, Tuple

import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm_mem
import torch.nn.functional as F
import torch.utils.checkpoint
from symm_mem_recipes import OnDeviceAllToAllV
from torch import nn
from torch.distributed._functional_collectives import all_to_all_single_autograd

from torchtitan.experiments.kernels.group_gemms.group_gemms import (
    DSGroupGEMM,
    TorchAOBF16GroupGEMM,
    TorchBF16GroupGEMM,
    TorchFP8GroupGEMM,
    TritonCGBF16GroupGEMM,
)

from torchtitan.experiments.kernels.moe.indices import generate_permute_indices
from torchtitan.experiments.kernels.triton_mg_group_gemm.torchao_pr import ALIGN_SIZE_M
from torchtitan.tools.logging import logger

from .mlp import MLP


# Get model parallel subgroup by name:
# e.g. "pp", "ep", None
def get_group(dim_name: Optional[str] = None) -> dist.ProcessGroup:
    glob = torch.distributed.device_mesh._mesh_resources.get_current_mesh()
    return glob.get_group(dim_name)


class MoEGate(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.top_k = config.num_experts_per_tok
        self.n_routed_experts = config.n_routed_experts
        self.routed_scaling_factor = config.routed_scaling_factor
        self.scoring_func = config.scoring_func
        self.seq_aux = config.seq_aux
        self.topk_method = config.topk_method
        self.n_group = config.n_group
        self.topk_group = config.topk_group

        # topk selection algorithm
        self.norm_topk_prob = config.norm_topk_prob
        self.gating_dim = config.hidden_size
        self.weight = nn.Parameter(
            torch.empty((self.n_routed_experts, self.gating_dim))
        )
        if self.topk_method == "noaux_tc":
            self.e_score_correction_bias = nn.Parameter(
                # Changed from torch.empty to torch.rand to avoid non-even
                # distribution for runs without actual weigths
                torch.rand((self.n_routed_experts))
            )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        import torch.nn.init as init

        init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def forward(self, hidden_states):
        bsz, seq_len, h = hidden_states.shape
        # compute gating score
        hidden_states = hidden_states.view(-1, h)
        logits = F.linear(
            hidden_states.type(torch.float32), self.weight.type(torch.float32), None
        )
        if self.scoring_func == "sigmoid":
            scores = logits.sigmoid()
        elif self.scoring_func == "softmax":
            scores = logits.softmax(dim=-1, dtype=torch.float32)
        else:
            raise NotImplementedError(
                f"insupportable scoring function for MoE gating: {self.scoring_func}"
            )

        # select top-k experts
        if self.topk_method == "noaux_tc":
            scores_for_choice = scores.view(
                bsz * seq_len, -1
            ) + self.e_score_correction_bias.unsqueeze(0)
            group_scores = (
                scores_for_choice.view(bsz * seq_len, self.n_group, -1)
                .topk(2, dim=-1)[0]
                .sum(dim=-1)
            )  # [n, n_group]
            group_idx = torch.topk(
                group_scores, k=self.topk_group, dim=-1, sorted=False
            )[
                1
            ]  # [n, top_k_group]
            group_mask = torch.zeros_like(group_scores)  # [n, n_group]
            group_mask.scatter_(1, group_idx, 1)  # [n, n_group]
            score_mask = (
                group_mask.unsqueeze(-1)
                .expand(
                    bsz * seq_len, self.n_group, self.n_routed_experts // self.n_group
                )
                .reshape(bsz * seq_len, -1)
            )  # [n, e]
            tmp_scores = scores_for_choice.masked_fill(
                ~score_mask.bool(), 0.0
            )  # [n, e]
            _, topk_idx = torch.topk(tmp_scores, k=self.top_k, dim=-1, sorted=False)
            topk_weight = scores.gather(1, topk_idx)
        elif self.topk_method == "greedy":
            topk_weight, topk_idx = torch.topk(
                scores, k=self.top_k, dim=-1, sorted=False
            )
        else:
            raise NotImplementedError(
                f"insupportable TopK function for MoE gating: {self.topk_method}"
            )

        # norm gate to sum 1
        if self.top_k > 1 and self.norm_topk_prob:
            denominator = topk_weight.sum(dim=-1, keepdim=True) + 1e-20
            topk_weight = topk_weight / denominator
        topk_weight = (
            topk_weight * self.routed_scaling_factor
        )  # must multiply the scaling factor

        return topk_idx, topk_weight


class MoE(nn.Module):
    """
    A mixed expert module containing shared experts.
    """

    # Class attributes:
    # Two shuffle method supported:
    # 1. "torch_all_to_all"
    # 2. "symm_mem" (see `setup_symm_mem` below)
    shuffle_method = "torch_all_to_all"

    # Symmetric memory buffers shared by all MoE instances across layers
    token_send_buf: Optional[torch.Tensor] = None
    token_gather_buf: Optional[torch.Tensor] = None

    # Group GEMM strategies
    group_gemm_strategies = None
    # which group gemm to use?
    group_mm = "torch"  # fp8 options = ["torchfp8", "dsgemm"] bf16 = ["torch", , "torchao", "tritoncg"]

    def __init__(self, config, layer_idx, token_tracker):
        super().__init__()
        self.config = config
        self.num_experts_per_tok = config.num_experts_per_tok
        # do we use triton kernel for input(activation) quantization or the default dsgemm utils (Pytorch eager based)
        self.activation_function = MLP.act_fn
        self.layer_idx = layer_idx
        self.layer_token_tracker = token_tracker

        # ep_size is the number of ranks in expert dimension
        if config.ep_size <= 1:
            raise ValueError(
                "For code simplicity, this model only supports distributed experts, "
                "thus EP size must be > 1, please modify your model config"
            )
        self.ep_group = get_group("ep")
        assert config.ep_size == self.ep_group.size()
        self.ep_size = config.ep_size
        self.ep_rank = self.ep_group.rank()
        self.experts_per_rank = config.n_routed_experts // config.ep_size

        # Use ModuleDict instead of ModuleList to preserve absolute expert
        # IDs while avoiding `None` experts. The absolute expert IDs match
        # with checkpoint FQNs.
        self.experts = nn.ModuleDict()
        for i in range(self.experts_per_rank):
            abs_expert_id = self.ep_rank * self.experts_per_rank + i
            self.experts[str(abs_expert_id)] = MLP(
                config, intermediate_size=config.moe_intermediate_size
            )
        self.gate = MoEGate(config)
        if config.n_shared_experts is not None:
            intermediate_size = config.moe_intermediate_size * config.n_shared_experts
            self.shared_experts = MLP(
                config=config, intermediate_size=intermediate_size
            )

        # Group Gemm
        # Initialize group GEMM strategies if not already loaded
        if MoE.group_gemm_strategies is None:
            MoE._initialize_group_gemm_strategies()

        assert (
            MoE.group_mm in MoE.group_gemm_strategies
        ), f"selected group gemm {self.group_mm} is not avaiable!"
        # keep active gg ready
        self.group_gemm_instance = MoE.group_gemm_strategies[MoE.group_mm]
        self._buffer_initialized = False

    @classmethod
    def _initialize_group_gemm_strategies(cls):
        """Initialize available group GEMM strategies"""
        cls.group_gemm_strategies = {
            "torch": TorchBF16GroupGEMM(MLP.act_fn),
            "torchao": (
                TorchAOBF16GroupGEMM(MLP.act_fn)
                if TorchAOBF16GroupGEMM.is_available()
                else None
            ),
            "torchfp8": (
                TorchFP8GroupGEMM(MLP.act_fn)
                if TorchFP8GroupGEMM.is_available()
                else None
            ),
            "dsgemm": (
                DSGroupGEMM(MLP.act_fn, use_triton_quant=True)
                if DSGroupGEMM.is_available()
                else None
            ),
            "tritoncg": (
                TritonCGBF16GroupGEMM(
                    MLP.act_fn,
                )
                if TritonCGBF16GroupGEMM.is_available()
                else None
            ),
        }

    def combine_experts(self, submod_name: str):
        all_weights = []
        for expert in self.experts.values():

            lin = expert.get_submodule(submod_name)
            all_weights.append(lin.weight)
            lin.weight = None

        # let the group gemm strategy prep the final weight layout
        combined_weight = self.group_gemm_instance.arrange_expert_weights(
            all_weights, submod_name, self
        )

        if combined_weight is None:
            raise NotImplementedError("expert weights not handled by group gemmm")

        self.register_parameter(f"{submod_name}_weight", nn.Parameter(combined_weight))

    # This function is used to create a symm mem buffer for MoE's. It is for
    # shuffling tokens fully "on-device", as compared to traditional torch
    # all_to_all APIs which require a GPU-to-CPU sync of the splits.  If a user
    # calls this function, the `shuffle_method` would switch from
    # `torch_all_to_all` to `symm_mem`.
    def setup_symm_mem(self, dtype: torch.dtype, device: torch.device):
        # Switch shuffle method
        self.shuffle_method = "symm_mem"

        # Combine expert weights
        self.combine_experts("gate_proj")
        self.combine_experts("up_proj")
        self.combine_experts("down_proj")

        # Assuming worst case, 2x tokens are routed to one EP rank
        overflow = 2
        OnDeviceAllToAllV.max_output_len = (
            self.config.max_seq_len * self.num_experts_per_tok * overflow
        )

        # Symmetric memory buffers are shared by all MoE instances across
        # layers, we only need to initialize them once
        if MoE.token_send_buf is not None:
            return

        # Input buffer for DP-to-EP shuffle
        MoE.token_send_buf = symm_mem.empty(
            self.config.max_seq_len
            * self.num_experts_per_tok,  # seq len * top k (flattened)
            self.config.hidden_size,  # hidden dim
            dtype=dtype,
            device=device,
        )
        # Input buffer for EP-to-DP shuffle
        MoE.token_gather_buf = symm_mem.empty(
            self.config.max_seq_len
            * self.num_experts_per_tok  # seq len * top k (flattened)
            * overflow,
            self.config.hidden_size,  # hidden dim
            dtype=dtype,
            device=device,
        )

    def get_send_buf(self):
        # [Why detach?] During a first forward-backward step, the buffer would
        # be included in a computational graph. In a second step, autograd will
        # return an error saying "Trying to backward through the graph a second
        # time (or directly access saved tensors more than once)". This is
        # because the buffer is still in the graph, and autograd is trying to
        # backward through the graph a second time. To avoid this, we detach the
        # buffer from the graph. `detach()` returns a new tensor, which shares
        # the same storage with the original one.
        self.token_send_buf.grad = None
        return self.token_send_buf.detach()

    def get_gather_buf(self):
        # See [Why detach?] in `get_send_buf`
        self.token_gather_buf.grad = None
        return self.token_gather_buf.detach()

    def forward(self, hidden_states):
        identity = hidden_states
        orig_shape = hidden_states.shape
        # for each token, select top-k experts, and compute the weight for each expert
        topk_idx, topk_weight = self.gate(hidden_states)
        # token tracking
        logger.info(f"MoE layer {self.layer_idx} topk_idx: {topk_idx}")
        self.layer_token_tracker.record_expert_assignment(self.layer_idx, topk_idx)

        hidden_states = hidden_states.view(-1, hidden_states.shape[-1])
        if self.shuffle_method == "symm_mem":
            y = self.moe_on_device(hidden_states, topk_idx, topk_weight)
        else:  # "torch_all_to_all"
            y = self.moe_forward(hidden_states, topk_idx, topk_weight)

        y = y.view(*orig_shape)
        if self.config.n_shared_experts is not None:
            y = y + self.shared_experts(identity)
        return y

    def moe_forward(self, x, topk_ids, topk_weight):
        (
            sorted_tokens,
            token_indices,
            tokens_per_expert,
        ) = self.sort_tokens(x, topk_ids, topk_weight)

        # keep the seqlen dimension for later use without holding onto the sorted tokens
        seqlen_sorted_tokens = sorted_tokens.shape[0]

        # all to all
        # This part exchange the information about the number of tokens send and
        # received by each expert. We can understand this information as "side
        # band", which is not part of the actual data. Thus no gradient is
        # needed.

        # Sum the tokens over local experts, then we get tokens per EP rank,
        # which is the input splits
        with torch.no_grad():
            tokens_per_expert_group = tokens_per_expert.new_empty(
                tokens_per_expert.shape[0]
            )
            dist.all_to_all_single(
                tokens_per_expert_group, tokens_per_expert, group=self.ep_group
            )
            input_splits = tokens_per_expert.view(self.ep_size, -1).sum(dim=1)

        # DP to EP token shuffle. This part needs gradient.
        if self.shuffle_method == "symm_mem":
            # Move input to the `token_send_buf` symm mem
            token_send_buf = self.get_send_buf()
            token_send_buf[: token_indices.shape[0]].copy_(sorted_tokens)
            # Note: `out=` avoids copy, but it is not differentiable
            # torch.index_select(x, 0, idxs // topk_ids.shape[1], out=self.token_send_buf[: idxs.shape[0]])
            token_gather_buf, output_splits = OnDeviceAllToAllV.apply(
                token_send_buf,
                input_splits,
                self.ep_group,
            )
            with torch.no_grad():
                # Received tokens from all other ranks. TODO: use mask instead
                received = output_splits.sum()
            # TODO: don't use `received`
            gathered_tokens = token_gather_buf[:received]
        else:  # "torch_all_to_all"
            # Prepare input ans output splits
            with torch.no_grad():
                output_splits = tokens_per_expert_group.view(self.ep_size, -1).sum(
                    dim=1
                )
            gathered_tokens = all_to_all_single_autograd(
                sorted_tokens,
                output_splits.tolist(),
                input_splits.tolist(),
                self.ep_group,
            )

        # This part prepares a 1D tensor with the same length as
        # `gathered_tokens`. The 1D tensor is filled with local expert IDs which
        # the tokens in `gathered_tokens` are headed for. This part doesn't need
        # gradient.
        with torch.no_grad():
            gatherd_idxs = (
                torch.arange(
                    tokens_per_expert_group.numel(),
                    device=tokens_per_expert_group.device,
                )
                % self.experts_per_rank
            )
            gatherd_idxs = gatherd_idxs.repeat_interleave(tokens_per_expert_group)

        # Prepare buffer for tokens processed by experts
        if self.shuffle_method == "symm_mem":
            # Take necessary space from `token_gather_buf` symm mem because we are
            # going to send them out after expert processing
            processed_tokens = self.get_gather_buf()[: gathered_tokens.shape[0]]
        else:  # "torch_all_to_all"
            processed_tokens = torch.empty_like(gathered_tokens)

        # This part processes the tokens routed to the local experts.
        # TODO: can we use group GEMM here?
        for i, expert in enumerate(self.experts.values()):
            processed_tokens[gatherd_idxs == i] = expert(
                gathered_tokens[gatherd_idxs == i]
            )

        # Now shuffle the tokens back to their original owner, i.e. EP to DP shuffle.
        # The input/output splits are just a reverse of the previous shuffle.
        if self.shuffle_method == "symm_mem":
            token_return_buf, _ = OnDeviceAllToAllV.apply(
                processed_tokens,
                output_splits,
                self.ep_group,
            )
            returned_tokens = token_return_buf[:seqlen_sorted_tokens]
        else:  # "torch_all_to_all"
            returned_tokens = all_to_all_single_autograd(
                processed_tokens,
                input_splits.tolist(),
                output_splits.tolist(),
                self.ep_group,
            )

        output_tokens = torch.empty_like(returned_tokens)
        output_tokens[token_indices] = returned_tokens
        final_out = (
            output_tokens.view(*topk_ids.shape, -1)
            .type(topk_weight.dtype)
            .mul_(topk_weight.unsqueeze(dim=-1))
            .sum(dim=1)
            .type(returned_tokens.dtype)
        )
        return final_out

    def sort_tokens(self, x, topk_ids, topk_weights):
        # This part sorts the token indices so that tokens routed to the same expert reside consecutively.
        # An implication is that tokens to the same "expert group" (i.e., device) are also consecutive.
        # Since this is an "aritificial" index creation (final outcome being
        # `idxs`), we don't need gradients here.

        with torch.no_grad():
            # [seq_len, n_routed_experts]
            expert_counts = topk_ids.new_zeros(
                (topk_ids.shape[0], self.config.n_routed_experts)
            )
            # Fill 1 to the selected experts
            expert_counts.scatter_(1, topk_ids, 1)
            tokens_per_expert = expert_counts.sum(dim=0)
            # Token indices for each expert
            token_indices = topk_ids.view(-1).argsort()

        sorted_tokens = x[token_indices // topk_ids.shape[1]]
        # assert sorted_tokens.shape == sorted_tokens_shape

        return (sorted_tokens, token_indices, tokens_per_expert)

    # ------- Group GEMM implementation ------

    def _run_group_gemm(self, contig_tokens, m_sizes, m_offsets):
        """Run the appropriate group GEMM implementation based on configuration"""

        try:
            return self.group_gemm_strategies[self.group_mm].execute(
                contig_tokens, m_sizes, m_offsets, self
            )
        except Exception as e:
            # Flag the error
            print(f"Error using {self.group_mm} strategy: {e}")

    def moe_on_device(self, x, topk_ids, topk_weight):
        (
            sorted_tokens,
            token_indices,
            tokens_per_expert,
        ) = self.sort_tokens(x, topk_ids, topk_weight)

        # keep the seqlen dimension for later use without holding onto the sorted tokens
        seqlen_sorted_tokens = sorted_tokens.shape[0]

        # This part exchange the information about the number of tokens send and
        # received by each expert. We can understand this information as "side
        # band", which is not part of the actual data. Thus no gradient is
        # needed.

        # Sum the tokens over local experts, then we get tokens per EP rank,
        # which is the input splits
        with torch.no_grad():
            tokens_per_expert_group = tokens_per_expert.new_empty(
                tokens_per_expert.shape[0]
            )
            dist.all_to_all_single(
                tokens_per_expert_group, tokens_per_expert, group=self.ep_group
            )
            input_splits = tokens_per_expert.view(self.ep_size, -1).sum(dim=1)

        # Move input to the `token_send_buf` symm mem
        token_send_buf = self.get_send_buf()
        token_send_buf[: token_indices.shape[0]].copy_(sorted_tokens)
        # Note: `out=` avoids copy, but it is not differentiable
        # torch.index_select(x, 0, idxs // topk_ids.shape[1], out=token_send_buf[: idxs.shape[0]])
        token_gather_buf, output_splits = OnDeviceAllToAllV.apply(
            token_send_buf,
            input_splits,
            self.ep_group,
        )

        # We need to permute the received tokens so that tokens for the same expert are contiguous.
        # This part prepares a 1D tensor `permuted_indices` for such permutation.
        # This part doesn't need gradient.
        with torch.no_grad():
            permuted_indices, m_sizes, m_offsets = generate_permute_indices(
                tokens_per_expert_group,
                self.experts_per_rank,
                self.ep_size,
                token_gather_buf.shape[0],
                ALIGN_SIZE_M,
            )

        # Permute the received tokens so that tokens for the same expert are contiguous.
        contig_tokens = token_gather_buf[permuted_indices]

        # group gemm - handle all three group gemms (up, gate, down for all experts)
        hidden_outputs = self._run_group_gemm(
            contig_tokens,
            m_sizes,
            m_offsets,
        )

        # Prepare buffer for tokens processed by experts
        processed_tokens = self.get_gather_buf()

        # Move into Symmetric Memory for the return shuffle
        processed_tokens[permuted_indices] = hidden_outputs

        # Now shuffle the tokens back to their original owner, i.e. EP to DP shuffle.
        # The input/output splits are just a reverse of the previous shuffle.
        token_return_buf, _ = OnDeviceAllToAllV.apply(
            processed_tokens,
            output_splits,
            self.ep_group,
        )

        returned_tokens = token_return_buf[:seqlen_sorted_tokens]
        output_tokens = torch.empty_like(returned_tokens)
        output_tokens[token_indices] = returned_tokens

        final_out = (
            output_tokens.view(*topk_ids.shape, -1)
            .type(topk_weight.dtype)
            .mul_(topk_weight.unsqueeze(dim=-1))
            .sum(dim=1)
            .type(returned_tokens.dtype)
        )

        return final_out
