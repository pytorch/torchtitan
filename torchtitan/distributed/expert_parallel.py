# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from abc import ABC, abstractmethod

import torch
import torch.nn as nn
from torch import Tensor
from torch.distributed._functional_collectives import (
    all_to_all_single,
    all_to_all_single_autograd,
)
from torch.distributed.tensor import (
    DeviceMesh,
    distribute_module,
    distribute_tensor,
    DTensor,
    Shard,
)
from torch.distributed.tensor.parallel import ParallelStyle


def _generate_permute_indices(
    tokens_per_expert_group: torch.Tensor,
    experts_per_rank: int,
    num_ranks: int,
):
    """Generate indices to reorder tokens from rank-major to expert-major layout.

    Args:
        tokens_per_expert_group: shape ``[num_ranks * experts_per_rank]``.

    Input layout:  (e0,r0), (e1,r0), ..., (e0,r1), (e1,r1), ...  (rank-major)
    Output layout: (e0,r0), (e0,r1), ..., (e1,r0), (e1,r1), ...  (expert-major)
    """
    device = tokens_per_expert_group.device
    total = tokens_per_expert_group.sum()

    # [R, E] matrix of token counts per (rank, expert)
    t_mat = tokens_per_expert_group.view(num_ranks, experts_per_rank)

    # Where each (r, e) segment starts in the input (rank-major order)
    input_starts = (tokens_per_expert_group.cumsum(0) - tokens_per_expert_group).view(
        num_ranks, experts_per_rank
    )

    # Transpose to expert-major [E, R] and flatten
    segment_lens = t_mat.t().reshape(-1)
    input_starts = input_starts.t().reshape(-1)

    # For each output position, find its input position:
    #   output[p] = input[input_starts[seg] + (p - output_starts[seg])]
    seg_ids = torch.arange(segment_lens.shape[0], device=device).repeat_interleave(
        segment_lens
    )
    output_starts = segment_lens.cumsum(0) - segment_lens
    permuted_indices = (
        input_starts[seg_ids]
        + torch.arange(total, device=device)  # pyrefly: ignore [no-matching-overload]
        - output_starts[seg_ids]
    )

    num_tokens_per_expert = t_mat.sum(0)
    return permuted_indices, num_tokens_per_expert


def _permute(x, num_tokens_per_expert, ep_degree, num_local_experts):
    permuted_indices, num_tokens_per_expert = _generate_permute_indices(
        num_tokens_per_expert, num_local_experts, ep_degree
    )
    return x.shape, x[permuted_indices, :], permuted_indices, num_tokens_per_expert


def _unpermute(out, input_shape, permuted_indices):
    out_unpermuted = out.new_empty(input_shape)
    out_unpermuted[permuted_indices, :] = out
    return out_unpermuted


class BaseExpertParallel(ParallelStyle, ABC):
    @abstractmethod
    def _partition_fn(self, name: str, mod: nn.Module, device_mesh: DeviceMesh) -> None:
        ...

    @abstractmethod
    def _token_dispatch(
        self, mod: nn.Module, inputs: tuple, device_mesh: DeviceMesh
    ) -> tuple[Tensor, Tensor]:
        ...

    @abstractmethod
    def _token_combine(
        self, mod: nn.Module, routed_output: Tensor, device_mesh: DeviceMesh
    ) -> Tensor:
        ...


# implementation of Tensor Parallel for the GroupedExperts in MoE
class TensorParallel(ParallelStyle):
    def _partition_fn(self, name, module, device_mesh):
        # w1 shape = (experts, out_dim, in_dim)
        module.register_parameter(
            "w1", nn.Parameter(distribute_tensor(module.w1, device_mesh, [Shard(1)]))
        )  # Column-wise sharding

        # w2 shape = (experts, in_dim, out_dim)
        module.register_parameter(
            "w2",
            nn.Parameter(distribute_tensor(module.w2, device_mesh, [Shard(2)])),
        )  # Row-wise sharding

        # w3 shape = (experts, out_dim, in_dim)
        module.register_parameter(
            "w3",
            nn.Parameter(distribute_tensor(module.w3, device_mesh, [Shard(1)])),
        )  # Column-wise sharding

    def _apply(self, module: nn.Module, device_mesh: DeviceMesh) -> nn.Module:
        return distribute_module(
            module,
            device_mesh,
            self._partition_fn,
        )


class ExpertParallel(BaseExpertParallel):
    def __init__(self):
        super().__init__()
        self.input_splits = None
        self.output_splits = None
        self.input_shape: torch.Size | None = None
        self.permuted_indices = None

    def _partition_fn(self, name: str, mod: nn.Module, device_mesh: DeviceMesh) -> None:
        for param_name, param in mod.named_parameters(recurse=False):
            dist_param = nn.Parameter(distribute_tensor(param, device_mesh, [Shard(0)]))
            mod.register_parameter(param_name, dist_param)

    def _token_dispatch(
        self, mod: nn.Module, inputs: tuple, device_mesh: DeviceMesh
    ) -> tuple[Tensor, Tensor]:
        # annotate module input placements/sharding with input_layouts
        routed_input, num_tokens_per_expert = inputs
        ep_degree = device_mesh.shape[0]
        num_local_experts = num_tokens_per_expert.shape[0] // ep_degree

        # generate the input splits and output splits for all-to-all
        with torch.no_grad():
            num_tokens_per_expert_group = all_to_all_single(
                num_tokens_per_expert,
                None,
                None,
                group=device_mesh.get_group(),
            )
            # Need to wait explicitly because it is used by a triton kernel later
            # which doesn't realize that AsyncCollectiveTensor needs unwrapping
            num_tokens_per_expert_group = torch.ops._c10d_functional.wait_tensor(
                num_tokens_per_expert_group
            )
            input_splits = (
                num_tokens_per_expert.view(ep_degree, -1)
                .sum(dim=1)
                .to(torch.device("cpu"), non_blocking=True)
            )
            # NOTE: this would incur a device-to-host sync
            output_splits = (
                num_tokens_per_expert_group.view(ep_degree, -1)
                .sum(dim=1)
                .to(torch.device("cpu"), non_blocking=False)
            )
            self.input_splits = input_splits.tolist()
            self.output_splits = output_splits.tolist()

        # perform all-to-all
        routed_input = all_to_all_single_autograd(
            routed_input,
            self.output_splits,
            self.input_splits,
            device_mesh.get_group(),
        )

        # NOTE: After this all-to-all, the routed input is put on proper EP rank.
        # However, the num_tokens_per_expert_group is not of the final target format
        # [#tokens for local expert 0, #tokens for local expert 1, ...]
        # Rather, it is of the format
        # [#tokens for local expert 0 from EP rank 0, #tokens for local expert 1 from EP rank 0, ...,
        #  #tokens for local expert 0 from EP rank 1, #tokens for local expert 1 from EP rank 1, ...]
        # We need to perform another shuffle to get the correct layout, via the _permute function below.

        (
            self.input_shape,
            routed_input,
            self.permuted_indices,
            num_tokens_per_expert_group,
        ) = _permute(
            routed_input, num_tokens_per_expert_group, ep_degree, num_local_experts
        )

        return routed_input, num_tokens_per_expert_group

    def _token_combine(
        self, mod: nn.Module, routed_output: Tensor, device_mesh: DeviceMesh
    ) -> Tensor:
        routed_output = _unpermute(
            routed_output, self.input_shape, self.permuted_indices
        )

        routed_output = all_to_all_single_autograd(
            routed_output,
            self.input_splits,
            self.output_splits,
            device_mesh.get_group(),
        )
        return routed_output

    def _apply(self, module: nn.Module, device_mesh: DeviceMesh) -> nn.Module:
        return distribute_module(
            module,
            device_mesh,
            partition_fn=self._partition_fn,
            # pyrefly: ignore [bad-argument-type]
            input_fn=self._token_dispatch,
            # pyrefly: ignore [bad-argument-type]
            output_fn=self._token_combine,
        )


# This class is for dp2ep with TP (without TP we can just use ExpertParallel)
class ExpertTensorParallel(ExpertParallel):
    def _token_dispatch(self, mod, inputs, device_mesh):
        # token dispatch happens on the EP mesh, whereas device_mesh is [ep, tp] mesh
        return super()._token_dispatch(mod, inputs, device_mesh["ep"])

    def _partition_fn(self, name: str, mod: nn.Module, device_mesh: DeviceMesh) -> None:
        # w1 shape = (experts, out_dim, in_dim)
        mod.register_parameter(
            "w1",
            # pyrefly: ignore [bad-argument-type]
            nn.Parameter(distribute_tensor(mod.w1, device_mesh, [Shard(0), Shard(1)])),
        )  # Column-wise sharding

        # w2 shape = (experts, in_dim, out_dim)
        mod.register_parameter(
            "w2",
            # pyrefly: ignore [bad-argument-type]
            nn.Parameter(distribute_tensor(mod.w2, device_mesh, [Shard(0), Shard(2)])),
        )  # Row-wise sharding

        # w3 shape = (experts, out_dim, in_dim)
        mod.register_parameter(
            "w3",
            # pyrefly: ignore [bad-argument-type]
            nn.Parameter(distribute_tensor(mod.w3, device_mesh, [Shard(0), Shard(1)])),
        )  # Column-wise sharding

    def _token_combine(self, mod, routed_output, device_mesh):
        # token combine happens on the EP mesh, whereas device_mesh is [ep, tp] mesh
        return super()._token_combine(mod, routed_output, device_mesh["ep"])

    def _apply(self, module: nn.Module, device_mesh: DeviceMesh) -> nn.Module:
        return distribute_module(
            module,
            device_mesh,
            partition_fn=self._partition_fn,
            # pyrefly: ignore [bad-argument-type]
            input_fn=self._token_dispatch,
            # pyrefly: ignore [bad-argument-type]
            output_fn=self._token_combine,
        )


# This class is to support Sequence Parallel for ETP=1
# when EP borrows from all TP and part of DP
class ReordererSequenceParallel(ParallelStyle):
    def __init__(self):
        super().__init__()

    def _prepare_inputput_fn(self, mod, inputs, device_mesh):
        # shape (batch_size*seq_len, top_k)
        top_scores, selected_experts_indices = inputs
        num_tokens, _ = top_scores.shape

        # NOTE: If needed, we can pad tokens in case bs*slen is not divisible by TP degree
        # if top_scores.shape[0] % device_mesh.size() != 0:
        #     num_tokens = top_scores.shape[0]
        #     tp_size = device_mesh.size()
        #     n_pad = (num_tokens // tp_size + 1) * tp_size - num_tokens
        #     selected_experts_indices = F.pad(selected_experts_indices, [0, 0, 0, n_pad])
        #     top_scores = F.pad(top_scores, [0, 0, 0, n_pad])

        def _split_along_first_dim(x: torch.Tensor) -> torch.Tensor:
            assert x.is_contiguous()
            if num_tokens % device_mesh.size() != 0:
                raise ValueError(
                    "Uneven split of tokens of is not supported yet. "
                    "Requires EP degree dividing batch size * seq len."
                )
            local_num_tokens = num_tokens // device_mesh.size()
            local_rank = device_mesh.get_local_rank()
            offset = local_rank * local_num_tokens
            output = x[offset : offset + local_num_tokens]

            return output

        top_scores = _split_along_first_dim(top_scores)
        selected_experts_indices = _split_along_first_dim(selected_experts_indices)

        # shape (batch_size * seq_len // ep_degree, top_k)
        return top_scores, selected_experts_indices

    def _prepare_output_fn(self, mod, outputs, device_mesh):
        # shape (batch_size * seq_len * top_k // ep_degree)
        top_scores, token_indices_experts_sorted, num_tokens_per_expert = outputs

        # NOTE: As we shard routed tokens along bs*slen dim across the TP ranks,
        #       the MoE gather and scatter still require global token indices.
        local_rank = device_mesh.get_local_rank()
        if not hasattr(mod, "top_k"):
            raise ValueError(
                "TokenReorderer class in MoE should always have top_k attribute."
            )
        token_indices_experts_sorted = (
            token_indices_experts_sorted + top_scores.shape[0] // mod.top_k * local_rank
        )

        return top_scores, token_indices_experts_sorted, num_tokens_per_expert

    def _apply(self, module: nn.Module, device_mesh: DeviceMesh) -> nn.Module:
        return distribute_module(
            module,
            device_mesh,
            partition_fn=None,
            # pyrefly: ignore [bad-argument-type]
            input_fn=self._prepare_inputput_fn,
            # pyrefly: ignore [bad-argument-type]
            output_fn=self._prepare_output_fn,
        )


class DeepEPExpertParallel(BaseExpertParallel):
    """Expert Parallel using DeepEP/HybridEP for efficient token dispatch/combine.

    Uses DeepEP library kernels instead of standard all-to-all collectives for
    token dispatch and combine. Designed for use with DeepEPMoE, which passes
    routing info through the experts call so dispatch/combine hooks can handle
    communication.

    Expects inputs as:
        (hidden_states, num_tokens_per_expert, selected_experts_indices, top_scores, num_experts)

    Args:
        score_before_experts: If True, apply routing scores before expert computation.
        comm_backend: "deepep" for H100/NVLink Switch, "hybridep" for GB200/NVLink72.
        hybridep_non_blocking_expert_capacity_factor: None = blocking mode (default).
            float in (0, 1] = non-blocking mode; controls the fused-permute
            output tensor size (num_permuted_tokens). Only used with hybridep.
        pad_multiple: Alignment size for token groups needed by quantized grouped
            GEMMs (e.g. 16 for FP8, 32 for MXFP8). Only supported with hybridep.
            None means no padding.
    """

    def __init__(
        self,
        score_before_experts: bool = True,
        comm_backend: str = "deepep",
        hybridep_non_blocking_expert_capacity_factor: float | None = None,
        pad_multiple: int | None = None,
    ):
        super().__init__()
        self._state = None  # State preserved between dispatch and combine
        self.score_before_experts = score_before_experts
        self.comm_backend = comm_backend
        self.hybridep_non_blocking_expert_capacity_factor = (
            hybridep_non_blocking_expert_capacity_factor
        )
        self.pad_multiple = pad_multiple
        # Import to register custom ops so SAC saves communication outputs
        # instead of recomputing them. This must happen before apply_ac.
        if comm_backend == "hybridep":
            from torchtitan.distributed.deepep import hybridep  # noqa: F401
        else:
            from torchtitan.distributed.deepep import deepep  # noqa: F401

    def _token_dispatch(self, mod, inputs, device_mesh):
        """Dispatch tokens via DeepEP or HybridEP based on configured backend."""
        hidden_states, _, selected_experts_indices, top_scores, num_experts = inputs
        if isinstance(mod.w1, DTensor):
            num_local_experts = mod.w1.to_local().shape[0]
        else:
            num_local_experts = mod.w1.shape[0]
        ep_group = device_mesh.get_group()

        if self.comm_backend == "hybridep":
            from torchtitan.distributed.deepep.hybridep import dispatch_tokens

            hidden_states, tokens_per_expert, self._state = dispatch_tokens(
                hidden_states,
                selected_experts_indices,
                top_scores,
                num_local_experts,
                num_experts,
                ep_group,
                score_before_experts=self.score_before_experts,
                non_blocking_expert_capacity_factor=self.hybridep_non_blocking_expert_capacity_factor,
                pad_multiple=self.pad_multiple,
            )
        else:
            from torchtitan.distributed.deepep.deepep import dispatch_tokens

            hidden_states, tokens_per_expert, self._state = dispatch_tokens(
                hidden_states,
                selected_experts_indices,
                top_scores,
                num_local_experts,
                num_experts,
                ep_group,
                score_before_experts=self.score_before_experts,
            )

        return hidden_states, tokens_per_expert

    @staticmethod
    def _partition_fn(name, mod, device_mesh):
        """Shard expert weights on expert dimension."""
        for param_name, param in mod.named_parameters(recurse=False):
            mod.register_parameter(
                param_name,
                nn.Parameter(distribute_tensor(param, device_mesh, [Shard(0)])),
            )

    def _token_combine(self, mod, routed_output, device_mesh):
        """Combine tokens via DeepEP or HybridEP based on configured backend."""
        if self.comm_backend == "hybridep":
            from torchtitan.distributed.deepep import hybridep

            routed_output = hybridep.combine_tokens(
                routed_output,
                self._state,  # pyrefly: ignore [bad-argument-type]
                pad_multiple=self.pad_multiple,
            )
        else:
            from torchtitan.distributed.deepep.deepep import combine_tokens

            # pyrefly: ignore [bad-argument-type]
            routed_output = combine_tokens(routed_output, self._state)

        self._state = None
        return routed_output

    def _apply(self, module: nn.Module, device_mesh: DeviceMesh) -> nn.Module:
        """Apply DeepEP/HybridEP parallelization."""
        return distribute_module(
            module,
            device_mesh,
            partition_fn=DeepEPExpertParallel._partition_fn,
            input_fn=self._token_dispatch,  # pyrefly: ignore [bad-argument-type]
            output_fn=self._token_combine,  # pyrefly: ignore [bad-argument-type]
        )


class TorchAOExpertParallel(ExpertParallel):
    """Expert Parallel with token group padding for quantized grouped GEMMs.

    Extends ExpertParallel by using torchao's ``permute_and_pad`` to reorder
    tokens into expert-major order and pad each expert's token group to a
    multiple of ``pad_multiple``. This alignment is required by FP8/MXFP8
    quantized grouped GEMM kernels (e.g. 16 for FP8, 32 for MXFP8).

    Args:
        pad_multiple: Alignment size for per-expert token groups.
    """

    def __init__(self, pad_multiple: int) -> None:
        super().__init__()
        self.pad_multiple = pad_multiple

    def _token_dispatch(
        self, mod: nn.Module, inputs: tuple, device_mesh: DeviceMesh
    ) -> tuple[Tensor, Tensor]:
        # annotate module input placements/sharding with input_layouts
        routed_input, num_tokens_per_expert = inputs
        ep_degree = device_mesh.shape[0]
        num_local_experts = num_tokens_per_expert.shape[0] // ep_degree

        # generate the input splits and output splits for all-to-all
        with torch.no_grad():
            num_tokens_per_expert_group = all_to_all_single(
                num_tokens_per_expert,
                None,
                None,
                group=device_mesh.get_group(),
            )
            # Need to wait explicitly because it is used by a triton kernel later
            # which doesn't realize that AsyncCollectiveTensor needs unwrapping
            num_tokens_per_expert_group = torch.ops._c10d_functional.wait_tensor(
                num_tokens_per_expert_group
            )
            input_splits = (
                num_tokens_per_expert.view(ep_degree, -1)
                .sum(dim=1)
                .to(torch.device("cpu"), non_blocking=True)
            )
            # NOTE: this would incur a device-to-host sync
            output_splits = (
                num_tokens_per_expert_group.view(ep_degree, -1)
                .sum(dim=1)
                .to(torch.device("cpu"), non_blocking=False)
            )
            self.input_splits = input_splits.tolist()
            self.output_splits = output_splits.tolist()

        # perform all-to-all
        routed_input = all_to_all_single_autograd(
            routed_input,
            self.output_splits,
            self.input_splits,
            device_mesh.get_group(),
        )

        # NOTE: After this all-to-all, the routed input is put on proper EP rank.
        # However, the num_tokens_per_expert_group is not of the final target format
        # [#tokens for local expert 0, #tokens for local expert 1, ...]
        # Rather, it is of the format
        # [#tokens for local expert 0 from EP rank 0, #tokens for local expert 1 from EP rank 0, ...,
        #  #tokens for local expert 0 from EP rank 1, #tokens for local expert 1 from EP rank 1, ...]
        # We need to perform another shuffle to get the correct layout, via the _permute function below.

        # FP8/MXFP8 require groups to be permuted to expert major order AND padded to nearest multiple of 16.
        # It also does padding to make sure the number of tokens each expert gets locally
        # is a multiple of `self.token_group_alignment_size`.
        # Note that this will create side effects when wrapping the for-loop implementation
        # of GroupedExperts, as it does not need padding.

        from torchao.prototype.moe_training.ep.permute import permute_and_pad

        (
            self.input_shape,
            routed_input,
            self.permuted_indices,
            num_tokens_per_expert_group_padded,
            group_offsets,
        ) = permute_and_pad(
            routed_input,
            num_tokens_per_expert_group,
            ep_degree,
            num_local_experts,
            self.pad_multiple,
        )
        return routed_input, num_tokens_per_expert_group_padded

    def _token_combine(
        self, mod: nn.Module, routed_output: Tensor, device_mesh: DeviceMesh
    ) -> Tensor:
        def _unpermute(out, input_shape, permuted_indices):
            """Unpermute tokens from expert-major to rank-major order."""
            out_unpermuted = out.new_empty(input_shape)
            out_unpermuted[permuted_indices, :] = out
            out_unpermuted = out_unpermuted[:-1]
            return out_unpermuted

        routed_output = _unpermute(
            routed_output, self.input_shape, self.permuted_indices
        )

        routed_output = all_to_all_single_autograd(
            routed_output,
            self.input_splits,
            self.output_splits,
            device_mesh.get_group(),
        )
        return routed_output
