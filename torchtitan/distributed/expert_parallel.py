# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


from functools import partial
from typing import Callable, Literal

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed.tensor import (
    DeviceMesh,
    distribute_module,
    distribute_tensor,
    DTensor,
    Replicate,
    Shard,
)
from torch.distributed.tensor.parallel import ParallelStyle
from torch.distributed.tensor.placement_types import Placement


# from torch.distributed._functional_collectives import all_to_all_single_autograd
# TODO: there is memory leak issue with AC + all_to_all_single_autograd
# This is a temporary fix by @rakkit https://github.com/pytorch/torchtitan/issues/1467
class _A2A(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, out_splits, in_splits, group):
        T_out = int(sum(out_splits))
        y = x.new_empty((T_out,) + tuple(x.shape[1:]))  # allocate by output splits
        dist.all_to_all_single(y, x.contiguous(), out_splits, in_splits, group=group)

        ctx.in_splits = in_splits
        ctx.out_splits = out_splits
        ctx.group = group
        return y

    @staticmethod
    def backward(ctx, grad_y):
        # grad wrt input has length sum(in_splits)
        T_in = int(sum(ctx.in_splits))
        grad_x = grad_y.new_empty((T_in,) + tuple(grad_y.shape[1:]))
        dist.all_to_all_single(
            grad_x, grad_y.contiguous(), ctx.in_splits, ctx.out_splits, group=ctx.group
        )
        return grad_x, None, None, None


def all_to_all_single_autograd(x, out_splits, in_splits, group):
    return _A2A.apply(x, out_splits, in_splits, group)


TOKEN_GROUP_ALIGN_SIZE_M = 8
ValidTokenGroupAlignmentSize = Literal[8, 16, 32]


def set_token_group_alignment_size_m(
    alignment_size: ValidTokenGroupAlignmentSize,
) -> None:
    """
    Set the token group alignment size for token groups in MoE. This is implemented by
    padding each token group size to the next multiple of TOKEN_GROUP_ALIGN_SIZE_M.

    Valid values are: 8, 16, or 32.
    Different values are needed for different cases:

    * For bf16, 8 is enough (16 byte alignment / 2 bytes per elem = 8 elements).
    * For fp8, 16 byte alignment / 1 byte per elem = 16 elements.
    * For mxfp8, we need 32 (or block_size) because scaling block size is (1 x 32),
      so when doing per-token-group quantization on each logically distinct subtensor,
      we need to ensure the contracting dim is divisible by block_size.
      In the backward pass, grad_weight = (grad_output_t @ input).t() has gemm dims
      of (N, M) @ (M, K) so M is the contracting dim, and group offsets are along M,
      so we need 32 element alignment.
    """
    global TOKEN_GROUP_ALIGN_SIZE_M
    TOKEN_GROUP_ALIGN_SIZE_M = alignment_size


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


# NOTE: This is to achieve replicate computation on the gate module in the MoE router.
# It does nothing other than (1) setting the module parameters as DTensors on the given mesh
# and (2) inserting hooks to module boundary to change torch.Tensor to DTensor and back.
# The reason we need this wrapping is to ensure all parameters are on the same 1D/2D mesh,
# which is assumed by (1) gradient norm clipping, and (2) optimizer fused implementation.
class NoParallel(ParallelStyle):
    def __init__(
        self,
        *,
        input_layout: Placement | None = None,
        output_layout: Placement | None = None,
        use_local_output: bool = True,
    ):
        super().__init__()
        self.input_layout = input_layout or Replicate()
        self.output_layout = output_layout or Replicate()
        self.desired_input_layout = Replicate()
        self.use_local_output = use_local_output

    @staticmethod
    def _prepare_input_fn(input_layout, desired_input_layout, mod, inputs, device_mesh):
        # annotate module input placements/sharding with input_layouts
        input_tensor = inputs[0]
        if not isinstance(input_tensor, DTensor):
            input_tensor = DTensor.from_local(
                input_tensor, device_mesh, (input_layout,), run_check=False
            )

        if input_layout != desired_input_layout:
            input_tensor = input_tensor.redistribute(
                placements=(desired_input_layout,), async_op=True
            )
        return (input_tensor, *inputs[1:])

    @staticmethod
    def _prepare_output_fn(output_layout, use_local_output, mod, outputs, device_mesh):
        if outputs.placements != (output_layout,):
            outputs = outputs.redistribute(placements=(output_layout,), async_op=True)
        # back to local tensor
        return outputs.to_local() if use_local_output else outputs

    def _apply(self, module: nn.Module, device_mesh: DeviceMesh) -> nn.Module:
        return distribute_module(
            module,
            device_mesh,
            None,
            partial(
                self._prepare_input_fn, self.input_layout, self.desired_input_layout
            ),
            partial(self._prepare_output_fn, self.output_layout, self.use_local_output),
        )


class ExpertParallel(ParallelStyle):
    def __init__(self):
        super().__init__()
        self.input_splits = None
        self.output_splits = None
        self.input_shape = None
        self.permuted_indices = None

    # performing all-to-all dispatch on the input
    def _token_dispatch(self, mod, inputs, device_mesh):
        # annotate module input placements/sharding with input_layouts
        routed_input, num_tokens_per_expert = inputs
        ep_degree = device_mesh.shape[0]
        num_local_experts = num_tokens_per_expert.shape[0] // ep_degree

        # generate the input splits and output splits for all-to-all
        with torch.no_grad():
            num_tokens_per_expert_group = num_tokens_per_expert.new_empty(
                num_tokens_per_expert.shape[0]
            )
            dist.all_to_all_single(
                num_tokens_per_expert_group,
                num_tokens_per_expert,
                group=device_mesh.get_group(),
            )
            input_splits = (
                num_tokens_per_expert.view(ep_degree, -1)
                .sum(dim=1)
                .to(torch.device("cpu"), non_blocking=True)
            )
            output_splits = (
                num_tokens_per_expert_group.view(ep_degree, -1)
                .sum(dim=1)
                .to(torch.device("cpu"), non_blocking=True)
            )
            # NOTE: this would incur a device-to-host sync
            torch.cuda.current_stream().synchronize()
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
        # We need to perform another shuffle to get the correct layout, via the _permute function
        # below, which also does padding to make sure the number of tokens each expert gets locally
        # is a multiple of TOKEN_GROUP_ALIGN_SIZE_M.

        (
            self.input_shape,
            routed_input,
            self.permuted_indices,
            num_tokens_per_expert_group,
            offsets,
        ) = _permute(
            routed_input, num_tokens_per_expert_group, ep_degree, num_local_experts
        )

        return routed_input, num_tokens_per_expert_group, offsets

    @staticmethod
    def _partition_fn(name, mod, device_mesh):
        # shard on the expert dimension
        for name, param in mod.named_parameters(recurse=False):
            dist_param = nn.Parameter(distribute_tensor(param, device_mesh, [Shard(0)]))
            mod.register_parameter(name, dist_param)

    # performing all-to-all combine on the output
    def _token_combine(self, mod, routed_output, device_mesh):
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
            partition_fn=ExpertParallel._partition_fn,
            input_fn=self._token_dispatch,
            output_fn=self._token_combine,
        )


# This class is for dp2ep with TP (without TP we can just use ExpertParallel)
class ExpertTensorParallel(ExpertParallel):
    def _token_dispatch(self, mod, inputs, device_mesh):
        # token dispatch happens on the EP mesh, whereas device_mesh is [ep, tp] mesh
        return super()._token_dispatch(mod, inputs, device_mesh["ep"])

    def _partition_fn_2d(self, name, mod, ep_tp_mesh):
        # w1 shape = (experts, out_dim, in_dim)
        mod.register_parameter(
            "w1",
            nn.Parameter(distribute_tensor(mod.w1, ep_tp_mesh, [Shard(0), Shard(1)])),
        )  # Column-wise sharding

        # w2 shape = (experts, in_dim, out_dim)
        mod.register_parameter(
            "w2",
            nn.Parameter(distribute_tensor(mod.w2, ep_tp_mesh, [Shard(0), Shard(2)])),
        )  # Row-wise sharding

        # w3 shape = (experts, out_dim, in_dim)
        mod.register_parameter(
            "w3",
            nn.Parameter(distribute_tensor(mod.w3, ep_tp_mesh, [Shard(0), Shard(1)])),
        )  # Column-wise sharding

    def _token_combine(self, mod, routed_output, device_mesh):
        # token combine happens on the EP mesh, whereas device_mesh is [ep, tp] mesh
        return super()._token_combine(mod, routed_output, device_mesh["ep"])

    def _apply(self, module: nn.Module, device_mesh: DeviceMesh) -> nn.Module:
        return distribute_module(
            module,
            device_mesh,
            partition_fn=self._partition_fn_2d,
            input_fn=self._token_dispatch,
            output_fn=self._token_combine,
        )


def _permute(x, num_tokens_per_expert, ep_degree, num_local_experts):
    # TODO: move to core
    from torchtitan.experiments.kernels.moe.indices import generate_permute_indices

    global TOKEN_GROUP_ALIGN_SIZE_M
    with torch.no_grad():
        (permuted_indices, num_tokens_per_expert, offsets,) = generate_permute_indices(
            num_tokens_per_expert,
            num_local_experts,
            ep_degree,
            x.shape[0] + num_local_experts * TOKEN_GROUP_ALIGN_SIZE_M,
            TOKEN_GROUP_ALIGN_SIZE_M,
        )

    x = torch.vstack((x, x.new_zeros((x.shape[-1]))))
    input_shape = x.shape
    x = x[permuted_indices, :]

    return input_shape, x, permuted_indices, num_tokens_per_expert, offsets


def _unpermute(out, input_shape, permuted_indices):
    out_unpermuted = out.new_empty(input_shape)
    out_unpermuted[permuted_indices, :] = out
    out = out_unpermuted[:-1]
    return out


def indices_permutation_wrapper(func: Callable) -> Callable:
    """
    In order to use torch._grouped_mm, we need to make sure the number of
    tokens each expert gets is a multiple of TOKEN_GROUP_ALIGN_SIZE_M. The
    generate_permute_indices kernel also helps achieve this via padding,
    without incurring synchronization between device and host. Note that
    this will create side effects when wrapping the for-loop implementation
    of GroupedExperts, as it does not need padding.
    """

    def wrapper(
        w1: torch.Tensor,
        w2: torch.Tensor,
        w3: torch.Tensor,
        x: torch.Tensor,
        num_tokens_per_expert: torch.Tensor,
        _offsets: torch.Tensor | None = None,
    ) -> torch.Tensor:
        num_local_experts = w1.shape[0]
        ep_degree = num_tokens_per_expert.shape[0] // num_local_experts

        input_shape, x, permuted_indices, num_tokens_per_expert, offsets = _permute(
            x, num_tokens_per_expert, ep_degree, num_local_experts
        )

        out = func(w1, w2, w3, x, num_tokens_per_expert, offsets)

        out = _unpermute(out, input_shape, permuted_indices)

        return out

    return wrapper


# This class is to support Sequence Parallel for ETP=1
# when EP borrows from all TP and part of DP
class ReordererSequenceParallel(ParallelStyle):
    def _prepare_inputput_fn(self, mod, inputs, device_mesh):
        top_scores, selected_experts_indices = inputs

        top_scores = DTensor.from_local(top_scores, device_mesh, (Replicate(),))
        selected_experts_indices = DTensor.from_local(
            selected_experts_indices, device_mesh, (Replicate(),)
        )

        # NOTE: If needed, we can pad tokens in case bs*slen is not divisible by TP degree
        # if top_scores.shape[0] % device_mesh.size() != 0:
        #     num_tokens = top_scores.shape[0]
        #     tp_size = device_mesh.size()
        #     n_pad = (num_tokens // tp_size + 1) * tp_size - num_tokens
        #     selected_experts_indices = F.pad(selected_experts_indices, [0, 0, 0, n_pad])
        #     top_scores = F.pad(top_scores, [0, 0, 0, n_pad])
        assert top_scores.shape[0] % device_mesh.size() == 0

        # split on the bs*slen dimension
        top_scores = top_scores.redistribute(device_mesh, (Shard(0),)).to_local()
        selected_experts_indices = selected_experts_indices.redistribute(
            device_mesh, (Shard(0),)
        ).to_local()

        return top_scores, selected_experts_indices

    def _prepare_output_fn(self, mod, outputs, device_mesh):
        top_scores, token_indices_experts_sorted, num_tokens_per_expert = outputs

        # NOTE: As we shard routed tokens along bs*slen dim across the TP ranks,
        #       the MoE gather and scatter still require global token indices.
        num_tokens = top_scores.shape[0]
        local_rank = device_mesh.get_local_rank()
        token_indices_experts_sorted += num_tokens // device_mesh.size() * local_rank

        return top_scores, token_indices_experts_sorted, num_tokens_per_expert

    def _apply(self, module: nn.Module, device_mesh: DeviceMesh) -> nn.Module:
        return distribute_module(
            module,
            device_mesh,
            partition_fn=None,
            input_fn=self._prepare_inputput_fn,
            output_fn=self._prepare_output_fn,
        )


# TODO: let multiple MoE layers share the same input / output buffer
# TODO: add NVSHMEMExpertTensorParallel support
class NVSHMEMExpertParallel(ParallelStyle):
    def __init__(
        self,
        num_tokens: int,
        dim: int,
        num_experts: int,
        ep_mesh: DeviceMesh,
        dtype: torch.dtype,
    ):
        import torch.distributed._symmetric_memory as symm_mem

        from torchtitan.experiments.kernels.moe.combine import TokenCombiner
        from torchtitan.experiments.kernels.moe.dispatch import TokenDispatcher
        from torchtitan.tools.utils import device_type

        super().__init__()

        ep_degree = ep_mesh.shape[0]
        # bs * slen * top_k, or
        # (bs * slen // tp_degree) * top_k if ReordererSequenceParallel is used
        input_length = num_tokens
        # TODO: make overflow_factor configurable? but can cause IMA
        # worst case: one rank receives all data
        overflow_factor = ep_degree
        max_output_length = input_length * overflow_factor

        device = torch.device(device_type)
        self.input_buffer = symm_mem.empty(
            num_tokens,
            dim,
            dtype=dtype,
            device=device,
        )
        self.output_buffer = symm_mem.empty(
            max_output_length, dim, dtype=dtype, device=device
        )
        # two rows: input splits, input offsets
        self.in_splits_offsets_buffer = symm_mem.empty(
            (2, num_experts), dtype=torch.int64, device=device
        )
        # two rows: output splits, output offsets
        self.out_splits_offsets_buffer = symm_mem.empty(
            (2, num_experts), dtype=torch.int64, device=device
        )

        group_name = ep_mesh.get_group().group_name
        num_local_experts = num_experts // ep_degree

        global TOKEN_GROUP_ALIGN_SIZE_M
        self.dispatcher = TokenDispatcher(
            group_name,
            TOKEN_GROUP_ALIGN_SIZE_M,
            input_length,
            max_output_length,
            [dim],
            ep_degree,
            num_local_experts,
            dtype,
            device,
        )
        self.combiner = TokenCombiner(
            group_name,
            TOKEN_GROUP_ALIGN_SIZE_M,
            max_output_length,
            input_length,
            [dim],
            ep_degree,
            num_local_experts,
            dtype,
            device,
        )

        self.input_splits = None
        self.output_splits = None

    # performing all-to-all dispatch on the input
    def _token_dispatch(self, mod, inputs, device_mesh):
        # annotate module input placements/sharding with input_layouts
        routed_input, num_tokens_per_expert = inputs
        ep_degree = device_mesh.shape[0]

        self.input_splits = num_tokens_per_expert
        self.in_splits_offsets_buffer[0].copy_(self.input_splits)
        input_buffer = self.input_buffer.detach()
        output_buffer = self.output_buffer.detach()
        input_buffer.copy_(routed_input)
        output = self.dispatcher(
            input_buffer,
            output_buffer,
            self.in_splits_offsets_buffer[0],
            self.out_splits_offsets_buffer,
        )

        # NOTE: output_splits layout:
        # for i in range(num_local_experts):
        #     for j in range(ep_degree):
        #         output_splits[i * ep_degree + j] denotes:
        # number of tokens passed from EP rank j to local expert i
        output_splits = self.out_splits_offsets_buffer[0]
        output_offsets = self.out_splits_offsets_buffer[1]

        # TODO: need to simplify this
        offsets = torch.zeros_like(output_offsets[::ep_degree])
        offsets[:-1] = output_offsets[ep_degree::ep_degree]
        offsets[-1] = output_offsets[-1] + output_splits[-1]

        return output, None, offsets.to(dtype=torch.int32)

    @staticmethod
    def _partition_fn(name, mod, device_mesh):
        # shard on the expert dimension
        for name, param in mod.named_parameters(recurse=False):
            dist_param = nn.Parameter(distribute_tensor(param, device_mesh, [Shard(0)]))
            mod.register_parameter(name, dist_param)

    # performing all-to-all combine on the output
    def _token_combine(self, mod, routed_output, device_mesh):
        input_buffer = self.input_buffer.detach()
        output_buffer = self.output_buffer.detach()
        output_buffer.copy_(routed_output)

        routed_output = self.combiner(
            output_buffer,
            input_buffer,
            self.out_splits_offsets_buffer,
            self.in_splits_offsets_buffer,
        )

        return routed_output

    def _apply(self, module: nn.Module, device_mesh: DeviceMesh) -> nn.Module:
        return distribute_module(
            module,
            device_mesh,
            partition_fn=ExpertParallel._partition_fn,
            input_fn=self._token_dispatch,
            output_fn=self._token_combine,
        )
