# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


from functools import partial
from typing import Callable

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed._functional_collectives import all_to_all_single_autograd
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


# implementation of Tensor Parallel for the GroupedExperts in MoE
class TensorParallel(ParallelStyle):
    def _partition_fn(self, name, module, device_mesh):
        module.register_parameter(
            "w1", nn.Parameter(distribute_tensor(module.w1, device_mesh, [Shard(2)]))
        )  # Column-wise sharding
        module.register_parameter(
            "w2",
            nn.Parameter(distribute_tensor(module.w2, device_mesh, [Shard(1)])),
        )  # Row-wise sharding
        module.register_parameter(
            "w3",
            nn.Parameter(distribute_tensor(module.w3, device_mesh, [Shard(2)])),
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

    # performing all-to-all dispatch on the input
    def _token_dispatch(self, mod, inputs, device_mesh):
        # annotate module input placements/sharding with input_layouts
        routed_input, num_tokens_per_expert = inputs

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
            # NOTE: this would incur a device-to-host sync
            self.input_splits = (
                num_tokens_per_expert.view(device_mesh.shape[0], -1).sum(dim=1).tolist()
            )
            self.output_splits = (
                num_tokens_per_expert_group.view(device_mesh.shape[0], -1)
                .sum(dim=1)
                .tolist()
            )

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
        # We need to perform another shuffle to get the correct format -- this is done via the function
        # generate_permute_indices in moe.py, which also does padding to make sure the number of tokens
        # each expert gets locally is a multiple of ALIGN_SIZE_M.

        return routed_input, num_tokens_per_expert_group

    @staticmethod
    def _partition_fn(name, mod, device_mesh):
        # shard on the expert dimension
        for name, param in mod.named_parameters(recurse=False):
            dist_param = nn.Parameter(distribute_tensor(param, device_mesh, [Shard(0)]))
            mod.register_parameter(name, dist_param)

    # performing all-to-all combine on the output
    def _token_combine(self, mod, routed_output, device_mesh):
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
    def __init__(
        self,
        tp_mesh: DeviceMesh,
        ep_mesh: DeviceMesh,
    ):
        super().__init__()
        # TODO: has to pass in the meshes in addition to the [ep, tp] device_mesh,
        #       as DeviceMesh doesn't support slicing from a submesh.
        self.tp_mesh = tp_mesh
        self.ep_mesh = ep_mesh

    def _token_dispatch(self, mod, inputs, device_mesh):
        # token dispatch happens on the EP mesh, whereas device_mesh is [ep, tp] mesh
        return super()._token_dispatch(mod, inputs, self.ep_mesh)

    def _partition_fn_2d(self, name, mod, ep_tp_mesh):
        mod.register_parameter(
            "w1",
            nn.Parameter(distribute_tensor(mod.w1, ep_tp_mesh, [Shard(0), Shard(2)])),
        )  # Column-wise sharding
        mod.register_parameter(
            "w2",
            nn.Parameter(distribute_tensor(mod.w2, ep_tp_mesh, [Shard(0), Shard(1)])),
        )  # Row-wise sharding
        mod.register_parameter(
            "w3",
            nn.Parameter(distribute_tensor(mod.w3, ep_tp_mesh, [Shard(0), Shard(2)])),
        )  # Column-wise sharding

    def _token_combine(self, mod, routed_output, device_mesh):
        # token combine happens on the EP mesh, whereas device_mesh is [ep, tp] mesh
        return super()._token_combine(mod, routed_output, self.ep_mesh)

    def _apply(self, module: nn.Module, device_mesh: DeviceMesh) -> nn.Module:
        return distribute_module(
            module,
            device_mesh,
            partition_fn=self._partition_fn_2d,
            input_fn=self._token_dispatch,
            output_fn=self._token_combine,
        )


def expert_parallel(func: Callable) -> Callable:
    """
    This is a wrapper applied to the GroupedExperts computation, serving
    the following three purposes:
    1. Convert parameters from DTensors to plain Tensors, to work with
    dynamic-shape inputs which cannot be easily expressed as DTensors.
    2. In Expert Parallel, apply the generate_permute_indices kernel to
    permute the inputs to be ordered by local experts (see the _token_dispatch
    function in ExpertParallel) and permute the outputs back.
    3. In order to use torch._grouped_mm, we need to make sure the number of
    tokens each expert gets is a multiple of ALIGN_SIZE_M. The generate_permute_indices
    kernel also helps achieve this via padding, without incurring synchronization
    between device and host. Note that this will create side effects when wrapping
    the for-loop implementation of GroupedExperts, as it does not need padding.

    Among the above:
    1 and 2 are needed only when expert_parallel_degree > 1.
    3 is needed even for single-device computation.
    2 can be moved to ExpertParallel _token_dispatch if not coupled with 3.
    """

    def wrapper(
        w1: torch.Tensor,
        w2: torch.Tensor,
        w3: torch.Tensor,
        x: torch.Tensor,
        num_tokens_per_expert: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if isinstance(w1, DTensor):
            w1 = w1.to_local()
            w2 = w2.to_local()
            w3 = w3.to_local()

        if num_tokens_per_expert is not None:
            from torchtitan.experiments.kernels.moe.indices import (
                generate_permute_indices,
            )

            experts_per_ep_rank = w1.shape[0]
            num_ep_ranks = num_tokens_per_expert.shape[0] // experts_per_ep_rank

            ALIGN_SIZE_M = 16
            with torch.no_grad():
                (
                    permuted_indices,
                    num_tokens_per_expert,
                    _,  # offsets,
                ) = generate_permute_indices(
                    num_tokens_per_expert,
                    experts_per_ep_rank,
                    num_ep_ranks,
                    x.shape[0] + experts_per_ep_rank * ALIGN_SIZE_M,
                    ALIGN_SIZE_M,
                )

            x = torch.vstack((x, x.new_zeros((x.shape[-1]))))
            input_shape = x.shape
            x = x[permuted_indices, :]

        out = func(w1, w2, w3, x, num_tokens_per_expert)

        if num_tokens_per_expert is not None:
            out_unpermuted = out.new_empty(input_shape)
            out_unpermuted[permuted_indices, :] = out
            out = out_unpermuted[:-1]

        return out

    return wrapper
