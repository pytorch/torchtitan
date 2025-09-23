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
from torchtitan.distributed.expert_parallel import ExpertParallel


# implementation of Tensor Parallel for the GroupedExperts in MoE
class TensorParallel(ParallelStyle):
    def _partition_fn(self, name, module, device_mesh):
        module.register_parameter(
            "mlp1_weight", nn.Parameter(distribute_tensor(module.mlp1_weight, device_mesh, [Shard(2)]))
        )  # Column-wise sharding
        module.register_parameter(
            "mlp1_bias",
            nn.Parameter(distribute_tensor(module.mlp1_bias, device_mesh, [Shard(1)])),
        )  # Column-wise sharding
        module.register_parameter(
            "mlp2_weight",
            nn.Parameter(distribute_tensor(module.mlp2_weight, device_mesh, [Shard(1)])),
        )  # Row-wise sharding
        module.register_parameter(
            "mlp2_bias",
            nn.Parameter(distribute_tensor(module.mlp2_bias, device_mesh, [Replicate()])),
        )  # Replicate

    def _apply(self, module: nn.Module, device_mesh: DeviceMesh) -> nn.Module:
        return distribute_module(
            module,
            device_mesh,
            self._partition_fn,
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
            "mlp1_weight",
            nn.Parameter(distribute_tensor(mod.mlp1_weight, ep_tp_mesh, [Shard(0), Shard(2)])),
        )  # Column-wise sharding
        mod.register_parameter(
            "mlp1_bias",
            nn.Parameter(distribute_tensor(mod.mlp1_bias, ep_tp_mesh, [Shard(0), Shard(1)])),
        )  # Row-wise sharding
        mod.register_parameter(
            "mlp2_weight",
            nn.Parameter(distribute_tensor(mod.mlp2_weight, ep_tp_mesh, [Shard(0), Shard(2)])),
        )  # Column-wise sharding
        mod.register_parameter(
            "mlp2_bias",
            nn.Parameter(distribute_tensor(mod.mlp2_bias, ep_tp_mesh, [Shard(0), Shard(1)])),
        )  # Row-wise sharding

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


# TODO(jianiw): This need to be merged with 
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
        mlp1_weight: torch.Tensor,
        mlp1_bias: torch.Tensor,
        mlp2_weight: torch.Tensor,
        mlp2_bias: torch.Tensor,
        swiglu_limit: float,
        x: torch.Tensor,
        num_tokens_per_expert: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if isinstance(mlp1_weight, DTensor):
            mlp1_weight = mlp1_weight.to_local()
            mlp1_bias = mlp1_bias.to_local()
            mlp2_weight = mlp2_weight.to_local()
            mlp2_bias = mlp2_bias.to_local()

        if num_tokens_per_expert is not None:
            from torchtitan.experiments.kernels.moe.indices import (
                generate_permute_indices,
            )

            experts_per_ep_rank = mlp1_weight.shape[0]
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

        out = func(mlp1_weight, mlp1_bias, mlp2_weight, mlp2_bias, swiglu_limit, x, num_tokens_per_expert)

        if num_tokens_per_expert is not None:
            out_unpermuted = out.new_empty(input_shape)
            out_unpermuted[permuted_indices, :] = out
            out = out_unpermuted[:-1]

        return out

    return wrapper
