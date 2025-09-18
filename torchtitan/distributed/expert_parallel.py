# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


from typing import Callable, Literal, Dict

import torch
import torch.nn as nn
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

import threading
import torch
from typing import Optional
import time

class HookSequenceCoordinator:
    """Coordinates hooks based on a predefined sequence"""
    
    def __init__(self):
        self._lock = threading.Lock()
        self._condition = threading.Condition(self._lock)
        
        # Define your desired execution sequence matching:
        # stageB.combine() -> stageA.forward_attention() -> stageB.backward_moe() -> 
        # stageA.dispatch() -> stageB.dispatch() -> stageA.forward_moe() -> 
        # stageB.backward_attention() -> stageA.combine()
        self._hook_sequence = [
            "combine_D_bwd",
            "dispatch_A_fwd",
            "combine_C_bwd",
            "dispatch_B_fwd",
            "dispatch_B_bwd",
            "combine_C_fwd",
            "dispatch_A_bwd",
            "combine_D_fwd",
        ]
        # Create a semaphore for each hook in the sequence
        self._semaphores: Dict[str, threading.Semaphore] = {}
        self._reset_semaphores()
        
        # Coordination control - disabled by default
        self._coordination_enabled = False
        self._cycle_count = 0
        
    def _reset_semaphores(self):
        """Reset all semaphores - first one gets 1 permit, others get 0"""
        self._semaphores.clear()
        for i, hook_name in enumerate(self._hook_sequence):
            # First semaphore starts with 1 permit, others start with 0
            initial_permits = 1 if i == 0 else 0
            self._semaphores[hook_name] = threading.Semaphore(initial_permits)
        
    def enable_coordination(self):
        """Enable hook coordination"""
        self._coordination_enabled = True
        self._reset_semaphores()  # Reset semaphores when enabling
        print("[COORDINATION] Hook coordination ENABLED")
    
    def disable_coordination(self):
        """Disable hook coordination"""
        self._coordination_enabled = False
        # Release all semaphores so no threads get stuck
        for semaphore in self._semaphores.values():
            try:
                semaphore.release()
            except ValueError:
                pass  # Semaphore was already at max value
        print("[COORDINATION] Hook coordination DISABLED")
    
    def is_coordination_enabled(self) -> bool:
        """Check if coordination is currently enabled"""
        return self._coordination_enabled
    
    def reset_coordination(self):
        """Reset coordination state (useful between training runs)"""
        self._cycle_count = 0
        self._reset_semaphores()
        print("[COORDINATION] Hook coordination state RESET")
    
    def acquire_execution(self, hook_name: str):
        """Acquire execution permission using semaphores"""
        # If coordination is disabled, just pass through
        if not self._coordination_enabled:
            print(f"[PASSTHROUGH] {hook_name} executing (coordination disabled)")
            return
        
        # Check if hook is in our sequence
        if hook_name not in self._semaphores:
            print(f"[WARNING] {hook_name} not in sequence, executing without coordination")
            return
        
        # Acquire the semaphore for this hook (blocks until available)
        print(f"[WAITING] {hook_name} waiting for semaphore")
        self._semaphores[hook_name].acquire()
        print(f"[EXECUTING] {hook_name} acquired semaphore")
    
    def release_execution(self, hook_name: str):
        """Release execution and signal next hook"""
        # If coordination is disabled, just pass through
        if not self._coordination_enabled:
            return
        
        # Check if hook is in our sequence
        if hook_name not in self._semaphores:
            return
        
        # Find the next hook in the sequence and release its semaphore
        try:
            current_index = self._hook_sequence.index(hook_name)
            next_index = (current_index + 1) % len(self._hook_sequence)
            next_hook = self._hook_sequence[next_index]
            
            print(f"[COMPLETED] {hook_name} completed, signaling {next_hook}")
            self._semaphores[next_hook].release()
            
            # Check if we completed a full cycle
            if next_index == 0:
                self._cycle_count += 1
                print(f"[CYCLE] Completed cycle {self._cycle_count}")
                
        except ValueError:
            print(f"[ERROR] {hook_name} not found in sequence")

# Global coordinator
_hook_coordinator = HookSequenceCoordinator()

class SyncHook(torch.autograd.Function):
    """Sync hook that follows a predefined execution sequence"""
    
    @staticmethod
    def forward(ctx, x, hook_name):
        ctx.hook_name = hook_name
        
        # Use forward-specific hook name
        forward_hook_name = f"{hook_name}_fwd"
        _hook_coordinator.acquire_execution(forward_hook_name)
        
        try:
            if _hook_coordinator.is_coordination_enabled():
                print(f"[FORWARD HOOK] {forward_hook_name} (coordinated)")
            else:
                print(f"[FORWARD HOOK] {forward_hook_name} (uncoordinated)")
            return x
        finally:
            _hook_coordinator.release_execution(forward_hook_name)
    
    @staticmethod
    def backward(ctx, grad_output):
        hook_name = ctx.hook_name
        
        # Use backward-specific hook name
        backward_hook_name = f"{hook_name}_bwd"
        _hook_coordinator.acquire_execution(backward_hook_name)
        
        try:
            if _hook_coordinator.is_coordination_enabled():
                print(f"[BACKWARD HOOK] {backward_hook_name} (coordinated)")
            else:
                print(f"[BACKWARD HOOK] {backward_hook_name} (uncoordinated)")
            return grad_output, None
        finally:
            _hook_coordinator.release_execution(backward_hook_name)



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

class ExpertParallel(ParallelStyle):
    def __init__(self):
        super().__init__()
        self.input_splits = None
        self.output_splits = None

    # performing all-to-all dispatch on the input
    def _token_dispatch(self, mod, inputs, device_mesh):
        # annotate module input placements/sharding with input_layouts
        routed_input, num_tokens_per_expert = inputs
        ep_size = device_mesh.shape[0]

        # HOOK: signal ready for sync
        routed_input = SyncHook.apply(routed_input, "dispatch_A")

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
                num_tokens_per_expert.view(ep_size, -1)
                .sum(dim=1)
                .to(torch.device("cpu"), non_blocking=True)
            )
            # NOTE: this would incur a device-to-host sync
            output_splits = (
                num_tokens_per_expert_group.view(ep_size, -1)
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
        # We need to perform another shuffle to get the correct format -- this is done via the function
        # generate_permute_indices in moe.py, which also does padding to make sure the number of tokens
        # each expert gets locally is a multiple of ALIGN_SIZE_M.

        # HOOK: signal ready for sync
        routed_input = SyncHook.apply(routed_input, "dispatch_B")

        return routed_input, num_tokens_per_expert_group

    @staticmethod
    def _partition_fn(name, mod, device_mesh):
        # shard on the expert dimension
        for name, param in mod.named_parameters(recurse=False):
            dist_param = nn.Parameter(distribute_tensor(param, device_mesh, [Shard(0)]))
            mod.register_parameter(name, dist_param)

    # performing all-to-all combine on the output
    def _token_combine(self, mod, routed_output, device_mesh):
        # HOOK: signal ready for sync
        routed_output = SyncHook.apply(routed_output, "combine_C")
        routed_output = all_to_all_single_autograd(
            routed_output,
            self.input_splits,
            self.output_splits,
            device_mesh.get_group(),
        )
        # HOOK: signal ready for sync
        routed_output = SyncHook.apply(routed_output, "combine_D")
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
        num_tokens_per_expert: torch.Tensor,
    ) -> torch.Tensor:
        global TOKEN_GROUP_ALIGN_SIZE_M
        if isinstance(w1, DTensor):
            w1 = w1.to_local()
            w2 = w2.to_local()
            w3 = w3.to_local()

        from torchtitan.experiments.kernels.moe.indices import generate_permute_indices

        experts_per_ep_rank = w1.shape[0]
        num_ep_ranks = num_tokens_per_expert.shape[0] // experts_per_ep_rank

        with torch.no_grad():
            (
                permuted_indices,
                num_tokens_per_expert,
                _,  # offsets,
            ) = generate_permute_indices(
                num_tokens_per_expert,
                experts_per_ep_rank,
                num_ep_ranks,
                x.shape[0] + experts_per_ep_rank * TOKEN_GROUP_ALIGN_SIZE_M,
                TOKEN_GROUP_ALIGN_SIZE_M,
            )

        x = torch.vstack((x, x.new_zeros((x.shape[-1]))))
        input_shape = x.shape
        x = x[permuted_indices, :]

        out = func(w1, w2, w3, x, num_tokens_per_expert)

        out_unpermuted = out.new_empty(input_shape)
        out_unpermuted[permuted_indices, :] = out
        out = out_unpermuted[:-1]

        return out

    return wrapper


# This class is to support Sequence Parallel for ETP=1
# when EP borrows from all TP and part of DP
class ReordererSequenceParallel(ParallelStyle):
    def __init__(self):
        super().__init__()
        self.num_tokens = None

    def _prepare_inputput_fn(self, mod, inputs, device_mesh):
        top_scores, selected_experts_indices = inputs
        self.num_tokens = top_scores.shape[0]

        # NOTE: If needed, we can pad tokens in case bs*slen is not divisible by TP degree
        # if top_scores.shape[0] % device_mesh.size() != 0:
        #     num_tokens = top_scores.shape[0]
        #     tp_size = device_mesh.size()
        #     n_pad = (num_tokens // tp_size + 1) * tp_size - num_tokens
        #     selected_experts_indices = F.pad(selected_experts_indices, [0, 0, 0, n_pad])
        #     top_scores = F.pad(top_scores, [0, 0, 0, n_pad])

        def _split_along_first_dim(x: torch.Tensor) -> torch.Tensor:
            assert x.is_contiguous()
            assert self.num_tokens % device_mesh.size() == 0
            local_num_tokens = self.num_tokens // device_mesh.size()
            local_rank = device_mesh.get_local_rank()
            offset = local_rank * local_num_tokens
            output = x[offset : offset + local_num_tokens]

            return output

        top_scores = _split_along_first_dim(top_scores)
        selected_experts_indices = _split_along_first_dim(selected_experts_indices)

        return top_scores, selected_experts_indices

    def _prepare_output_fn(self, mod, outputs, device_mesh):
        top_scores, token_indices_experts_sorted, num_tokens_per_expert = outputs

        # NOTE: As we shard routed tokens along bs*slen dim across the TP ranks,
        #       the MoE gather and scatter still require global token indices.
        local_rank = device_mesh.get_local_rank()
        token_indices_experts_sorted += (
            self.num_tokens // device_mesh.size() * local_rank
        )

        return top_scores, token_indices_experts_sorted, num_tokens_per_expert

    def _apply(self, module: nn.Module, device_mesh: DeviceMesh) -> nn.Module:
        return distribute_module(
            module,
            device_mesh,
            partition_fn=None,
            input_fn=self._prepare_inputput_fn,
            output_fn=self._prepare_output_fn,
        )
