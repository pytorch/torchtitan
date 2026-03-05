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
    Partial,
    Replicate,
    Shard,
)
from torch.distributed.tensor.parallel import ParallelStyle

from torchtitan.models.moe.utils import _permute, _unpermute


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
    def _prepare_input_fn(self, mod, inputs, device_mesh):
        routed_input, num_tokens_per_expert = inputs
        # NOTE: Currently in MoE TP, experts multiplication runs in plain Tensors.
        #       The grad_placements on inputs is set to Partial so that necessary
        #       reductions are performed during backward.
        routed_input = DTensor.from_local(
            routed_input, device_mesh, (Replicate(),)
        ).to_local(grad_placements=(Partial(),))

        return routed_input, num_tokens_per_expert

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
        # check for lora parameters
        if hasattr(module, "w1_lora_a"):
            module.register_parameter(
                "w1_lora_a",
                nn.Parameter(
                    distribute_tensor(module.w1_lora_a, device_mesh, [Replicate()])
                ),
            )
            module.register_parameter(
                "w1_lora_b",
                nn.Parameter(
                    distribute_tensor(module.w1_lora_b, device_mesh, [Shard(1)])
                ),
            )
            module.register_parameter(
                "w2_lora_a",
                nn.Parameter(
                    distribute_tensor(module.w2_lora_a, device_mesh, [Shard(2)])
                ),
            )
            module.register_parameter(
                "w2_lora_b",
                nn.Parameter(
                    distribute_tensor(module.w2_lora_b, device_mesh, [Replicate()])
                ),
            )
            module.register_parameter(
                "w3_lora_a",
                nn.Parameter(
                    distribute_tensor(module.w3_lora_a, device_mesh, [Replicate()])
                ),
            )
            module.register_parameter(
                "w3_lora_b",
                nn.Parameter(
                    distribute_tensor(module.w3_lora_b, device_mesh, [Shard(1)])
                ),
            )

    def _apply(self, module: nn.Module, device_mesh: DeviceMesh) -> nn.Module:
        return distribute_module(
            module,
            device_mesh,
            self._partition_fn,
            # pyrefly: ignore [bad-argument-type]
            self._prepare_input_fn,
        )


class ExpertParallel(BaseExpertParallel):
    def __init__(self):
        super().__init__()
        self.input_splits = None
        self.output_splits = None
        self.input_shape = None
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
        # We need to perform another shuffle to get the correct layout, via the _permute function
        # below, which also does padding to make sure the number of tokens each expert gets locally
        # is a multiple of TOKEN_GROUP_ALIGN_SIZE_M.
        # Note that this will create side effects when wrapping the for-loop implementation
        # of GroupedExperts, as it does not need padding.

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
        routed_input, num_tokens_per_expert = inputs

        # NOTE: Currently in MoE TP, experts multiplication runs in plain Tensors.
        #       The grad_placements on inputs is set to Partial so that necessary
        #       reductions are performed during backward.

        # NOTE: The mesh used here should be dense_mesh["tp"] as routed_input is
        #       technically wrapped with the dense_mesh["tp"] but this complicates
        #       the interface of ExpertTensorParallel and it doesn't matter as etp
        #       is almost always the same as tp or is 1. To avoid the complexity,
        #       we use the etp mesh here.
        routed_input = DTensor.from_local(
            routed_input, device_mesh["etp"], (Replicate(),)
        ).to_local(grad_placements=(Partial(),))

        inputs = (routed_input, num_tokens_per_expert)

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
        token_indices_experts_sorted = (
            token_indices_experts_sorted + top_scores.shape[0] * local_rank
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
    """Expert Parallel using DeepEP for efficient token dispatch/combine.

    Expects inputs as:
        (hidden_states, num_tokens_per_expert, selected_experts_indices, top_scores, num_experts)

    Args:
        score_before_experts: If True, apply routing scores before expert computation.
    """

    def __init__(self, score_before_experts: bool = True):
        super().__init__()
        self._state = None  # State preserved between dispatch and combine
        self.score_before_experts = score_before_experts

    def _token_dispatch(self, mod, inputs, device_mesh):
        """Dispatch tokens via DeepEP."""
        from torchtitan.distributed.deepep import dispatch_tokens

        hidden_states, _, selected_experts_indices, top_scores, num_experts = inputs
        if isinstance(mod.w1, DTensor):
            num_local_experts = mod.w1.to_local().shape[0]
        else:
            num_local_experts = mod.w1.shape[0]
        ep_group = device_mesh.get_group()

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
        """Combine tokens via DeepEP."""
        from torchtitan.distributed.deepep import combine_tokens

        # pyrefly: ignore [bad-argument-type]
        routed_output = combine_tokens(routed_output, self._state)
        self._state = None
        return routed_output

    def _apply(self, module: nn.Module, device_mesh: DeviceMesh) -> nn.Module:
        """Apply DeepEP parallelization."""
        return distribute_module(
            module,
            device_mesh,
            partition_fn=DeepEPExpertParallel._partition_fn,
            input_fn=self._token_dispatch,  # pyrefly: ignore [bad-argument-type]
            output_fn=self._token_combine,  # pyrefly: ignore [bad-argument-type]
        )


class DeepEPLLEPExpertParallel(BaseExpertParallel):
    """Adaptive Expert Parallel: DeepEP when balanced, LLEP when imbalanced.

    Expects DeepEP-style 5-tuple inputs:
        (hidden_states, num_tokens_per_expert, selected_experts_indices, top_scores, num_experts)

    Each step:
    1. All-gather expert counts → compute imbalance ratio
    2. If balanced (imbalance < threshold): use DeepEP dispatch/combine
    3. If imbalanced (imbalance >= threshold): reorder tokens inline, use LLEP dispatch/combine

    Both paths produce (bs*slen, dim) output for uniform MoE post-processing.

    Args:
        score_before_experts: If True, apply routing scores before expert computation.
        max_tokens_factor: LLEP alpha parameter.
        min_tokens_per_gemm: LLEP minimum tokens per GEMM.
        adaptive_threshold: Imbalance ratio threshold to switch to LLEP.
        verbose: Enable per-step logging.
    """

    def __init__(
        self,
        score_before_experts: bool = True,
        max_tokens_factor: float = 1.1,
        min_tokens_per_gemm: int = 1024,
        adaptive_threshold: float = 1.3,
        verbose: bool = False,
    ):
        super().__init__()
        self.score_before_experts = score_before_experts
        self._max_tokens_factor = max_tokens_factor
        self._min_tokens_per_gemm = min_tokens_per_gemm
        self._adaptive_threshold = adaptive_threshold
        self._verbose = verbose

        # Per-step state
        self._use_llep_path = False
        self._deepep_state = None  # DispatchState when DeepEP path
        self._llep_state = None  # LLEPState when LLEP path
        # Saved for LLEP combine → unsort + top_k reduction
        self._token_indices_sorted = None
        self._num_src_tokens = 0
        self._top_k = 0
        self._top_scores = None  # saved for score_before_experts=False

    def _partition_fn(self, name: str, mod: nn.Module, device_mesh: DeviceMesh) -> None:
        for param_name, param in mod.named_parameters(recurse=False):
            dist_param = nn.Parameter(distribute_tensor(param, device_mesh, [Shard(0)]))
            mod.register_parameter(param_name, dist_param)

    def _token_dispatch(self, mod, inputs, device_mesh):
        """Dispatch tokens: DeepEP if balanced, LLEP if imbalanced."""
        hidden_states, _, selected_experts_indices, top_scores, num_experts = inputs

        if isinstance(mod.w1, DTensor):
            num_local_experts = mod.w1.to_local().shape[0]
        else:
            num_local_experts = mod.w1.shape[0]
        ep_group = device_mesh.get_group()
        ep_size = device_mesh.shape[0]

        # Compute imbalance ratio from expert counts
        num_tokens_per_expert = torch.histc(
            selected_experts_indices.float(),
            bins=num_experts,
            min=0,
            max=num_experts,
        )
        # All-gather counts across EP ranks for global view
        from torchtitan.distributed.llep import compute_gpu_imbalance_ratio

        with torch.no_grad():
            local_counts = num_tokens_per_expert.to(torch.int64)
            all_counts = [torch.zeros_like(local_counts) for _ in range(ep_size)]
            torch.distributed.all_gather(all_counts, local_counts, group=ep_group)
            global_counts = torch.stack(all_counts).sum(dim=0)
            imbalance = compute_gpu_imbalance_ratio(
                global_counts, ep_size, num_local_experts
            )

        use_llep = (
            self._adaptive_threshold > 0 and imbalance >= self._adaptive_threshold
        )
        self._use_llep_path = use_llep

        if not use_llep:
            # === DeepEP path (balanced) ===
            from torchtitan.distributed.deepep import dispatch_tokens

            hidden_states, tokens_per_expert, self._deepep_state = dispatch_tokens(
                hidden_states,
                selected_experts_indices,
                top_scores,
                num_local_experts,
                num_experts,
                ep_group,
                score_before_experts=self.score_before_experts,
            )
            self._llep_state = None
            return hidden_states, tokens_per_expert
        else:
            # === LLEP path (imbalanced) ===
            from torchtitan.distributed.llep import llep_dispatch_tokens

            num_src_tokens = hidden_states.shape[0]
            top_k = selected_experts_indices.shape[1]
            self._num_src_tokens = num_src_tokens
            self._top_k = top_k

            # Inline reorder: sort tokens by expert (mimics TokenReorderer)
            token_indices_sorted = torch.argsort(
                selected_experts_indices.view(-1), stable=True
            )
            self._token_indices_sorted = token_indices_sorted

            num_tokens_per_expert_local = torch.histc(
                selected_experts_indices.view(-1).float(),
                bins=num_experts,
                min=0,
                max=num_experts,
            )

            routed_input = hidden_states[token_indices_sorted // top_k]

            if self.score_before_experts:
                scores_sorted = top_scores.view(-1)[token_indices_sorted]
                routed_input = (
                    routed_input.to(torch.float32) * scores_sorted.reshape(-1, 1)
                ).to(hidden_states.dtype)

            # Save scores for post-combine weighting if needed
            if not self.score_before_experts:
                self._top_scores = top_scores

            dispatched_tokens, padded_counts, llep_state = llep_dispatch_tokens(
                routed_input,
                num_tokens_per_expert_local,
                ep_group,
                max_tokens_factor=self._max_tokens_factor,
                min_tokens_per_gemm=self._min_tokens_per_gemm,
                adaptive_threshold=0.0,  # Already decided to use LLEP
                verbose=self._verbose,
            )
            self._llep_state = llep_state
            self._deepep_state = None

            return dispatched_tokens, padded_counts, llep_state

    def _token_combine(self, mod, routed_output, device_mesh):
        """Combine tokens: DeepEP or LLEP depending on dispatch path."""
        if not self._use_llep_path:
            # === DeepEP path ===
            from torchtitan.distributed.deepep import combine_tokens

            routed_output = combine_tokens(routed_output, self._deepep_state)
            self._deepep_state = None
            return routed_output
        else:
            # === LLEP path: combine + unsort + top_k reduction ===
            from torchtitan.distributed.llep import llep_combine_output

            # llep_combine_output returns tokens in original routed_input order
            routed_output = llep_combine_output(routed_output, self._llep_state)

            # Unsort: scatter back to (bs*slen*top_k, dim) then reduce
            num_src_tokens = self._num_src_tokens
            top_k = self._top_k
            dim = routed_output.shape[1]

            unsorted = torch.zeros(
                (num_src_tokens * top_k, dim),
                dtype=routed_output.dtype,
                device=routed_output.device,
            )
            unsorted[self._token_indices_sorted] = routed_output

            unsorted = unsorted.reshape(num_src_tokens, top_k, dim)
            if self.score_before_experts:
                # Scores already applied: sum across top_k
                result = unsorted.sum(dim=1)
            else:
                # Apply scores via bmm: (N, 1, top_k) @ (N, top_k, dim) → (N, 1, dim)
                result = (
                    torch.bmm(
                        self._top_scores.float().reshape(-1, 1, top_k),
                        unsorted.float(),
                    )
                    .to(unsorted.dtype)
                    .squeeze(1)
                )

            self._llep_state = None
            self._token_indices_sorted = None
            self._top_scores = None
            return result

    def _apply(self, module: nn.Module, device_mesh: DeviceMesh) -> nn.Module:
        return distribute_module(
            module,
            device_mesh,
            partition_fn=self._partition_fn,
            input_fn=self._token_dispatch,  # pyrefly: ignore [bad-argument-type]
            output_fn=self._token_combine,  # pyrefly: ignore [bad-argument-type]
        )


class ExpertParallelLLEP(BaseExpertParallel):
    """Expert Parallelism with Least-Loaded Expert Parallelism (LLEP).

    Shards expert weights across EP ranks (Shard(0)) and installs
    dispatch/combine hooks that use LPT-based routing instead of
    naive EP routing.

    The dispatch hook handles:
    - all_gather expert counts → imbalance check → LPT plan
    - token assignment to GPUs → AllToAll dispatch → sort/pad

    The combine hook handles:
    - AllToAll combine → unsort to original token order

    Weight transfer (P2P) happens inside GroupedExperts.forward() after
    FSDP has unsharded the weights, via llep_prepare_weights().
    """

    def __init__(
        self,
        max_tokens_factor: float = 1.1,
        min_tokens_per_gemm: int = 1024,
        adaptive_threshold: float = 0.0,
        verbose: bool = False,
    ):
        super().__init__()
        self._max_tokens_factor = max_tokens_factor
        self._min_tokens_per_gemm = min_tokens_per_gemm
        self._adaptive_threshold = adaptive_threshold
        self._verbose = verbose
        self._llep_state = None

    def _partition_fn(self, name: str, mod: nn.Module, device_mesh: DeviceMesh) -> None:
        for param_name, param in mod.named_parameters(recurse=False):
            dist_param = nn.Parameter(distribute_tensor(param, device_mesh, [Shard(0)]))
            mod.register_parameter(param_name, dist_param)

    def _token_dispatch(
        self, mod: nn.Module, inputs: tuple, device_mesh: DeviceMesh
    ) -> tuple[Tensor, Tensor]:
        from torchtitan.distributed.llep import llep_dispatch_tokens

        routed_input, num_tokens_per_expert = inputs

        dispatched_tokens, padded_counts, llep_state = llep_dispatch_tokens(
            routed_input,
            num_tokens_per_expert,
            device_mesh.get_group(),
            max_tokens_factor=self._max_tokens_factor,
            min_tokens_per_gemm=self._min_tokens_per_gemm,
            adaptive_threshold=self._adaptive_threshold,
            verbose=self._verbose,
        )
        self._llep_state = llep_state

        # Return 3-tuple: GroupedExperts.forward() gets llep_state as 3rd arg
        return dispatched_tokens, padded_counts, llep_state

    def _token_combine(
        self, mod: nn.Module, routed_output: Tensor, device_mesh: DeviceMesh
    ) -> Tensor:
        from torchtitan.distributed.llep import llep_combine_output

        result = llep_combine_output(routed_output, self._llep_state)
        self._llep_state = None
        return result

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
