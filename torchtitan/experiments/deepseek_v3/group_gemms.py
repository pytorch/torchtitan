# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
[CUTLASS] Kernel compilation failed: DSLRuntimeError: 704 to integer conversion is not supported
[CUTLASS] Tensor shapes:
  Group 0: Atorch.Size([128, 2048, 1]), Btorch.Size([2048, 1408, 1]), Ctorch.Size([128, 2048, 1])
  Group 1: Atorch.Size([128, 2048, 1]), Btorch.Size([2048, 1408, 1]), Ctorch.Size([128, 2048, 1])
  Group 2: Atorch.Size([128, 2048, 1]), Btorch.Size([2048, 1408, 1]), Ctorch.Size([128, 2048, 1])
  Group 3: Atorch.Size([128, 2048, 1]), Btorch.Size([2048, 1408, 1]), Ctorch.Size([128, 2048, 1])
  Group 4: Atorch.Size([128, 2048, 1]), Btorch.Size([2048, 1408, 1]), Ctorch.Size([128, 2048, 1])
  Group 5: Atorch.Size([128, 2048, 1]), Btorch.Size([2048, 1408, 1]), Ctorch.Size([128, 2048, 1])
  Group 6: Atorch.Size([128, 2048, 1]), Btorch.Size([2048, 1408, 1]), Ctorch.Size([128, 2048, 1])
  Group 7: Atorch.Size([128, 2048, 1]), Btorch.Size([2048, 1408, 1]), Ctorch.Size([128, 2048, 1])
  Group 8: Atorch.Size([128, 2048, 1]), Btorch.Size([2048, 1408, 1]), Ctorch.Size([128, 2048, 1])
  Group 9: Atorch.Size([128, 2048, 1]), Btorch.Size([2048, 1408, 1]), Ctorch.Size([128, 2048, 1])
  Group 10: Atorch.Size([128, 2048, 1]), Btorch.Size([2048, 1408, 1]), Ctorch.Size([128, 2048, 1])
  Group 11: Atorch.Size([896, 2048, 1]), Btorch.Size([2048, 1408, 1]), Ctorch.Size([896, 2048, 1])
  Group 12: Atorch.Size([128, 2048, 1]), Btorch.Size([2048, 1408, 1]), Ctorch.Size([128, 2048, 1])
  Group 13: Atorch.Size([128, 2048, 1]), Btorch.Size([2048, 1408, 1]), Ctorch.Size([128, 2048, 1])
  Group 14: Atorch.Size([128, 2048, 1]), Btorch.Size([2048, 1408, 1]), Ctorch.Size([128, 2048, 1])
  Group 15: Atorch.Size([128, 2048, 1]), Btorch.Size([2048, 1408, 1]), Ctorch.Size([128, 2048, 1])
Error using cutlass strategy: DSLRuntimeError: 704 to integer conversion is not supported
[CUTLASS] Kernel compilation failed: DSLRuntimeError: 928 to integer conversion is not supported
[CUTLASS] Tensor shapes:
  Group 0: Atorch.Size([128, 2048, 1]), Btorch.Size([2048, 1408, 1]), Ctorch.Size([128, 2048, 1])
  Group 1: Atorch.Size([128, 2048, 1]), Btorch.Size([2048, 1408, 1]), Ctorch.Size([128, 2048, 1])
  Group 2: Atorch.Size([128, 2048, 1]), Btorch.Size([2048, 1408, 1]), Ctorch.Size([128, 2048, 1])
  Group 3: Atorch.Size([128, 2048, 1]), Btorch.Size([2048, 1408, 1]), Ctorch.Size([128, 2048, 1])
  Group 4: Atorch.Size([128, 2048, 1]), Btorch.Size([2048, 1408, 1]), Ctorch.Size([128, 2048, 1])
  Group 5: Atorch.Size([128, 2048, 1]), Btorch.Size([2048, 1408, 1]), Ctorch.Size([128, 2048, 1])
  Group 6: Atorch.Size([128, 2048, 1]), Btorch.Size([2048, 1408, 1]), Ctorch.Size([128, 2048, 1])
  Group 7: Atorch.Size([896, 2048, 1]), Btorch.Size([2048, 1408, 1]), Ctorch.Size([896, 2048, 1])
  Group 8: Atorch.Size([256, 2048, 1]), Btorch.Size([2048, 1408, 1]), Ctorch.Size([256, 2048, 1])
  Group 9: Atorch.Size([128, 2048, 1]), Btorch.Size([2048, 1408, 1]), Ctorch.Size([128, 2048, 1])
  Group 10: Atorch.Size([128, 2048, 1]), Btorch.Size([2048, 1408, 1]), Ctorch.Size([128, 2048, 1])
  Group 11: Atorch.Size([128, 2048, 1]), Btorch.Size([2048, 1408, 1]), Ctorch.Size([128, 2048, 1])
  Group 12: Atorch.Size([896, 2048, 1]), Btorch.Size([2048, 1408, 1]), Ctorch.Size([896, 2048, 1])
  Group 13: Atorch.Size([128, 2048, 1]), Btorch.Size([2048, 1408, 1]), Ctorch.Size([128, 2048, 1])
  Group 14: Atorch.Size([128, 2048, 1]), Btorch.Size([2048, 1408, 1]), Ctorch.Size([128, 2048, 1])
  Group 15: Atorch.Size([128, 2048, 1]), Btorch.Size([2048, 1408, 1]), Ctorch.Size([128, 2048, 1])
Error using cutlass strategy: DSLRuntimeError: 928 to integer conversion is not supported
[CUTLASS] Kernel compilation failed: DSLRuntimeError: 704 to integer conversion is not supported
[CUTLASS] Tensor shapes:
  Group 0: Atorch.Size([128, 2048, 1]), Btorch.Size([2048, 1408, 1]), Ctorch.Size([128, 2048, 1])
  Group 1: Atorch.Size([128, 2048, 1]), Btorch.Size([2048, 1408, 1]), Ctorch.Size([128, 2048, 1])
  Group 2: Atorch.Size([128, 2048, 1]), Btorch.Size([2048, 1408, 1]), Ctorch.Size([128, 2048, 1])
  Group 3: Atorch.Size([128, 2048, 1]), Btorch.Size([2048, 1408, 1]), Ctorch.Size([128, 2048, 1])
  Group 4: Atorch.Size([896, 2048, 1]), Btorch.Size([2048, 1408, 1]), Ctorch.Size([896, 2048, 1])
  Group 5: Atorch.Size([128, 2048, 1]), Btorch.Size([2048, 1408, 1]), Ctorch.Size([128, 2048, 1])
  Group 6: Atorch.Size([128, 2048, 1]), Btorch.Size([2048, 1408, 1]), Ctorch.Size([128, 2048, 1])
  Group 7: Atorch.Size([128, 2048, 1]), Btorch.Size([2048, 1408, 1]), Ctorch.Size([128, 2048, 1])
  Group 8: Atorch.Size([128, 2048, 1]), Btorch.Size([2048, 1408, 1]), Ctorch.Size([128, 2048, 1])
  Group 9: Atorch.Size([128, 2048, 1]), Btorch.Size([2048, 1408, 1]), Ctorch.Size([128, 2048, 1])
  Group 10: Atorch.Size([128, 2048, 1]), Btorch.Size([2048, 1408, 1]), Ctorch.Size([128, 2048, 1])
  Group 11: Atorch.Size([128, 2048, 1]), Btorch.Size([2048, 1408, 1]), Ctorch.Size([128, 2048, 1])
  Group 12: Atorch.Size([128, 2048, 1]), Btorch.Size([2048, 1408, 1]), Ctorch.Size([128, 2048, 1])
  Group 13: Atorch.Size([128, 2048, 1]), Btorch.Size([2048, 1408, 1]), Ctorch.Size([128, 2048, 1])
  Group 14: Atorch.Size([128, 2048, 1]), Btorch.Size([2048, 1408, 1]), Ctorch.Size([128, 2048, 1])
  Group 15: Atorch.Size([128, 2048, 1]), Btorch.Size([2048, 1408, 1]), Ctorch.Size([128, 2048, 1])
Error using cutlass strategy: DSLRuntimeError: 704 to integer conversion is not supported

and tiler_m and tiler_n = 16,16

[CUTLASS] Kernel compilation failed: DSLRuntimeError: 💥💥💥 Error during runtime code generation for function `__call__` 💥💥💥
[CUTLASS] Tensor shapes:
  Group 0: Atorch.Size([128, 2048, 1]), Btorch.Size([2048, 1408, 1]), Ctorch.Size([128, 2048, 1])
  Group 1: Atorch.Size([128, 2048, 1]), Btorch.Size([2048, 1408, 1]), Ctorch.Size([128, 2048, 1])
  Group 2: Atorch.Size([128, 2048, 1]), Btorch.Size([2048, 1408, 1]), Ctorch.Size([128, 2048, 1])
  Group 3: Atorch.Size([128, 2048, 1]), Btorch.Size([2048, 1408, 1]), Ctorch.Size([128, 2048, 1])
  Group 4: Atorch.Size([128, 2048, 1]), Btorch.Size([2048, 1408, 1]), Ctorch.Size([128, 2048, 1])
  Group 5: Atorch.Size([128, 2048, 1]), Btorch.Size([2048, 1408, 1]), Ctorch.Size([128, 2048, 1])
  Group 6: Atorch.Size([128, 2048, 1]), Btorch.Size([2048, 1408, 1]), Ctorch.Size([128, 2048, 1])
  Group 7: Atorch.Size([896, 2048, 1]), Btorch.Size([2048, 1408, 1]), Ctorch.Size([896, 2048, 1])
  Group 8: Atorch.Size([256, 2048, 1]), Btorch.Size([2048, 1408, 1]), Ctorch.Size([256, 2048, 1])
  Group 9: Atorch.Size([128, 2048, 1]), Btorch.Size([2048, 1408, 1]), Ctorch.Size([128, 2048, 1])
  Group 10: Atorch.Size([128, 2048, 1]), Btorch.Size([2048, 1408, 1]), Ctorch.Size([128, 2048, 1])
  Group 11: Atorch.Size([128, 2048, 1]), Btorch.Size([2048, 1408, 1]), Ctorch.Size([128, 2048, 1])
  Group 12: Atorch.Size([896, 2048, 1]), Btorch.Size([2048, 1408, 1]), Ctorch.Size([896, 2048, 1])
  Group 13: Atorch.Size([128, 2048, 1]), Btorch.Size([2048, 1408, 1]), Ctorch.Size([128, 2048, 1])
  Group 14: Atorch.Size([128, 2048, 1]), Btorch.Size([2048, 1408, 1]), Ctorch.Size([128, 2048, 1])
  Group 15: Atorch.Size([128, 2048, 1]), Btorch.Size([2048, 1408, 1]), Ctorch.Size([128, 2048, 1])
Error using cutlass strategy: DSLRuntimeError: 💥💥💥 Error during runtime code generation for function `__call__` 💥💥💥
[CUTLASS] Kernel compilation failed: DSLRuntimeError: 💥💥💥 Error during runtime code generation for function `__call__` 💥💥💥
"""

import torch
import torch.nn as nn

from torchtitan.tools.logging import logger

try:
    import deep_gemm

    DEEPGEMM_AVAILABLE = True
except ImportError:
    DEEPGEMM_AVAILABLE = False

if DEEPGEMM_AVAILABLE:
    import dsgemm_kernels
    import dsgemm_utils

try:
    from torchao.float8.config import ScalingGranularity
    from torchao.float8.float8_utils import tensor_to_scale, to_fp8_saturated

    TORCHAO_FP8_GG_AVAILABLE = True

except ImportError:
    TORCHAO_FP8_GG_AVAILABLE = False
    # raise NotImplementedError("Missing TorchAO")

try:
    from torchtitan.experiments.kernels.triton_mg_group_gemm.torchao_pr import (
        # ALIGN_SIZE_M,
        grouped_gemm_forward,
    )

    TRITON_MG_GROUP_GEMM_AVAILABLE = True
except ImportError:
    TRITON_MG_GROUP_GEMM_AVAILABLE = False

try:
    from torchtitan.experiments.kernels.triton_contiguous_group_gemm.cg_forward import (
        cg_grouped_gemm_forward,
    )

    TRITON_CONTIGUOUS_GROUP_GEMM_AVAILABLE = True
except ImportError:
    TRITON_CONTIGUOUS_GROUP_GEMM_AVAILABLE = False

try:
    import cuda.bindings.driver as cuda
    import cutlass
    import cutlass.cute as cute
    import cutlass.torch as cutlass_torch
    import cutlass.utils as utils
    from cutlass.cute.runtime import from_dlpack
    from torchtitan.experiments.kernels.blackwell_group_gemms.cute_grouped_gemm import (
        GroupedGemmKernel,
    )

    CUTLASS_AVAILABLE = True
except ImportError:
    CUTLASS_AVAILABLE = False


# Strategy base class for GroupGEMM implementations
class GroupGEMMStrategy:
    """Base class for group gemm strategies"""

    def __init__(self, custom_activation):
        self.activation_function = custom_activation

    def arrange_expert_weights(self, all_weights, submod_name, module):
        """prepare expert weights, including prescaling

        Args:
            all_weights: List of weight tensors from each expert
            submod_name: Name of the submodule (e.g., 'gate_proj', 'up_proj', 'down_proj')
            module: The parent module that will store the arranged weights

        Returns:
            Tensor: The arranged weights in the format required by the specific strategy
        """

        raise NotImplementedError("Requires arrange_expert_weights method")

    def execute(self, contig_tokens, m_sizes, m_offsets, module):
        """Execute the group gemm operation

        Args:
            contig_tokens: The input tokens, arranged contiguously by expert
            m_sizes: Sizes of each group
            m_offsets: Offsets of each group
            module: The MoE module containing weights and parameters

        Returns:
            The processed tokens
        """
        raise NotImplementedError("GroupGEMM strategy must implement execute method")

    @staticmethod
    def is_available() -> bool:
        """Check if this strategy is available on the current system"""
        return False


# ========= Implementations ===================

__all__ = [
    "TorchFP8GroupGEMM",
    "DSGroupGEMM",
    "TorchBF16GroupGEMM",
    "TorchAOBF16GroupGEMM",
    "TritonCGBF16GroupGEMM",
    "CUTLASSGroupGEMM",
]


class CUTLASSGroupGEMM(GroupGEMMStrategy):
    """Implementation using CUTLASS GroupedGemmKernel for BF16"""

    def __init__(self, custom_activation):
        super().__init__(custom_activation)
        self.dtype_torch = torch.bfloat16
        self.dtype_cutlass = cutlass.BFloat16
        self.acc_dtype = cutlass.Float32
        self.m_tiler = 128
        self.n_tiler = 64

        logger.info("Using CUTLASS GroupedGemmKernel for BF16, init")

        # CUTLASS kernel configuration - optimized for typical MoE workloads
        self.grouped_gemm = GroupedGemmKernel(
            acc_dtype=self.acc_dtype,
            use_2cta_instrs=True,  # Set to True for larger problems if beneficial
            mma_tiler_mn=(
                self.m_tiler,
                self.n_tiler,
            ),  # Can be tuned based on problem sizes
            cluster_shape_mn=(2, 2),  # Can be increased for larger problems
            tensormap_update_mode=utils.TensorMapUpdateMode.SMEM,
        )

        # Hardware info for tensormap allocation
        self.hardware_info = utils.HardwareInfo()
        self.max_active_clusters = self.hardware_info.get_max_active_clusters(1)

        # Compiled kernels cache
        self._compiled_kernels = {}
        self._tensormap_buffers = {}

    def arrange_expert_weights(self, all_weights, submod_name, module):
        """Prepare expert weights for CUTLASS grouped GEMM"""
        # Stack weights for easier manipulation
        combined_weights = torch.stack(all_weights)
        # combined_weights shape: torch.Size([16, 1408, 2048])
        # logger.info(f"cutlass log: combined_weights shape: {combined_weights.shape}")

        # CUTLASS expects specific tensor layouts - we'll handle conversion in execute()
        return combined_weights

    def _prepare_cutlass_tensors(self, torch_tensor, is_weight=False):
        """Convert PyTorch tensor to CUTLASS format with proper layout marking"""
        # Ensure tensor is contiguous
        torch_tensor = torch_tensor.contiguous()

        # Create CUTE tensor
        cute_tensor = from_dlpack(torch_tensor, assumed_align=16)
        cute_tensor.element_type = self.dtype_cutlass

        # Mark appropriate dimension as dynamic for grouped operations
        if is_weight:
            # logger.info("cutlass log: is_weight")
            # For weights, find the dimension with stride 1
            for i, stride in enumerate(torch_tensor.stride()):
                if stride == 1:
                    cute_tensor = cute_tensor.mark_layout_dynamic(leading_dim=i)
                    break
        else:
            # For activations, find the dimension with stride 1
            # logger.info("cutlass log: not is_weight")
            for i, stride in enumerate(torch_tensor.stride()):
                if stride == 1:
                    cute_tensor = cute_tensor.mark_layout_dynamic(leading_dim=i)
                    break

        return cute_tensor

    def _setup_cutlass_metadata(self, torch_tensors_abc, problem_sizes):
        """Setup metadata tensors required by CUTLASS"""
        device = torch_tensors_abc[0][0].device

        # Problem sizes tensor
        problem_sizes_tensor = torch.tensor(
            problem_sizes, dtype=torch.int32, device=device
        )
        problem_sizes_cute = from_dlpack(problem_sizes_tensor, assumed_align=16)

        # Extract strides from tensors
        strides = []
        pointers = []

        for torch_a, torch_b, torch_c in torch_tensors_abc:
            # For MNKL tensors, we need strides for M,N,K dimensions (ignoring L=1)
            a_strides = [torch_a.stride(0), torch_a.stride(1)]  # M, K strides
            b_strides = [torch_b.stride(0), torch_b.stride(1)]  # N, K strides
            c_strides = [torch_c.stride(0), torch_c.stride(1)]  # M, N strides

            strides.append([a_strides, b_strides, c_strides])
            pointers.append(
                [torch_a.data_ptr(), torch_b.data_ptr(), torch_c.data_ptr()]
            )

        # Convert to tensors
        strides_tensor = torch.tensor(strides, dtype=torch.int32, device=device)
        logger.info(f"cutlass log: strides_tensor: {strides_tensor}")
        strides_cute = from_dlpack(strides_tensor, assumed_align=16)
        logger.info(f"cutlass log: strides_cute: {strides_cute}")

        pointers_tensor = torch.tensor(pointers, dtype=torch.int64, device=device)
        pointers_cute = from_dlpack(pointers_tensor, assumed_align=16)

        logger.info(f"{pointers_tensor=}")
        logger.info(f"cutlass log: pointers_cute: {pointers_cute}")

        return problem_sizes_cute, strides_cute, pointers_cute

    def _get_tensormap_buffer(self, num_groups, device):
        """Get or create tensormap buffer for given configuration"""
        cache_key = (num_groups, device)

        if cache_key not in self._tensormap_buffers:
            sm_count = self.hardware_info.get_max_active_clusters(1)
            tensormap_tensor = torch.zeros(
                (sm_count, 3, 128 // 8),  # 3 tensormaps (A, B, C), 128 bytes each
                dtype=torch.int64,
                device=device,
            )
            self._tensormap_buffers[cache_key] = from_dlpack(
                tensormap_tensor, assumed_align=16
            )

        return self._tensormap_buffers[cache_key]

    def _compute_total_clusters(self, problem_sizes):
        """Compute total number of clusters needed"""
        # Use same calculation as in cute_grouped_gemm.py
        cluster_tile_m = self.m_tiler  # matches mma_tiler_mn[0]
        cluster_tile_n = self.n_tiler  # matches mma_tiler_mn[1]

        total = 0
        for m, n, k, l in problem_sizes:
            clusters_m = (m + cluster_tile_m - 1) // cluster_tile_m
            clusters_n = (n + cluster_tile_n - 1) // cluster_tile_n
            total += clusters_m * clusters_n

        return total

    def _execute_grouped_gemm(self, input_tokens, weights, m_sizes, is_down_proj=False):
        """Execute a single grouped GEMM operation using CUTLASS"""
        device = input_tokens.device
        num_groups = len([s for s in m_sizes if s > 0])  # Only count non-empty groups

        if num_groups == 0:
            # No valid groups, return zeros
            if is_down_proj:
                return torch.zeros_like(input_tokens[:, : weights.shape[-1]])
            else:
                return torch.zeros(
                    input_tokens.shape[0],
                    weights.shape[-1],
                    dtype=self.dtype_torch,
                    device=device,
                )

        # Create contiguous input tensor by filtering out empty groups
        valid_tokens = []
        valid_weights = []
        valid_problem_sizes = []
        padded_m_sizes = []

        offset = 0
        for i, size in enumerate(m_sizes):
            if size > 0:
                # Get tokens for this group
                group_tokens = input_tokens[offset : offset + size].contiguous()

                # Pad M dimension to nearest multiple of 128 (or power of 2)
                padded_size = ((size + 127) // 128) * 128
                padded_tokens = torch.zeros(
                    padded_size,
                    group_tokens.shape[1],
                    dtype=group_tokens.dtype,
                    device=device,
                )
                padded_tokens[:size] = group_tokens

                valid_tokens.append(padded_tokens)
                padded_m_sizes.append(padded_size)

                # Get weight for this expert
                expert_weight = weights[i].contiguous()
                valid_weights.append(expert_weight)

                # Create problem size specification with padded M
                if is_down_proj:
                    # down: (intermediate_size, hidden_size)
                    m, n, k = (
                        padded_size,
                        expert_weight.shape[1],
                        expert_weight.shape[0],
                    )
                else:
                    # gate/up: (hidden_size, intermediate_size)
                    m, n, k = (
                        padded_size,
                        expert_weight.shape[1],
                        expert_weight.shape[0],
                    )
                valid_problem_sizes.append((m, n, k, 1))

            offset += size

        # Concatenate all valid inputs
        concat_tokens = torch.cat(valid_tokens, dim=0)

        # Create output tensor for concatenated results
        if is_down_proj:
            output_dim = valid_weights[0].shape[1]  # hidden_size
        else:
            output_dim = valid_weights[0].shape[1]  # intermediate_size

        concat_output = torch.zeros(
            concat_tokens.shape[0], output_dim, dtype=self.dtype_torch, device=device
        )

        # Prepare tensors in MNKL format as expected by CUTLASS
        torch_tensors_abc = []
        token_offset = 0

        for i, (m, n, k, l) in enumerate(valid_problem_sizes):
            # Get tokens for this problem
            group_tokens = concat_tokens[token_offset : token_offset + m]
            group_output = concat_output[token_offset : token_offset + m]

            # Reshape tensors to MNKL format (add L=1 dimension)
            group_tokens_mnkl = group_tokens.unsqueeze(-1)  # (M, K) -> (M, K, 1)
            expert_weight_mnkl = (
                valid_weights[i].transpose(0, 1).unsqueeze(-1)
            )  # (K, N) -> (N, K, 1)
            group_output_mnkl = group_output.unsqueeze(-1)  # (M, N) -> (M, N, 1)

            torch_tensors_abc.append(
                (group_tokens_mnkl, expert_weight_mnkl, group_output_mnkl)
            )
            token_offset += m

        # Convert to CUTLASS format
        cute_tensors_abc = []
        for torch_a, torch_b, torch_c in torch_tensors_abc:
            cute_a = self._prepare_cutlass_tensors(torch_a, is_weight=False)
            cute_b = self._prepare_cutlass_tensors(torch_b, is_weight=True)
            cute_c = self._prepare_cutlass_tensors(torch_c, is_weight=False)
            cute_tensors_abc.append((cute_a, cute_b, cute_c))

        # Setup metadata
        problem_sizes_cute, strides_cute, pointers_cute = self._setup_cutlass_metadata(
            torch_tensors_abc, valid_problem_sizes
        )

        # Get tensormap buffer
        tensormap_cute = self._get_tensormap_buffer(num_groups, device)

        # Compute grid parameters
        total_clusters = self._compute_total_clusters(valid_problem_sizes)

        # Choose initial tensors (use first group)
        initial_a, initial_b, initial_c = cute_tensors_abc[0]

        # Setup CUDA stream
        torch_stream = torch.cuda.current_stream()
        stream = cuda.CUstream(torch_stream.cuda_stream)

        # Compile kernel (cache based on problem configuration)
        cache_key = (num_groups, tuple(valid_problem_sizes))
        if cache_key not in self._compiled_kernels:
            try:

                self._compiled_kernels[cache_key] = cute.compile(
                    self.grouped_gemm,
                    initial_a,
                    initial_b,
                    initial_c,
                    num_groups,
                    problem_sizes_cute,
                    strides_cute,
                    pointers_cute,
                    total_clusters,
                    tensormap_cute,
                    self.max_active_clusters,
                    stream,
                )

                print(f"[CUTLASS] Successfully compiled kernel")

            except Exception as e:
                print(f"[CUTLASS] Kernel compilation failed: {e}")
                print(f"[CUTLASS] Tensor shapes:")
                for i, (ta, tb, tc) in enumerate(torch_tensors_abc):
                    print(f"  Group {i}: A{ta.shape}, B{tb.shape}, C{tc.shape}")
                raise e

        compiled_kernel = self._compiled_kernels[cache_key]

        # Execute kernel
        try:

            logger.info(f"[CUTLASS] Executing kernel for {num_groups} groups")

            compiled_kernel(
                initial_a,
                initial_b,
                initial_c,
                problem_sizes_cute,
                strides_cute,
                pointers_cute,
                tensormap_cute,
                stream,
            )

            logger.info(f"[CUTLASS] Kernel execution completed")

        except Exception as e:
            print(f"[CUTLASS] Kernel execution failed: {e}")
            raise e

        # Extract results from MNKL format back to MN
        """for i, (torch_a, torch_b, torch_c) in enumerate(torch_tensors_abc):
            start_idx = sum(valid_problem_sizes[j][0] for j in range(i))
            end_idx = start_idx + valid_problem_sizes[i][0]
            concat_output[start_idx:end_idx] = torch_c.squeeze(
                -1
            )  # (M, N, 1) -> (M, N)

        return concat_output
        """
        full_output = torch.zeros(
            sum(m_sizes),
            hidden_size,
            dtype=self.dtype_torch,
            device=contig_tokens.device,
        )

        # Fill in results for valid groups, removing padding
        valid_offset = 0
        full_offset = 0
        for i, (size, padded_size) in enumerate(zip(m_sizes, padded_m_sizes)):
            if size > 0:
                # Only copy the non-padded part of the result
                full_output[full_offset : full_offset + size] = final_output[
                    valid_offset : valid_offset + size
                ]
                valid_offset += padded_size
            full_offset += size

        return full_output

    def execute(self, contig_tokens, m_sizes, m_offsets, module):
        """Execute the complete MoE forward pass using CUTLASS grouped GEMM"""
        # Get weights
        w_gate = module.get_parameter("gate_proj_weight")
        w_up = module.get_parameter("up_proj_weight")
        w_down = module.get_parameter("down_proj_weight")

        # Get dimensions
        hidden_size = w_gate.shape[1]  # input dimension

        # Split contiguous tokens back into groups
        token_groups = []
        offset = 0
        for size in m_sizes:
            if size > 0:
                token_groups.append(contig_tokens[offset : offset + size])
            offset += size

        # Filter out empty groups for CUTLASS execution
        valid_indices = [i for i, size in enumerate(m_sizes) if size > 0]
        valid_m_sizes = [m_sizes[i] for i in valid_indices]

        if not valid_m_sizes:
            # No valid tokens, return zeros
            return torch.zeros(
                sum(m_sizes),
                hidden_size,
                dtype=self.dtype_torch,
                device=contig_tokens.device,
            )

        # Concatenate valid tokens
        valid_tokens = torch.cat([token_groups[i] for i in valid_indices], dim=0)

        # Prepare weights for valid experts only
        valid_w_gate = w_gate[valid_indices]
        valid_w_up = w_up[valid_indices]
        valid_w_down = w_down[valid_indices]

        # Execute gate and up projections
        gate_output = self._execute_grouped_gemm(
            valid_tokens, valid_w_gate, valid_m_sizes, is_down_proj=False
        )
        up_output = self._execute_grouped_gemm(
            valid_tokens, valid_w_up, valid_m_sizes, is_down_proj=False
        )

        # Apply activation and element-wise multiplication
        hidden_states = self.activation_function(gate_output) * up_output

        # Execute down projection
        final_output = self._execute_grouped_gemm(
            hidden_states, valid_w_down, valid_m_sizes, is_down_proj=True
        )

        # Reconstruct full output with zeros for empty groups
        full_output = torch.zeros(
            sum(m_sizes),
            hidden_size,
            dtype=self.dtype_torch,
            device=contig_tokens.device,
        )

        # Fill in results for valid groups
        valid_offset = 0
        full_offset = 0
        for i, size in enumerate(m_sizes):
            if size > 0:
                full_output[full_offset : full_offset + size] = final_output[
                    valid_offset : valid_offset + size
                ]
                valid_offset += size
            full_offset += size

        return full_output

    @staticmethod
    def is_available() -> bool:
        return CUTLASS_AVAILABLE


class TritonCGBF16GroupGEMM(GroupGEMMStrategy):
    """Implementation of Triton Contiguous group Gemm"""

    def arrange_expert_weights(self, all_weights, submod_name, module):
        """prep the expert weights for group gemm usage"""

        return torch.stack(all_weights)

    def execute(self, contig_tokens, m_sizes, m_offsets, module):
        # Get weights
        w_gate = module.get_parameter("gate_proj_weight")
        w_up = module.get_parameter("up_proj_weight")
        w_down = module.get_parameter("down_proj_weight")

        # Run first two GEMMs (gate and up projections)
        # Get only valid tokens
        valid_tokens = contig_tokens[: m_offsets[-1]]

        # Create indices from offsets without CPU-GPU sync
        m_indices = dsgemm_utils.create_indices_from_offsets_nosync(m_offsets)

        gate_proj = cg_grouped_gemm_forward(valid_tokens, w_gate, m_indices)

        up_proj = cg_grouped_gemm_forward(valid_tokens, w_up, m_indices)

        # Apply activation
        hidden_outputs = self.activation_function(gate_proj) * up_proj

        # Run the third GEMM (down projection)

        down_proj_out = cg_grouped_gemm_forward(hidden_outputs, w_down, m_indices)

        # Copy results back to contig_tokens
        contig_tokens[: m_offsets[-1]] = down_proj_out
        return contig_tokens

    @staticmethod
    def is_available() -> bool:
        return TRITON_CONTIGUOUS_GROUP_GEMM_AVAILABLE


class TorchBF16GroupGEMM(GroupGEMMStrategy):
    """Implementation for PyTorch native BF16  _grouped_mm"""

    def arrange_expert_weights(self, all_weights, submod_name, module):
        """prep the expert weights for group gemm usage"""
        return torch.stack(all_weights)

    def execute(self, contig_tokens, m_sizes, m_offsets, module):
        # Get weights
        w_gate = module.get_parameter("gate_proj_weight")
        w_up = module.get_parameter("up_proj_weight")
        w_down = module.get_parameter("down_proj_weight")

        # Run first two GEMMs (gate and up projections)
        gate_proj = torch._grouped_mm(
            contig_tokens,
            w_gate.transpose(-2, -1),
            m_offsets,
            out_dtype=torch.bfloat16,
        )
        up_proj = torch._grouped_mm(
            contig_tokens,
            w_up.transpose(-2, -1),
            m_offsets,
            out_dtype=torch.bfloat16,
        )

        # Apply activation
        hidden_outputs = self.activation_function(gate_proj) * up_proj

        # Run the third GEMM (down projection)
        hidden_outputs = torch._grouped_mm(
            hidden_outputs,
            w_down.transpose(-2, -1),
            m_offsets,
            out_dtype=torch.bfloat16,
        )

        return hidden_outputs


class TorchAOBF16GroupGEMM(GroupGEMMStrategy):
    """Implementation using TorchAO's grouped_gemm_forward"""

    def arrange_expert_weights(self, all_weights, submod_name, module):
        """prep the expert weights for group gemm usage"""
        return torch.cat(all_weights)

    def execute(self, contig_tokens, m_sizes, m_offsets, module):
        # Get weights
        w_gate = module.get_parameter("gate_proj_weight")
        w_up = module.get_parameter("up_proj_weight")
        w_down = module.get_parameter("down_proj_weight")

        # Run first two GEMMs (gate and up projections)
        gate_proj = grouped_gemm_forward(contig_tokens, w_gate, m_sizes)
        up_proj = grouped_gemm_forward(contig_tokens, w_up, m_sizes)

        # Apply activation
        hidden_outputs = self.activation_function(gate_proj) * up_proj

        # Run the third GEMM (down projection)
        hidden_outputs = grouped_gemm_forward(hidden_outputs, w_down, m_sizes)

        return hidden_outputs

    @staticmethod
    def is_available() -> bool:
        return TRITON_MG_GROUP_GEMM_AVAILABLE


class TorchFP8GroupGEMM(GroupGEMMStrategy):
    """Implementation using TorchAO's _scaled_grouped_mm with FP8 rowwise precision and weight prescaling"""

    def arrange_expert_weights(self, all_weights, submod_name, module):
        """prep the expert weights for group gemm usage with prescaling"""
        # Stack weights as in the original implementation
        combined_weights = torch.stack(all_weights)

        # Transpose weights for column-major format
        transposed_weights = combined_weights.transpose(-2, -1)

        # Convert weights to float8 format with appropriate scaling
        weight_scales = tensor_to_scale(
            transposed_weights,
            torch.float8_e4m3fn,
            scaling_granularity=ScalingGranularity.AXISWISE,
            axiswise_dim=-2,  # Use -2 for transposed weights
            round_scales_to_power_of_2=True,
        )

        # Scale the weights
        scaled_weights = transposed_weights.to(torch.float32) * weight_scales

        # Convert to FP8
        fp8_weights = to_fp8_saturated(scaled_weights, torch.float8_e4m3fn)

        # Register as module parameters
        module.register_parameter(
            f"{submod_name}_fp8",
            nn.Parameter(
                fp8_weights,
            ),
        )

        module.register_parameter(
            f"{submod_name}_scales",
            nn.Parameter(
                weight_scales,
            ),
        )

        return combined_weights

    def execute(self, contig_tokens, m_sizes, m_offsets, module):
        # Get prescaled transposed weights and scales
        gate_proj_fp8 = module.get_parameter("gate_proj_fp8")
        gate_proj_scales = module.get_parameter("gate_proj_scales")
        up_proj_fp8 = module.get_parameter("up_proj_fp8")
        up_proj_scales = module.get_parameter("up_proj_scales")
        down_proj_fp8 = module.get_parameter("down_proj_fp8")
        down_proj_scales = module.get_parameter("down_proj_scales")

        # Convert input tokens to FP8 with appropriate scaling
        token_scales = tensor_to_scale(
            contig_tokens,
            torch.float8_e4m3fn,
            scaling_granularity=ScalingGranularity.AXISWISE,
            axiswise_dim=-1,
            round_scales_to_power_of_2=True,
        )
        scaled_tokens = contig_tokens.to(torch.float32) * token_scales
        fp8_tokens = to_fp8_saturated(scaled_tokens, torch.float8_e4m3fn)

        # Run first two GEMMs (gate and up projections) using prescaled weights
        gate_proj = torch._scaled_grouped_mm(
            fp8_tokens,
            gate_proj_fp8,
            token_scales.squeeze().reciprocal(),
            gate_proj_scales.squeeze().reciprocal(),
            m_offsets,
            out_dtype=torch.bfloat16,
            use_fast_accum=True,
        )

        up_proj = torch._scaled_grouped_mm(
            fp8_tokens,
            up_proj_fp8,
            token_scales.squeeze().reciprocal(),
            up_proj_scales.squeeze().reciprocal(),
            m_offsets,
            out_dtype=torch.bfloat16,
            use_fast_accum=True,
        )

        # Apply activation
        hidden_outputs = self.activation_function(gate_proj) * up_proj

        # Convert hidden_outputs to FP8 for the third GEMM
        hidden_scales = tensor_to_scale(
            hidden_outputs,
            torch.float8_e4m3fn,
            scaling_granularity=ScalingGranularity.AXISWISE,
            axiswise_dim=-1,
            round_scales_to_power_of_2=True,
        )
        scaled_hidden = hidden_outputs.to(torch.float32) * hidden_scales
        fp8_hidden = to_fp8_saturated(scaled_hidden, torch.float8_e4m3fn)

        # Run the third GEMM (down projection)
        result = torch._scaled_grouped_mm(
            fp8_hidden,
            down_proj_fp8,
            hidden_scales.squeeze().reciprocal(),
            down_proj_scales.squeeze().reciprocal(),
            m_offsets,
            out_dtype=torch.bfloat16,
            use_fast_accum=True,
        )

        return result

    @staticmethod
    def is_available() -> bool:
        return TORCHAO_FP8_GG_AVAILABLE


class DSGroupGEMM(GroupGEMMStrategy):
    """Implementation using DeepGEMM with FP8 quantization"""

    def __init__(self, custom_activation, use_triton_quant=True):
        self.activation_function = custom_activation
        self.use_triton_quant = use_triton_quant

    def arrange_expert_weights(self, all_weights, submod_name, module):
        """prep the expert weights for group gemm usage"""
        combined_weights = torch.stack(all_weights)

        fp8, scales = dsgemm_utils.prepare_fp8_weight(combined_weights)

        # prescale weights
        # TODO - this creates 2 sets of weights, need to resolve this for traiing aspect.
        module.register_parameter(
            f"{submod_name}_fp8",
            nn.Parameter(
                fp8,
            ),
        )

        module.register_parameter(
            f"{submod_name}_scales",
            nn.Parameter(
                scales,
            ),
        )

        return combined_weights

    def execute(self, contig_tokens, m_sizes, m_offsets, module):
        # Get only valid tokens
        valid_tokens = contig_tokens[: m_offsets[-1]]

        # Create indices from offsets without CPU-GPU sync
        m_indices = dsgemm_utils.create_indices_from_offsets_nosync(m_offsets)

        # Get expert weights for all projections
        gate_proj_weight_fp8 = module.get_parameter("gate_proj_fp8")
        gate_proj_scales = module.get_parameter("gate_proj_scales")
        up_proj_weight_fp8 = module.get_parameter("up_proj_fp8")
        up_proj_scales = module.get_parameter("up_proj_scales")
        down_proj_weight_fp8 = module.get_parameter("down_proj_fp8")
        down_proj_scales = module.get_parameter("down_proj_scales")

        # Get dimensions
        m_actual_tokens = valid_tokens.shape[0]
        intermediate_size = module.get_parameter("gate_proj_weight").shape[1]
        hidden_size = module.get_parameter("down_proj_weight").shape[1]

        # Allocate output buffers
        gate_proj_out = torch.empty(
            (m_actual_tokens, intermediate_size),
            device=contig_tokens.device,
            dtype=contig_tokens.dtype,
        )
        up_proj_out = torch.empty_like(gate_proj_out)

        # Allocate output buffer for down projection
        down_proj_out = torch.empty(
            (m_actual_tokens, hidden_size),
            device=contig_tokens.device,
            dtype=contig_tokens.dtype,
        )

        # Prepare input in FP8 format (shared by gate and up projections)
        if self.use_triton_quant:
            gate_up_input = dsgemm_kernels.groupwise_activation_quant(valid_tokens)
        else:
            gate_up_input = dsgemm_utils.prepare_fp8_input(valid_tokens)

        # Run first GEMM (gate projection)
        deep_gemm.m_grouped_gemm_fp8_fp8_bf16_nt_contiguous(
            gate_up_input,
            (gate_proj_weight_fp8, gate_proj_scales),
            gate_proj_out,
            m_indices,
        )

        # Run second GEMM (up projection)
        deep_gemm.m_grouped_gemm_fp8_fp8_bf16_nt_contiguous(
            gate_up_input,
            (up_proj_weight_fp8, up_proj_scales),
            up_proj_out,
            m_indices,
        )

        # Apply activation
        hidden_states = self.activation_function(gate_proj_out) * up_proj_out

        # Run third GEMM (down projection)
        if self.use_triton_quant:
            hidden_states_quantized = dsgemm_kernels.groupwise_activation_quant(
                hidden_states
            )
        else:
            hidden_states_quantized = dsgemm_utils.prepare_fp8_input(hidden_states)

        deep_gemm.m_grouped_gemm_fp8_fp8_bf16_nt_contiguous(
            hidden_states_quantized,
            (down_proj_weight_fp8, down_proj_scales),
            down_proj_out,
            m_indices,
        )

        # Copy results back to contig_tokens
        contig_tokens[: m_offsets[-1]] = down_proj_out
        return contig_tokens

    @staticmethod
    def is_available() -> bool:
        return DEEPGEMM_AVAILABLE
