# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Fixed Group GEMM implementations with proper CUTLASS integration and manual baseline
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

try:
    from torchtitan.experiments.kernels.triton_mg_group_gemm.torchao_pr import (
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
    from torchtitan.experiments.kernels.blackwell_group_gemms.dense_gemm import (
        DenseGemmKernel,
    )

    CUTLASS_AVAILABLE = True
except ImportError:
    CUTLASS_AVAILABLE = False
    assert False, "CUTLASS is not available"


# Export all strategies
__all__ = [
    "ManualLoopGroupGEMM",
    "CuteDenseLoopingGroupGEMM",
    "CUTLASSGroupGEMM",
    "TorchBF16GroupGEMM",
    "TorchAOBF16GroupGEMM",
    "TritonCGBF16GroupGEMM",
    "DSGroupGEMM",
]


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


# ========= Manual Looping Baseline ===================
class CuteDenseLoopingGroupGEMM(GroupGEMMStrategy):
    """CuteDenseLoopingGroupGEMM baseline implementation for comparison"""

    def arrange_expert_weights(self, all_weights, submod_name, module):
        """Store weights in a simple list format"""
        return torch.stack(all_weights)

    def execute(self, contig_tokens, m_sizes, m_offsets, module):
        w_gate = module.get_parameter("gate_proj_weight")
        w_up = module.get_parameter("up_proj_weight")
        w_down = module.get_parameter("down_proj_weight")

        # Prepare output tensor
        hidden_size = w_gate.shape[
            2
        ]  # Assuming stacked weights shape [num_experts, out_dim, in_dim]
        output = torch.zeros(
            contig_tokens.shape[0],
            hidden_size,
            dtype=contig_tokens.dtype,
            device=contig_tokens.device,
        )

        # Process each expert sequentially
        offset = 0
        for expert_idx, size in enumerate(m_sizes):
            if size > 0:
                # Get tokens for this expert
                expert_tokens = contig_tokens[offset : offset + size]

                # Get weights for this expert
                gate_weight = w_gate[expert_idx]  # [out_dim, in_dim]
                up_weight = w_up[expert_idx]
                down_weight = w_down[expert_idx]

                # Forward pass: gate and up projections
                gate_out = torch.mm(expert_tokens, gate_weight.t())
                up_out = torch.mm(expert_tokens, up_weight.t())

                # Apply activation and combine
                hidden = self.activation_function(gate_out) * up_out

                # Down projection
                expert_output = torch.mm(hidden, down_weight.t())

                # Store results
                output[offset : offset + size] = expert_output

            offset += size

        return output

    @staticmethod
    def is_available() -> bool:
        return True


class ManualLoopGroupGEMM(GroupGEMMStrategy):
    """Manual looping baseline implementation for comparison"""

    def arrange_expert_weights(self, all_weights, submod_name, module):
        """Store weights in a simple list format"""
        return torch.stack(all_weights)

    def execute(self, contig_tokens, m_sizes, m_offsets, module):
        """Execute using manual loops over experts"""
        # Get weights and ensure they're on the same device as tokens
        device = contig_tokens.device
        w_gate = module.get_parameter("gate_proj_weight").to(device)
        w_up = module.get_parameter("up_proj_weight").to(device)
        w_down = module.get_parameter("down_proj_weight").to(device)

        # Prepare output tensor
        hidden_size = w_gate.shape[
            2
        ]  # Assuming stacked weights shape [num_experts, out_dim, in_dim]
        output = torch.zeros(
            contig_tokens.shape[0],
            hidden_size,
            dtype=contig_tokens.dtype,
            device=contig_tokens.device,
        )

        # Process each expert sequentially
        offset = 0
        for expert_idx, size in enumerate(m_sizes):
            if size > 0:
                # Get tokens for this expert
                expert_tokens = contig_tokens[offset : offset + size]

                # Get weights for this expert
                gate_weight = w_gate[expert_idx]  # [out_dim, in_dim]
                up_weight = w_up[expert_idx]
                down_weight = w_down[expert_idx]

                # Forward pass: gate and up projections
                gate_out = torch.mm(expert_tokens, gate_weight.t())
                up_out = torch.mm(expert_tokens, up_weight.t())

                # Apply activation and combine
                hidden = self.activation_function(gate_out) * up_out

                # Down projection
                expert_output = torch.mm(hidden, down_weight.t())

                # Store results
                output[offset : offset + size] = expert_output

            offset += size

        return output

    @staticmethod
    def is_available() -> bool:
        return True


# ========= Cutlass Python Implementation ===================


class CUTLASSGroupGEMM(GroupGEMMStrategy):
    """Fixed CUTLASS GroupedGemmKernel implementation"""

    def __init__(self, custom_activation):
        super().__init__(custom_activation)
        self.dtype_torch = torch.bfloat16
        self.dtype_cutlass = cutlass.BFloat16
        self.acc_dtype = cutlass.Float32

        logger.info("Initializing CUTLASS GroupedGemmKernel")

        # Kernel configuration
        self.grouped_gemm = GroupedGemmKernel(
            acc_dtype=self.acc_dtype,
            use_2cta_instrs=False,
            mma_tiler_mn=(128, 64),
            cluster_shape_mn=(1, 1),
            tensormap_update_mode=utils.TensorMapUpdateMode.SMEM,
        )

        # Hardware info
        self.hardware_info = utils.HardwareInfo()
        self.max_active_clusters = self.hardware_info.get_max_active_clusters(1)

        # Buffers
        self._tensormap_buffers = {}
        self._compiled_kernels = {}

    def arrange_expert_weights(self, all_weights, submod_name, module):
        """Prepare expert weights for CUTLASS grouped GEMM"""
        # Stack weights: [num_experts, out_dim, in_dim]
        combined_weights = torch.stack(all_weights)
        logger.info(f"CUTLASS arranged weights {submod_name}: {combined_weights.shape}")
        return combined_weights

    def _create_tensor_metadata(self, tokens, w_gate, w_up, w_down, m_sizes, device):
        """Create the metadata tensors required by CUTLASS GroupedGEMM"""
        # Filter out empty groups and create contiguous data
        valid_groups = []
        problem_sizes = []
        strides_abc = []
        ptrs_abc = []
        output_tensors = []

        offset = 0
        for expert_idx, size in enumerate(m_sizes):
            if size > 0:
                # Get input tokens for this expert
                expert_tokens = tokens[offset : offset + size].contiguous()

                # Get weights for this expert
                gate_weight = w_gate[
                    expert_idx
                ].contiguous()  # [intermediate_size, hidden_size]
                up_weight = w_up[expert_idx].contiguous()
                down_weight = w_down[
                    expert_idx
                ].contiguous()  # [hidden_size, intermediate_size]

                # Create tensors for each GEMM operation in MoE forward pass
                M = size  # Number of tokens for this expert
                K_in = expert_tokens.shape[1]  # hidden_size
                N_intermediate = gate_weight.shape[0]  # intermediate_size
                N_out = down_weight.shape[0]  # hidden_size (output)
                L = 1  # Batch dimension

                # Store group info
                group_info = {
                    "expert_idx": expert_idx,
                    "tokens": expert_tokens,
                    "gate_weight": gate_weight,
                    "up_weight": up_weight,
                    "down_weight": down_weight,
                    "M": M,
                    "K_in": K_in,
                    "N_intermediate": N_intermediate,
                    "N_out": N_out,
                }
                valid_groups.append(group_info)

                # For gate projection: A=[M,K], B=[N,K], C=[M,N]
                # CUTLASS expects B in [N,K] format (already correct)
                gate_A = expert_tokens  # [M, K_in]
                gate_B = gate_weight  # [N_intermediate, K_in]
                gate_C = torch.empty(
                    M, N_intermediate, dtype=self.dtype_torch, device=device
                )

                problem_sizes.append([M, N_intermediate, K_in, L])
                strides_abc.append(
                    [
                        [gate_A.stride(0), gate_A.stride(1)],  # A strides
                        [gate_B.stride(0), gate_B.stride(1)],  # B strides
                        [gate_C.stride(0), gate_C.stride(1)],  # C strides
                    ]
                )
                ptrs_abc.append(
                    [gate_A.data_ptr(), gate_B.data_ptr(), gate_C.data_ptr()]
                )
                output_tensors.append(gate_C)

                # For up projection: same dimensions as gate
                up_A = expert_tokens  # [M, K_in]
                up_B = up_weight  # [N_intermediate, K_in]
                up_C = torch.empty(
                    M, N_intermediate, dtype=self.dtype_torch, device=device
                )

                problem_sizes.append([M, N_intermediate, K_in, L])
                strides_abc.append(
                    [
                        [up_A.stride(0), up_A.stride(1)],
                        [up_B.stride(0), up_B.stride(1)],
                        [up_C.stride(0), up_C.stride(1)],
                    ]
                )
                ptrs_abc.append([up_A.data_ptr(), up_B.data_ptr(), up_C.data_ptr()])
                output_tensors.append(up_C)

            offset += size

        if not valid_groups:
            return None, None, None, None, None

        # Convert to tensors
        problem_sizes_tensor = torch.tensor(
            problem_sizes, dtype=torch.int32, device=device
        )
        strides_tensor = torch.tensor(strides_abc, dtype=torch.int32, device=device)
        ptrs_tensor = torch.tensor(ptrs_abc, dtype=torch.int64, device=device)

        # Convert to CUTE tensors
        problem_sizes_cute = from_dlpack(problem_sizes_tensor, assumed_align=16)
        strides_cute = from_dlpack(strides_tensor, assumed_align=16)
        ptrs_cute = from_dlpack(ptrs_tensor, assumed_align=16)

        return problem_sizes_cute, strides_cute, ptrs_cute, valid_groups, output_tensors

    def _create_initial_tensors(self, tokens, weights, device):
        """Create initial tensors for tensormap setup"""
        # Use smallest problem size for initial setup
        M, K = 128, tokens.shape[1]
        N = weights.shape[1]

        A_init = torch.randn(M, K, dtype=self.dtype_torch, device=device)
        B_init = torch.randn(
            N, K, dtype=self.dtype_torch, device=device
        )  # Note: N,K format
        C_init = torch.zeros(M, N, dtype=self.dtype_torch, device=device)

        # Convert to MNKL format and mark dynamic
        A_mnkl = A_init.unsqueeze(-1).contiguous()
        B_mnkl = B_init.unsqueeze(-1).contiguous()
        C_mnkl = C_init.unsqueeze(-1).contiguous()

        A_cute = from_dlpack(A_mnkl, assumed_align=16)
        B_cute = from_dlpack(B_mnkl, assumed_align=16)
        C_cute = from_dlpack(C_mnkl, assumed_align=16)

        # Set CUTLASS data types
        A_cute.element_type = self.dtype_cutlass
        B_cute.element_type = self.dtype_cutlass
        C_cute.element_type = self.dtype_cutlass

        # Mark layouts as dynamic
        A_cute = A_cute.mark_layout_dynamic(leading_dim=1)
        B_cute = B_cute.mark_layout_dynamic(leading_dim=1)
        C_cute = C_cute.mark_layout_dynamic(leading_dim=1)

        return A_cute, B_cute, C_cute

    def _setup_tensormap_buffer(self, num_groups, device):
        """Setup tensormap buffer for CUTLASS"""
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

    def _compute_total_clusters(self, problem_sizes, cluster_shape_mn=(1, 1)):
        """Compute total number of clusters needed"""
        cluster_tile_m = 128  # From mma_tiler_mn[0]
        cluster_tile_n = 64  # From mma_tiler_mn[1]

        total = 0
        for m, n, k, l in problem_sizes:
            clusters_m = (m + cluster_tile_m - 1) // cluster_tile_m
            clusters_n = (n + cluster_tile_n - 1) // cluster_tile_n
            total += clusters_m * clusters_n

        return total

    def execute(self, contig_tokens, m_sizes, m_offsets, module):
        """Execute the complete MoE forward pass using CUTLASS grouped GEMM"""
        try:
            # Get weights
            w_gate = module.get_parameter("gate_proj_weight")
            w_up = module.get_parameter("up_proj_weight")
            w_down = module.get_parameter("down_proj_weight")

            device = contig_tokens.device
            num_valid_experts = len([s for s in m_sizes if s > 0])

            if num_valid_experts == 0:
                return torch.zeros_like(contig_tokens)

            logger.info(f"CUTLASS executing with {num_valid_experts} experts")

            # Step 1: Create metadata for gate and up projections (can be batched)
            gate_up_metadata = self._create_tensor_metadata(
                contig_tokens, w_gate, w_up, w_down, m_sizes, device
            )

            if gate_up_metadata[0] is None:
                return torch.zeros_like(contig_tokens)

            (
                problem_sizes_cute,
                strides_cute,
                ptrs_cute,
                valid_groups,
                gate_up_outputs,
            ) = gate_up_metadata

            # Step 2: Create initial tensors for tensormap setup
            first_group = valid_groups[0]
            initial_A, initial_B, initial_C = self._create_initial_tensors_from_group(
                first_group, device
            )

            # Step 3: Setup tensormap buffer
            num_operations = len(gate_up_outputs)  # gate + up operations
            tensormap_cute = self._setup_tensormap_buffer(num_operations, device)

            # Step 4: Compute total clusters and setup kernel
            total_clusters = self._compute_total_clusters_from_metadata(
                problem_sizes_cute
            )

            # Step 5: Execute gate and up projections using CUTLASS
            gate_up_results = self._execute_cutlass_grouped_gemm(
                initial_A,
                initial_B,
                initial_C,
                num_operations,
                problem_sizes_cute,
                strides_cute,
                ptrs_cute,
                total_clusters,
                tensormap_cute,
            )

            if gate_up_results is None:
                logger.warning(
                    "CUTLASS kernel execution failed, falling back to manual"
                )
                return self._manual_fallback_full(
                    contig_tokens, m_sizes, w_gate, w_up, w_down
                )

            # Step 6: Apply activation and combine gate/up results
            intermediate_results = self._apply_activation_and_combine(
                gate_up_outputs, valid_groups
            )

            # Step 7: Execute down projections
            final_output = self._execute_down_projections(
                intermediate_results, valid_groups, device
            )

            # Step 8: Reconstruct full output tensor
            return self._reconstruct_output(final_output, m_sizes, contig_tokens)

        except Exception as e:
            logger.error(f"CUTLASS execution failed: {e}")
            # Fall back to manual implementation
            return self._manual_fallback_full(
                contig_tokens, m_sizes, w_gate, w_up, w_down
            )

    def _create_initial_tensors_from_group(self, group_info, device):
        """Create initial CUTE tensors from group information"""
        M, K_in, N_intermediate = (
            group_info["M"],
            group_info["K_in"],
            group_info["N_intermediate"],
        )

        # Create initial tensors with proper dimensions
        A_init = torch.randn(M, K_in, dtype=self.dtype_torch, device=device)
        B_init = torch.randn(
            N_intermediate, K_in, dtype=self.dtype_torch, device=device
        )
        C_init = torch.zeros(M, N_intermediate, dtype=self.dtype_torch, device=device)

        # Convert to MNKL format and mark dynamic
        A_mnkl = A_init.unsqueeze(-1).contiguous()
        B_mnkl = B_init.unsqueeze(-1).contiguous()
        C_mnkl = C_init.unsqueeze(-1).contiguous()

        A_cute = from_dlpack(A_mnkl, assumed_align=16)
        B_cute = from_dlpack(B_mnkl, assumed_align=16)
        C_cute = from_dlpack(C_mnkl, assumed_align=16)

        # Set CUTLASS data types
        A_cute.element_type = self.dtype_cutlass
        B_cute.element_type = self.dtype_cutlass
        C_cute.element_type = self.dtype_cutlass

        # Mark layouts as dynamic
        A_cute = A_cute.mark_layout_dynamic(leading_dim=1)
        B_cute = B_cute.mark_layout_dynamic(leading_dim=1)
        C_cute = C_cute.mark_layout_dynamic(leading_dim=1)

        return A_cute, B_cute, C_cute

    def _compute_total_clusters_from_metadata(self, problem_sizes_cute):
        """Compute total clusters from problem sizes metadata"""
        # Convert CUTE tensor back to Python list for computation
        problem_sizes_torch = problem_sizes_cute.to_torch_tensor()
        total = 0

        cluster_tile_m = 128  # From mma_tiler_mn[0]
        cluster_tile_n = 64  # From mma_tiler_mn[1]

        for i in range(problem_sizes_torch.shape[0]):
            m, n, k, l = problem_sizes_torch[i].tolist()
            clusters_m = (m + cluster_tile_m - 1) // cluster_tile_m
            clusters_n = (n + cluster_tile_n - 1) // cluster_tile_n
            total += clusters_m * clusters_n

        return total

    def _execute_cutlass_grouped_gemm(
        self,
        initial_A,
        initial_B,
        initial_C,
        num_groups,
        problem_sizes_cute,
        strides_cute,
        ptrs_cute,
        total_clusters,
        tensormap_cute,
    ):
        """Execute the CUTLASS grouped GEMM kernel"""
        try:
            # Setup CUDA stream
            torch_stream = torch.cuda.current_stream()
            stream = cuda.CUstream(torch_stream.cuda_stream)

            # Compile kernel if not already cached
            cache_key = (num_groups, total_clusters)
            if cache_key not in self._compiled_kernels:
                logger.info(
                    f"Compiling CUTLASS kernel for {num_groups} groups, {total_clusters} clusters"
                )

                self._compiled_kernels[cache_key] = cute.compile(
                    self.grouped_gemm,
                    initial_A,
                    initial_B,
                    initial_C,
                    num_groups,
                    problem_sizes_cute,
                    strides_cute,
                    ptrs_cute,
                    total_clusters,
                    tensormap_cute,
                    self.max_active_clusters,
                    stream,
                )

                logger.info("CUTLASS kernel compilation successful")

            compiled_kernel = self._compiled_kernels[cache_key]

            # Execute kernel
            logger.info(f"Executing CUTLASS grouped GEMM kernel")
            compiled_kernel(
                initial_A,
                initial_B,
                initial_C,
                problem_sizes_cute,
                strides_cute,
                ptrs_cute,
                tensormap_cute,
                stream,
            )

            # Synchronize to ensure completion
            torch.cuda.synchronize()
            logger.info("CUTLASS kernel execution completed")

            return True

        except Exception as e:
            logger.error(f"CUTLASS kernel execution failed: {e}")
            return None

    def _apply_activation_and_combine(self, gate_up_outputs, valid_groups):
        """Apply activation function and combine gate/up projection results"""
        intermediate_results = []

        # gate_up_outputs contains interleaved gate and up results
        for i in range(0, len(gate_up_outputs), 2):
            gate_output = gate_up_outputs[i]
            up_output = gate_up_outputs[i + 1]

            # Apply activation to gate output and multiply with up output
            activated_gate = self.activation_function(gate_output)
            combined = activated_gate * up_output

            intermediate_results.append(combined)

        return intermediate_results

    def _execute_down_projections(self, intermediate_results, valid_groups, device):
        """Execute down projections manually (could be another grouped GEMM)"""
        final_outputs = []

        for i, (intermediate, group_info) in enumerate(
            zip(intermediate_results, valid_groups)
        ):
            down_weight = group_info["down_weight"]  # [hidden_size, intermediate_size]

            # Manual matrix multiplication for down projection
            final_output = torch.mm(intermediate, down_weight.t())
            final_outputs.append(final_output)

        return final_outputs

    def _reconstruct_output(self, final_outputs, m_sizes, contig_tokens):
        """Reconstruct the full output tensor from expert results"""
        total_tokens = sum(m_sizes)
        hidden_size = (
            final_outputs[0].shape[1] if final_outputs else contig_tokens.shape[1]
        )

        output = torch.zeros(
            total_tokens,
            hidden_size,
            dtype=contig_tokens.dtype,
            device=contig_tokens.device,
        )

        output_idx = 0
        result_idx = 0

        for size in m_sizes:
            if size > 0:
                if result_idx < len(final_outputs):
                    output[output_idx : output_idx + size] = final_outputs[result_idx]
                    result_idx += 1
                output_idx += size

        return output

    def _manual_fallback_full(self, tokens, m_sizes, w_gate, w_up, w_down):
        """Complete manual fallback implementation"""
        total_tokens = sum(m_sizes)
        hidden_size = w_gate.shape[2] if len(w_gate.shape) > 2 else w_gate.shape[1]

        output = torch.zeros(
            total_tokens, hidden_size, dtype=tokens.dtype, device=tokens.device
        )

        offset = 0
        expert_idx = 0

        for size in m_sizes:
            if size > 0:
                if expert_idx < w_gate.shape[0]:  # Check bounds
                    expert_tokens = tokens[offset : offset + size]

                    # Forward pass through expert
                    gate_out = torch.mm(expert_tokens, w_gate[expert_idx].t())
                    up_out = torch.mm(expert_tokens, w_up[expert_idx].t())
                    hidden = self.activation_function(gate_out) * up_out
                    expert_output = torch.mm(hidden, w_down[expert_idx].t())

                    output[offset : offset + size] = expert_output
                    expert_idx += 1

            offset += size

        return output

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
        device = contig_tokens.device
        w_gate = module.get_parameter("gate_proj_weight").to(device)
        w_up = module.get_parameter("up_proj_weight").to(device)
        w_down = module.get_parameter("down_proj_weight").to(device)

        # Run first two GEMMs (gate and up projections)
        # Get only valid tokens
        valid_tokens = contig_tokens[: m_offsets[-1]]

        # Create indices from offsets without CPU-GPU sync
        m_indices = torch.tensor(
            dsgemm_utils.create_indices_from_offsets_nosync(m_offsets)
        )

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
    """Implementation for PyTorch native BF16 _grouped_mm"""

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
