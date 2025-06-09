# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn

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

    # Import our strategy - UPDATE PATH AS NEEDED

    from cutlass.cute.runtime import from_dlpack
    from torchtitan.experiments.kernels.blackwell.cute_grouped_gemm import (
        GroupedGemmKernel,
    )

    HAS_CUTLASS = True
    print("✓ CUTLASS and strategies imported successfully")
except ImportError as e:
    HAS_CUTLASS = False
    print(f"✗ Import failed: {e}")
    print("Using PyTorch fallback implementations only")


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
    "CUTLASSGroupedGemmStrategy",
    "ManualLoopGroupGEMM",
]


class ManualLoopGroupGEMM(GroupGEMMStrategy):
    """Manual looping baseline implementation for any arch (esp Blackwell) support"""

    def arrange_expert_weights(self, all_weights, submod_name, module):
        """Store weights in a stacked format"""
        return torch.stack(all_weights)

    def execute(self, contig_tokens, m_sizes, m_offsets, module):
        """Execute using manual loops over experts"""
        # Get weights

        w_gate = module.get_parameter("gate_proj_weight")
        w_up = module.get_parameter("up_proj_weight")
        w_down = module.get_parameter("down_proj_weight")

        # Prepare output tensor
        hidden_size = w_gate.shape[
            2
        ]  # stacked weights shape [num_experts, out_dim, in_dim]
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


class CUTLASSGroupedGemmStrategy(GroupGEMMStrategy):
    """
    Strategy using CUTLASS GroupedGemmKernel for group GEMM operations

    """

    def __init__(self, custom_activation):
        super().__init__(custom_activation)
        self.dtype_torch = torch.bfloat16
        self.dtype_cutlass = cutlass.BFloat16
        self.acc_dtype = cutlass.Float32
        self.alignment = 16

        # Create grouped GEMM kernel
        self.grouped_gemm = GroupedGemmKernel(
            acc_dtype=self.acc_dtype,
            use_2cta_instrs=False,
            mma_tiler_mn=(128, 64),
            cluster_shape_mn=(1, 1),
            tensormap_update_mode=utils.TensorMapUpdateMode.SMEM,
        )

        # Setup hardware info and stream
        self.hardware_info = utils.HardwareInfo()
        self.max_active_clusters = self.hardware_info.get_max_active_clusters(1)

        torch_stream = torch.cuda.current_stream()
        self.stream = cuda.CUstream(torch_stream.cuda_stream)

        # Cache for compiled kernels and tensormap buffers
        self._compiled_kernels = {}
        self._tensormap_buffers = {}

        print("Initialized CUTLASSGroupedGemmStrategy with GroupedGemmKernel")

    def arrange_expert_weights(self, all_weights, submod_name, module):
        """Store weights in stacked format"""
        return torch.stack(all_weights)

    def execute(self, contig_tokens, m_sizes, m_offsets, module):
        """Execute using CUTLASS grouped GEMM kernel"""
        # Get weights
        w_gate = module.get_parameter("gate_proj_weight")
        w_up = module.get_parameter("up_proj_weight")
        w_down = module.get_parameter("down_proj_weight")

        device = contig_tokens.device
        hidden_size = w_gate.shape[2]

        # Prepare output tensor
        output = torch.zeros(
            contig_tokens.shape[0], hidden_size, dtype=self.dtype_torch, device=device
        )

        # Filter valid experts
        valid_experts = [(i, size) for i, size in enumerate(m_sizes) if size > 0]
        if not valid_experts:
            return output

        # Step 1: Execute gate and up projections using grouped GEMM
        gate_outputs, up_outputs = self._execute_gate_up_projections(
            contig_tokens, w_gate, w_up, m_sizes, device
        )

        # Step 2: Apply activation and combine
        hidden_states = self._apply_activation_and_combine(
            gate_outputs, up_outputs, m_sizes
        )

        # Step 3: Execute down projection using grouped GEMM
        final_outputs = self._execute_down_projection(
            hidden_states, w_down, m_sizes, device
        )

        # Step 4: Reconstruct full output
        return self._reconstruct_output(final_outputs, m_sizes, output)

    def _execute_gate_up_projections(
        self, contig_tokens, w_gate, w_up, m_sizes, device
    ):
        """Execute gate and up projections using grouped GEMM"""
        # Prepare tensors and metadata for gate and up projections
        problem_sizes = []
        strides_abc = []
        ptrs_abc = []
        gate_outputs = []
        up_outputs = []

        offset = 0
        for expert_idx, size in enumerate(m_sizes):
            if size > 0:
                # Get expert tokens
                expert_tokens = contig_tokens[offset : offset + size].contiguous()
                gate_weight = w_gate[
                    expert_idx
                ].contiguous()  # [intermediate_size, hidden_size]
                up_weight = w_up[expert_idx].contiguous()

                M, K = expert_tokens.shape
                N = gate_weight.shape[0]  # intermediate_size
                L = 1

                # Create output tensors for gate and up projections
                gate_output = torch.empty(M, N, dtype=self.dtype_torch, device=device)
                up_output = torch.empty(M, N, dtype=self.dtype_torch, device=device)

                # Convert to MNKL format (following bench_group_gemm.py)
                expert_tokens_mnkl = expert_tokens.unsqueeze(
                    -1
                ).contiguous()  # (M, K, 1)
                gate_weight_mnkl = gate_weight.unsqueeze(-1).contiguous()  # (N, K, 1)
                up_weight_mnkl = up_weight.unsqueeze(-1).contiguous()  # (N, K, 1)
                gate_output_mnkl = gate_output.unsqueeze(-1).contiguous()  # (M, N, 1)
                up_output_mnkl = up_output.unsqueeze(-1).contiguous()  # (M, N, 1)

                # Extract 2D strides from MNKL tensors (following bench_group_gemm.py)
                expert_tokens_strides = expert_tokens_mnkl.stride()[:2]
                gate_weight_strides = gate_weight_mnkl.stride()[:2]
                up_weight_strides = up_weight_mnkl.stride()[:2]
                gate_output_strides = gate_output_mnkl.stride()[:2]
                up_output_strides = up_output_mnkl.stride()[:2]

                # Gate projection metadata
                problem_sizes.append([M, N, K, L])
                strides_abc.append(
                    [
                        list(expert_tokens_strides),  # A strides
                        list(gate_weight_strides),  # B strides
                        list(gate_output_strides),  # C strides
                    ]
                )
                ptrs_abc.append(
                    [
                        expert_tokens.data_ptr(),
                        gate_weight.data_ptr(),
                        gate_output.data_ptr(),
                    ]
                )

                # Up projection metadata
                problem_sizes.append([M, N, K, L])
                strides_abc.append(
                    [
                        list(expert_tokens_strides),  # A strides
                        list(up_weight_strides),  # B strides
                        list(up_output_strides),  # C strides
                    ]
                )
                ptrs_abc.append(
                    [
                        expert_tokens.data_ptr(),
                        up_weight.data_ptr(),
                        up_output.data_ptr(),
                    ]
                )

                gate_outputs.append(gate_output)
                up_outputs.append(up_output)

            offset += size

        if not problem_sizes:
            return [], []

        # Execute grouped GEMM for gate and up projections
        self._execute_grouped_gemm(problem_sizes, strides_abc, ptrs_abc, device)

        return gate_outputs, up_outputs

    def _execute_down_projection(self, hidden_states, w_down, m_sizes, device):
        """Execute down projection using grouped GEMM"""
        problem_sizes = []
        strides_abc = []
        ptrs_abc = []
        down_outputs = []

        expert_idx = 0
        for size in m_sizes:
            if size > 0 and expert_idx < len(hidden_states):
                hidden = hidden_states[expert_idx]
                down_weight = w_down[
                    expert_idx
                ].contiguous()  # [hidden_size, intermediate_size]

                M, K = hidden.shape
                N = down_weight.shape[0]  # hidden_size
                L = 1

                # Create output tensor
                down_output = torch.empty(M, N, dtype=self.dtype_torch, device=device)

                # Convert to MNKL format (following bench_group_gemm.py)
                hidden_mnkl = hidden.unsqueeze(-1).contiguous()  # (M, K, 1)
                down_weight_mnkl = down_weight.unsqueeze(-1).contiguous()  # (N, K, 1)
                down_output_mnkl = down_output.unsqueeze(-1).contiguous()  # (M, N, 1)

                # Extract 2D strides from MNKL tensors (following bench_group_gemm.py)
                hidden_strides = hidden_mnkl.stride()[:2]
                down_weight_strides = down_weight_mnkl.stride()[:2]
                down_output_strides = down_output_mnkl.stride()[:2]

                # Down projection metadata
                problem_sizes.append([M, N, K, L])
                strides_abc.append(
                    [
                        list(hidden_strides),  # A strides
                        list(down_weight_strides),  # B strides
                        list(down_output_strides),  # C strides
                    ]
                )
                ptrs_abc.append(
                    [hidden.data_ptr(), down_weight.data_ptr(), down_output.data_ptr()]
                )

                down_outputs.append(down_output)
                expert_idx += 1

        if not problem_sizes:
            return []

        # Execute grouped GEMM for down projection
        self._execute_grouped_gemm(problem_sizes, strides_abc, ptrs_abc, device)

        return down_outputs

    def _execute_grouped_gemm(self, problem_sizes, strides_abc, ptrs_abc, device):
        """Execute the grouped GEMM kernel"""
        num_groups = len(problem_sizes)

        # Convert to tensors
        problem_sizes_tensor = torch.tensor(
            problem_sizes, dtype=torch.int32, device=device
        )
        strides_tensor = torch.tensor(strides_abc, dtype=torch.int32, device=device)
        ptrs_tensor = torch.tensor(ptrs_abc, dtype=torch.int64, device=device)

        # Convert to CUTE tensors
        problem_sizes_cute = from_dlpack(
            problem_sizes_tensor, assumed_align=self.alignment
        )
        strides_cute = from_dlpack(strides_tensor, assumed_align=self.alignment)
        ptrs_cute = from_dlpack(ptrs_tensor, assumed_align=self.alignment)

        # Setup tensormap buffer
        tensormap_cute = self._get_tensormap_buffer(device)

        # Compute total clusters
        total_clusters = self._compute_total_clusters(problem_sizes)

        # Create initial tensors for kernel compilation (use first problem for shapes)
        initial_A, initial_B, initial_C = self._create_initial_tensors(
            problem_sizes[0], device
        )

        # Get or compile kernel
        cache_key = (num_groups, total_clusters)
        if cache_key not in self._compiled_kernels:
            print(f"Compiling grouped GEMM kernel for {num_groups} groups")
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
                self.stream,
            )
            print("Kernel compilation successful")

        # Execute kernel
        compiled_kernel = self._compiled_kernels[cache_key]
        compiled_kernel(
            initial_A,
            initial_B,
            initial_C,
            problem_sizes_cute,
            strides_cute,
            ptrs_cute,
            tensormap_cute,
            self.stream,
        )

        # Synchronize to ensure completion
        torch.cuda.synchronize()

    def _create_initial_tensors(self, problem_shape, device):
        """Create initial CUTE tensors for kernel compilation"""
        M, N, K, L = problem_shape

        # Create tensors with the right shapes
        # A: tokens [M, K], B: weights [N, K], C: output [M, N]
        A_init = torch.randn(M, K, dtype=self.dtype_torch, device=device)
        B_init = torch.randn(
            N, K, dtype=self.dtype_torch, device=device
        )  # Already (N, K) format
        C_init = torch.zeros(M, N, dtype=self.dtype_torch, device=device)

        # Convert to MNKL format
        A_mnkl = A_init.unsqueeze(-1).contiguous()  # (M, K) -> (M, K, 1)
        B_mnkl = B_init.unsqueeze(
            -1
        ).contiguous()  # (N, K) -> (N, K, 1) - no transpose needed
        C_mnkl = C_init.unsqueeze(-1).contiguous()  # (M, N) -> (M, N, 1)

        # Create CUTE tensors
        A_cute = from_dlpack(A_mnkl, assumed_align=self.alignment)
        B_cute = from_dlpack(B_mnkl, assumed_align=self.alignment)
        C_cute = from_dlpack(C_mnkl, assumed_align=self.alignment)

        # Set CUTLASS data types
        A_cute.element_type = self.dtype_cutlass
        B_cute.element_type = self.dtype_cutlass
        C_cute.element_type = self.dtype_cutlass

        # Mark layouts as dynamic
        A_cute = A_cute.mark_layout_dynamic(leading_dim=1)
        B_cute = B_cute.mark_layout_dynamic(leading_dim=1)
        C_cute = C_cute.mark_layout_dynamic(leading_dim=1)

        return A_cute, B_cute, C_cute

    def _get_tensormap_buffer(self, device):
        """Get or create tensormap buffer"""
        if device not in self._tensormap_buffers:
            sm_count = self.hardware_info.get_max_active_clusters(1)
            tensormap_tensor = torch.zeros(
                (sm_count, 3, 128 // 8),  # 3 tensormaps (A, B, C), 128 bytes each
                dtype=torch.int64,
                device=device,
            )
            self._tensormap_buffers[device] = from_dlpack(
                tensormap_tensor, assumed_align=self.alignment
            )

        return self._tensormap_buffers[device]

    def _compute_total_clusters(self, problem_sizes):
        """Compute total number of clusters needed"""
        cluster_tile_m = 128  # From mma_tiler_mn[0]
        cluster_tile_n = 64  # From mma_tiler_mn[1]

        total = 0
        for M, N, K, L in problem_sizes:
            clusters_m = (M + cluster_tile_m - 1) // cluster_tile_m
            clusters_n = (N + cluster_tile_n - 1) // cluster_tile_n
            total += clusters_m * clusters_n

        return total

    def _apply_activation_and_combine(self, gate_outputs, up_outputs, m_sizes):
        """Apply activation and combine gate/up outputs"""
        hidden_states = []

        for gate_out, up_out in zip(gate_outputs, up_outputs):
            # Apply activation to gate output and multiply with up output
            activated_gate = self.activation_function(gate_out)
            combined = activated_gate * up_out
            hidden_states.append(combined)

        return hidden_states

    def _reconstruct_output(self, final_outputs, m_sizes, output):
        """Reconstruct the full output tensor from expert results"""
        offset = 0
        expert_idx = 0

        for size in m_sizes:
            if size > 0 and expert_idx < len(final_outputs):
                output[offset : offset + size] = final_outputs[expert_idx]
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
