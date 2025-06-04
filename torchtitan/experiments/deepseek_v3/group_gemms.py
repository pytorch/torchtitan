# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Current cutlass issue:
Generating: NCCL version 2.26.5+cuda12.9
loc("kernel"("/data/users/less/torchtitan/torchtitan/experiments/kernels/blackwell_group_gemms/cute_grouped_gemm.py":651:0)): error: failed to construct a valid coordinate from #cute.coord<"(_,_,_)"> resulting in an incorrect profile
Error using cutlass strategy: DSLRuntimeError: 💥💥💥 Error during runtime code generation for function `__call__` 💥💥💥
loc("kernel"("/data/users/less/torchtitan/torchtitan/experiments/kernels/blackwell_group_gemms/cute_grouped_gemm.py":651:0)): error: failed to construct a valid coordinate from #cute.coord<"(_,_,_)"> resulting in an incorrect profile
Error using cutlass strategy: DSLRuntimeError: 💥💥💥 Error during runtime code generation for function `__call__` 💥💥💥
loc("kernel"("/data/users/less/torchtitan/torchtitan/experiments/kernels/blackwell_group_gemms/cute_grouped_gemm.py":651:0)): error: failed to construct a valid coordinate from #cute.coord<"(_,_,_)"> resulting in an incorrect profile
Error using cutlass strategy: DSLRuntimeError: 💥💥💥 Error during runtime code generation for function `__call__` 💥💥💥
loc("kernel"("/data/users/less/torchtitan/torchtitan/experiments/kernels/blackwell_group_gemms/cute_grouped_gemm.py":651:0)): error: failed to construct a valid coordinate from #cute.coord<"(_,_,_)"> resulting in an incorrect profile
Error using cutlass strategy: DSLRuntimeError: 💥💥💥 Error during runtime code generation for function `__call__` 💥💥💥

"""

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

        # CUTLASS kernel configuration - optimized for typical MoE workloads
        self.grouped_gemm = GroupedGemmKernel(
            acc_dtype=self.acc_dtype,
            use_2cta_instrs=False,  # Set to True for larger problems if beneficial
            mma_tiler_mn=(128, 128),  # Can be tuned based on problem sizes
            cluster_shape_mn=(1, 1),  # Can be increased for larger problems
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

        # CUTLASS expects specific tensor layouts -  handle conversion in execute()
        return combined_weights

    def _prepare_cutlass_tensors(self, torch_tensor, is_weight=False):
        """Convert PyTorch tensor to CUTLASS format with proper layout marking"""
        # Ensure tensor is contiguous
        torch_tensor = torch_tensor.contiguous()

        # Create CUTE tensor
        cute_tensor = from_dlpack(torch_tensor, assumed_align=16)
        cute_tensor.element_type = self.dtype_cutlass

        # Mark appropriate dimension as dynamic for grouped operations
        # For PyTorch tensors in row-major order, the last dimension has stride 1
        # Use the dimension index directly instead of -1
        dim_with_stride_1 = torch_tensor.dim() - 1
        cute_tensor = cute_tensor.mark_layout_dynamic(leading_dim=dim_with_stride_1)

        return cute_tensor

    def _create_problem_specification(
        self, m_sizes, hidden_size, intermediate_size, is_down_proj=False
    ):
        """Create CUTLASS problem specification from MoE group sizes"""
        num_groups = len(m_sizes)
        problem_sizes = []

        for m in m_sizes:
            if is_down_proj:
                # down projection: intermediate_size -> hidden_size
                problem_sizes.append(
                    (int(m), int(hidden_size), int(intermediate_size), 1)
                )
            else:
                # gate/up projections: hidden_size -> intermediate_size
                problem_sizes.append(
                    (int(m), int(intermediate_size), int(hidden_size), 1)
                )

        return problem_sizes

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
            # Get 2D strides (ignore batch dimension if present)
            a_strides = torch_a.stride()[-2:] if torch_a.dim() > 2 else torch_a.stride()
            b_strides = torch_b.stride()[-2:] if torch_b.dim() > 2 else torch_b.stride()
            c_strides = torch_c.stride()[-2:] if torch_c.dim() > 2 else torch_c.stride()

            strides.append([a_strides, b_strides, c_strides])
            pointers.append(
                [torch_a.data_ptr(), torch_b.data_ptr(), torch_c.data_ptr()]
            )

        # Convert to tensors
        strides_tensor = torch.tensor(strides, dtype=torch.int32, device=device)
        strides_cute = from_dlpack(strides_tensor, assumed_align=16)

        pointers_tensor = torch.tensor(pointers, dtype=torch.int64, device=device)
        pointers_cute = from_dlpack(pointers_tensor, assumed_align=16)

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
        cluster_tile_m = 128  # matches mma_tiler_mn[0]
        cluster_tile_n = 128  # matches mma_tiler_mn[1]

        total = 0
        for m, n, k, l in problem_sizes:
            clusters_m = (m + cluster_tile_m - 1) // cluster_tile_m
            clusters_n = (n + cluster_tile_n - 1) // cluster_tile_n
            total += clusters_m * clusters_n

        return total

    def _execute_grouped_gemm(self, input_tokens, weights, problem_sizes, bias=None):
        """Execute a single grouped GEMM operation using CUTLASS"""
        device = input_tokens.device
        num_groups = len(problem_sizes)

        # Prepare tensor groups for each problem
        torch_tensors_abc = []

        token_offset = 0
        for i, (m, n, k, l) in enumerate(problem_sizes):
            # Get tokens for this group
            group_tokens = input_tokens[token_offset : token_offset + m].contiguous()

            # Get weight for this expert
            expert_weight = weights[i].contiguous()

            # Create output tensor
            group_output = torch.zeros(m, n, dtype=self.dtype_torch, device=device)

            torch_tensors_abc.append((group_tokens, expert_weight, group_output))
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
            torch_tensors_abc, problem_sizes
        )

        # Get tensormap buffer
        tensormap_cute = self._get_tensormap_buffer(num_groups, device)

        # Compute grid parameters
        total_clusters = self._compute_total_clusters(problem_sizes)

        # Choose initial tensors (use first group)
        initial_a, initial_b, initial_c = cute_tensors_abc[0]

        # Setup CUDA stream
        torch_stream = torch.cuda.current_stream()
        stream = cuda.CUstream(torch_stream.cuda_stream)

        # Compile kernel (cache based on problem configuration)
        cache_key = (num_groups, tuple(problem_sizes))
        if cache_key not in self._compiled_kernels:
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

        compiled_kernel = self._compiled_kernels[cache_key]

        # Execute kernel
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

        # Collect results
        results = []
        for torch_a, torch_b, torch_c in torch_tensors_abc:
            results.append(torch_c)

        return torch.cat(results, dim=0)

    def execute(self, contig_tokens, m_sizes, m_offsets, module):
        """Execute the complete MoE forward pass using CUTLASS grouped GEMM"""
        # Get weights
        w_gate = module.get_parameter("gate_proj_weight")
        w_up = module.get_parameter("up_proj_weight")
        w_down = module.get_parameter("down_proj_weight")

        # Get dimensions
        hidden_size = w_gate.shape[1]  # input dimension
        intermediate_size = w_gate.shape[2]  # output dimension for gate/up

        # Split contiguous tokens back into groups
        token_groups = []
        offset = 0
        for size in m_sizes:
            if size > 0:
                token_groups.append(contig_tokens[offset : offset + size])
                offset += size
            else:
                token_groups.append(
                    torch.empty(
                        0,
                        hidden_size,
                        dtype=self.dtype_torch,
                        device=contig_tokens.device,
                    )
                )

        # Filter out empty groups for CUTLASS execution
        valid_indices = [i for i, size in enumerate(m_sizes) if size > 0]
        valid_m_sizes = [m_sizes[i] for i in valid_indices]
        valid_token_groups = [token_groups[i] for i in valid_indices]

        if not valid_m_sizes:
            # No valid tokens, return zeros
            return torch.zeros_like(contig_tokens[: sum(m_sizes)])

        # Concatenate valid tokens
        valid_tokens = torch.cat(valid_token_groups, dim=0)

        # Prepare weights for valid experts
        valid_w_gate = w_gate[valid_indices]
        valid_w_up = w_up[valid_indices]
        valid_w_down = w_down[valid_indices]

        # Create problem specifications
        gate_up_problems = self._create_problem_specification(
            valid_m_sizes, hidden_size, intermediate_size, is_down_proj=False
        )
        down_problems = self._create_problem_specification(
            valid_m_sizes, hidden_size, intermediate_size, is_down_proj=True
        )

        # Execute gate projection
        gate_output = self._execute_grouped_gemm(
            valid_tokens, valid_w_gate, gate_up_problems
        )

        # Execute up projection
        up_output = self._execute_grouped_gemm(
            valid_tokens, valid_w_up, gate_up_problems
        )

        # Apply activation and element-wise multiplication
        hidden_states = self.activation_function(gate_output) * up_output

        # Execute down projection
        final_output = self._execute_grouped_gemm(
            hidden_states, valid_w_down, down_problems
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
