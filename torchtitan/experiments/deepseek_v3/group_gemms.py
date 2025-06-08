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

# Cutlass Cute DSL
try:
    import cuda.bindings.driver as cuda
    import cutlass
    import cutlass.cute as cute
    import cutlass.torch as cutlass_torch
    import cutlass.utils as utils
    from cutlass.cute.runtime import from_dlpack

    from torchtitan.experiments.kernels.blackwell.cute_dense_gemm import DenseGemmKernel

    CUTLASS_AVAILABLE = True
except ImportError as e:
    CUTLASS_AVAILABLE = False
    print(f"Cutlass imports not available: {e}`")
    print("Please run `pip install nvidia-cutlass-dsl`")


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
    "CuteDenseLoopingGroupGEMM",
]


# requires pip install nvidia-cutlass-dsl
class CuteDenseLoopingGroupGEMM(GroupGEMMStrategy):
    """
    Implementation of grouped GEMM using Blackwell Dense GEMM kernel with manual looping.

    High level overview:
    - Compiled kernels via Kernel caching: Compiled kernels are cached and reused
    - Expert token tensor reuse: For MoE forward pass, expert_tokens are converted to CUTE
      format once and reused for both gate and up projections
    - Backup: Falls back to PyTorch implementation if CUTE kernels fail
    """

    def __init__(self, custom_activation):
        """
        Initialize the CuteDenseLoopingGroupGEMM.

        Args:
            custom_activation: Activation function to use
        """
        super().__init__(custom_activation)

        # Kernel configuration
        self.alignment = 16
        self.dtype = torch.bfloat16
        self.cutlass_dtype = cutlass.BFloat16

        # Initialize Cute Dense GEMM kernel
        try:
            self.gemm_kernel = DenseGemmKernel(
                acc_dtype=cutlass.Float32,
                use_2cta_instrs=False,
                mma_tiler_mn=(128, 128),
                cluster_shape_mn=(1, 1),
                use_tma_store=False,
            )
        except Exception as e:
            raise RuntimeError(f"Failed to initialize GEMM kernel: {e}") from e

        # Setup CUDA stream
        torch_stream = torch.cuda.Stream()
        self.stream = cuda.CUstream(torch_stream.cuda_stream)

        # Cache for compiled kernels
        self._compiled_kernels = {}

        # debug monitoring
        self.debug_mode = True

    def arrange_expert_weights(self, all_weights, submod_name, module):
        """Store weights in a simple list format"""
        return torch.stack(all_weights)

    def _create_cute_tensor(self, tensor: torch.Tensor) -> cute.Tensor:
        """
        Convert a PyTorch tensor to a CUTE tensor with proper formatting.

        Args:
            tensor: PyTorch tensor to convert

        Returns:
            CUTE tensor ready for kernel execution
        """
        if not tensor.is_contiguous():
            tensor = tensor.contiguous()

        # Convert to MNKL format
        tensor_mnkl = tensor.unsqueeze(-1).contiguous().detach()

        # Create CUTE tensor
        cute_tensor = from_dlpack(tensor_mnkl, assumed_align=self.alignment)
        cute_tensor.element_type = self.cutlass_dtype
        cute_tensor = cute_tensor.mark_layout_dynamic(leading_dim=1)

        return cute_tensor

    def _get_or_compile_kernel(self, a_cute, b_cute, c_cute, operation_name: str):
        """
        Get a compiled kernel from cache or compile a new one.

        Args:
            a_cute: Input tensor A in CUTE format
            b_cute: Input tensor B in CUTE format
            c_cute: Output tensor C in CUTE format
            operation_name: Name of the operation for caching

        Returns:
            Compiled CUTE kernel
        """
        cache_key = f"{operation_name}_{a_cute.shape}_{b_cute.shape}_{c_cute.shape}"

        if cache_key not in self._compiled_kernels:
            try:
                self._compiled_kernels[cache_key] = cute.compile(
                    self.gemm_kernel, a_cute, b_cute, c_cute, self.stream
                )
                if self.debug_mode:
                    print(f"✓ Compiled kernel for {operation_name}")
            except Exception as e:
                raise RuntimeError(
                    f"Failed to compile {operation_name} kernel: {e}"
                ) from e

        return self._compiled_kernels[cache_key]

    def _execute_gemm_operation(
        self, input_tensor: torch.Tensor, weight: torch.Tensor, operation_name: str
    ) -> torch.Tensor:
        """
        Execute a single GEMM operation using cute dense kernel.

        Args:
            input_tensor: Input tensor [M, K]
            weight: Weight tensor [N, K]
            operation_name: Name of the operation for debugging

        Returns:
            Output tensor [M, N]
        """
        batch_size, input_dim = input_tensor.shape
        output_dim = weight.shape[0]

        # Create output tensor
        output = torch.zeros(
            (batch_size, output_dim),
            device=input_tensor.device,
            dtype=self.dtype,
            requires_grad=False,
        )

        # Convert tensors to cute format
        try:
            a_cute = self._create_cute_tensor(input_tensor)
            b_cute = self._create_cute_tensor(weight)
            c_cute = self._create_cute_tensor(output)
        except Exception as e:
            raise RuntimeError(
                f"Failed to create CUTE tensors for {operation_name}: {e}"
            ) from e

        # Get or compile kernel
        compiled_kernel = self._get_or_compile_kernel(
            a_cute, b_cute, c_cute, operation_name
        )

        # Execute kernel
        try:
            compiled_kernel(a_cute, b_cute, c_cute, self.stream)
            if self.debug_mode:
                print(f"✓ Executed {operation_name} kernel successfully")
        except Exception as e:
            raise RuntimeError(f"Failed to execute {operation_name} kernel: {e}") from e

        return output.squeeze(-1) if output.dim() > 2 else output

    def _execute_gemm_with_cute_input(
        self,
        input_cute: cute.Tensor,
        weight: torch.Tensor,
        operation_name: str,
        output_shape: tuple,
    ) -> torch.Tensor:
        """
        Execute a GEMM operation with pre-converted cute input tensor.

        Args:
            input_cute: Input cute tensor (already in cute format)
            weight: Weight tensor [N, K]
            operation_name: Name of the operation for debugging
            output_shape: Shape of output tensor (batch_size, output_dim)

        Returns:
            Output tensor [M, N]
        """
        batch_size, output_dim = output_shape

        # Create output tensor
        output = torch.zeros(
            (batch_size, output_dim),
            device=weight.device,
            dtype=self.dtype,
            requires_grad=False,
        )

        # Convert weight and output tensors to CUTE format
        try:
            b_cute = self._create_cute_tensor(weight)
            c_cute = self._create_cute_tensor(output)
        except Exception as e:
            raise RuntimeError(
                f"Failed to create CUTE tensors for {operation_name}: {e}"
            ) from e

        # Get or compile kernel
        compiled_kernel = self._get_or_compile_kernel(
            input_cute, b_cute, c_cute, operation_name
        )

        # Execute kernel
        try:
            compiled_kernel(input_cute, b_cute, c_cute, self.stream)
            if self.debug_mode:
                print(f"✓ Executed {operation_name} kernel successfully")
        except Exception as e:
            raise RuntimeError(f"Failed to execute {operation_name} kernel: {e}") from e

        return output.squeeze(-1) if output.dim() > 2 else output

    def _process_expert(
        self,
        expert_tokens: torch.Tensor,
        expert_idx: int,
        w_gate: torch.Tensor,
        w_up: torch.Tensor,
        w_down: torch.Tensor,
    ) -> torch.Tensor:
        """
        Process tokens through a single expert using cute dense kernels.

        Args:
            expert_tokens: Tokens for this expert [num_tokens, hidden_size]
            expert_idx: Index of the expert
            w_gate: Gate projection weights [intermediate_size, hidden_size]
            w_up: Up projection weights [intermediate_size, hidden_size]
            w_down: Down projection weights [hidden_size, intermediate_size]

        Returns:
            Expert output [num_tokens, hidden_size]
        """
        num_tokens = expert_tokens.shape[0]
        intermediate_size = w_gate.shape[0]
        hidden_size = w_down.shape[0]

        # Convert expert_tokens to CUTE format once for reuse
        # OPTIMIZATION: Gate and up projections share the same input tensor,
        # so we convert to CUTE format once and reuse to avoid redundant overhead
        try:
            expert_tokens_cute = self._create_cute_tensor(expert_tokens)
        except BaseException as e:
            raise RuntimeError(
                f"Failed to create CUTE tensor for expert {expert_idx} input: {e}"
            ) from e

        # Gate projection - reuse the CUTE input tensor
        gate_out = self._execute_gemm_with_cute_input(
            expert_tokens_cute,
            w_gate,
            f"gate_expert_{expert_idx}",
            (num_tokens, intermediate_size),
        )

        # Up projection - reuse the same CUTE input tensor
        up_out = self._execute_gemm_with_cute_input(
            expert_tokens_cute,
            w_up,
            f"up_expert_{expert_idx}",
            (num_tokens, intermediate_size),
        )

        # Apply activation and combine
        hidden = self.activation_function(gate_out) * up_out

        # Down projection - create new CUTE tensor for hidden state
        expert_output = self._execute_gemm_operation(
            hidden, w_down, f"down_expert_{expert_idx}"
        )

        return expert_output

    def execute(self, contig_tokens, m_sizes, m_offsets, module):
        """
        Execute the complete grouped GEMM operation via looping.

        Args:
            contig_tokens: Input tokens arranged contiguously by expert
            m_sizes: Sizes of each group
            m_offsets: Offsets of each group
            module: MoE module containing weights and parameters

        Returns:
            Processed tokens
        """
        try:
            # Get weights
            device = contig_tokens.device
            w_gate = module.get_parameter("gate_proj_weight")
            w_up = module.get_parameter("up_proj_weight")
            w_down = module.get_parameter("down_proj_weight")

            # Validate inputs
            if len(m_sizes) != w_gate.shape[0]:
                raise ValueError(
                    f"Number of experts mismatch: {len(m_sizes)} vs {w_gate.shape[0]}"
                )

            # Prepare output tensor
            hidden_size = w_gate.shape[2] if len(w_gate.shape) > 2 else w_gate.shape[1]
            output = torch.zeros(
                contig_tokens.shape[0],
                hidden_size,
                dtype=contig_tokens.dtype,
                device=device,
            )

            # Process each expert
            offset = 0
            active_experts = 0

            for expert_idx, size in enumerate(m_sizes):
                if size > 0:
                    # Get tokens and weights for this expert
                    expert_tokens = contig_tokens[offset : offset + size]
                    expert_gate_weight = w_gate[expert_idx]
                    expert_up_weight = w_up[expert_idx]
                    expert_down_weight = w_down[expert_idx]

                    # Process through expert
                    expert_output = self._process_expert(
                        expert_tokens,
                        expert_idx,
                        expert_gate_weight,
                        expert_up_weight,
                        expert_down_weight,
                    )

                    # Store results
                    output[offset : offset + size] = expert_output
                    active_experts += 1

                offset += size

            if self.debug_mode:
                print(
                    f"Processed {active_experts} active experts out of {len(m_sizes)} total"
                )

            return output

        except Exception as e:
            # Fallback to PyTorch implementation on error
            if self.debug_mode:
                print(f"CUTE kernel failed, falling back to PyTorch: {e}")
            return self._fallback_pytorch(contig_tokens, m_sizes, module)

    def _fallback_pytorch(self, contig_tokens, m_sizes, module):
        """
        Fallback implementation using standard PyTorch operations.

        Args:
            contig_tokens: Input tokens
            m_sizes: Group sizes
            module: MoE module

        Returns:
            Processed tokens using PyTorch mm
        """
        print("\nWARNING:  Cute GEMM issue -- Falling back to PyTorch implementation\n")
        device = contig_tokens.device
        w_gate = module.get_parameter("gate_proj_weight")
        w_up = module.get_parameter("up_proj_weight")
        w_down = module.get_parameter("down_proj_weight")

        hidden_size = w_gate.shape[2] if len(w_gate.shape) > 2 else w_gate.shape[1]
        output = torch.zeros(
            contig_tokens.shape[0],
            hidden_size,
            dtype=contig_tokens.dtype,
            device=device,
        )

        offset = 0
        for expert_idx, size in enumerate(m_sizes):
            if size > 0:
                expert_tokens = contig_tokens[offset : offset + size]

                # Standard PyTorch forward pass
                gate_out = torch.mm(expert_tokens, w_gate[expert_idx].t())
                up_out = torch.mm(expert_tokens, w_up[expert_idx].t())
                hidden = self.activation_function(gate_out) * up_out
                expert_output = torch.mm(hidden, w_down[expert_idx].t())

                output[offset : offset + size] = expert_output

            offset += size

        return output

    def clear_cache(self):
        """Clear the compiled kernel cache to free memory."""
        self._compiled_kernels.clear()
        if self.debug_mode:
            print("Cleared compiled kernel cache")

    def set_debug_mode(self, enabled: bool = False):
        """Enable or disable debug mode."""
        self.debug_mode = enabled

    @staticmethod
    def is_available() -> bool:
        """Check if this strategy is available on the current system."""
        try:
            return CUTLASS_AVAILABLE and torch.cuda.is_available()
        except Exception:
            return False


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
