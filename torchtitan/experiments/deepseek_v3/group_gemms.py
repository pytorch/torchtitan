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
]


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
