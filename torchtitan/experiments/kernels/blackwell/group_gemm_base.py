import torch


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
