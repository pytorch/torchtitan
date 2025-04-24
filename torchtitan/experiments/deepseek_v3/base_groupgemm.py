import torch


# Strategy pattern for GroupGEMM implementations
class GroupGEMMStrategy:
    """Base class for group gemm strategies"""

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
        return True
