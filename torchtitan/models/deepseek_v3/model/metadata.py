"""
Unified metadata for DeepSeek V3 checkpoint loading.

This module provides a unified metadata class that inherits from DCP's Metadata
while maintaining both HuggingFace and TorchTitan metadata formats internally.
This allows seamless integration with DCP's API while providing access to both formats.
"""

from torch.distributed.checkpoint.metadata import Metadata


class DeepSeekV3Metadata(Metadata):
    """
    Unified metadata class that inherits from DCP's Metadata.

    This class presents state dict metadata to DCP while internally
    maintaining both IO and SD metadata for expert tensor loading and other operations.
    """

    def __init__(
        self,
        io_metadata: Metadata,
        sd_metadata: Metadata,
    ):
        """
        Initialize unified metadata with both IO and SD metadata.

        Args:
            io_metadata: Original HuggingFace metadata for storage IO
            sd_metadata: State dict formatted metadata for tensor loading
        """
        # Initialize parent with state dict metadata (what DCP will see)
        super().__init__(
            state_dict_metadata=sd_metadata.state_dict_metadata,
            storage_data=sd_metadata.storage_data,
        )

        # Store both metadata formats internally
        self._io_metadata = io_metadata
        self._sd_metadata = sd_metadata

    @property
    def io_metadata(self) -> Metadata:
        """Get the original IO metadata for storage operations."""
        return self._io_metadata

    @property
    def sd_metadata(self) -> Metadata:
        """Get the state dict metadata for planning operations."""
        return self._sd_metadata

