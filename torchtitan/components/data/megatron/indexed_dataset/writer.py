# Copyright (c) 2026, trainstation team
# The following code is copied from https://github.com/open-lm-engine/lm-engine

# **************************************************
# Copyright (c) 2026, Mayank Mishra
# **************************************************

from __future__ import annotations

import struct
from types import TracebackType

import numpy as np

from ..dtype import DType

from ..utils import is_multi_storage_client_available, MSC_PREFIX
from .reader import _INDEX_HEADER


if is_multi_storage_client_available():
    import multistorageclient as msc


class _IndexWriter:
    """class to write the index (.idx) file

    Args:
        idx_path (str): The path to the index file

        dtype (type[np.number]): The dtype of the index file
    """

    def __init__(self, idx_path: str, dtype: type[np.number]) -> _IndexWriter:
        self.idx_path = idx_path
        self.dtype = dtype

    def __enter__(self) -> _IndexWriter:
        """Enter the context introduced by the 'with' keyword

        Returns:
            _IndexWriter: The instance
        """
        self.idx_writer = (msc.open if self.idx_path.startswith(MSC_PREFIX) else open)(
            self.idx_path, "wb"
        )
        # fixed, vestigial practice
        self.idx_writer.write(_INDEX_HEADER)
        # fixed, vestigial practice
        self.idx_writer.write(struct.pack("<Q", 1))
        # the numeric code for the dtype
        self.idx_writer.write(struct.pack("<B", DType.code_from_dtype(self.dtype)))
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> bool | None:
        """Exit the context introduced by the 'with' keyword

        Args:
            exc_type (type[BaseException] | None): Exception type

            exc_val (BaseException | None): Exception value

            exc_tb (TracebackType | None): Exception traceback object

        Returns:
            bool | None: Whether to silence the exception
        """
        self.idx_writer.close()

    def write(
        self,
        sequence_lengths: list[int],
        sequence_modes: list[int],
        document_indices: list[int],
    ) -> None:
        """Write the index (.idx) file

        Args:
            sequence_lengths (list[int]): The length of each sequence

            sequence_modes (list[int]): The mode of each sequences

            document_indices (list[int]): The sequence indices demarcating the end of each document
        """
        sequence_pointers = self._sequence_pointers(sequence_lengths)

        # the number of sequences in the dataset
        sequence_count = len(sequence_lengths)
        self.idx_writer.write(struct.pack("<Q", sequence_count))

        # the number of documents in the dataset
        document_count = len(document_indices)
        self.idx_writer.write(struct.pack("<Q", document_count))

        # the number of tokens per sequence
        assert (
            max(sequence_lengths) <= np.iinfo(np.int32).max
        ), "sequence lengths are assumed to be smaller than the max value of np.int32"
        sequence_lengths = np.array(sequence_lengths, dtype=np.int32)
        self.idx_writer.write(sequence_lengths.tobytes(order="C"))
        del sequence_lengths

        # the byte offsets for all sequences
        sequence_pointers = np.array(sequence_pointers, dtype=np.int64)
        self.idx_writer.write(sequence_pointers.tobytes(order="C"))
        del sequence_pointers

        # the sequence indices marking the end of each document
        document_indices = np.array(document_indices, dtype=np.int64)
        self.idx_writer.write(document_indices.tobytes(order="C"))

        # the mode per sequence
        if sequence_modes is not None:
            sequence_modes = np.array(sequence_modes, dtype=np.int8)
            self.idx_writer.write(sequence_modes.tobytes(order="C"))
            del sequence_modes

    def _sequence_pointers(self, sequence_lengths: list[int]) -> list[int]:
        """Build the sequence pointers per the sequence lengths and dtype size

        Args:
            sequence_lengths (list[int]): The length of each sequence

        Returns:
            list[int]: The pointer to the beginning of each sequence
        """
        itemsize = DType.size(self.dtype)
        curr_ptr = 0
        list_ptr = []
        for length in sequence_lengths:
            list_ptr.append(curr_ptr)
            curr_ptr += (
                length if isinstance(length, int) else length.item()
            ) * itemsize
        return list_ptr
