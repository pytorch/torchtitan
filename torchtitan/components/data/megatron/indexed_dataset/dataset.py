# Copyright (c) 2026, trainstation team
# The following code is copied from https://github.com/open-lm-engine/lm-engine

# **************************************************
# Copyright (c) 2026, Mayank Mishra
# **************************************************

from __future__ import annotations

import os
from itertools import accumulate

import numpy as np
import torch

from ..bin import _MMapBinReader, _MultiStorageClientBinReader, get_bin_path
from ..dtype import DType

from ..utils import MSC_PREFIX
from .reader import _IndexReader
from .utils import get_idx_path


class MMapIndexedDataset(torch.utils.data.Dataset):
    """The low-level interface dataset class

    Args:
        path_prefix (str): The index (.idx) and data (.bin) prefix

        multimodal (bool, optional): Whether the dataset is multimodal. Defaults to False.
    """

    def __init__(
        self, path_prefix: str, multimodal: bool = False, idx_path: str | None = None
    ) -> MMapIndexedDataset:
        super().__init__()
        self.initialize(path_prefix, multimodal, idx_path)

    def initialize(
        self, path_prefix: str, multimodal: bool, idx_path: str | None
    ) -> None:
        is_object_storage = path_prefix.startswith(MSC_PREFIX)

        self.path_prefix = path_prefix
        self.multimodal = multimodal
        self.idx_path = idx_path

        self.index = _IndexReader(
            get_idx_path(path_prefix) if idx_path is None else idx_path, self.multimodal
        )
        self.bin_reader = (
            _MultiStorageClientBinReader if is_object_storage else _MMapBinReader
        )(get_bin_path(self.path_prefix))

    def __getstate__(self) -> tuple[str, bool]:
        """Get the state during pickling

        Returns:
            tuple[str, bool]: The state tuple
        """
        return self.path_prefix, self.multimodal, self.idx_path

    def __setstate__(self, state: tuple[str, bool]) -> None:
        """Set the state during un-pickling

        Args:
            state (tuple[str, bool]): The state tuple
        """
        path_prefix, multimodal, idx_path = state
        self.initialize(path_prefix, multimodal, idx_path)

    def __len__(self) -> int:
        """Return the length of the dataset i.e. the number of sequences in the index

        Returns:
            int: The length of the dataset
        """
        return len(self.index)

    def __getitem__(
        self, idx: int | np.integer | slice
    ) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
        """Return from the dataset

        Args:
            idx (int | np.integer | slice): The index or index slice into the dataset

        Raises:
            ValueError: When the index slice is non-contiguous
            TypeError: When the index is of an unexpected type

        Returns:
            np.ndarray | tuple[np.ndarray, np.ndarray]: The sequence tokens and
            modes at the index or index slice
        """
        if isinstance(idx, (int, np.integer)):
            sequence_pointer, sequence_length, sequence_mode = self.index[idx]
            sequence = self.bin_reader.read(
                dtype=self.index.dtype, count=sequence_length, offset=sequence_pointer
            )
            return (sequence, sequence_mode) if sequence_mode is not None else sequence
        elif isinstance(idx, slice):
            start, stop, step = idx.indices(len(self))
            if step != 1:
                raise ValueError("Slices into indexed_dataset must be contiguous")
            sequence_lengths = self.index.sequence_lengths[idx]
            sequence_modes = self.index.sequence_modes[idx] if self.multimodal else None
            sequence_offsets = list(accumulate(sequence_lengths))
            sequences = np.split(
                self.bin_reader.read(
                    dtype=self.index.dtype,
                    count=sum(sequence_lengths),
                    offset=self.index.sequence_pointers[start],
                ),
                sequence_offsets[:-1],
            )
            return (
                (sequences, sequence_modes) if sequence_modes is not None else sequences
            )
        else:
            raise TypeError("Unexpected type received for idx: {}".format(type(idx)))

    def get(self, idx: int, offset: int = 0, length: int | None = None) -> np.ndarray:
        """Retrieve a single item from the dataset with the option to only
        return a portion of the item.

        get(idx) is the same as [idx] but get() does not support slicing.
        """

        offset = int(offset)

        sequence_pointer, sequence_length, sequence_mode = self.index[idx]
        if length is None:
            length = sequence_length - offset
        sequence_pointer += offset * DType.size(self.index.dtype)
        sequence = self.bin_reader.read(
            dtype=self.index.dtype, count=length, offset=sequence_pointer
        )
        return (sequence, sequence_mode) if sequence_mode is not None else sequence

    @property
    def sequence_lengths(self) -> np.ndarray:
        """Get the sequence lengths

        Returns:
            np.ndarray: The sequence lengths
        """
        return self.index.sequence_lengths

    @property
    def document_indices(self) -> np.ndarray:
        """Get the document indices

        Returns:
            np.ndarray: The document indices
        """
        return self.index.document_indices

    @property
    def sequence_modes(self) -> np.ndarray:
        """Get the sequence modes

        Returns:
            np.ndarray: The sequence modes
        """
        return self.index.sequence_modes

    @staticmethod
    def exists(path_prefix: str) -> bool:
        """Return whether the MMapIndexedDataset exists on disk at the prefix

        Args:
            path_prefix (str): The prefix to the index (.idx) and data (.bin) files

        Returns:
            bool: Whether the MMapIndexedDataset exists on disk at the prefix
        """
        return os.path.exists(get_idx_path(path_prefix)) and os.path.exists(
            get_bin_path(path_prefix)
        )
