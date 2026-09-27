# Copyright (c) 2026, trainstation team
# The following code is copied from https://github.com/open-lm-engine/lm-engine

# **************************************************
# Copyright (c) 2026, Mayank Mishra
# **************************************************

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from .dtype import DType

from .utils import is_multi_storage_client_available, MSC_PREFIX


if is_multi_storage_client_available():
    import multistorageclient as msc


class _BinReader(ABC):
    """Abstract class to read the data (.bin) file"""

    @abstractmethod
    def read(self, dtype: type[np.number], count: int, offset: int) -> np.ndarray:
        ...


class _MMapBinReader(_BinReader):
    """A _BinReader that memory maps the data (.bin) file

    Args:
        bin_path (str): bin_path (str): The path to the data (.bin) file.
    """

    def __init__(self, bin_path: str) -> _MMapBinReader:
        self._bin_file_reader = (msc.open if bin_path.startswith(MSC_PREFIX) else open)(
            bin_path, mode="rb"
        )
        self._bin_buffer_mmap = np.memmap(self._bin_file_reader, mode="r", order="C")
        self._bin_buffer = memoryview(self._bin_buffer_mmap)

    def read(self, dtype: type[np.number], count: int, offset: int) -> np.ndarray:
        """Read bytes into a np array.

        Args:
            dtype (type[np.number]): Data-type of the returned array.
            count (int): Number of items to read.
            offset (int): Start reading from this offset (in bytes).

        Returns:
            np.ndarray: An array with `count` items and data-type `dtype` constructed from
                reading bytes from the data file starting at `offset`.
        """
        return np.frombuffer(self._bin_buffer, dtype=dtype, count=count, offset=offset)

    def __del__(self) -> None:
        """Clean up the object."""
        if self._bin_buffer_mmap is not None:
            self._bin_buffer_mmap._mmap.close()  # type: ignore[attr-defined]
        if self._bin_file_reader is not None:
            self._bin_file_reader.close()
        del self._bin_buffer_mmap
        del self._bin_file_reader


class _MultiStorageClientBinReader(_BinReader):
    """A _BinReader that reads from the data (.bin) file using the multi-storage client.

    Args:
        bin_path (str): bin_path (str): The path to the data (.bin) file.
    """

    def __init__(self, bin_path: str) -> _MultiStorageClientBinReader:
        self._client, self._bin_path = msc.resolve_storage_client(bin_path)

    def read(self, dtype: type[np.number], count: int, offset: int) -> np.ndarray:
        size = count * DType.size(dtype)
        buffer = self._client.read(
            path=self._bin_path, byte_range=msc.types.Range(offset=offset, size=size)
        )
        return np.frombuffer(buffer, dtype=dtype)


def get_bin_path(path_prefix: str) -> str:
    return f"{path_prefix}.bin"
