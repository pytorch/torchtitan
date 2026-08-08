# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import bisect
import json
import os
import random
import threading
from collections import OrderedDict, deque
from collections.abc import Generator, Iterator
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import torch
from torch.distributed.checkpoint.stateful import Stateful
from torch.utils.data import get_worker_info, IterableDataset

from torchtitan.components.dataloader import ParallelAwareDataloader
from torchtitan.components.tokenizer import BaseTokenizer
from torchtitan.tools.logging import logger

# Tokens are stored as uint32 and read back as int32 (same width, and the
# OLMo3 vocabulary fits well inside the positive int32 range).
_TOKEN_NBYTES = 4

# Reader threads per dataloader worker when `num_threads` is left unset.
# Mirrors OLMo-core's `_IterableDatasetWrapper.__iter__`, which only guesses a
# thread count when multiprocessing is off. Backends with high per-read latency
# need threads on top of worker processes; set `num_threads` explicitly there.
_DEFAULT_NUM_THREADS = 4

# Reads submitted ahead of the consumer, per reader thread, when `read_ahead` is
# left unset. Deeper queues absorb bursts (every rank crosses a step boundary at
# the same instant) but cannot raise sustained throughput past what the storage
# backend serves.
_DEFAULT_READ_AHEAD_PER_THREAD = 4


@dataclass(frozen=True, slots=True)
class _TokenFile:
    path: Path
    num_tokens: int
    start: int


class PreTokenizedTextDataset(IterableDataset, Stateful):
    """Iterate fixed-length sequences out of pre-tokenized uint32 token files.

    The data order follows OLMo-core's ``NumpyFSLDataLoader``:

    * Instances (``seq_len``-token sequences) are shuffled individually. With
      ``chunk_size > 1``, runs of that many consecutive instances stay together
      in the global order, matching OLMo-core's ``chunk_size`` knob, which
      exists for long-context recipes rather than for IO.
    * The global order is cut into rows of ``dp_world_size * local_batch_size``
      instances. Dataloader worker ``w`` takes rows ``w::num_workers`` and rank
      ``r`` takes columns ``r::dp_world_size`` of each row -- the same striding
      as OLMo-core's ``_get_local_instance_indices``. The per-rank, per-step
      instance set is therefore identical to OLMo-core's.
    * The tail that does not fill a whole row is dropped, as in OLMo-core's
      ``total_size``.

    Two implementation details differ from OLMo-core, neither of which changes
    what a rank sees:

    * The permutation is a stateless Feistel network evaluated per position
      instead of a materialized ``global_indices.npy`` in a shared work_dir. It
      is the same kind of object (a uniform permutation of the instance space,
      redrawn per epoch) without a 3GB per-epoch index file, a startup barrier,
      or regeneration on epoch rollover.
    * Instances are read with ``pread`` rather than ``open``/``seek``/``read``
      per call, reusing a small descriptor cache.
    """

    def __init__(
        self,
        dataset_path: str,
        seq_len: int,
        local_batch_size: int,
        dp_rank: int = 0,
        dp_world_size: int = 1,
        infinite: bool = False,
        shuffle: bool = False,
        shuffle_seed: int = 42,
        shuffle_strategy: Literal["block", "global"] = "block",
        shuffle_block_size: int = 1024,
        chunk_size: int = 1,
        num_threads: int | None = None,
        read_ahead: int | None = None,
        max_open_files: int = 256,
    ) -> None:
        self.dataset_path = Path(dataset_path)
        self.seq_len = seq_len
        self.local_batch_size = local_batch_size
        self.dp_rank = dp_rank
        self.dp_world_size = dp_world_size
        self.infinite = infinite
        self.shuffle = shuffle
        self.shuffle_seed = shuffle_seed
        self.shuffle_strategy = shuffle_strategy
        self.shuffle_block_size = shuffle_block_size
        self.chunk_size = chunk_size
        self.num_threads = num_threads
        self.read_ahead = read_ahead
        self.max_open_files = max_open_files

        self._epoch = 0
        self._stream_row = 0
        self._row_offset = 0

        if self.local_batch_size < 1:
            raise ValueError("local_batch_size must be at least 1")
        if self.shuffle_block_size < 1:
            raise ValueError("shuffle_block_size must be at least 1 shuffle unit")
        if self.shuffle_strategy not in ("block", "global"):
            raise ValueError(
                "shuffle_strategy must be either 'block' or 'global', got "
                f"{self.shuffle_strategy!r}"
            )
        if self.chunk_size < 1:
            raise ValueError("chunk_size must be at least 1 instance")
        if self.num_threads is not None and self.num_threads < 0:
            raise ValueError("num_threads must be non-negative")
        if self.read_ahead is not None and self.read_ahead < 1:
            raise ValueError("read_ahead must be at least 1")
        if self.max_open_files < 1:
            raise ValueError("max_open_files must be at least 1")

        metadata_paths = self._discover_metadata_paths(self.dataset_path)
        self._token_files, self.num_tokens = self._load_metadata(metadata_paths)
        self._file_starts = [token_file.start for token_file in self._token_files]

        if (
            self.shuffle
            and self.shuffle_strategy == "global"
            and self.max_open_files < len(self._token_files)
        ):
            logger.warning(
                f"max_open_files={self.max_open_files} is below the "
                f"{len(self._token_files)} token files at {self.dataset_path}. "
                "A global shuffle reaches them in random order, so the fd cache "
                "misses on most reads and every miss pays an open on the storage "
                "backend. Raise max_open_files to at least the file count."
            )

        self._fd_lock = threading.Lock()
        self._open_fds: OrderedDict[int, int] = OrderedDict()
        self._fd_inflight: dict[int, int] = {}

        self.num_sequences = max(0, (self.num_tokens - 1) // self.seq_len)

        # The shuffle unit is a run of `chunk_size` consecutive instances; a
        # trailing partial unit (or, for 'block', a trailing partial block)
        # cannot be permuted without moving indices out of range.
        num_units = self.num_sequences // self.chunk_size
        if self.shuffle and self.shuffle_strategy == "block":
            num_units = (num_units // self.shuffle_block_size) * self.shuffle_block_size
        self._permutation_domain = num_units

        # Drop the tail that does not fill a whole row, like OLMo-core's
        # `total_size`. A row is one microbatch for every rank.
        self._row_size = self.dp_world_size * self.local_batch_size
        self._num_rows = (num_units * self.chunk_size) // self._row_size
        if self._num_rows == 0:
            raise ValueError(
                f"Pre-tokenized dataset at {self.dataset_path} has {self.num_tokens} "
                f"tokens, which yields {self.num_sequences} sequences at "
                f"seq_len={self.seq_len}. At least {self._row_size} are required "
                f"for dp_world_size={self.dp_world_size} and "
                f"local_batch_size={self.local_batch_size}."
            )

        self._block_order_epoch: int | None = None
        self._block_order: list[int] = []
        self._positions = torch.arange(self.seq_len, dtype=torch.long)

        dropped = self.num_sequences - self._num_rows * self._row_size
        logger.info(
            f"Loaded pre-tokenized dataset from {self.dataset_path} with "
            f"{self.num_tokens} tokens, {self.num_sequences} sequences, "
            f"{self._num_rows} rows of {self._row_size} instances, "
            f"{len(self._token_files)} token files, and {len(metadata_paths)} "
            f"metadata files. {dropped} sequences "
            f"({100 * dropped / self.num_sequences:.4f}%) fall in the unaligned "
            f"tail of each epoch"
        )

    @staticmethod
    def _discover_metadata_paths(dataset_path: Path) -> list[Path]:
        if dataset_path.is_file():
            return [dataset_path]

        metadata_path = dataset_path / "metadata.json"
        if metadata_path.is_file():
            return [metadata_path]

        metadata_paths = sorted(dataset_path.glob("*/metadata.json"))
        if not metadata_paths:
            raise ValueError(
                f"No metadata.json files found in pre-tokenized dataset path "
                f"{dataset_path}"
            )
        return metadata_paths

    @staticmethod
    def _resolve_data_path(metadata_path: Path, data_file: str) -> Path:
        data_path = Path(data_file)
        if not data_path.is_absolute():
            data_path = metadata_path.parent / data_path
        return data_path

    @classmethod
    def _load_metadata(cls, metadata_paths: list[Path]) -> tuple[list[_TokenFile], int]:
        token_files: list[_TokenFile] = []
        start = 0

        for metadata_path in metadata_paths:
            with metadata_path.open() as f:
                metadata = json.load(f)

            dtype = metadata.get("dtype", "uint32")
            if dtype != "uint32":
                raise ValueError(
                    f"PreTokenizedTextDataset only supports dtype='uint32', got "
                    f"{dtype!r} in {metadata_path}"
                )

            metadata_num_tokens = int(metadata["num_tokens"])
            metadata_file_tokens = 0
            data_files = metadata.get("data_files")
            if data_files is None:
                data_files = [
                    {
                        "data_file": metadata["data_file"],
                        "num_tokens": metadata_num_tokens,
                    }
                ]

            for data_file in data_files:
                num_tokens = int(data_file["num_tokens"])
                token_files.append(
                    _TokenFile(
                        path=cls._resolve_data_path(
                            metadata_path, data_file["data_file"]
                        ),
                        num_tokens=num_tokens,
                        start=start,
                    )
                )
                start += num_tokens
                metadata_file_tokens += num_tokens

            if metadata_file_tokens != metadata_num_tokens:
                raise ValueError(
                    f"metadata num_tokens={metadata_num_tokens} does not match "
                    f"data_files total={metadata_file_tokens} in {metadata_path}"
                )

        if not token_files:
            raise ValueError("No token files found in pre-tokenized dataset metadata")
        return token_files, start

    # ------------------------------------------------------------------
    # IO
    # ------------------------------------------------------------------

    @contextmanager
    def _borrow_fd(self, file_idx: int) -> Generator[int]:
        """Lend out a cached read-only fd, keeping it alive for the read.

        Reader threads share the fd cache, so eviction has to skip descriptors
        that a concurrent ``pread`` is still using.

        ``os.open`` runs outside ``_fd_lock``. On a network FUSE mount an open
        costs about as much as the read itself, and a global shuffle misses the
        cache on most reads whenever ``max_open_files`` is below the file count,
        so holding the lock across the open would collapse ``num_threads``
        reader threads into a single serial stream. Two threads racing to open
        the same file is harmless: the loser closes its duplicate and uses the
        winner's fd. Which descriptor a read gets does not affect what it reads,
        because ``pread`` takes an explicit offset and never touches the file
        position.
        """
        with self._fd_lock:
            fd = self._open_fds.get(file_idx)
            if fd is not None:
                self._open_fds.move_to_end(file_idx)
                self._fd_inflight[file_idx] = self._fd_inflight.get(file_idx, 0) + 1

        if fd is None:
            opened = os.open(self._token_files[file_idx].path, os.O_RDONLY)
            # Every read is a seq_len-sized pread at a permuted offset, so
            # kernel readahead only inflates what the backend has to fetch.
            # posix_fadvise is Linux-only; elsewhere the reads just keep the
            # default readahead.
            if hasattr(os, "posix_fadvise"):
                os.posix_fadvise(opened, 0, 0, os.POSIX_FADV_RANDOM)
            with self._fd_lock:
                fd = self._open_fds.get(file_idx)
                if fd is None:
                    fd = self._open_fds[file_idx] = opened
                else:
                    self._open_fds.move_to_end(file_idx)
                    os.close(opened)
                self._fd_inflight[file_idx] = self._fd_inflight.get(file_idx, 0) + 1
                # The cache only grows here, so this is the only path that can
                # push it over max_open_files.
                self._evict_fds()

        try:
            yield fd
        finally:
            with self._fd_lock:
                self._fd_inflight[file_idx] -= 1

    def _evict_fds(self) -> None:
        """Close least-recently-used idle fds. Caller must hold ``_fd_lock``."""
        if len(self._open_fds) <= self.max_open_files:
            return
        for idx in list(self._open_fds):
            if len(self._open_fds) <= self.max_open_files:
                return
            if self._fd_inflight.get(idx, 0) > 0:
                continue
            os.close(self._open_fds.pop(idx))
            self._fd_inflight.pop(idx, None)

    def _pread(self, file_idx: int, token_offset: int, num_tokens: int) -> torch.Tensor:
        num_bytes = num_tokens * _TOKEN_NBYTES
        byte_offset = token_offset * _TOKEN_NBYTES
        buffer = bytearray(num_bytes)
        view = memoryview(buffer)
        try:
            with self._borrow_fd(file_idx) as fd:
                read = 0
                while read < num_bytes:
                    data = os.pread(fd, num_bytes - read, byte_offset + read)
                    if not data:
                        raise ValueError(
                            f"Unexpected EOF reading {num_bytes} bytes at offset "
                            f"{byte_offset} from {self._token_files[file_idx].path}"
                        )
                    view[read : read + len(data)] = data
                    read += len(data)
        finally:
            view.release()
        return torch.frombuffer(buffer, dtype=torch.int32)

    def _read_tokens(self, start: int, end: int) -> torch.Tensor:
        file_idx = bisect.bisect_right(self._file_starts, start) - 1
        pieces: list[torch.Tensor] = []

        while start < end:
            token_file = self._token_files[file_idx]
            file_end = token_file.start + token_file.num_tokens
            take_end = min(end, file_end)
            pieces.append(
                self._pread(file_idx, start - token_file.start, take_end - start)
            )
            start = take_end
            file_idx += 1

        if len(pieces) == 1:
            return pieces[0]
        return torch.cat(pieces)

    def _read_instance(self, instance_idx: int) -> torch.Tensor:
        start = instance_idx * self.seq_len
        # One extra token so the last position has its label.
        return self._read_tokens(start, start + self.seq_len + 1)

    # ------------------------------------------------------------------
    # Ordering
    # ------------------------------------------------------------------

    @staticmethod
    def _mix64(value: int) -> int:
        mask = (1 << 64) - 1
        value = (value + 0x9E3779B97F4A7C15) & mask
        value = ((value ^ (value >> 30)) * 0xBF58476D1CE4E5B9) & mask
        value = ((value ^ (value >> 27)) * 0x94D049BB133111EB) & mask
        return value ^ (value >> 31)

    def _permute_unit(self, index: int, epoch: int) -> int:
        """Map a position through a stateless permutation of the shuffle units."""
        num_units = self._permutation_domain
        num_bits = max(2, (num_units - 1).bit_length())
        if num_bits % 2:
            num_bits += 1
        half_bits = num_bits // 2
        half_mask = (1 << half_bits) - 1
        epoch_seed = self._mix64(self.shuffle_seed + epoch)

        value = index
        while True:
            left, right = value >> half_bits, value & half_mask
            for round_idx in range(4):
                round_key = self._mix64(epoch_seed + round_idx)
                round_value = self._mix64(right ^ round_key) & half_mask
                left, right = right, left ^ round_value
            value = (left << half_bits) | right
            if value < num_units:
                return value

    def _get_block_order(self, epoch: int) -> list[int]:
        if self._block_order_epoch != epoch:
            num_blocks = self._permutation_domain // self.shuffle_block_size
            block_order = list(range(num_blocks))
            random.Random(self.shuffle_seed + epoch).shuffle(block_order)
            self._block_order = block_order
            self._block_order_epoch = epoch
        return self._block_order

    def _shuffled_unit(self, unit: int, epoch: int) -> int:
        if self.shuffle_strategy == "global":
            return self._permute_unit(unit, epoch)
        block_order = self._get_block_order(epoch)
        block_id = block_order[unit // self.shuffle_block_size]
        return block_id * self.shuffle_block_size + unit % self.shuffle_block_size

    def _global_index_at(self, position: int, epoch: int) -> int:
        """Map a position in the epoch's global order to an instance index.

        Mirrors OLMo-core's ``_build_global_indices``: with ``chunk_size == 1``
        this is a plain permutation of the instance space, and with
        ``chunk_size > 1`` the permutation applies to runs of that many
        consecutive instances.
        """
        if not self.shuffle:
            return position
        if self.chunk_size == 1:
            return self._shuffled_unit(position, epoch)
        unit, offset = divmod(position, self.chunk_size)
        return self._shuffled_unit(unit, epoch) * self.chunk_size + offset

    def _iter_stream_instances(
        self, worker_id: int, num_workers: int
    ) -> Iterator[tuple[int, int, int, int]]:
        """Yield ``(epoch, stream_row, offset, instance_idx)`` for this stream.

        Rows of ``dp_world_size * local_batch_size`` instances are dealt to
        dataloader workers, and each rank takes a ``dp_world_size``-strided
        column slice of its rows. All ranks therefore consume the same rows in
        the same order and stay in lockstep.
        """
        epoch = self._epoch
        stream_row = self._stream_row
        offset = self._row_offset

        while True:
            row = stream_row * num_workers + worker_id
            if row < self._num_rows:
                base = row * self._row_size + self.dp_rank
                while offset < self.local_batch_size:
                    position = base + offset * self.dp_world_size
                    yield (
                        epoch,
                        stream_row,
                        offset,
                        self._global_index_at(position, epoch),
                    )
                    offset += 1
                stream_row += 1
                offset = 0
                continue

            if not self.infinite:
                logger.warning(
                    f"Pre-tokenized dataset {self.dataset_path} has run out of data"
                )
                return

            epoch += 1
            stream_row = 0
            offset = 0
            logger.warning(
                f"Pre-tokenized dataset {self.dataset_path} is being re-looped "
                f"(epoch {epoch})"
            )

    def _iter_prefetched(
        self, worker_id: int, num_workers: int, num_threads: int, read_ahead: int
    ) -> Iterator[tuple[int, int, int, torch.Tensor]]:
        """Yield instances in order, reading ahead on ``num_threads`` threads.

        ``num_threads`` bounds how many reads are in flight at once;
        ``read_ahead`` bounds how many are submitted ahead of the consumer. The
        two are separate because the useful depth depends on read latency and
        on how long the consumer stalls, not on the thread count.
        """
        stream = self._iter_stream_instances(worker_id, num_workers)

        if num_threads == 0:
            for epoch, stream_row, offset, instance_idx in stream:
                yield epoch, stream_row, offset, self._read_instance(instance_idx)
            return

        depth = read_ahead
        pending: deque[tuple[int, int, int, Future[torch.Tensor]]] = deque()
        pool = ThreadPoolExecutor(num_threads, thread_name_prefix="pretokenized-read")
        try:
            for epoch, stream_row, offset, instance_idx in stream:
                pending.append(
                    (
                        epoch,
                        stream_row,
                        offset,
                        pool.submit(self._read_instance, instance_idx),
                    )
                )
                if len(pending) <= depth:
                    continue
                ready = pending.popleft()
                yield ready[0], ready[1], ready[2], ready[3].result()
            while pending:
                ready = pending.popleft()
                yield ready[0], ready[1], ready[2], ready[3].result()
        finally:
            for _, _, _, future in pending:
                future.cancel()
            pool.shutdown(wait=False, cancel_futures=True)

    # ------------------------------------------------------------------
    # Iteration
    # ------------------------------------------------------------------

    def _make_sample(
        self, tokens: torch.Tensor
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
        window = tokens.long()
        return {"input": window[:-1], "positions": self._positions}, window[1:]

    def __iter__(self):
        worker_info = get_worker_info()
        worker_id = 0 if worker_info is None else worker_info.id
        num_workers = 1 if worker_info is None else worker_info.num_workers

        num_threads = self.num_threads
        if num_threads is None:
            num_threads = _DEFAULT_NUM_THREADS if worker_info is None else 0

        read_ahead = self.read_ahead
        if read_ahead is None:
            read_ahead = _DEFAULT_READ_AHEAD_PER_THREAD * num_threads

        for epoch, stream_row, offset, tokens in self._iter_prefetched(
            worker_id, num_workers, num_threads, read_ahead
        ):
            sample = self._make_sample(tokens)
            self._epoch = epoch
            if offset + 1 < self.local_batch_size:
                self._stream_row, self._row_offset = stream_row, offset + 1
            else:
                self._stream_row, self._row_offset = stream_row + 1, 0
            yield sample

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        if "sequence_idx" in state_dict or "stream_position" in state_dict:
            raise ValueError(
                "This checkpoint was written by an earlier version of "
                "PreTokenizedTextDataset whose data order cannot be reproduced "
                "by the current implementation. Resume from a newer checkpoint "
                "or restart the run."
            )
        self._epoch = state_dict["epoch"]
        self._stream_row = state_dict["stream_row"]
        self._row_offset = state_dict["row_offset"]

    def state_dict(self) -> dict[str, Any]:
        return {
            "epoch": self._epoch,
            "stream_row": self._stream_row,
            "row_offset": self._row_offset,
        }

    # `os` may already be torn down when __del__ runs at interpreter exit, so
    # bind os.close at definition time.
    def __del__(self, _close=os.close) -> None:
        for fd in getattr(self, "_open_fds", {}).values():
            _close(fd)


class PreTokenizedTextDataLoader(ParallelAwareDataloader):
    @dataclass(kw_only=True, slots=True)
    class Config(ParallelAwareDataloader.Config):
        infinite: bool = True
        """Whether to loop the dataset infinitely."""

        shuffle: bool = False
        """Shuffle instances at each epoch."""

        shuffle_seed: int = 42
        """Base seed used to deterministically shuffle instances."""

        shuffle_strategy: Literal["block", "global"] = "block"
        """Shuffle blocks of units or use a global, no-replacement permutation."""

        shuffle_block_size: int = 1024
        """Number of shuffle units per block, for shuffle_strategy='block'."""

        chunk_size: int = 1
        """
        Number of consecutive instances kept together in the global shuffle
        order, matching OLMo-core's `NumpyFSLDataLoader.chunk_size`. Only useful
        for long-context recipes that concatenate adjacent instances; leave at 1
        for pretraining.
        """

        num_threads: int | None = None
        """
        Reader threads per dataloader worker. Defaults to 4 when num_workers is
        0 and to 0 otherwise, matching OLMo-core. Set it explicitly on
        high-latency storage, where worker processes alone do not supply enough
        concurrency to hide read latency.
        """

        read_ahead: int | None = None
        """
        Reads submitted ahead of the consumer per dataloader worker. Defaults to
        4x num_threads. Raising it absorbs bursts -- all ranks cross a step
        boundary together -- but it cannot raise sustained throughput past what
        the storage backend serves, and the worker's own output queue
        (prefetch_factor x local_batch_size instances) caps how far ahead the
        pipeline can actually run.
        """

        max_open_files: int = 256
        """Maximum number of token file descriptors kept open at once."""

    def __init__(
        self,
        config: Config,
        *,
        dp_world_size: int,
        dp_rank: int,
        tokenizer: BaseTokenizer,
        seq_len: int,
        local_batch_size: int,
        snapshot_every_n_steps: int | None = 1,
        **kwargs,
    ):
        del tokenizer, kwargs

        if config.dataset_path is None:
            raise ValueError(
                "PreTokenizedTextDataLoader requires config.dataset_path to point "
                "to a pre-tokenized dataset directory or metadata.json file"
            )

        ds = PreTokenizedTextDataset(
            dataset_path=config.dataset_path,
            seq_len=seq_len,
            local_batch_size=local_batch_size,
            dp_rank=dp_rank,
            dp_world_size=dp_world_size,
            infinite=config.infinite,
            shuffle=config.shuffle,
            shuffle_seed=config.shuffle_seed,
            shuffle_strategy=config.shuffle_strategy,
            shuffle_block_size=config.shuffle_block_size,
            chunk_size=config.chunk_size,
            num_threads=config.num_threads,
            read_ahead=config.read_ahead,
            max_open_files=config.max_open_files,
        )

        dataloader_kwargs = {
            "num_workers": config.num_workers,
            "persistent_workers": config.persistent_workers,
            "pin_memory": config.pin_memory,
            "prefetch_factor": config.prefetch_factor,
            "snapshot_every_n_steps": snapshot_every_n_steps,
            "batch_size": local_batch_size,
        }

        super().__init__(
            ds,
            dp_rank=dp_rank,
            dp_world_size=dp_world_size,
            **dataloader_kwargs,
        )
