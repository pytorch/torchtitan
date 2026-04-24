import torch
from torch.utils.data import Dataset
import multiprocessing as mp
import os
import time
import random
import queue
from typing import Any, Union, List, Protocol, MutableMapping

class SizedDataset(Protocol):
    def __len__(self) -> int: ...
    def __getitem__(self, index: int) -> Any: ...

class AsyncPrefetchedDatasetWrapper(Dataset):
    """
    A dataset wrapper that asynchronously prefetches and buffers samples from
    an underlying dataset using background worker processes.

    This class is useful when validation data loading becomes a bottleneck due to
    CPU or disk contention, especially when using multiple GPUs or slow storage.
    It offloads sample preparation into persistent background processes with
    lower priority, maintaining a cyclic buffer of preloaded samples.

    **Notes**
    ---------
    - This dataset uses multiprocessing primitives and shared memory; it is
      **NOT thread-safe**.
    - Do **NOT** use with `DataLoader` workers > 1, as that duplicates the
      dataset instance and buffer.
      Instead, control the number of prefetch workers via `num_prefetch_workers`
      during initialization.
    - The class returns **individual samples only**, not batches; use a
      `DataLoader` with `num_workers=0` or `1` to handle batching.
    - Samples returned early during buffer warm-up or under heavy load **may be
      repeated or stale**.
    - Samples marked as unseen in the buffer are never overwritten; if too many
      unseen samples accumulate, workers pause until more entries are consumed.
    - Background workers sleep for a configurable duration (`default_sleep_time`)
      during backoff, which prevents high CPU usage and allows for gradual 
      recovery under resource pressure.
    - Data augmentation or non-deterministic transformations inside the base 
      dataset may produce unexpected repeated samples due to buffering.

    **Multiprocessing details**
    ---------------------------
    - Uses `multiprocessing.Manager` and shared `Array` / `Value` for 
      inter-process communication.
    - Workers run with optionally lowered OS priority (`os.nice`), which only 
      works on Unix-like systems.
    - Workers back off and sleep if the buffer is too full or cannot be updated
      using specific event handlers.
    - Workers exit cleanly on `shutdown()`, but this should be called explicitly
      to avoid orphan processes.

    See Also
    --------
    torch.utils.data.DataLoader : Combine this dataset with a DataLoader to 
    perform batching.
    Remember to always set `num_workers` to 0 or 1 when using this wrapper.
    """

    base_dataset: SizedDataset
    """The wrapped dataset from which this wrapper requests the actual datapoints
    during prefetching"""
    buffer: Any  # mp.managers.ListProxy
    """Shared buffered list storing prefetched samples from the underlying dataset."""
    unseen_flags: Any #mp.Array('b')
    """Shared boolean array (dtype 'b') indicating whether a buffer slot contains
    an unseen (1) or seen (0) sample."""
    next_fill_idx: Any #mp.Value('i')
    """Shared integer pointer to the next buffer index to fill with a new sample."""
    next_get_idx: Any #mp.Value('i')
    """Shared integer pointer to the next buffer index to retrieve."""
    unseen_count: Any #mp.Value('i')
    """Shared integer tracking how many unseen samples are currently buffered."""
    skipped_idx_count: Any #mp.Value('i')
    """Shared integer tracking how many indices where requested in :py:meth:`_getitem`
    but couldn't be placed into the waiting queue due to the size limit."""
    worker_idle_time: Any #mp.Value('d')
    """Shared double float tracking the datatset workers spend idle (normalised
    by number of workers)"""
    request_queue: mp.Queue
    """Queue where sample indices are pushed for prefetching."""
    stop_event: Any #mp.Event
    """Event used to signal background workers to exit."""
    buffer_element_seen: Any #mp.Event
    """Event used to signal that a sample was consumed from the buffer. Workers 
       wait on this when the buffer is too full to avoid redundant computation."""
    has_data_event: Any #mp.Event
    """Event used to notify that at least one sample has been added to the buffer.
       Prevents getitem from blocking indefinitely during warm-up."""
    prefetch_workers: List[mp.Process]
    """List of launched background prefetching processes."""
    
    def __init__(
        self, 
        base_dataset: SizedDataset,
        buffer_size: int = 1024,
        queue_size: int = 1024,
        priority: int = 10,
        num_prefetch_workers: int = 2,
        wait_till_buffer_has: int = 1,
        buffer_fill_pause_threshold: float = 0.95,
        default_sleep_time: Union[int,float] = 60,
    ):
        """
        Parameters
        ----------
        base_dataset : torch.utils.data.Dataset with __len__ implemented
            The underlying dataset from which samples are drawn.
        buffer_size : int, default=1024
            Maximum number of prefetched samples to keep in memory.
        queue_size : int, default=1024
            Maximum number of sample indices to queue for prefetching.
        priority : int, default=10
            Niceness value for background workers (higher means lower priority).
            Passed to `os.nice()` on Unix-like systems only.
            Typical niceness values range from -20 (highest priority) to 19 
            (lowest priority). Negative values require superuser privileges.
            On Windows or restricted environments, this value is ignored.
        num_prefetch_workers : int > 0, default=2
            Number of background processes used for sample prefetching.
        wait_till_buffer_has : int > 0, default=1
            Number of elements the buffer needs to have loaded before cycling
            through the buffer. Usefull if the dataset is used to to populate a
            batch directly after initialisation. With value 1, it will simply
            populate the first batch with just repetitions of the first element.
            With value 5, it will be a cyclic repetition of at least 5 elements,
            but will pause the process longer in the first step. If e.g. 64 and
            used to populate batches of size e.g. 32, then the first two batches
            will carry completely unique samples without any repetition.
            Will not have an effect later as the buffer size never shrinks again
            and just replaces the oldest element.
        buffer_fill_pause_threshold : float, default=0.95
            Fraction of the buffer that, when filled with unseen samples,
            causes workers to pause until more entries are consumed.
            Must be a float between 0 and 1.
        default_sleep_time : int | float, 60
            Sleep time (in seconds) used by workers when backing off during 
            buffer fill or data wait. Processes wake up earlier when the state of
            the buffer changes accordingly.
        """
        #print(f"AsyncPrefetchedDatasetWrapper on ({len(base_dataset)}) {base_dataset}")
        self.base_dataset = base_dataset
        self._buffer_size = buffer_size
        self._priority = priority
        self._num_prefetch_workers = max(num_prefetch_workers,1)
        self._wait_till_buffer_has = min(max(wait_till_buffer_has,1), buffer_size)
        self._buffer_fill_pause_threshold = buffer_fill_pause_threshold
        self._sleep_time = default_sleep_time

        # Shared buffer (data + state)
        self.buffer = mp.Manager().list()
        self.unseen_flags = mp.Array('b', [0]*buffer_size)  # boolean buffer: 1 = unseen, 0 = seen

        # Cyclic buffer index for replacement and providing
        self.next_fill_idx = mp.Value('i', 0)
        self.next_get_idx = mp.Value('i', 0)
        
        # Number of prepared samples that have not yet been provided
        self.unseen_count = mp.Value('i', 0)
        # Number of sample indices that have been requested but will not be able
        # to be returned this epoch
        self.skipped_idx_count = mp.Value('i', 0)
        # time the workers were idle
        self.worker_idle_time = mp.Value('d', 0.0)

        # Prefetch request queue (non-blocking on push)
        self._queue_size = queue_size
        self.request_queue = mp.Queue(maxsize=queue_size)

        # Stop signal for the workers
        self.stop_event = mp.Event()
        # Signals workers that getitem looked at a previously unseen data
        self.buffer_element_seen = mp.Event()
        self.buffer_element_seen.clear()  # Initially buffer completely unseen
        # Signals __getitem__ that buffer has data
        self.has_data_event = mp.Event()
        self.has_data_event.clear() # Initially empty buffer

        # Launch background workers
        self.prefetch_workers = []
        for _ in range(num_prefetch_workers):
            p = mp.Process(target=self._prefetch_worker, daemon=True)
            p.start()
            self.prefetch_workers.append(p)

    def _prefetch_worker(self):
        """
        Background worker process function.

        Continuously fetches indices from the request queue (or samples randomly
        if queue is empty), retrieves corresponding samples from the base dataset,
        and inserts them into the shared buffer if the buffer slot is free (seen).
        The worker respects the unseen sample fill threshold to avoid wasteful
        computation and lowers its OS priority for minimal interference.
        """
        try:
            # Lower process priority
            os.nice(self._priority)
        except Exception:
            pass  # Might not work on Windows or restricted environments

        while not self.stop_event.is_set():
            # If too many unseen samples, back off
            if self.unseen_count.value >= int(self._buffer_fill_pause_threshold * self._buffer_size):
                sleep_start = time.time()
                self.buffer_element_seen.wait(timeout=self._sleep_time)
                sleep_end = time.time()
                with self.worker_idle_time.get_lock():
                    self.worker_idle_time.value += (sleep_end - sleep_start) / self._num_prefetch_workers
                self.buffer_element_seen.clear()
                continue

            # Try to get an index to prefetch
            try:
                index = self.request_queue.get_nowait()
            except Exception as e: #queue.Empty:
                # Queue is empty, just sample a random index
                if not isinstance(e, queue.Empty):
                    print(f"Unexpected exception when getting the next sample index: {e}")
                index = random.randint(0, len(self.base_dataset) - 1)

            sample = self.base_dataset[index]

            while not self.stop_event.is_set():
                # wait until the sample can be added to the buffer without 
                # wasting computational ressources. Free resources otherwise
                with self.next_fill_idx.get_lock(), self.unseen_count.get_lock():
                    i = self.next_fill_idx.value
                    if not self.unseen_flags[i]:
                        if len(self.buffer) < self._buffer_size:
                            self.buffer.append(sample)
                        else:
                            self.buffer[i] = sample
                        self.has_data_event.set()
                        self.unseen_flags[i] = 1     # mark as unseen
                        self.unseen_count.value += 1
                        self.next_fill_idx.value = (i + 1) % self._buffer_size
                        break
                sleep_start = time.time()
                self.buffer_element_seen.wait(timeout=self._sleep_time)
                sleep_end = time.time()
                with self.worker_idle_time.get_lock():
                    self.worker_idle_time.value += (sleep_end - sleep_start) / self._num_prefetch_workers
                self.buffer_element_seen.clear()

            # Alternative code for extracting idle time for when the process is 
            # interrupted at a different place. However has a problem when the
            # .get_item() function uses multiple CPUs as process_time sums these times
            #with self.worker_idle_time.get_lock():
            #    wall_start = time.time()           # line at start of while
            #    cpu_start = time.process_time()    # line at start of while
            #    wall_time = time.time() - wall_start
            #    cpu_time = time.process_time() - cpu_start
            #    self.worker_idle_time.value += max(0.0, wall_time - cpu_time) / self._num_prefetch_workers


    def __getitem__(self, idx):
        """
        Retrieve a sample from the buffer asynchronously.

        Requests asynchronous prefetch of the sample at `idx` by putting
        `idx` into the request queue (non-blocking).

        If the buffer has samples ready, returns the next buffered sample in a
        cyclic manner and marks it as seen.

        If the buffer is still warming up (empty), blocks briefly until a sample
        is available, then returns it.

        **Important:**
        - Samples returned early during buffer warm-up or under high load may be repeated.
        - The sample returned may not correspond to the requested index `idx` immediately,
        but the requested index is queued for prefetching.

        Parameters
        ----------
        idx : int
            Index of the sample to request prefetching for.

        Returns
        -------
        sample : any
            Sample retrieved from the buffer (original type from base_dataset).
            if the sample is a dictionary, it will add the key, value pairs:
            - `"prefetch_unseen"`, integer, one if the sample has never before been returned, zero otherwise
            - `"prefetch_unseen_count"`, integer, how many unseen samples are in the buffer
            - `"prefetch_queue"`, integer, the number of indices currently in the queue
            - `"prefetch_skipped"`, integer, the number of indices that couldn't be buffered in the queue
            - `"prefetch_idle_time"` float, the time in seconds that the dataset was idle
            values will be returned as floats for easy use in mean operations
        """
        # Request prefetch of this index (non-blocking put)
        try:
            self.request_queue.put_nowait(idx)
        except Exception as e: #queue.Full:
            if not isinstance(e, queue.Full):
                print(f"Unexpected exception while pushing index {idx}: {e}")
            # Silently drop if queue is full
            with self.skipped_idx_count.get_lock():
                self.skipped_idx_count.value += 1

        while not self.stop_event.is_set():
            with self.next_get_idx.get_lock():
                i = self.next_get_idx.value
                if i < len(self.buffer):
                    # Safe to access
                    sample = self.buffer[i]
                    self.next_get_idx.value = (i + 1) % max(len(self.buffer), self._wait_till_buffer_has)
                    # NOTE: We use `len(self.buffer)` instead of `self._buffer_size` 
                    # here on purpose. This allows validation to proceed as soon 
                    # as *any* samples are available, rather than waiting for the 
                    # full buffer to be populated. It's a trade-off: early batches 
                    # may contain repeated samples, but training can continue 
                    # without stalling on slow validation prefetch.
                    with self.unseen_count.get_lock():
                        unseen = self.unseen_flags[i]
                        if unseen:
                            self.unseen_flags[i] = 0
                            self.unseen_count.value -= 1
                            self.buffer_element_seen.set()
                        if isinstance(sample, MutableMapping):
                            sample["prefetch_unseen"]  = float(unseen)
                            sample["prefetch_unseen_count"] = float(self.unseen_count.value)
                            sample["prefetch_queue"]   = float(self.request_queue.qsize())
                            sample["prefetch_skipped"] = float(self.skipped_idx_count.value)
                            sample["prefetch_idle_time"] = float(self.worker_idle_time.value)

                    return sample # Exit while loop
                # Else: i==0 and not a single sample has been gathered yet, will sleep below
            # Buffer empty as still warming up, wait for data event or timeout
            self.has_data_event.wait(timeout=self._sleep_time)


    def __len__(self):
        """
        Returns
        -------
        int
            Length of the underlying base dataset.
        """
        return len(self.base_dataset)

    def shutdown(self):
        """
        Signal all background prefetch worker processes to stop and wait for
        their termination.

        **Important:** This method should be called explicitly before program exit
        or dataset deletion to ensure all worker processes terminate cleanly and
        no orphaned processes remain.

        If not called, the destructor `__del__` will attempt shutdown but
        explicit call is recommended for robust resource management.
        """
        self.stop_event.set()
        for p in self.prefetch_workers:
            if p.is_alive():
                p.terminate()
                p.join()

    def __repr__(self):
        try:
            current_qsize = self.request_queue.qsize()
        except NotImplementedError:
            current_qsize = "N/A"
        return (f"{self.__class__.__name__}("
                f"buffer_filled={len(self.buffer)}/{self._buffer_size}, "
                f"queue_indices={current_qsize}/{self._queue_size}, "
                f"prefetch_workers={len(self.prefetch_workers)}, "
                f"unseen_count={self.unseen_count.value})")

    def __del__(self):
        """
        Destructor to ensure proper shutdown of background worker processes.
        """
        self.shutdown()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.shutdown()