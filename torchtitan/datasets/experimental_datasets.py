import csv
import logging
import math
import os
import random
import time
from typing import Any, Callable, List, Optional, Set, Type, Union

import pyarrow as pa
import torch
import torch.utils.data as data

"""
The following distributed dataloaders are designed around 3 main principles:

1. Efficient, asynchronous operation. Workers on different devices do not communicate. 
2. Modularity. Data loading pipeline is composed of wrapped iterators, the base iterator 
    loading from disk and additional layers adding levels of post-processing (shuffling, 
    packing, padding, etc.).
3. Seamless resumption from checkpoint. Each stage of the pipeline maintains an internal 
    state that can be written/read on disk via implemented recursive `state_dict()` and 
    `load_state_dict()` calls.
4. Rescalability. Users can save and load checkpoints to/from different numbers of workers 
    without losing the global state. This is accomplished by splitting state fields for each 
    layer into `state_params`, which are typically scalar-valued and can be discarded when 
    rescaling (i.e. counters, RNG states), and `reshard_params`, which are lists that can be 
    re-distributed over workers (i.e. buffers).

Our loaders obey the following type heirarchy: 
torch.data.IterableDataset -> _Stateful_Dataset -> _Wrapper_Dataset. 
`_Stateful_Dataset` implements state and checkpointing logic. A `_Wrapper_Dataset` holds a 
single `_Stateful_Dataset` and iterates via calling its wrapped dataset any number of times, 
then applying some sort of post-processing and yielding the result. Users build data processing 
pipelines by wrapping a base `_Stateful_Dataset` in any number of `_Wrapper_Dataset` layers, 
which is then passed to the torch DataLoader.

NOTE: `_Wrapper_Dataset` currently only implements wrapping a single instantiated sub-dataset layer. 
Many layers need multiple sub-layers (i.e. random sampling from distinct data sources). These are 
currently implemented as base `_Stateful_Datasets` that take the class of their sub-layers plus any 
pass-through arguments, and instantiate all those sub-layers. This is easy on the user, who no longer 
needs to instantiate large sets of sub-layers in their code, but leads to awkwardness in this file. 
Cleanup is planned for the future. 
"""


# --------------  UTILITIES  --------------


def _get_latest(targdir, qualifier=lambda x: True):
    """Fetch the latest file or folder written to target directory, subject to name passing the qualifier fn.
    If directory is empty or nonexistent or no items qualify, return None."""
    if os.path.exists(targdir) and len(os.listdir(targdir)) > 0:
        latest = max(
            [
                os.path.join(targdir, x)
                for x in os.listdir(targdir)
                if qualifier(os.path.join(targdir, x))
            ],
            key=lambda path: int(path.split("/")[-1].split("-")[1]),
        )
        return os.path.join(targdir, latest)
    return None


def _shard_partition(itemlist: List[Any], rank: int, worldsize: int) -> List[Any]:
    """
    Partition itemlist into worldsize chunks, grab chunk corresponding to rank and return.
    """
    return itemlist[
        (rank * len(itemlist)) // worldsize : ((rank + 1) * len(itemlist)) // worldsize
    ]


def _shard_inclusive(itemlist: List[Any], rank: int, worldsize: int) -> List[Any]:
    """
    In cases where len(itemlist) % worldsize != 0, allow for fractional ownership of items,
    and return the span including all owned items, fractional or otherwise.
    """
    start = math.floor(len(itemlist) * rank / worldsize)
    end = math.ceil(len(itemlist) * (rank + 1) / worldsize)
    return itemlist[start:end]


class _Stateful_Dataset(data.IterableDataset):
    """
    Stub for stateful datasets, extends data.IterableDataset with state_dict methods.
    All subclasses should specify the params to be considered stateful or reshardable in the
    self.state_params and self.reshard_params lists.
    """

    def __init__(
        self,
        rank: int,
        worldsize: int,
    ):
        assert rank >= 0, f"Rank {rank} must be a positive integer"
        assert (
            worldsize > rank
        ), f"Worldsize {worldsize} must be greater than rank {rank}"
        self.state_params: List[str] = []
        self.reshard_params: List[str] = []
        self.rank = rank
        self.worldsize = worldsize
        self.load_worldsize = (
            worldsize  # Enable calling load_state_dict() directly, assume no rescaling
        )

    def statename(self, x: str):
        # Note that this naming convention implicitly disallows repeated layers in the dataset pipeline
        return self.__class__.__name__ + "." + x

    def state_dict(self):
        """
        Retrieve all state and reshard flags (each worker/process saves its own state dict shard)
        """
        return {
            self.statename(flag): getattr(self, flag)
            for flag in self.state_params + self.reshard_params
        }

    def _reshard(self, sharded_list):
        """
        Sharded_list is a list of lists, where each "shard" sublist must have the same length.
        These shards should tightly span only the partition of data owned by this worker.
        (i.e. if global_list is the list of all entries, sharded_list = _shard_inclusive(global_list) ).
        Determine fractional ownership of shards, and get the flattened partition owned by this worker.
        """
        # How many shards did _shard_inclusive() drop to the left of sharded_list?
        shard_offset = math.floor(self.load_worldsize * self.rank / self.worldsize)
        # How long are the list shards?
        shard_len = len(sharded_list[0])
        for i, shard in enumerate(sharded_list):
            assert (
                len(shard) == shard_len
            ), f"Shard {i} with length {len(shard)} does not match expected {shard_len}"
        # How many list items did _shard_inclusive() drop to the left of the flattened sharded_list?
        item_offset = shard_len * shard_offset
        # How many list items are there in total?
        n_items = self.load_worldsize * shard_len
        # The indices of the flattened sharded_list that this worker owns
        my_items = range(
            int(n_items * self.rank / self.worldsize) - item_offset,
            int(n_items * (self.rank + 1) / self.worldsize) - item_offset,
        )
        # Pull out owned items
        return [sharded_list[i // shard_len][i % shard_len] for i in my_items]

    def load_state_dict(self, state_dicts, sharded_input=False):
        """
        Input state_dicts is a list of state_dicts. If sharded_input=False, this is expected to be the
        global list of states across all checkpoint shard files. If sharded_input=True, this expects
        _shard_inclusive(global_state_list). Handling reduced inputs allows for much more efficient loading.
        Workflow:
        1. if sharded_inputs is false, shard the inputs.
        2. If worldsize matches checkpoint, pull state and reshard params from the given checkpoint
            shard (state_dicts is a singleton list).
        3. If worldsize does not match checkpoint, toss state params and assemble reshard params from
            across given state_dicts. In this case state_dicts may be singleton (for fractional ownership)
            or multi-element (for multiple/partitioned ownership).
        4. Return reduced input for use by downstream loading functions
        """
        if not sharded_input:
            self.load_worldsize = len(state_dicts)
            state_dicts = _shard_inclusive(state_dicts, self.rank, self.worldsize)
        if self.load_worldsize == self.worldsize:
            [
                setattr(self, flag, state_dicts[0][self.statename(flag)])
                for flag in self.state_params + self.reshard_params
            ]
        else:
            for flag in self.reshard_params:
                reshard = self._reshard(
                    [sd[self.statename(flag)] for sd in state_dicts]
                )
                setattr(self, flag, reshard)
        return state_dicts

    def load_from_path(self, path: str):
        """
        Count shard files in the specified checkpoint folder and determine overlap with current
        rank and worldsize partition. Load only matching shardfile(s) and pass to load_state_dict.
        This is more efficient than sharding the full loaded state.
        """
        assert os.path.exists(path), "Specified checkpoint does not exist"
        assert not os.path.isfile(path), "Checkpoint should be a folder of shard states"
        fileshards = [x for x in os.listdir(path) if "loader" in x]
        fileshards = sorted(fileshards, key=lambda x: int(x.split("_")[2][:-4]))
        assert (
            len(fileshards) > 0
        ), "Checkpoint directory must contain checkpoint files with 'loader' in the name"
        self.load_worldsize = len(fileshards)
        # Grab only the shard files holding data we currently own
        my_fileshards = _shard_inclusive(fileshards, self.rank, self.worldsize)
        states = [torch.load(os.path.join(path, x)) for x in my_fileshards]
        self.load_state_dict(states, True)

    def save_to_path(self, path: str):
        """
        Grab recursive shard states and save all shard states to the specified checkpoint folder
        """
        os.makedirs(path, exist_ok=True)
        state = self.state_dict()
        torch.save(state, os.path.join(path, f"loader_state_{self.rank}.pth"))


class _Wrapper_Dataset(_Stateful_Dataset):
    """
    Stub for nested wrappers of _Stateful_Datasets. Extends state fns with recursion.
    Requires a single instantiated sub-dataset.
    """

    def __init__(
        self,
        dataset: _Stateful_Dataset,
    ):
        self.dataset = dataset
        super().__init__(self.dataset.rank, self.dataset.worldsize)

    def load_state_dict(self, state_dicts, sharded_input=False):
        """
        Sets all specified flags at the current level, then recurses into wrapped dataset.
        """
        sharded_dicts = super().load_state_dict(state_dicts, sharded_input)
        self.dataset.load_worldsize = self.load_worldsize
        self.dataset.load_state_dict(sharded_dicts, True)
        return sharded_dicts

    def state_dict(self):
        """
        Fetches state dict recursively from wrapped layers, then adds specified flags.
        Overlapping flags are overwritten with a warning.
        """
        out = self.dataset.state_dict()
        state = super().state_dict()
        for flag in self.state_params + self.reshard_params:
            if flag in out:
                logging.warning(
                    f"Loader {self.rank}: flag {flag} already present in state_dict with value {out[flag]}. "
                    + f"Overwriting with value {state[flag]}"
                )
        out.update(state)
        return out


# --------------  DATASET LAYERS  --------------


class Preprocess_Dataset(_Wrapper_Dataset):
    """
    Wrapper for a _Stateful_Dataset that applies a specified preprocessing
    or augmentation function to dataset outputs.
    ...
    Args
    ----
    dataset : _Stateful_Dataset
        Fully instantiated dataset
    aug_fn : function (any -> any)
        The augmentation function to apply to each dataset item.
    """

    def __init__(
        self,
        dataset: _Stateful_Dataset,
        aug_fn: Callable,
    ):
        super().__init__(dataset)
        self.aug_fn = aug_fn

    def __iter__(self):
        dataset = iter(self.dataset)
        while True:
            out = next(dataset)
            yield self.aug_fn(out)


class Checkpoint_Dataset(_Wrapper_Dataset):
    """
    Wrapper for a _Stateful_Dataset that implements auto-checkpoint saving every n steps.
    Useful for setting n_workers > 0, so that workers do not rely on the master process
    for state saving (inter-process communication unsupported in PyTorch datasets).
    ...
    Args
    ----
    dataset : _Stateful_Dataset
        Fully instantiated dataset
    load_path : str
        Absolute path to checkpoint load directory. If a checkpoint exists, loads it.
    interval : int
        Saves a new checkpoint every interval.
    steps_per_batch : optional[int]
        Number of steps required to fill a single batch. Increments interval only
        when a full batch is formed. Defaults to 1.
    save_path : optional[str]
        Absolute path to checkpoint save directory. Defaults to load_path.
    """

    def __init__(
        self,
        dataset: _Stateful_Dataset,
        load_path: str,
        interval: int,
        steps_per_batch: int = 1,
        save_path: str = "",
    ):
        super().__init__(dataset)
        self.interval = interval
        self.spb = steps_per_batch
        if len(save_path) == 0:
            save_path = load_path
        self.path = save_path
        self.step = 0
        self.ministep = 0
        self.load_from_path(load_path)

    def __iter__(self):
        dataset = iter(self.dataset)
        while True:
            yield next(dataset)
            self.ministep += 1
            if self.ministep == self.spb:
                self.ministep = 0
                self.step += 1
                if self.step % self.interval == 0:
                    newpath = os.path.join(self.path, "step-" + str(self.step))
                    self.save_to_path(newpath)

    def report(self, msg):
        if self.rank == 0:
            print(msg)

    def save_to_path(self, path: str):
        self.report(f"Saving dataset to {path}")
        start = time.time()
        super().save_to_path(path)
        self.report(
            f"Dataset successfully saved to {path}! Save time: {time.time() - start}"
        )

    def load_from_path(self, path: str):
        # If path does not exist, or exists but is empty, exit early
        if not os.path.exists(path) or len(os.listdir(path)) == 0:
            self.report(
                f"No valid checkpoint detected at {path}, dataset starting from scratch."
            )
            return
        # Grab latest item in path
        latest = os.path.join(path, _get_latest(path))
        self.report(f"Dataset checkpoint detected at {latest}")
        # If item is not a folder, exit early
        if os.path.isfile(latest):
            self.report(
                f"Checkpoint exists but contains no dataset! Dataset starting from scratch."
            )
            return
        # If item is a folder, get the step count
        self.step = int(latest.split("-")[-1])
        # Proceed
        start = time.time()
        self.dataset.load_from_path(latest)
        self.report(f"Dataset checkpoint loaded! Load time: {time.time() - start}")


class Preload_Buffer_Dataset(_Wrapper_Dataset):
    """
    Wrapper for a Stateful_Dataset that implements data shuffling via a single in/out buffer.
    Fills buffer two at a time, up to desired size, then switches to one at a time to maintain size.
    Passes randomly sampled outputs one by one.
    Ensures local mixing of data without relying on sliding windows or shuffling of large buffers.
    Any two consecutive inputs will be separated by window_size steps in expectation.
    Rescaling-enabled: buffers that shrink will re-grow to window_size, buffers that expand stay large.
    ...
    Args
    ----
    dataset : _Stateful_Dataset
        Fully instantiated dataset
    window_size : int
        Max size of input/output buffer
    """

    def __init__(self, dataset: _Stateful_Dataset, window_size: int):
        super().__init__(dataset)
        assert (
            window_size > 1
        ), f"Window size {window_size} must be greater than 1 for shuffling to occur"
        self.window_size = window_size
        self.g_state = None
        self.generator = torch.Generator().manual_seed(self.rank)
        self.buffer: List[List[Any]] = []
        self.buffer_size = 0
        self.state_params = ["g_state"]
        self.reshard_params = ["buffer"]

    def __iter__(self):
        dataset = iter(self.dataset)
        while True:
            # Pad out buffer if needed
            self._pad_buffer()

            # Load a point to buffer if necessary
            if self.buffer_size < self.window_size:
                self.buffer[self.buffer_size] = next(dataset)
                self.buffer_size += 1

            # Swap out randomly sampled value from buffer
            i = torch.randint(self.buffer_size, (1,), generator=self.generator).item()
            out = self.buffer[i]
            self.buffer[i] = next(dataset)
            yield out

    def _pad_buffer(self):
        if self.buffer_size < self.window_size:
            self.buffer += [
                [],
            ] * (self.window_size - self.buffer_size)

    def state_dict(self):
        # Write generator state manually
        self.g_state = self.generator.get_state()
        # Prune buffer so it can be resharded in future
        self.buffer = self.buffer[: self.buffer_size]
        out = super().state_dict()
        return out

    def load_state_dict(self, state_dicts, sharded_input=False):
        sharded_dicts = super().load_state_dict(state_dicts, sharded_input)
        # Manually set generator state if it exists
        if self.g_state is not None:
            self.generator.set_state(self.g_state)
        # Manually set buffer size
        self.buffer_size = len(self.buffer)
        return sharded_dicts


class Buffer_Dataset(_Wrapper_Dataset):
    """
    Wrapper for a _Stateful_Dataset that takes in sequences of varying lengths, and packs/pads them
    into sequences of desired length. Input sequences are packed greedily until the buffer would
    otherwise overrun, then remaining values are filled depending on initialization flags.
    Also injects BOS/EOS into the packed output sequence if desired, and if BOS/EOS tokens are
    not already in those positions. Implements rescaling by simply dropping (buffer) state.
    ...
    Args
    ----
    dataset : _Stateful_Dataset
        Fully instantiated dataset
    seq_len : int
        The desired sequence length
    pack_hard : bool
        Split input sequences to fill output buffer, or use pad tokens to fill remaining space?
    bos_token : any | None
        Token to prepend to every output sequence. If None, no token is added. Type should match data type.
    eos_token : any | None
        Token to append to every output sequence. If None, no token is added. Type should match data type.
    pad_token : any | None
        Token used to fill out output sequence. Type should match data type.
    drop_final_token : any | None
        Drop the final token of each document if it matches this value?
        (For edge case where bos=eos=None, and sep already appears at beginning of each doc -
        drop added extra sep from end of doc)
    """

    def __init__(
        self,
        dataset: _Stateful_Dataset,
        seq_len: int,
        pack_hard: bool,
        bos_token=None,
        eos_token=None,
        pad_token=None,
    ):
        super().__init__(dataset)
        self.len = seq_len

        # Buffer args
        self.buffer: List[str] = []
        self.bos = bos_token
        self.eos = eos_token
        self.pad = pad_token
        self.pack_hard = pack_hard
        if not pack_hard:
            assert (
                pad_token is not None
            ), "Error: if using pads, you must supply a pad_token"

        self.state_params = ["buffer"]

    def _get_buffer(self, iterable, length, buffer):
        # Pull data until buffer is about to overrun, return exactly proper length
        new = []
        while len(buffer) + len(new) < length:
            buffer += new
            new = next(iterable)

        # Add bos if needed
        if self.bos is not None and (len(buffer) == 0 or buffer[0] != self.bos):
            buffer = [self.bos] + buffer

        # Handle buffer splitting
        if len(buffer) >= length:
            # If buffer is too long, force split
            out = buffer[:length]
            buffer = buffer[length:]
            if self.eos is not None and out[-1] != self.eos:
                buffer = [out[-1]] + buffer
                out[-1] = self.eos
            buffer = buffer + new
        else:
            if self.pack_hard:
                # Pack in as much of new sequence as will fit
                buffer = buffer + new
                out = buffer[:length]
                buffer = buffer[length:]
                if self.eos is not None and out[-1] != self.eos:
                    buffer = [out[-1]] + buffer
                    out[-1] = self.eos
            else:
                # Fill out with pads as needed
                if self.eos is not None and buffer[-1] != self.eos:
                    buffer.append(self.eos)
                if self.pad is not None:
                    out = buffer + [self.pad] * (length - len(buffer))
                else:
                    out = buffer
                buffer = new
        return out, buffer

    # Fill buffer line by line, delimiters and packing/splitting as appropriate
    def __iter__(self):
        dataset = iter(self.dataset)
        while True:
            out, buffer = self._get_buffer(dataset, self.len, self.buffer)
            self.buffer = buffer
            yield out


class Streaming_Doc_Dataset(_Stateful_Dataset):
    """
    The base distributed dataset for loading sequences/documents from pyarrow shards.
    Pyarrow shard files are expected to hold multiple recordBatches, where each recordBatch has a "tokens"
    field consisting of a single token list. (i.e. each document is a single sequence under a "token" field,
    and the file is a list of such sequences)
    Relies on a compiled metadata file to fetch shardfile lengths, assumes file already exists in the parent directory,
    and is in proper csv format (first row "dataset/filename,documents,tokens", subsequent rows these values).

    For a single dataset directory, splits shard files into x=worldsize fragments and grabs a 1/n contiguous
    span of shard fragments (contiguous to limit file reads from cloud/disk).
    Logs the number of documents owned from each shardfile, and relies on ZCG random bijection to
    map contiguous range of indices to shuffled, noncontiguous set of documents from each shard file.
    Shuffles the file list deterministically to hop from file to file.

    At runtime, iterates through documents in each shuffled shard file, pulling each shard on demand.
    Shards are thus pulled no more than once per epoch.
    Returns documents in chunks up to size max_chunksize, and handles delimiter token placement between documents.

    Streaming_Doc_Dataset grabs files from a flat directory representing a single dataset.
    For percentage-based sampling of multiple subdatasets, see Sampling_Dataset.
    ...
    Args
    ----
    datapath : str
        Absolute path to the dataset directory. Expects directory containing pyarrow shardfiles.
        Parent directory should contain 'meta' folder with metadata csv file inside.
    rank : int
        Current worker index
    worldsize : int
        Total number of workers
    delimiter_token : Any
        Token used to indicate sequence/document breaks. Type should match data type. Required for downstream
        sampling logic (can be removed later via PreProcess_Dataset if needed).
    bos_token : Any | None
        Optional token used to indicate sequence/document start. Type should match data type.
    strip_tokens : set[Any]
        Token values that should be removed if detected at beginning or end of document
        (i.e. any eos/bos tokens already present in the data). Type should match data type.
    seed : int
        The random seed for deterministic shuffling/sharding
    min_length : int
        Sequences below this length are skipped
    max_chunksize : int
        Maximum sequence length to return. Break long docs into chunks of this size or shorter.
    verbose : bool
        Track setup progress?
    shuffle : bool
        Shuffle shard file and document orders? (Disable for simple testing)
    """

    def __init__(
        self,
        datapath: str,
        rank: int,
        worldsize: int,
        delimiter_token: Any,
        bos_token: Optional[Any] = None,
        strip_tokens: Optional[Set[Any]] = set(),
        seed: int = 42,
        min_length: int = 1,
        max_chunksize: int = 1024,
        verbose: bool = False,
        shuffle: bool = True,
    ):
        super(Streaming_Doc_Dataset, self).__init__(rank, worldsize)
        self.seed = seed
        self.data = datapath
        self.min_length = min_length
        assert max_chunksize > 0, f"Max chunksize must be a nonzero positive integer"
        self.chunksize = max_chunksize
        self.eos = delimiter_token
        self.bos = bos_token
        self.drop = strip_tokens
        self.verbose = verbose
        self.docset: List[
            Any
        ] = []  # map of doc indices to (shardid, min docid, max docid)
        self.docs_per_shard = {}

        # Guaranteed inconsistent shuffling across workers
        random.seed(self.seed + rank)

        # Gather per-file document counts from metadata count file(s)
        countfiles = [
            x
            for x in os.listdir(os.path.join(os.path.dirname(datapath), "meta"))
            if "counts" in x and "csv" in x
        ]
        assert len(countfiles) == 1
        doc_counts = {}
        pathsplit = (datapath, "")
        while len(pathsplit[1]) == 0:
            pathsplit = os.path.split(pathsplit[0])
        pardir, dataset = pathsplit
        self.dataset = dataset
        with open(os.path.join(pardir, "meta", countfiles[0]), "r") as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                fullpath = row["dataset/filename"]
                prefix = fullpath.find("/" + dataset) + 1
                if prefix > 0:
                    key = fullpath[prefix:]
                    doc_counts[key] = int(row["documents"])

        # Assemble document set owned by this worker:
        # listdir, assemble shardfraglist (ind -> shard, frag)
        shards = [
            shard
            for shard in os.listdir(datapath)
            if os.path.isfile(os.path.join(datapath, shard))
            and "arrow" in os.path.join(datapath, shard)
        ]
        shards.sort()  # Ensure consistent sharding across machines
        start_frag = (rank * worldsize * len(shards)) // worldsize
        end_frag = ((rank + 1) * worldsize * len(shards)) // worldsize
        shardfrags = [
            (shards[i // worldsize], i % worldsize) for i in range(start_frag, end_frag)
        ]

        # Read shardfrags, assemble doc list for each file shard (aggregating over fragments):
        ndocs = -1
        docset = {}  # shardid -> (min docid, max docid)
        for i, (shard, frag) in enumerate(shardfrags):
            ndocs = doc_counts[os.path.join(dataset, shard)]
            self.docs_per_shard[shard] = ndocs
            doc_start = (ndocs * frag) // worldsize
            doc_end = (ndocs * frag + ndocs) // worldsize - 1  # Inclusive upper bound
            if shard not in docset:
                docset[shard] = [doc_start, doc_end]
            min_d, max_d = docset[shard]
            if doc_start < min_d:
                docset[shard][0] = doc_start
            if doc_end > max_d:
                docset[shard][1] = doc_end

        # Add all of this dataset's shard entries to self.docset
        doccount = 0
        for shardid in docset:
            min_d = docset[shardid][0]
            max_d = docset[shardid][1]
            self.docset.append((shardid, min_d, max_d))
            doccount += max_d - min_d + 1
        self._len = doccount

        if verbose:
            logging.info(
                f"    Worker {rank} ingested {len(shardfrags)} shard fragments from {dataset}"
            )

        # Shuffle shard files
        if shuffle:
            random.shuffle(self.docset)

        self.docset_index = 0
        self.chunk_index = -1

        # Stats
        self.epochs_seen = -1
        self.tokens_seen = 0
        self.docs_seen = 0
        self.percent_seen = 0
        self.lcg_state = seed + rank

        self.state_params = [
            "dataset",
            "docset_index",
            "chunk_index",
            "epochs_seen",
            "tokens_seen",
            "docs_seen",
            "percent_seen",
            "lcg_state",
        ]

    def _get_docid(self, i):
        """
        Given a global doc index over the set of docs owned by this worker,
        return the corresponding data/shard/local index
        """
        cur = 0
        assert (
            i <= self._len
        ), f"You have requested an illegal doc index {i}, docset length is {self._len}"
        for shardid, min_d, max_d in self.docset:
            docrange = max_d - min_d + 1
            cur += docrange
            if cur > i:
                return shardid, docrange, min_d

    def _get_reader(self, path, newpath, reader):
        """
        If new filepath does not match the current one,
        open a new reader on that filepath (pull file on demand)
        """
        if newpath != path:
            del reader
            if self.verbose:
                logging.info(f"Worker {self.rank} opening new file {newpath}")
            reader = pa.ipc.open_file(newpath)
            path = newpath
        return path, reader

    def _construct_chunk(self, j, doc, n_chunks):
        """
        Grab a chunk of the desired size from the pyarrow document,
        avoiding unnecessary overhead in case of large docs
        """
        start_index = j * self.chunksize
        n_pull = self.chunksize
        if self.bos is not None:
            if j == 0:
                n_pull -= 1
            else:
                start_index -= 1
        chunk = doc.slice(start_index, n_pull).to_pylist()
        self.tokens_seen += len(chunk)
        # Add bos/eos tokens if needed
        if self.bos is not None and j == 0:
            chunk = [self.bos] + chunk
        if j == n_chunks - 1:
            chunk = chunk + [self.eos]
        return chunk

    def _random_map_docid(self, size):
        """
        Given size of document pool, use saved state (prior index) to generate the next index via LCG.
        Implements within-shard document shuffling without materializing any large doc lists.
        """
        m = 2 ** math.ceil(math.log2(size))  # Round up to nearest power of 2
        a = 5  # A,C values known to work well with powers of 2 (Knuth, 1997, 3.2.1.3)
        c = (self.rank + self.seed) * 2 + 1
        state = self.lcg_state
        while True:
            state = (a * state + c) % m
            if state < size:
                return state

    def __iter__(self):
        docset_offset = self.docset_index
        lcg_offset = self.lcg_state
        residual_chunks = self.chunk_index + 1  # pick up AFTER where the ckp left off
        ndocs = self._len
        path = ""
        reader = None
        while True:
            # Iterate through docs, starting at desired offset
            for i in range(ndocs):
                doc_index = (docset_offset + i) % ndocs

                # Update stats
                if doc_index == 0:
                    self.epochs_seen += 1
                self.docset_index = doc_index
                # Map doc id to shard, id in file
                shardid, docrange, mindoc = self._get_docid(doc_index)

                # Read doc
                newpath = os.path.join(self.data, shardid)
                path, reader = self._get_reader(path, newpath, reader)
                # Map id in range of owned docs to new (consistently) shuffled id
                doclcg = self._random_map_docid(docrange)
                docid = doclcg + mindoc
                doc = reader.get_batch(docid)["tokens"]
                if doc[0].as_py() in self.drop:
                    doc = doc.slice(1, len(doc) - 1)
                if doc[-1].as_py() in self.drop:
                    doc = doc.slice(0, len(doc) - 1)
                doclen = len(doc) + 1 if self.bos is None else len(doc) + 2
                if doclen >= self.min_length:
                    n_chunks = math.ceil(doclen / self.chunksize)
                    for j in range(n_chunks):
                        if i == 0 and j < residual_chunks:
                            pass
                        else:
                            self.chunk_index = j
                            # Document complete, update stats
                            if j == n_chunks - 1:
                                self.docs_seen += 1
                                self.percent_seen = (
                                    self.docs_seen * 100 / (self._len + 1e-9)
                                )
                            yield self._construct_chunk(j, doc, n_chunks)

                # Advance RNG state
                self.lcg_state = doclcg

            # Load any chunks initially skipped in first doc
            self.docset_index = docset_offset
            self.lcg_state = lcg_offset
            shardid, docrange, mindoc = self._get_docid(docset_offset)
            docid = self._random_map_docid(docrange) + mindoc
            newpath = os.path.join(self.data, shardid)
            path, reader = self._get_reader(path, newpath, reader)
            doc = reader.get_batch(docid)["tokens"]
            if doc[0].as_py() in self.drop:
                doc = doc.slice(1, len(doc) - 1)
            if doc[-1].as_py() in self.drop:
                doc = doc.slice(0, len(doc) - 1)
            doclen = len(doc) + 1 if self.bos is None else len(doc) + 2
            if doclen >= self.min_length:
                n_chunks = math.ceil(doclen / self.chunksize)
                for j in range(residual_chunks):
                    self.chunk_index = j
                    yield self._construct_chunk(j, doc, n_chunks)

    def load_state_dict(self, state_dicts, sharded_input=False):
        assert (
            self.load_worldsize == self.worldsize
        ), "Streaming_Doc_Dataset does not support rescaling. Please use a Scalable_Shard_Dataset."
        d = self.dataset
        out = super().load_state_dict(state_dicts, sharded_input)
        assert (
            d == self.dataset
        ), f"Dataset mismatch: checkpoint contains {self.dataset}, expected {d}"
        return out


class Sampling_Dataset(_Stateful_Dataset):
    """
    A _Stateful_Dataset implementing percentage-based sampling: weights can be floats, and the
    number of tokens seen from each subdataset will match those weights as closely as possible.
    This is accomplished by maintaining a _Stateful_Dataset for each subdataset, and tracking
    the number of tokens emitted by each. Whichever loader is furthest from its target will be
    the next to pass a document.

    All args except for dataset_type, datasets, weights and delimiter are pass-through args for
    the component _Stateful_Datasets and are documented in the appropriate classes.
    ...
    Args
    ----
    dataset_type : Scalable_Shard_Dataset | Streaming_Doc_Dataset
        Underlying iterator for each desired subdataset
    delimiter_token : Any
        Token used to indicate sequence/document breaks. Type should match data type.
    datasets : list[str] | None
        A list of subdatasets to draw from. If None, draws from all subfolders of datapath.
    weights : list(float) | None
        Weights describing what percent of emitted tokens should come from each subdataset.
        Need not sum to 1. If None, tokens are drawn evenly.
    ...
        Pass-through args, see Streaming_Doc_Dataset or Scalable_Shard_Dataset
    """

    def __init__(
        self,
        datapath: str,
        dataset_type: Union[
            Type["Streaming_Doc_Dataset"],
            Type["Scalable_Shard_Dataset"],
        ],
        rank: int,
        worldsize: int,
        delimiter_token: Any,
        datasets=None,
        weights=None,
        verbose=False,
        **kwargs,
    ):
        super().__init__(rank, worldsize)
        self.delimiter = delimiter_token
        self.datasets = (
            datasets
            if datasets is not None
            else [
                f
                for f in os.listdir(datapath)
                if not os.path.isfile(os.path.join(datapath, f)) and "meta" not in f
            ]
        )
        assert len(self.datasets) > 0, "You must specify at least one dataset"

        if weights is not None:
            assert len(weights) == len(
                self.datasets
            ), f"Number of oversample weights {len(weights)} must match number of datasets {len(self.datasets)}"
            for w in weights:
                assert w > 0, f"Sampling rate {w} must be positive"
        self.weights = [1] * len(self.datasets) if weights is None else weights
        self.weights = [w / sum(self.weights) for w in self.weights]

        self.tokens_seen = [0] * len(self.datasets)

        # Build subdataset iterators
        self.data = []
        for i, d in enumerate(self.datasets):
            self.data.append(
                dataset_type(
                    datapath=os.path.join(datapath, d),
                    rank=rank,
                    worldsize=worldsize,
                    delimiter_token=delimiter_token,
                    verbose=verbose,
                    **kwargs,
                )
            )
            if verbose:
                logging.info(
                    f"Worker {rank} assembled subdataset iterator for {d}, {i+1} of {len(self.datasets)}"
                )

        self.current_iterator = -1
        self.state_params = ["tokens_seen", "current_iterator"]

    def __iter__(self):
        # Grab one doc at a time in random order
        data = [iter(d) for d in self.data]
        while True:
            if self.current_iterator != -1:
                # Finish current document
                out = next(data[self.current_iterator])
                self.tokens_seen[self.current_iterator] += len(out)
                if out[-1] == self.delimiter:
                    self.current_iterator = -1
                yield out
            else:
                # Choose new subdataset to draw from
                # (whichever is currently most underrepresented compared to target rate)
                offset = [
                    self.weights[i]
                    - self.tokens_seen[i] / (sum(self.tokens_seen) + 1e-9)
                    for i in range(len(self.datasets))
                ]
                offset_argmax = max((diff, i) for i, diff in enumerate(offset))[1]
                self.current_iterator = offset_argmax

    def state_dict(self):
        # Manually add state of all subloaders to self state
        out = {
            self.statename("sample_iterator_states"): [
                d.state_dict() for d in self.data
            ]
        }
        out.update(super().state_dict())
        return out

    def load_state_dict(self, state_dicts, sharded_input=False):
        # Load stats
        sharded_dicts = super().load_state_dict(state_dicts, sharded_input)
        # Load sub-iterator states
        for i, subdata in enumerate(self.data):
            # Grab just that sub-iterator across all ranks
            subdata.load_worldsize = self.load_worldsize
            subdata.load_state_dict(
                [
                    sd[self.statename("sample_iterator_states")][i]
                    for sd in sharded_dicts
                ],
                True,
            )
        return sharded_dicts


class Scalable_Shard_Dataset(_Stateful_Dataset):
    """
    A _Stateful_Dataset implementing rescalability: loading from checkpoint into a different
    number of gpus will nonetheless keep avoiding all data previously seen in the current epoch.
    This is accomplished by maintaining a large number of small Streaming_Doc_Datasets, which track
    state individually and reshard over n_gpus.

    All keywords except the first are simple pass-through arguments and are documented in Streaming_Doc_Dataset.
    ...
    Args
    ----
    datapath : str
        Absolute path to the dataset directory. Expects folder containing pyarrow shardfiles.
    rank : int
        Current worker index
    worldsize : int
        Total number of workers
    delimiter_token : Any
        Token used to indicate sequence/document breaks. Type should match data type.
    n_logical_shards : int
        Number of logical shards. Must be a multiple of world size.
    ...
        Pass-through args, see Streaming_Doc_Dataset
    """

    def __init__(
        self,
        datapath: str,
        rank: int,
        worldsize: int,
        delimiter_token: Any,
        n_logical_shards: int = 2048,
        verbose=False,
        **kwargs,
    ):
        assert (
            n_logical_shards % worldsize == 0
        ), f"World size {worldsize} must divide n_logical_shards {n_logical_shards} evenly"
        assert (
            n_logical_shards > 0
        ), f"n_logical_shards {n_logical_shards} must be a positive integer"

        super().__init__(rank, worldsize)
        self.data = []
        self.n_logicals = n_logical_shards // worldsize
        self.total_shards = n_logical_shards
        self.delimiter = delimiter_token

        logicals = list(range(n_logical_shards))
        self.logicals_owned = _shard_partition(logicals, self.rank, self.worldsize)
        assert len(self.logicals_owned) == self.n_logicals

        # Build logical shards
        for i in range(self.n_logicals):
            self.data.append(
                Streaming_Doc_Dataset(
                    datapath=datapath,
                    worldsize=n_logical_shards,
                    rank=self.logicals_owned[i],
                    delimiter_token=delimiter_token,
                    verbose=(rank == 0),
                    **kwargs,
                )
            )
            if verbose:
                logging.info(
                    f"Worker {rank} assembled logical shard {self.logicals_owned[i]}, {i+1} of {self.n_logicals}"
                )

        # Fetch logical shard sampling stats
        self.n_docs_remaining = [d._len for d in self.data]

        # Position "state", used only for maintaining order when n_workers is unchanged
        # For scaling up or down, logical position is meaningless, and reset
        self.current_reader = None
        self.logical_shard_states = None
        self.generator = torch.Generator().manual_seed(self.rank)
        self.g_state = None
        self.state_params = ["current_reader", "g_state"]
        self.reshard_params = ["n_docs_remaining", "logical_shard_states"]

    def __iter__(self):
        # Grab one doc at a time in random order
        data = [iter(d) for d in self.data]
        while True:
            # Sample logical shard (or load from ckp)
            if self.current_reader is not None:
                ind = self.current_reader
            else:
                ind = torch.multinomial(
                    torch.tensor(self.n_docs_remaining, dtype=torch.float),
                    1,
                    generator=self.generator,
                ).item()
            self.current_reader = ind
            # Read doc
            out = next(data[ind])
            while out[-1] != self.delimiter:
                yield out
                out = next(data[ind])
            # Update state to show we've finished the doc
            self.current_reader = None
            self.n_docs_remaining[ind] -= 1
            if sum(self.n_docs_remaining) == 0:
                self.n_docs_remaining = [d._len for d in self.data]
                self.generator.manual_seed(self.rank)
            # Return final piece of doc
            yield out

    def state_dict(self):
        # Write generator state manually
        self.g_state = self.generator.get_state()
        # Recursive fetch
        self.logical_shard_states = [d.state_dict() for d in self.data]
        return super().state_dict()

    def load_state_dict(self, state_dicts, sharded_input=False):
        sharded_dicts = super().load_state_dict(state_dicts, sharded_input)
        # Manually set generator state if it exists
        if self.g_state is not None:
            self.generator.set_state(self.g_state)
        # Recursive set
        for i in range(self.n_logicals):
            self.data[i].load_state_dict([self.logical_shard_states[i]], True)
        return sharded_dicts


# --------------  CONSTRUCTORS  --------------


def build_experimental_data_loader(cfg, rank, world_size):
    """
    Pytorch dataloader for stateful, distributed, and rescalable causal language model (CLM) training.
    Assumes underlying data is sequences of integer values.
    ...
    Args
    ----
    cfg : dataclass
        Training config containing seq len, dataset, dataset weight, datapath, etc. arguments
    rank : int
        Rank of current distributed worker. Used for handling dataset sharding logic.
    world_size : int
        Number of distributed workers. Used for handling dataset sharding logic.
    """

    datasets, weights = parse_data_args(
        cfg.dataset.datasets, cfg.dataset.dataset_weights
    )

    def causal_lm(data_seq, prompt_len=0):
        """
        Perform causal language modeling by right-shifting the input sequence.
        Sets first prompt_len tokens to be ignored by the loss.
        """
        data_seq = torch.LongTensor(data_seq)
        t = data_seq.clone()[1:]
        data_seq = data_seq[:-1]
        t[:prompt_len] = -100
        return data_seq, t

    # Base streaming dataset. Returns doc chunks in sequence.
    # Implements dataset sampling and rescalability.
    droplist = [
        int(x.strip())
        for x in cfg.dataset.drop_tokens.split(",")
        if len(x.strip()) > 0
    ]
    droplist = droplist + [cfg.dataset.bos_token, cfg.dataset.eos_token]
    data = Sampling_Dataset(
        cfg.dataset.dataset_path,
        Scalable_Shard_Dataset,
        rank,
        world_size,
        cfg.dataset.eos_token,
        bos_token=None if cfg.dataset.bos_token == -1 else cfg.dataset.bos_token,
        strip_tokens=set(droplist),
        min_length=3,
        datasets=datasets,
        weights=weights,
        seed=42,
        verbose=(rank == 0),
        n_logical_shards=cfg.dataset.data_logical_shards,
    )
    # Wrap above dataset in packing logic to form constant-length lines.
    data = Buffer_Dataset(
        data,
        cfg.training.seq_len + 1,
        pack_hard=True,
    )
    # Shuffle outputs in length 10k buffer. Consecutive lines appear 10k steps apart on average.
    data = Preload_Buffer_Dataset(data, 10000)
    # Split line into input and target for the CLM task.
    data = Preprocess_Dataset(data, causal_lm)
    # Enable auto-saving
    if cfg.checkpoint.enable_checkpoint and not cfg.checkpoint.model_weights_only:
        assert (
            cfg.checkpoint.interval_type == "steps"
        ), "Dataloader checkpointing supports only step-based interval"
        data = Checkpoint_Dataset(
            data,
            os.path.join(cfg.job.dump_folder, cfg.checkpoint.folder),
            cfg.checkpoint.interval,
            cfg.training.batch_size,
        )
    return torch.utils.data.DataLoader(
        data, num_workers=1, batch_size=cfg.training.batch_size
    )


def parse_data_args(datas, weights):
    # Convert csv inputs into corresponding lists of values
    def splitstrip(x):
        if isinstance(x, str):
            return [item.strip() for item in x.split(",")]
        elif isinstance(x, (list, tuple)):
            return list(x)
        elif isinstance(x, (int, float, complex)):
            return [x]
        else:
            raise ValueError(f"arg input {x} cannot be parsed.")

    datas = splitstrip(datas)
    weights = [float(x) for x in splitstrip(weights)]
    return datas, weights
