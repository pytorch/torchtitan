"""
An abstract framework for datasets composed of many uniform-format files.
Examples would be:

- Video frames across multiple video files.
- Rows spread across several .csv, .h5, or .parquet files.

:py:class:`MultiFileDataset` provides scalable bookkeeping for such datasets,
while leaving the actual loading logic to the subclass. It supports:

- Efficient global-to-local indexing.
- Deterministic or random sub-sampling.
- Optional per-file metadata extraction.
- Metadata caching via summary CSVs for faster reuse or distributed training.

This class is especially suitable for large-scale distributed setups and
integrates cleanly with PyTorch's `DataLoader`.

Subclasses must implement
-------------------------

1. ``_get_samples_metadata_from_filepath : path -> tuple[int, metadata | None]``

   Inspect a single file and return:

   - The number of valid samples in the file.
   - Optional per-file metadata (e.g., a 0-D structured `np.ndarray` or `None`). 
     Only simple structured metadata - that is unnested and scalar fields - are 
     supported.

2. ``_getitem(file_path, sample_idx, metadata) -> Any``

   Load and return a sample from a given file at a given index, optionally using
   the provided metadata.

3. ``_get_image_representation(file_path, sample_idx, metadata) -> ImageLike, str``

    Optional. A function that retuns a visualisation of a single datapoint with
    an image and a description string, which is then used for creating overview
    videos over the entire dataset. Usefull inspection tool.

Usage Lifecycle
---------------
1. **Construction**
    Instantiate with a root folder or a summary CSV:

    >>> dataset = MyDataset(root="data/")
    >>> # or for faster startup on distributed systems:
    >>> dataset = MyDataset("summary.csv", source_is_summary_csv=True)

    During construction:

    - Files are scanned and filtered.
    - Sample counts and metadata are collected.
    - Subsampling rules and RNG seeds are applied.

2. **Iteration**
    - ``__len__()`` returns total sample count *after* subsampling.
    - ``__getitem__(index)`` maps the global index to (file, local index) 
      and calls your subclass's ``_getitem(...)``.

3. **(Optional) Caching**
    - Use ``save("summary.csv")`` to persist file/sample/metadata info.
    - Later, call ``load("summary.csv")`` to skip expensive recursive file survey.

4. **(Optional) Video Generation**

    ``dataset.create_sample_video("survey.mp4", frames_per_file=15, fps=15)``
    Will create a video ``survey.mp4`` that shows 15 equidistantly sampled
    visualisations for datapoints in each file for one second.
    E.g. if the dataset contains 150 files, then it creates a 2:30 minute video
    of 2250 frames.

Distributed Training Support
----------------------------
To ensure reproducible subsampling across workers, use the provided:

>>> DataLoader(..., worker_init_fn=worker_init_fn)

This initializes the random number generator in each worker deterministically
from the base seed, so that subsampling decisions are reproducible.

Example
-------
See :class:`src.videodataset.VideoDataset` for a working subclass using video files.

Notes
-----
Actual data is never loaded or held in memory; this class only manages
bookkeeping and indexing. Subclasses should also not keep multiple files open,
as for our joined data training more files must be held open than is possible.
"""

import os, sys
from pathlib import Path
from typing import Any, Tuple, Union, Optional, cast
import re

import numpy as np
from numpy.typing import NDArray

import pandas as pd
from collections import defaultdict
from torch.utils.data import Dataset, get_worker_info
from tqdm.auto import tqdm

from .ffmpegvideowriter import FFMPEGVideoWriter


class MultiFileDataset(Dataset):
    """
    Abstract base class for datasets composed of multiple uniform-format files.

    This class handles:

    - Discovery and indexing of valid files.
    - Sample counting and per-file metadata.
    - Global-to-local sample indexing (O(log N)).
    - Sub-sampling (random or deterministic).
    - Metadata caching via CSV for distributed scalability.

    Subclasses must implement these two (three) methods:
    ----------------------------------------------------
    
    1. :py:meth:`_get_samples_metadata_from_filepath` ``: path -> tuple[int, metadata | None]``

       Inspect a single file and return:

       - The number of valid samples in the file.
       - Optional per-file metadata (e.g., a 0-D structured `np.ndarray` or `None`).
         Only simple structured metadata (unnested, scalar fields) is supported.

    2. :py:meth:`_getitem` ``: (file_path, sample_idx, metadata) -> Any``

       Load and return a sample from a given file at a given index, optionally using
       the provided metadata. This is the output the dataset should return when
       called via ``dataset[idx]`` or ``dataset.__getitem__(idx)`` just from file
       specific indexing information.
    
    (3.) :py:meth:`_get_image_representation` ``: (file_path, sample_idx, metadata, batch) -> ImageLike, str``

        Optional. A function that retuns a visualisation of a single datapoint with
        an image and a description string, which is then used for creating overview
        videos over the entire dataset. Usefull inspection tool.

    Notes
    -----
    This class does **not** load the entire dataset of each file. Subclasses
    should strive to implement methods for quick random access to the specific
    indexed datapoint via :py:meth:`_getitem` without loading each entire file
    into memory, this superclass handles everything else.
    """

    # --------------------------- constructor --------------------------- #

    _file_paths : NDArray[np.str_]
    """All valid file paths used in this dataset. 
       Sorted to be in lexicographic order."""

    _samples_per_file : NDArray[np.int64]
    """For each file in :py:attr:`_file_paths`, how many usable samples it contains.
       
       Yes, np.uint64 would make more sense, but because the result of an operation
       between an uint and an int is always upcasted into a float, its much easier
       this way"""

    _samples_after_file : NDArray[np.int64]
    """Cumulative sum of :py:attr:`_samples_per_file`. 
       Used to convert global indices to local ones and for O(log N) indexing."""

    _samples_before_file : NDArray[np.int64]
    """The number of samples in all files before the indexed one."""

    _file_metadatas : NDArray[np.void]
    """Optional structured array storing per-file metadata returned by the 
       subclass hook :py:meth:`_get_samples_metadata_from_filepath`. 
       Can be a N dimensional array of 0-byte entries if no metadata is returned.
       The structured datatype must have only non-nested and scalar values
       for easy interpretation as additional columns of a table."""

    _rng : np.random.Generator
    """RNG used for random sub-sampling if enabled. Manually handled for 
       reproducibility."""

    _max_files : int
    """Active limit on the number of files to sample from, in lexicographic
       order. For debugging purposes and for creating data-skaling plots.
       Set via the :py:meth:`set_parameters` function to respect constraints."""

    _subsampling_step : int
    """Step size for sub-sampling the dataset."""

    _random_subsampling : bool
    """If ``True`` and ``subsampling_step`` is greater than one, a *random*
       frame inside every contiguous block of length *k* is selected 
       instead of simply selecting the first one.
       In this way, subsampling can still see each datapoint over multiple
       epochs while distilling the dataset for individual epochs. 
       Useful if many datapoints are very similar if they appear
       consecutive, like in videos with high frame rate."""


    def __init__(
        self,
        source : Union[str, Path, list[Union[str,Path]], list[str], list[Path]],
        *,
        source_is_summary_csv : bool = False,
        valid_file_ending : str = "",
        valid_file_must_contain : str = "",
        recursive : bool = True,
        max_files : Optional[int] = None,
        subsampling_step : Optional[int] = None,
        target_dataset_size : Optional[int] = None,
        random_subsampling : bool = True,
        seed : Optional[int] = None,
        strict : bool = True,
        display_stats : bool = False,
        quiet : bool = False,
    ):
        """
        Initializes the MultiFileDataset.

        Parameters
        ----------
        source : str | Path | list[str | Path]
            Root directory that contains the data **or** path to a metadata CSV
            produced by :py:meth:`save` **or** a list of paths that should all
            be surveyed for valid source files.
        source_is_summary_csv : bool, default=False
            If ``True`` the expensive directory traversal is skipped and
            everything is restored from the CSV (recommended for clusters to
            prevent long initialisation periods on large numbers of GPUs).
            Such a summary can be created using the :py:meth:`save` function.
        valid_file_ending : str, default=""
            File suffix to accept (e.g. ``".mp4"`` or ``".jpg"``). The
            string is appended to the glob pattern, so wildcards are allowed.
        valid_file_must_contain : str, default=""
            Additional substring that must be present in the filename.
            E.g. if you specify that a valid file must contain e.g. 
            *"preprocessed"* then if the recursive search finds filenames like
            ``["vide01.mp4", "vide02.mp4", "vide02_preprocessed.mp4"]``
            only ``"vide02_preprocessed.mp4"`` would be used for the dataset.
        recursive : bool, default=True
            Whether to scan ``source`` recursively.
        max_files : int | None, default=None
            Active limit on the number of files to sample from, in lexicographic
            order. For debugging purposes and for creating data-skaling plots.
            When this many valid files were found during scanning already, the 
            scanning is halted prematurely. Defaults to no limit.
        subsampling_step : int | None, default=None
            Step size for sub-sampling the dataset.  If ``None``, defaults to 1.
        target_dataset_size : int | None, default=None
            Alternative to ``subsampling_step`` - chooses a step such that the
            visible dataset has roughly this many items.
        random_subsampling : bool, default=True
            If ``True`` and ``subsampling_step`` is greater than one, a *random*
            frame inside every contiguous block of length *k* is selected 
            instead of simply selecting the first one.
            In this way, subsampling can still see each datapoint over multiple
            epochs while distilling the dataset for individual epochs. 
            Useful if many datapoints are very similar if they appear
            consecutive, like in videos with high frame rate.
        seed : int | None, default=None
            RNG seed for *random* sub-sampling (ignored when deterministic).
            Defaults to a random seed.
        strict : bool, default=True
            If set, raises an error if any file satisfying the set criteria by
            ``valid_file_ending`` and ``valid_file_must_contain`` did not load
            properly.
        display_stats : bool, defaults=False
            Gives a status report about the total dataset size and restricted
            subsampled view after loading was successful.
        quiet : bool, defaults=False
            Whether to print a progress bar when surveying all files

        Raises
        ------
        RuntimeError
            When no valid files are found **or** strict mode encounters errors.
        NotImplementedError
            When the method :py:meth:`_get_samples_metadata_from_filepath`
            is not implemented yet.
        ValueError
            When the metadatas from :py:meth:`_get_samples_metadata_from_filepath`
            are either of inconsistent structured dtype, or contain nested types
            or non-scalar values, such as arrays.
        
        """
        if source_is_summary_csv:
            assert isinstance(source, (str, Path)), f"cannot load a list '{source}'"
            self.load(source)
        else:
            error_logs = defaultdict(list)
            if not isinstance(source, list):
                source = [source]

            possible_paths = []
            for p in source:
                path = Path(p)
                if not path.exists():
                    error_logs["FileNotFoundError"].append(str(path))
                    continue
                if path.is_dir():
                    found_paths = (
                            path.rglob("*" + valid_file_ending)
                        if recursive else
                            path.glob("*" + valid_file_ending)
                    )
                else:
                    # path is a file, check if it matches your criteria
                    if ( len(valid_file_ending) == 0
                        or path.name.lower().endswith(valid_file_ending.lower())
                    ):
                        found_paths = [path]
                    else:
                        found_paths = []

                valid_paths = sorted(
                    str(path) for path in found_paths
                    if valid_file_must_contain in path.stem
                )
                if len(valid_paths) == 0:
                    error_logs["NoValidFilesFound"].append(str(path))
                possible_paths.extend(valid_paths)

            possible_paths = sorted(possible_paths)
            if max_files is not None and  max_files < len(possible_paths):
                self._max_files = max_files
                print(f"only looking until we get {max_files} valid files "
                     +f"{len(possible_paths)} out of identified possible files")
            else:
                self._max_files = len(possible_paths)

            _file_paths, _samples_per_file, _file_metadatas = [], [], []

            # Collect number of datapoints and metadata for each file as long as
            # it does not throw an error
            pbar = tqdm(possible_paths, desc=f"{0} valid, {0} error files", disable=quiet, dynamic_ncols=True)
            for path in pbar:
                pbar.set_description(f"{len(_file_paths)} valid, {len(error_logs)} error files")
                if len(_file_paths) >= self._max_files:
                    break
                try:
                    samples, metadata = self._get_samples_metadata_from_filepath(path)
                    if metadata is not None and metadata.ndim > 0:
                        raise ValueError(f"requires 0-dimensional structured metadata, instead got '{metadata}'")
                    if samples > 0:
                        _file_paths.append(path)
                        _samples_per_file.append(samples)
                        _file_metadatas.append(metadata)
                except Exception as e:
                    if isinstance(e, NotImplementedError):
                        raise NotImplementedError(
                            f"Subclasses must implement _get_samples_metadata_from_filepath. "
                            f"Encountered while processing file: {path}"
                        ) from e
                    error_logs[f"{type(e).__name__}: {str(e)}"].append(path)
                    if strict:
                        raise e
            
            self._file_paths = np.array(_file_paths, dtype=str)
            self._samples_per_file = np.array(_samples_per_file, dtype=np.int64)
            self._samples_after_file = np.cumsum(_samples_per_file, dtype=np.int64)
            self._samples_before_file = np.concatenate([
                np.array([0], dtype=np.int64), self._samples_after_file[:-1]
            ])

            # Display any errors during surveying the files
            if len(error_logs) > 0:
                print("While scanning the files encountered errors:")
                for msg, paths in error_logs.items():
                    print(f"  {msg}:")
                    for p in paths:
                        print(f"    {p}")
                print(f"\nA total of {len(error_logs)} errors in "
                     +f"{len(possible_paths)} were encountered")
                if strict:
                    raise RuntimeError(
                        "\nAborted because strict=True and file errors occurred."+
                        " Please fix these errors first, or at least remove "+
                        "the offending files from your source data directory"
                    )

            # Transform the metadatas into a consistent structured numpy array
            if all(x is None for x in _file_metadatas):
                # if all metadatas are None, return
                self._file_metadatas = np.empty(
                    (self._samples_per_file.size,), dtype=[]
                )
            else:
                dtype = _file_metadatas[0].dtype
                try:
                    self._file_metadatas = np.array(_file_metadatas, dtype=dtype)
                except Exception as e:
                    raise ValueError("Could not combine all metadata into a "+
                        f"coherent array. Got error '{type(e).__name__}: {e}'."+
                        f"\nThis is probably because not all provided "+
                        f"metadatas have the same shema, got:\n   "+
                        "\n   ".join([str(m) for m in _file_metadatas]))
                if not (dtype.fields is None) and not all(
                    f[0].fields is None and f[0].subdtype is None
                    for f in dtype.fields.values()
                ):
                    raise ValueError(
                        f"Only unnested structured datatypes with scalar "+
                        f"values are allowed, not '{dtype}'"
                    )

        # honour user-provided view settings
        self.set_parameters(
            max_files=max_files,
            subsampling_step=subsampling_step,
            target_dataset_size=target_dataset_size,
            random_subsampling=random_subsampling,
            seed=seed,
        )
        # display loaded dataset stats
        if display_stats:
            print(
                f"{type(self).__name__}: loaded {self._samples_per_file.size} "
                f"files, {self._samples_after_file[-1]} samples total"
            )
            if self._max_files < self._samples_per_file.size:
                print(
                    f" (view limited to {self._max_files} files / "
                    f"{self._samples_after_file[self._max_files]} samples",
                    end=(")" if self._subsampling_step == 1 else 
                        f", subsampled to only {len(self)} samples)")
                )
        

    # ---------------------------- I/O hooks --------------------------- #

    def save(self, path : str) -> None:
        """
        Writes the gathered bookkeeping into a CSV so future runs can skip
        the expensive directory scan.

        Parameters
        ----------
        path : str | Path
            Destination **without** extension **or** ending in ``.csv``.
        """
        if not path.endswith(".csv"):
            path = f"{path}.csv"
        d = {
            'file_paths': self._file_paths,
            'samples_per_file': self._samples_per_file,
            **{f"{name}:{self._file_metadatas.dtype[name]}": self._file_metadatas[name] 
               for name in (self._file_metadatas.dtype.names or [])}
        }
        pd.DataFrame(d).to_csv(path, index=False)

    def load(self, 
             path: Union[str, Path],
             *,
             max_files : Optional[int] = None,
             subsampling_step : Optional[int] = None,
             target_dataset_size : Optional[int] = None,
             random_subsampling : Optional[bool] = None,
             seed : Optional[int] = None,
    ) -> None:
        """
        Restore a dataset description that was previously written by
        :py:meth:`save`.
        
        To load the dataset directly without first constructing
        a `MultiFileDataset` on which to call ``.load(path)`` on, simply provide
        the path to the summary ``.csv`` while creating the `MultiFileDataset`
        as parameter ``source`` and set ``source_is_summary_csv=True``.

        All keyword arguments except ``path`` mirror :py:meth:`set_parameters`
        so the view can be modified *while* loading.

        Parameters
        ----------
        path : str | Path
            Path to where the summary can be found on the file system.

        Notes
        -----

        """
        assert str(path).endswith('.csv'), f"expecting a .csv file, instead got '{path}'"
        df = pd.read_csv(path)
        
        self._max_files = len(df)
        self._file_paths = df['file_paths'].to_numpy().astype(str)
        self._samples_per_file = df['samples_per_file'].to_numpy()
        self._samples_after_file = np.cumsum(self._samples_per_file, dtype=np.int64)
        self._samples_before_file = np.concatenate([
            np.array([0], dtype=np.int64), self._samples_after_file[:-1]
        ])

        # reconstruct dtype from column names (<name>:<dtype>)
        metadata_cols = [(name, name, dtype) for name, dtype in 
                            zip(df.columns, df.dtypes.to_numpy())
                            if name not in ['file_paths', 'samples_per_file']]
        for i, (name, name, dtype) in enumerate(metadata_cols):
            splitname = name.split(":")
            if len(splitname) > 1:
                metadata_cols[i] = (name, ":".join(splitname[:-1]), np.dtype(splitname[-1]))
            else:
                print(
                    f"Could not infer dtype from '{name}', falling back to {dtype}"
                )
        metadata_dtype = np.dtype([(dtype_name, dt) for df_col_name, dtype_name, dt in metadata_cols])

        # reconstruct metadata as numpy structured arrray from the DataFrame 
        self._file_metadatas = np.empty(len(df), dtype=metadata_dtype)
        for df_col_name, dtype_name, dt in metadata_cols:
            self._file_metadatas[dtype_name] = df[df_col_name].to_numpy()

        self.set_parameters(max_files=max_files,
                            subsampling_step=subsampling_step,
                            target_dataset_size=target_dataset_size,
                            random_subsampling=random_subsampling,
                            seed=seed)

    # ------------------------- view parameters ------------------------ #

    def set_parameters(
        self,
        *,
        max_files : Optional[int] = None,
        subsampling_step : Optional[int] = None,
        target_dataset_size : Optional[int] = None,
        random_subsampling : Optional[bool] = None,
        seed : Optional[int] = None,
    ) -> None:
        """
        Define the *visible* part of the dataset. Should be called during
        set up, not while some process iterates over the dataset.
        Increasing the ``subsampling_step`` or decreasing the ``max_files``
        or ``target_dataset_size`` will most likely throw an error shortly
        afterwards.

        If any parameter is not specified in the function call, that parameter
        will not be modified, same if the parameters are tried to be set to be
        ``None``.

        Parameters
        ----------
        
        max_files : int | None, default=None
            Active limit on the number of files to sample from, in lexicographic
            order. For debugging purposes and for creating data-scaling plots.
        subsampling_step : int | None, default=None
            Step size for sub-sampling the dataset.
        target_dataset_size : int | None, default=None
            Alternative to ``subsampling_step`` - chooses a step such that the
            visible dataset has roughly this many items.
        random_subsampling : bool, default=True
            If ``True`` and ``subsampling_step`` is greater than one, a *random*
            frame inside every contiguous block of length *k* is selected 
            instead of simply selecting the first one.
        seed : int | None
            (Re-)seed the RNG for random sub-sampling.
        """
        if max_files is not None:
            self._max_files = max_files
        if (not hasattr(self, "_max_files") or 
            self._max_files > self._samples_after_file.size
        ):
            self._max_files = self._samples_after_file.size
        if self._max_files <= 0:
            raise RuntimeError(
                "No valid files found. If no prior error was thrown, check that "
                "your source directory  is not empty and contains valid files "
                "with the specified file ending and potentially containing "
                "specified parts in their names")
        
        if random_subsampling is not None:
            self._random_subsampling = random_subsampling
        elif not hasattr(self, "_random_subsampling"):
            self._random_subsampling = False

        if target_dataset_size is not None:
            subsampling_step = int(round(
                self._samples_after_file[self._max_files-1]/target_dataset_size
            ))
        if subsampling_step is not None:
            self._subsampling_step = subsampling_step
        if not hasattr(self, "_subsampling_step") or self._subsampling_step < 1:
            self._subsampling_step = 1

        if seed is not None:
            self._rng = np.random.default_rng(seed)
        elif not hasattr(self, "_rng"):
            self._rng = np.random.default_rng(None)  # non-deterministic

    # ------------------------- PyTorch hooks -------------------------- #


    def __len__(self) -> int:
        """Length *after* sub-sampling."""
        return int(self._samples_after_file[self._max_files-1]) // self._subsampling_step
    
    def nbr(self):
        """returns the number of files from which is sampled"""
        return self._max_files
    
    def _global2local_idx(self, idx:int) -> Tuple[int, int]:
        
        original_idx = idx
        idx *= self._subsampling_step
        if not (0 <= idx < self._samples_after_file[self._max_files - 1]):
            raise IndexError(
                f"index {idx} (pre-subsampling {original_idx}) out of range "+
                f"[0, {self._samples_after_file[self._max_files-1]})"
            )
        assert np.issubdtype(self._samples_after_file.dtype, np.integer)
        # _samples_before_file[key_index] <= idx < _samples_after_file[key_index]
        # _samples_before_file[key_index] = _samples_after_file[key_index-1] or 0
        key_idx = int(np.searchsorted(self._samples_after_file, idx, side="right"))
        frame_idx = idx - self._samples_before_file[key_idx]

        if self._random_subsampling and self._subsampling_step > 1:
            frame_idx += self._rng.integers(
                low=0,
                # guarantee frame_idx < self._samples_after_file[key_idx]
                high=min(
                    self._subsampling_step,
                    self._samples_after_file[key_idx] - idx,
                ),
            )
        return key_idx, frame_idx

    def __getitem__(self, idx : int):
        """
        Map a *global* index to a file-local sample and delegate to the
        subclass-provided loader :py:meth:`_getitem`.

        Raises
        ------
        IndexError
            If the index is out of range.
        NotImplementedError
            When the method :py:meth:`_getitem` is not implemented yet.
        """
        key_idx, frame_idx = self._global2local_idx(idx)
        
        try:
            return self._getitem(
                file_path=self._file_paths[key_idx],
                sample_idx=frame_idx,
                metadata=self._file_metadatas[key_idx],
            )
        except Exception as e:
            new_idx = self._rng.integers(low=0, high=len(self))
            print(f"\nError while loading file {self._file_paths[key_idx]} "+
                  f"at index {frame_idx} from global index {idx}\n"+
                  f"Error message was:\n  {e}\n"+
                  f"Resampling global index {new_idx} instead.")
            return self[new_idx]
    
    # ---------------- Debugging and Visualisation tools --------------- #

    
    def create_sample_video(
        self,
        vid_save_path: Union[str, Path],
        files_frac: float = 1.,
        fps: float = 5, 
        frames_per_file: int = 1,
        max_sample_distance: Optional[int] = None,
        start: Optional[int] = None,
        end: Optional[int] =None,
        random: bool = False,
        seed: Optional[int] = None,
    ):
        """
        Creates a video to visually inspect the dataset. The subclass must
        implement the method :py:meth:`_get_image_representation`, which returns
        the individual frames and titles for the video.
        Good when you have e.g. 2000 video files and only a small subset are 
        corrupt or static and need to be manually identified and removed from the dataset.
        For more custumisation in subclasses, you can call the initialisation
        hooks :py:meth:`_on_video_start` and :py:meth:`_on_video_end`.
        
        Parameters
        ----------
        vid_save_path : str | Path
            The output path where the video should be written to. Should not have
            a file ending like .mp4.
        video_frac : float
            Fraction of videos to subsample from. Defaults to subsample all videos
        fps : float
            How many samples are to be shown each second.
        frames_per_file : int
            How many frames should be extracted per video.
        max_sample_distance : int | None
            Only has an effect if not None. In that case the maximal distance
            between sampled frames will be ``max_sample_distance``.
            E.g. if a file will be sampled with 50 frames that contains 10000 
            datapoints then normally the frames 100, 300, ..., 9900 would be 
            sampled if ``max_sample_distance=None`` and ``random=False``. However,
            if for example ``max_sample_distance=10`` it will only sample the
            central datapoints between 4750 and 5250, e.g. if ``random=False``
            4755, 4765, ... 5245.
        start : int | None
            Start index of the video files to sample from. Usefull to parallelise
            the video creation accross multiple CPUs. Defaults to None, which is
            equivalent to 0.
        end : int | None
            End index of the video files to sample from. Usefull to parallelise
            the video creation accross multiple CPUs. Defaults to None, which is
            equivalent to the last file in the dataset
        random : bool, default=False
            Whether to sample random frames for each file, or to deterministically
            sample midpoints of ``frames_per_file`` many consecutive time blocks. 
            E.g. if a video of 15 frames should be subsampled with 3 frames per 
            video, then each block of 5 consecutive frames (0-4, 5-9, 10-14) 
            are represented by their midpoints: 2, 7 and 12.
        seed : int | None
            The seed for the random subsampling via frames_per_video
        """
        if files_frac < 0 or files_frac > 1:
            raise ValueError(f"parameter 'files_frac' needs to be between "
                             f"0 and 1, but is {files_frac}.")
        
        start = max(start,0) if start is not None else 0
        end = min(end, self._max_files) if end is not None else self._max_files
        
        if seed is not None:
            rng = np.random.default_rng(seed)
        else:
            rng = np.random.default_rng(self._rng.integers(0, 2**32-1))
        
        if files_frac < 1-1e-4:
            path_idxs = np.sort(rng.choice(
                np.arange(start, end, dtype=np.int64), 
                size=int(np.ceil(files_frac*(end-start))),
                replace=False
            ))
        else:
            path_idxs = np.arange(start, end, dtype=np.int64)
        
        pbar = tqdm(total=len(path_idxs)*frames_per_file, 
                    desc=f"Subsampling {len(path_idxs)} videos between {start} and {end}",
                    dynamic_ncols=True)
        k = frames_per_file
        vid_save_path = os.path.abspath(vid_save_path)
        os.makedirs(os.path.dirname(vid_save_path), exist_ok=True)

        self._on_video_start()

        with FFMPEGVideoWriter(
            video_path=f"{vid_save_path}_{len(path_idxs)}files_a{k}_[s{start}_e{end}]",
            fps=fps,
            font_size=12,
        ) as videowriter:
            for idx in path_idxs:
                file_start = 0
                file_end = int(self._samples_per_file[idx])
                file_path = str(self._file_paths[idx])
                meta = self._file_metadatas[idx]
                if max_sample_distance is not None:
                    file_start = max(0,int(np.floor((file_end-max_sample_distance*k)/2)))
                    file_end = min(file_end,int(np.ceil((file_end+max_sample_distance*k)/2)))
                if random:
                    frame_idxs = np.sort(rng.choice(
                        np.arange(file_start, file_end), size=k, replace=False
                    ))
                else:
                    frame_idxs = np.linspace(file_start,file_end-k,2*k+1)[1::2]+np.arange(k)
                    frame_idxs = np.round(frame_idxs).astype(int)

                for frame_idx in frame_idxs:
                    img, title = self.__get_image_representation(
                        file_path=file_path, 
                        sample_idx=frame_idx, 
                        metadata=meta
                    )
                    videowriter.write(img=img, title=title)
                    pbar.update()

        pbar.close()
        self._on_video_end()


    # -------------------- interface for sub-classes ------------------- #

    def _get_samples_metadata_from_filepath(self, path: str):
        """
        Abstract method, must be implemented by subclasses.

        Investigate the file at `path` and return the number of valid samples 
        and optionally a structured `np.ndarray` with additional metadata.
        For example, the video resolution or framerate for videos. The metadata
        needs to contain only basic numpy data types and no arrays - only scalar
        values.
        
        Parameters
        ----------
        file_path : str
            Path of the file that shall be inspected for the number of valid
            datapoints and any additional metadata

        Returns
        -------
        n_samples : int
            Number of valid samples.
        metadata : ndarray | None
            Structured numpy array with shape () but custom datatype containing
            additional metadata for this file. E.g.

            >>> np.array((720,1024,24), dtype=[('height', np.uint32), 
            >>>                                ('width',  np.uint32),
            >>>                                ('fps',    np.float32)])
        
        """
        raise NotImplementedError(
            "Sub-classes must implement _get_samples_metadata_from_filepath(...)"
        )

    def _getitem(
        self, 
        file_path: str,
        sample_idx: int,
        metadata: np.ndarray
    ) -> Any:
        """
        Abstract method, must be implemented by subclasses.

        Load the i-th valid datapoint in the file given by the path, can make
        use of additional metadata for the specific file computed by 
        :py:meth:`_get_samples_metadata_from_filepath`.

        Parameters
        ----------
        file_path : str
            Path of the file that contains the sample.
        sample_idx : int
            Index of the datapoint that shall be collected from that file.
            ``sample_idx = 0`` would request the first datapoint of this file,
            while ``sample_idx = n_samples-1`` would request the last one, where
            ``n_samples`` is the number of datapoints in this file as reported by
            :py:meth:`_get_samples_metadata_from_filepath`.
        metadata : structured ndarray | None
            Companion metadata for the file produced by
            :py:meth:`_get_samples_metadata_from_filepath`.
        """
        raise NotImplementedError
    
    def __get_image_representation(
        self, 
        file_path: Optional[str],
        sample_idx: Optional[int],
        metadata: Optional[np.ndarray], 
        batch = None,
        global_idx: Optional[int] = None,
    ): # -> Tuple[FFMPEG_VIDEO_WRITER_ACCEPTS, str]:
        """
        Helper interface for :py:meth:`_get_image_representation` where not 
        every input needs to be provided.
        
        Parameters
        ----------
        global_idx: int | None
            If the batch is known but not the file-level metadata 
            (file_path, sample_idx, metadata), the global index inside the dataset
            can be provided here so that this method figures them out itself.
        
        """

        if (file_path is None) or (sample_idx is None) or (metadata is None):
            assert global_idx is not None, ("if file level metadata is not provided, the global index must be given")
            key_idx, sample_idx = self._global2local_idx(global_idx)
            file_path = cast(str,self._file_paths[key_idx])
            metadata = self._file_metadatas[key_idx]
            assert metadata is not None # for typing

        if batch is None:
            batch = self._getitem(file_path=file_path, 
                                  sample_idx=sample_idx, 
                                  metadata=metadata,
                                  )
        return self._get_image_representation(
            file_path=file_path, 
            sample_idx=sample_idx, 
            metadata=metadata, 
            batch=batch,
        )
        
    def _get_image_representation(
        self, 
        file_path: str,
        sample_idx: int,
        metadata: np.ndarray, 
        batch,
    ): # -> Tuple[FFMPEG_VIDEO_WRITER_ACCEPTS, str]:
        """
        Abstract method, must be implemented by subclasses.

        Creates an image and additional text to visualise the specified data point
        and document from where it was taken for :py:meth:`create_sample_video`.

        Parameters
        ----------
        file_path : str
            Path of the file that contains the sample.
        sample_idx : int
            Index of the datapoint that shall be collected from that file.
            ``sample_idx = 0`` would request the first datapoint of this file,
            while ``sample_idx = n_samples-1`` would request the last one, where
            ``n_samples`` is the number of datapoints in this file as reported by
            :py:meth:`_get_samples_metadata_from_filepath`.
        metadata : structured ndarray | None
            Companion metadata for the file produced by
            :py:meth:`_get_samples_metadata_from_filepath`.
        batch : 
            If the batch is already known, e.g. the output of 
            _getitem(file_path, sample_idx, metadata).
        
        Returns
        -------
        A tupel with the first element being an image representation:
          - A numpy array of shape (H, W, C) with dtype uint8 of RGB values
          -  or an equivalent byte stream
          - A PIL Image object
          - A matplotlib Figure object
        the second element being a string containing text that shall be printed
        over the image.
        """
        raise NotImplementedError

    def _on_video_start(self):
        """
        Optional hook that is called before the video creation in 
        :py:meth:`create_sample_video` starts.
        Can be used to set up variables for the video creation, such as
        a matplotlib figure and artists to update.
        """
        pass

    def _on_video_end(self):
        """
        Optional hook that is called after the video creation in
        :py:meth:`create_sample_video` ended. Should be used to clean up the
        environment set up in :py:meth:`_on_video_start`.
        """
        pass

# ------------------ convenience for DataLoader workers ----------------

def worker_init_fn(worker_id):
    """
    Re-seed the RNG inside each worker for deterministic subsampling behavior
    in multi-worker PyTorch DataLoaders.

    Should be passed as the ``worker_init_fn`` argument to
    ``torch.utils.data.DataLoader``.

    Note that the ``torch.utils.data.get_worker_info`` method this function is
    based on only works properly, when there are multiple workers for dataloading.

    Parameters
    ----------
    worker_id : int
        ID assigned by PyTorch to the worker process.
    """

    worker_info = get_worker_info()
    if worker_info is not None:
        dataset = worker_info.dataset
        if not isinstance(dataset, MultiFileDataset):
            raise RuntimeError("Custom worker_init_fn is designed to only work"+
                               " with subclasses of MultiFileDataset")
        # worker_info.seed is already derived from base_seed + worker_id
        dataset.set_parameters( seed=worker_info.seed )