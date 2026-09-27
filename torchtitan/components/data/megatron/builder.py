# Copyright (c) 2026, trainstation team
# The following code is copied from https://github.com/open-lm-engine/lm-engine

# **************************************************
# Copyright (c) 2026, Mayank Mishra
# **************************************************

from __future__ import annotations

import logging
import math
import os

import numpy as np

from torchtitan.components.tokenizer import BaseTokenizer

from . import utils as dist_utils
from .blended_dataset import BlendedDataset
from .concatenated_dataset import ConcatenatedDataset
from .gpt_dataset import GPTDataset
from .indexed_dataset import get_idx_path, MMapIndexedDataset
from .utils import (
    is_multi_storage_client_available,
    log_rank_0,
    MSC_PREFIX,
    normalize,
    parse_and_normalize_split,
    Split,
)


if is_multi_storage_client_available():
    import multistorageclient as msc


def _resolve_idx_path(
    path_prefix: str, node_uses_local_storage: bool, path_to_cache: str
) -> str | None:
    """Download the .idx file for a dataset prefix to local disk if it lives on a multi-storage
    client, so that it can be opened by MMapIndexedDataset

    Args:
        path_prefix (str): The MMapIndexedDataset .bin and .idx file prefix
        node_uses_local_storage (bool): Whether the node caches data on local storage
        path_to_cache (str): Where all reusable dataset indices are to be cached

    Returns:
        str | None: The local .idx path, or None when the prefix already lives on local storage
    """

    if not path_prefix.startswith(MSC_PREFIX):
        return None

    remote_idx_path = get_idx_path(path_prefix)
    idx_path = os.path.join(
        path_to_cache, "cloud-idx-cache", remote_idx_path.removeprefix(MSC_PREFIX)
    )

    log_rank_0(logging.INFO, f"downloading {remote_idx_path} to {idx_path}")

    # lm-engine also downloads on every TP-first rank. In torchtitan every rank
    # reads its own data, so that would be every rank writing the same file;
    # download once globally, or once per node with node-local storage.
    if dist_utils.get_global_rank() == 0 or (
        node_uses_local_storage and dist_utils.get_local_rank() == 0
    ):
        msc.download_file(remote_idx_path, idx_path)

    dist_utils.barrier()

    assert os.path.exists(idx_path)

    return idx_path


def _get_blend_from_list(blend: list[str]) -> tuple[list[str], list[float] | None]:
    """Parse a blend list into dataset prefixes and, optionally, their sampling weights

    Args:
        blend (list[str]): Either a plain list of dataset prefixes, e.g.
        ["path/to/dataset_1_prefix", "path/to/dataset_2_prefix"], in which case the datasets are
        sampled equally, or a flattened sequence of weight-prefix pairs, e.g. ["30",
        "path/to/dataset_1_prefix", "70", "path/to/dataset_2_prefix"]

    Returns:
        tuple[list[str], list[float] | None]: The dataset prefixes, and the (unnormalized)
        weights, or None if the blend contains no weights and the datasets should be sampled
        equally
    """

    # An odd-length blend can't be weight-prefix pairs, so it must be a plain list of prefixes
    if len(blend) % 2 == 1:
        return [prefix.strip() for prefix in blend], None

    raw_weight_per_dataset, raw_prefix_per_dataset = zip(
        *[(blend[i], blend[i + 1]) for i in range(0, len(blend), 2)]
    )

    weight_per_dataset = []
    for weight in raw_weight_per_dataset:
        try:
            weight_per_dataset.append(float(weight))
        except ValueError:
            # not every entry parses as a weight, so treat the whole blend as a plain list of prefixes
            return [prefix.strip() for prefix in blend], None

    return [prefix.strip() for prefix in raw_prefix_per_dataset], weight_per_dataset


def _get_num_samples_per_prefix(
    prefixes: list[str], node_uses_local_storage: bool, path_to_cache: str
) -> list[int]:
    """Get the number of samples available in each dataset prefix, used to weight an unweighted
    blend as if the datasets were simply concatenated

    Args:
        prefixes (list[str]): The MMapIndexedDataset .bin and .idx file prefixes
        node_uses_local_storage (bool): Whether the node caches data on local storage
        path_to_cache (str): Where all reusable dataset indices are to be cached

    Returns:
        list[int]: The number of samples in each dataset prefix
    """

    num_samples = []
    for prefix in prefixes:
        idx_path = _resolve_idx_path(prefix, node_uses_local_storage, path_to_cache)
        num_samples.append(
            len(
                MMapIndexedDataset(
                    prefix, GPTDataset.is_multimodal(), idx_path=idx_path
                )
            )
        )

    return num_samples


def _get_sizes_for_blend(
    weights: list[float], target_num_samples_per_split: list[int]
) -> list[list[int]]:
    """Determine the contribution of the MegatronDataset splits to the BlendedDataset splits

    Args:
        weights (list[float]): The normalized weight for each dataset in the blend e.g. [0.3, 0.7]

        target_num_samples_per_split (list[int]): The number of samples to target for each
        BlendedDataset split

    Returns:
        list[list[int]]: The number of samples to request per MegatronDataset per split
    """

    # Use 0.5% target margin to ensure we satiate the network
    return [
        [
            int(math.ceil(target_num_samples * weight * 1.005))
            for target_num_samples in target_num_samples_per_split
        ]
        for weight in weights
    ]


def _get_appropriate_dtype_for_range(split_idx_bounds: list[int]) -> np.dtype:
    max_value = max(split_idx_bounds)

    if max_value <= np.iinfo(np.int32).max:
        dtype = np.int32
    elif max_value <= np.iinfo(np.int64).max:
        dtype = np.int64
    else:
        raise ValueError("value for split idx is too large")

    return dtype


def build(
    sizes: list[int],
    sequence_length: int,
    tokenizer: BaseTokenizer,
    node_uses_local_storage: bool,
    random_seed: int,
    name: str | None = None,
    blend: list[str] | None = None,
    blend_per_split: list[list[str] | None] | None = None,
    split: str | None = None,
    path_to_cache: str | None = None,
    fim_rate: float = 0,
    fim_spm_rate: float = 0,
) -> list[BlendedDataset | ConcatenatedDataset | GPTDataset | None]:
    if blend_per_split is not None and any(blend_per_split):
        assert blend is None, "blend and blend_per_split are incompatible"
        assert len(blend_per_split) == len(
            Split
        ), f"blend_per_split must contain {len(Split)} blends"
        if split is not None:
            split = None
            log_rank_0(logging.WARNING, f"Let split = {split}")
    elif blend is not None:
        assert split is not None, "both blend and split must be provided"

    dataset_kwargs = dict(
        sequence_length=sequence_length,
        name=name,
        split_str=split,
        path_to_cache=path_to_cache,
        fim_rate=fim_rate,
        fim_spm_rate=fim_spm_rate,
        tokenizer=tokenizer,
        random_seed=random_seed,
    )

    blended_datasets = []

    if blend is not None:
        split_vector = parse_and_normalize_split(split)
        log_rank_0(logging.INFO, f"Let split_vector = {split_vector}")

        prefix_per_dataset, weight_per_dataset = _get_blend_from_list(blend)

        # Blend consists of a single prefix
        if len(prefix_per_dataset) == 1:
            return _build_megatron_dataset_splits(
                prefix_per_dataset[0],
                split_vector,
                sizes,
                node_uses_local_storage=node_uses_local_storage,
                **dataset_kwargs,
            )

        # Blend consists of multiple prefixes; when no weights were provided, the datasets are
        # concatenated (each contributing samples proportional to its own size) rather than
        # weighted and randomly interleaved
        is_concatenated = weight_per_dataset is None
        if is_concatenated:
            weight_per_dataset = _get_num_samples_per_prefix(
                prefix_per_dataset, node_uses_local_storage, path_to_cache
            )

        weight_per_dataset = normalize(weight_per_dataset)
        sizes_per_dataset = _get_sizes_for_blend(weight_per_dataset, sizes)

        # Sum over all contributing datasets, per split
        size_per_split = list(map(sum, zip(*sizes_per_dataset)))
        megatron_datasets = [[] for _ in range(len(Split))]

        for i in range(len(prefix_per_dataset)):
            megatron_datasets_split = _build_megatron_dataset_splits(
                prefix_per_dataset[i],
                split_vector,
                sizes_per_dataset[i],
                node_uses_local_storage=node_uses_local_storage,
                **dataset_kwargs,
            )

            for j in range(len(megatron_datasets_split)):
                megatron_datasets[j].append(megatron_datasets_split[j])

        for i in range(len(megatron_datasets)):
            is_none = map(lambda _: _ is None, megatron_datasets[i])

            if split_vector[i] == 0.0 or size_per_split[i] == 0:
                assert all(is_none)
                blended_datasets.append(None)
            elif is_concatenated:
                assert all(is_none) or not any(is_none)
                blended_datasets.append(
                    _build_generic_dataset(
                        ConcatenatedDataset,
                        node_uses_local_storage=node_uses_local_storage,
                        datasets=megatron_datasets[i],
                    )
                )
            else:
                assert all(is_none) or not any(is_none)
                blended_datasets.append(
                    _build_generic_dataset(
                        BlendedDataset,
                        node_uses_local_storage=node_uses_local_storage,
                        datasets=megatron_datasets[i],
                        weights=weight_per_dataset,
                        size=size_per_split[i],
                        path_to_cache=path_to_cache,
                    )
                )
    else:
        for i in range(len(Split)):
            split_blend = blend_per_split[i]

            # Blend is not provided
            if not split_blend:
                blended_datasets.append(None)
                continue

            split_spoof = [0.0] * len(Split)
            split_spoof[i] = 1.0
            sizes_spoof = [0] * len(Split)
            sizes_spoof[i] = sizes[i]

            prefix_per_dataset, weight_per_dataset = _get_blend_from_list(split_blend)

            # Blend consists of a single prefix
            if len(prefix_per_dataset) == 1:
                blended_datasets.append(
                    _build_megatron_dataset_splits(
                        prefix_per_dataset[0],
                        split_spoof,
                        sizes_spoof,
                        node_uses_local_storage=node_uses_local_storage,
                        **dataset_kwargs,
                    )[i]
                )

            # Blend consists of multiple prefixes; when no weights were provided, the datasets are
            # concatenated (each contributing samples proportional to its own size) rather than
            # weighted and randomly interleaved
            else:
                is_concatenated = weight_per_dataset is None
                if is_concatenated:
                    weight_per_dataset = _get_num_samples_per_prefix(
                        prefix_per_dataset, node_uses_local_storage, path_to_cache
                    )

                weight_per_dataset = normalize(weight_per_dataset)
                sizes_per_dataset = _get_sizes_for_blend(
                    weight_per_dataset, sizes_spoof
                )

                size_per_split = list(map(sum, zip(*sizes_per_dataset)))
                megatron_datasets = []

                for j in range(len(prefix_per_dataset)):
                    megatron_datasets.append(
                        _build_megatron_dataset_splits(
                            prefix_per_dataset[j],
                            split_spoof,
                            sizes_per_dataset[j],
                            node_uses_local_storage=node_uses_local_storage,
                            **dataset_kwargs,
                        )[i]
                    )

                if is_concatenated:
                    blended_datasets.append(
                        _build_generic_dataset(
                            ConcatenatedDataset,
                            node_uses_local_storage=node_uses_local_storage,
                            datasets=megatron_datasets,
                        )
                    )
                else:
                    blended_datasets.append(
                        _build_generic_dataset(
                            BlendedDataset,
                            node_uses_local_storage=node_uses_local_storage,
                            datasets=megatron_datasets,
                            weights=weight_per_dataset,
                            size=size_per_split[i],
                            path_to_cache=path_to_cache,
                        )
                    )

    return blended_datasets


def _build_megatron_dataset_splits(
    path_prefix: str,
    split: list[float],
    sizes: list[int],
    node_uses_local_storage: bool,
    sequence_length: int,
    name: str | None,
    split_str: str | None,
    path_to_cache: str | None,
    fim_rate: float,
    fim_spm_rate: float,
    tokenizer: BaseTokenizer,
    random_seed: int,
) -> list[GPTDataset | None]:
    """Build each MegatronDataset split from a single MMapIndexedDataset

    Args:
        path_prefix (str): The MMapIndexedDataset .bin and .idx file prefix
        split (list[float]): The dataset split ratios (must sum to 1.00)
        sizes (list[int]): The number of total samples to draw from each split

    Returns:
        list[GPTDataset | None]: The GPTDataset (or None) per split
    """

    idx_path = _resolve_idx_path(path_prefix, node_uses_local_storage, path_to_cache)

    # lm-engine opens the dataset on TP-first ranks only and broadcasts batches to
    # the other TP ranks; in torchtitan every rank reads its own data.
    indexed_dataset = MMapIndexedDataset(
        path_prefix, GPTDataset.is_multimodal(), idx_path=idx_path
    )

    if GPTDataset.is_split_by_sequence():
        split_idx_bounds = _get_split_indices(
            split, indexed_dataset.sequence_lengths.shape[0]
        )
    else:
        split_idx_bounds = _get_split_indices(
            split, indexed_dataset.document_indices.shape[0] - 1
        )

    split_indices = [
        np.arange(
            start=split_idx_bounds[i],
            stop=split_idx_bounds[i + 1],
            step=1,
            dtype=_get_appropriate_dtype_for_range(split_idx_bounds),
        )
        for i, _ in enumerate(Split)
    ]

    megatron_datasets = []
    for i, _split in enumerate(Split):
        megatron_datasets.append(
            # A split with no requested samples is skipped like a 0-ratio split:
            # each torchtitan loader builds only the split it iterates.
            None
            if split[i] == 0.0 or sizes[i] == 0
            else _build_generic_dataset(
                GPTDataset,
                node_uses_local_storage=node_uses_local_storage,
                indexed_dataset=indexed_dataset,
                indexed_indices=split_indices[i],
                num_samples=sizes[i],
                index_split=_split,
                tokenizer=tokenizer,
                sequence_length=sequence_length,
                name=name,
                split=split_str,
                path_to_cache=path_to_cache,
                fim_rate=fim_rate,
                fim_spm_rate=fim_spm_rate,
                random_seed=random_seed,
            )
        )

    return megatron_datasets


def _build_generic_dataset(
    cls: type[BlendedDataset | ConcatenatedDataset | GPTDataset | MMapIndexedDataset],
    node_uses_local_storage: bool,
    **kwargs,
) -> BlendedDataset | ConcatenatedDataset | GPTDataset | MMapIndexedDataset | None:
    if dist_utils.is_distributed_initialized():
        caching_allowed = dist_utils.get_global_rank() == 0 or (
            node_uses_local_storage and dist_utils.get_local_rank() == 0
        )

        dataset = None

        # First, build on rank 0
        if caching_allowed:
            try:
                dataset = cls(**kwargs, caching_allowed=True)
            except OSError as err:
                log = (
                    "Failed to write dataset materials to the data cache directory. "
                    + "Please supply a directory to which you have write access via "
                    + "the path_to_cache argument and retry. Refer to the preserved "
                    + "traceback above for more information."
                )
                raise Exception(log) from err

        dist_utils.barrier()

        # After, build on other ranks
        if not caching_allowed:
            dataset = cls(**kwargs, caching_allowed=False)

        return dataset

    return cls(**kwargs, caching_allowed=True)


def _get_split_indices(split: list[float], num_elements: int) -> list[int]:
    """Determine the document index bounds per split

    Args:
        split (list[float]): The dataset split ratios (must sum to 1.00)
        num_elements (int): The number of elements, e.g. sequences or documents, available for
        the split

    Returns:
        list[int]: The indices for all three splits e.g. [0, 900, 990, 1000] for a 1000-document
        set and a [90.0, 9.0, 1.0] split
    """
    split_indices = [0]
    for split_pct in split:
        split_indices.append(
            split_indices[-1] + int(round(split_pct * float(num_elements)))
        )
    split_indices[1:] = list(
        map(lambda _: _ - (split_indices[-1] - num_elements), split_indices[1:])
    )

    assert len(split_indices) == len(split) + 1
    assert split_indices[-1] == num_elements

    return split_indices
