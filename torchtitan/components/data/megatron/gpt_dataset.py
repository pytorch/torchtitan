# Copyright (c) 2026, trainstation team
# The following code is copied from https://github.com/open-lm-engine/lm-engine

# **************************************************
# Copyright (c) 2026, Mayank Mishra
# **************************************************

from __future__ import annotations

import hashlib
import json
import logging
import os
import time
from collections import OrderedDict

import numpy as np
import torch

from torchtitan.components.tokenizer import BaseTokenizer

from .indexed_dataset import MMapIndexedDataset
from .utils import build_sample_idx, log_rank_0, Split


_FIM_PREFIX = "<fim_prefix>"
_FIM_MIDDLE = "<fim_middle>"
_FIM_SUFFIX = "<fim_suffix>"
_FIM_PAD = "<fim_pad>"


class GPTDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        indexed_dataset: MMapIndexedDataset,
        indexed_indices: np.ndarray,
        num_samples: int,
        index_split: Split,
        tokenizer: BaseTokenizer,
        sequence_length: int,
        name: str | None,
        split: str | None,
        path_to_cache: str | None,
        fim_rate: float,
        fim_spm_rate: float,
        caching_allowed: bool,
        random_seed: int,
    ) -> GPTDataset:
        assert indexed_indices.size > 0
        assert num_samples > 0
        assert self.is_multimodal() == indexed_dataset.multimodal

        self.indexed_dataset = indexed_dataset
        self.indexed_indices = indexed_indices
        self.num_samples = num_samples
        self.index_split = index_split
        self.sequence_length = sequence_length
        self.name = name
        self.split = split
        self.path_to_cache = path_to_cache
        self.random_seed = random_seed

        self.caching_allowed = caching_allowed
        # Only FIM uses the tokenizer. Not holding it otherwise keeps the dataset
        # picklable for DataLoader workers started with spawn/forkserver, since
        # torchtitan's HuggingFaceTokenizer is not picklable.
        self.tokenizer = tokenizer if fim_rate != 0 else None
        self.fim_rate = fim_rate
        self.fim_spm_rate = fim_spm_rate
        self.np_rng = np.random.RandomState(self.random_seed)  # rng state for FIM

        self.unique_identifiers = OrderedDict()
        self.unique_identifiers["class"] = type(self).__name__
        self.unique_identifiers["path_prefix"] = self.indexed_dataset.path_prefix
        self.unique_identifiers["num_samples"] = self.num_samples
        self.unique_identifiers["index_split"] = self.index_split.name
        self.unique_identifiers["name"] = self.name
        self.unique_identifiers["split"] = self.split
        self.unique_identifiers["random_seed"] = self.random_seed
        self.unique_identifiers["sequence_length"] = self.sequence_length

        self.unique_description = json.dumps(self.unique_identifiers, indent=4)
        self.unique_description_hash = hashlib.md5(
            self.unique_description.encode("utf-8")
        ).hexdigest()

        if self.fim_rate != 0:
            assert (
                self.fim_rate <= 1 and self.fim_rate >= 0
            ), "FIM rate must be a probability 0 <= rate <= 1"

            (
                self.suffix_tok_id,
                self.prefix_tok_id,
                self.middle_tok_id,
                self.pad_tok_id,
            ) = (
                _fim_token_id(self.tokenizer, tok)
                for tok in [_FIM_SUFFIX, _FIM_PREFIX, _FIM_MIDDLE, _FIM_PAD]
            )

            self.eos_token_id = self.tokenizer.eos_id
            if self.eos_token_id is None:
                raise ValueError(
                    "FIM (fim_rate > 0) needs a tokenizer with an EOS token"
                )

        (
            self.document_index,
            self.sample_index,
            self.shuffle_index,
        ) = self._build_document_sample_shuffle_indices()

    def __len__(self) -> int:
        return self.sample_index.shape[0] - 1

    def __getitem__(self, idx: int) -> dict[str, np.ndarray]:
        # Do the shuffle mapping
        idx = self.shuffle_index[idx]

        # Get the beginning and end documents and offsets
        doc_index_beg, doc_index_beg_offset = self.sample_index[idx]
        doc_index_end, doc_index_end_offset = self.sample_index[idx + 1]

        document_ids = []
        sample_parts = []

        # Sample spans a single document
        if doc_index_beg == doc_index_end:
            # Add the document id
            document_ids.append(self.document_index[doc_index_beg])

            # Add the entire sample
            sample_parts.append(
                self.indexed_dataset.get(
                    self.document_index[doc_index_beg],
                    offset=doc_index_beg_offset,
                    length=doc_index_end_offset - doc_index_beg_offset + 1,
                )
            )

        # Sample spans multiple documents
        else:
            for i in range(doc_index_beg, doc_index_end + 1):
                # Add the document id
                document_ids.append(self.document_index[i])

                # Add the sample part
                offset = 0 if i > doc_index_beg else doc_index_beg_offset
                length = None if i < doc_index_end else doc_index_end_offset + 1
                sample_parts.append(
                    self.indexed_dataset.get(
                        self.document_index[i], offset=offset, length=length
                    )
                )
        sample = np.array(np.concatenate(sample_parts), dtype=np.int64)

        # Code from: https://github.com/EleutherAI/gpt-neox/blob/FIM-clean/megatron/data/gpt2_dataset.py#L109
        # TODO(Hailey): can merge the code below this line with code above this line.
        # TODO(Hailey), cont: above already iterates through loop, so just add the permuting in there?
        # # print(sample, sample.shape)
        # # do FIM here, if enabled
        # TODO: Do we handle the following point from FIM paper?
        # To transform data in the character space for context-level FIM, the tokenized documents have to be decoded back into strings before FIM augmentation. Depending on the vocabulary, some care has to be given to ensure decoding does not introduce any spurious characters into training. For example, utf-8 characters are encoded as multiple tokens with a BPE vocabulary; they can result in fragments from chunking and fail to decode. To prevent unforeseen errors midway through training, we encourage checking for these fragments at the beginning or end of a context and removing them.

        if self.fim_rate != 0:
            sample_len = sample.shape[0]
            segment_breaks = np.argwhere(
                sample == self.eos_token_id
            )  # split sample by document

            if segment_breaks.shape != (
                0,
                1,
            ):  # then there is an EOD token in this example
                curr_start_position = 0
                new_samples = []
                for loc in np.nditer(segment_breaks):
                    # Only permute non-empty segments.
                    if loc - curr_start_position > 0:
                        # permute {prefix, suffix, middle} or {suffix, prefix, middle}
                        permuted, self.np_rng = permute(
                            sample[curr_start_position:loc],
                            self.np_rng,
                            self.fim_rate,
                            self.fim_spm_rate,
                            self.tokenizer,
                            truncate_or_pad=False,
                            suffix_tok_id=self.suffix_tok_id,
                            prefix_tok_id=self.prefix_tok_id,
                            middle_tok_id=self.middle_tok_id,
                            pad_tok_id=self.pad_tok_id,
                        )
                        new_samples += [permuted, [self.eos_token_id]]

                    curr_start_position = loc + 1  # jump over the EOD token
                # Permute the segment after the last EOD
                permuted, self.np_rng = permute(
                    sample[curr_start_position:],
                    self.np_rng,
                    self.fim_rate,
                    self.fim_spm_rate,
                    self.tokenizer,
                    truncate_or_pad=False,
                    suffix_tok_id=self.suffix_tok_id,
                    prefix_tok_id=self.prefix_tok_id,
                    middle_tok_id=self.middle_tok_id,
                    pad_tok_id=self.pad_tok_id,
                )
                new_samples.append(permuted)

                sample = np.concatenate(new_samples)
            else:
                sample, self.np_rng = permute(
                    sample,
                    self.np_rng,
                    self.fim_rate,
                    self.fim_spm_rate,
                    self.tokenizer,
                    truncate_or_pad=False,
                    suffix_tok_id=self.suffix_tok_id,
                    prefix_tok_id=self.prefix_tok_id,
                    middle_tok_id=self.middle_tok_id,
                    pad_tok_id=self.pad_tok_id,
                )

            # Truncate or pad sequence to max-length
            diff = sample.shape[0] - sample_len
            if diff > 0:  # too long
                sample = sample[:sample_len]
            elif diff < 0:  # too short
                sample = np.concatenate([sample, np.full((-1 * diff), self.pad_tok_id)])

            assert sample.shape[0] == sample_len

        document_ids = np.array(document_ids, dtype=np.int64)

        return {"text": sample}

    @staticmethod
    def is_multimodal() -> bool:
        return False

    @staticmethod
    def is_split_by_sequence() -> bool:
        return True

    def _build_document_sample_shuffle_indices(
        self,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Build the document index, the sample index, and the shuffle index

        The document index:
            -- 1-D
            -- An ordered array of document ids

        The sample index:
            -- 2-D
            -- The document indices and offsets which mark the start of every sample

        The shuffle index:
            -- 1-D
            -- A random permutation of index range of the sample index

        Returns:
            Tuple[np.ndarray, np.ndarray]: The document index, the sample index, and the
            shuffle index

        TODO: Explain the 80% threshold
        """

        path_to_cache = self.path_to_cache
        if path_to_cache is None:
            path_to_cache = os.path.join(
                self.indexed_dataset.path_prefix,
                "cache",
                f"{type(self).__name__}_indices",
            )

        def _get_path_to(suffix: str) -> str:
            return os.path.join(
                path_to_cache,
                f"{self.unique_description_hash}-{type(self).__name__}-{suffix}",
            )

        path_to_description = _get_path_to("description.txt")
        path_to_document_index = _get_path_to("document_index.npy")
        path_to_sample_index = _get_path_to("sample_index.npy")
        path_to_shuffle_index = _get_path_to("shuffle_index.npy")
        cache_hit = all(
            map(
                os.path.isfile,
                [
                    path_to_description,
                    path_to_document_index,
                    path_to_sample_index,
                    path_to_shuffle_index,
                ],
            )
        )

        num_tokens_per_epoch = np.sum(
            self.indexed_dataset.sequence_lengths[self.indexed_indices]
        )
        num_epochs = _get_num_epochs(
            num_tokens_per_epoch, self.sequence_length, self.num_samples
        )

        log_rank_0(logging.INFO, f"> Tokens per epoch: {num_tokens_per_epoch}")
        log_rank_0(logging.INFO, f"> Number of epochs: {num_epochs}")

        if not cache_hit and self.caching_allowed:
            log_rank_0(
                logging.INFO,
                f"Build and save the {type(self).__name__} {self.index_split.name} indices",
            )

            if num_epochs == 1:
                separate_final_epoch = False
            else:
                # Get the number of samples for the last epoch
                num_samples_sans_final_epoch = (
                    (num_epochs - 1) * num_tokens_per_epoch - 1
                ) // self.sequence_length
                num_samples_from_final_epoch = (
                    self.num_samples - num_samples_sans_final_epoch
                )
                num_samples_per_epoch = (
                    num_tokens_per_epoch - 1
                ) // self.sequence_length

                # num_samples_from_final_epoch should be non-negative
                assert num_samples_from_final_epoch >= 0

                # num_samples_from_final_epoch should not exceed max value
                assert num_samples_from_final_epoch <= num_samples_per_epoch + 1

                # Separate the final epoch if it falls below the threshold
                threshold = 0.80
                separate_final_epoch = num_samples_from_final_epoch < int(
                    threshold * num_samples_per_epoch
                )

                log_rank_0(
                    logging.DEBUG,
                    f"> num_samples_from_final_epoch: {num_samples_from_final_epoch}",
                )
                log_rank_0(logging.DEBUG, f"> threshold: {threshold}")
                log_rank_0(
                    logging.DEBUG, f"> num_samples_per_epoch: {num_samples_per_epoch}"
                )

            log_rank_0(logging.DEBUG, f"> separate_final_epoch: {separate_final_epoch}")

            numpy_random_state = np.random.RandomState(self.random_seed)

            os.makedirs(path_to_cache, exist_ok=True)

            # Write the description
            with open(path_to_description, "wt") as writer:
                writer.write(self.unique_description)

            # Build the document index
            log_rank_0(
                logging.INFO,
                f"\tBuild and save the document index to {os.path.basename(path_to_document_index)}",
            )
            t_beg = time.time()
            document_index = _build_document_index(
                self.indexed_indices,
                num_epochs,
                numpy_random_state,
                separate_final_epoch,
            )
            np.save(path_to_document_index, document_index, allow_pickle=True)
            t_end = time.time()
            log_rank_0(logging.DEBUG, f"\t> time elapsed: {t_end - t_beg:4f} seconds")

            # Build the sample index
            log_rank_0(
                logging.INFO,
                f"\tBuild and save the sample index to {os.path.basename(path_to_sample_index)}",
            )
            t_beg = time.time()

            assert self.indexed_dataset.sequence_lengths.dtype == np.int32
            sample_index = build_sample_idx(
                self.indexed_dataset.sequence_lengths,
                document_index,
                self.sequence_length,
                num_epochs,
                num_tokens_per_epoch,
            )
            np.save(path_to_sample_index, sample_index, allow_pickle=True)
            t_end = time.time()
            log_rank_0(logging.DEBUG, f"\t> time elapsed: {t_end - t_beg:4f} seconds")

            # Build the shuffle index
            log_rank_0(
                logging.INFO,
                f"\tBuild and save the shuffle index to {os.path.basename(path_to_shuffle_index)}",
            )
            t_beg = time.time()
            if separate_final_epoch:
                shuffle_index = _build_shuffle_index(
                    num_samples_sans_final_epoch,
                    sample_index.shape[0] - 1,
                    numpy_random_state,
                )
            else:
                shuffle_index = _build_shuffle_index(
                    sample_index.shape[0] - 1,
                    sample_index.shape[0] - 1,
                    numpy_random_state,
                )
            np.save(path_to_shuffle_index, shuffle_index, allow_pickle=True)
            t_end = time.time()
            log_rank_0(logging.DEBUG, f"\t> time elapsed: {t_end - t_beg:4f} seconds")

        log_rank_0(
            logging.INFO,
            f"Load the {type(self).__name__} {self.index_split.name} indices",
        )

        log_rank_0(
            logging.INFO,
            f"\tLoad the document index from {os.path.basename(path_to_document_index)}",
        )
        t_beg = time.time()
        document_index = np.load(
            path_to_document_index, allow_pickle=True, mmap_mode="r"
        )
        t_end = time.time()
        log_rank_0(logging.DEBUG, f"\t> time elapsed: {t_end - t_beg:4f} seconds")

        log_rank_0(
            logging.INFO,
            f"\tLoad the sample index from {os.path.basename(path_to_sample_index)}",
        )
        t_beg = time.time()
        sample_index = np.load(path_to_sample_index, allow_pickle=True, mmap_mode="r")
        t_end = time.time()
        log_rank_0(logging.DEBUG, f"\t> time elapsed: {t_end - t_beg:4f} seconds")

        log_rank_0(
            logging.INFO,
            f"\tLoad the shuffle index from {os.path.basename(path_to_shuffle_index)}",
        )
        t_beg = time.time()
        shuffle_index = np.load(path_to_shuffle_index, allow_pickle=True, mmap_mode="r")
        t_end = time.time()
        log_rank_0(logging.DEBUG, f"\t> time elapsed: {t_end - t_beg:4f} seconds")

        log_rank_0(
            logging.INFO, f"> total number of samples: {sample_index.shape[0] - 1}"
        )
        log_rank_0(logging.INFO, f"> total number of epochs: {num_epochs}")

        return document_index, sample_index, shuffle_index


def _get_num_epochs(
    num_tokens_per_epoch: int, seq_length: int, num_samples: int
) -> int:
    """Calculate the number of epochs

    Args:
        num_tokens_per_epoch (int): The number of tokens in a single epoch
        seq_length (int): The sequence length in tokens
        num_samples (int): The total number of samples

    Returns:
        int: The number of epochs
    """
    num_epochs = 0
    num_tokens = 0
    while True:
        num_epochs += 1
        num_tokens += num_tokens_per_epoch
        # -1 is because we need to retrieve seq_length + 1 token each time
        # but the last token will overlap with the first token of the next
        # sample except for the last sample.
        if ((num_tokens - 1) // seq_length) >= num_samples:
            return num_epochs


def _build_document_index(
    documents: np.ndarray,
    num_epochs: int,
    numpy_random_state: np.random.RandomState,
    separate_final_epoch: bool,
) -> np.ndarray:
    """Build an array with length = num epochs * num documents

    Args:
        documents (np.ndarray): the subset of exposed document indices

        num_epochs (int): The number of epochs

        numpy_random_state (np.random.RandomState): The NumPy random state

        separate_final_epoch (bool): Whether to exclude the last epoch from the global shuffle

    Returns:
        np.ndarray: The document index

    TODO: Explain separate_final_epoch
    """

    if not separate_final_epoch or num_epochs == 1:
        document_index = np.mgrid[0:num_epochs, 0 : len(documents)][1]
        document_index[:] = documents
        document_index = document_index.reshape(-1)
        document_index = document_index.astype(documents.dtype)
        numpy_random_state.shuffle(document_index)
        return document_index

    doc_idx_first = _build_document_index(
        documents, num_epochs - 1, numpy_random_state, False
    )
    doc_idx_last = _build_document_index(documents, 1, numpy_random_state, False)
    return np.concatenate((doc_idx_first, doc_idx_last))


def _build_shuffle_index(
    num_samples: int, total_size: int, numpy_random_state: np.random.RandomState
) -> np.ndarray:
    """Build the range [0, size) and shuffle

    Args:
        num_samples (int): The size of the first shuffle range [0, num_samples)

        total_size (int): The size of the entire index. If larger than 'num_samples', it defines

        the second shuffle range [num_samples, total_size)

        numpy_random_state (np.random.RandomState): The NumPy random state

    Returns:
        np.ndarray: The shuffle index

    TODO: Explain [0, num_samples) [num_samples, total_size) split
    """
    dtype_ = np.uint32
    if total_size >= (np.iinfo(np.uint32).max - 1):
        dtype_ = np.int64

    shuffle_idx_first = np.arange(start=0, stop=num_samples, step=1, dtype=dtype_)
    numpy_random_state.shuffle(shuffle_idx_first)
    if num_samples == total_size:
        return shuffle_idx_first

    shuffle_idx_last = np.arange(
        start=num_samples, stop=total_size, step=1, dtype=dtype_
    )
    numpy_random_state.shuffle(shuffle_idx_last)

    return np.concatenate((shuffle_idx_first, shuffle_idx_last))


def _fim_token_id(tokenizer: BaseTokenizer, token: str) -> int:
    """ID of a FIM sentinel token. lm-engine used HF ``convert_tokens_to_ids``;
    torchtitan's ``HuggingFaceTokenizer`` exposes ``token_to_id``."""
    token_to_id = getattr(tokenizer, "token_to_id", None)
    if token_to_id is None:
        raise ValueError(
            f"FIM (fim_rate > 0) needs a tokenizer with token_to_id, got {type(tokenizer).__qualname__}"
        )
    token_id = token_to_id(token)
    if token_id is None:
        raise ValueError(
            f"FIM (fim_rate > 0) needs the {token!r} token in the tokenizer vocabulary"
        )
    return token_id


# From https://github.com/EleutherAI/gpt-neox/blob/FIM-clean/megatron/data/gpt2_dataset.py#L339
def permute(
    sample: np.ndarray,
    np_rng: np.random.RandomState,
    fim_rate: float,
    fim_spm_rate: float,
    tokenizer: BaseTokenizer,
    truncate_or_pad: bool = True,
    suffix_tok_id: int | None = None,
    prefix_tok_id: int | None = None,
    middle_tok_id: int | None = None,
    pad_tok_id: int | None = None,
) -> tuple[np.ndarray, int]:
    """
    Take in a sample (np array w/ size (0,chunklength)) and perform a FIM transformation on it.
    Maintain the same sample length (if transform creates a few extra tokens, drop them).
    """

    if np_rng.binomial(1, fim_rate):  # sample bernoulli dist
        contents = tokenizer.decode(sample.tolist(), skip_special_tokens=False)

        try:
            # A boundary can be =0 (prefix will be empty)
            # a boundary can be =len(contents) (suffix will be empty)
            # The two boundaries can be equal (middle will be empty)
            boundaries = list(np_rng.randint(low=0, high=len(contents) + 1, size=2))
            boundaries.sort()
        except ValueError as e:
            print(len(contents), contents)
            print(e)
            raise e

        prefix = contents[: boundaries[0]]
        middle = contents[boundaries[0] : boundaries[1]]
        suffix = contents[boundaries[1] :]

        prefix = np.array(
            [*tokenizer.encode(prefix, add_bos=False, add_eos=False)], dtype=np.int64
        )
        middle = np.array(
            [*tokenizer.encode(middle, add_bos=False, add_eos=False)], dtype=np.int64
        )
        suffix = np.array(
            [*tokenizer.encode(suffix, add_bos=False, add_eos=False)], dtype=np.int64
        )

        # here we truncate each given segment to fit the same length as it was before
        # A consequence is that we never reach the end of a file?
        # we should rather truncate at the context-level
        if truncate_or_pad:
            # need to make same length as the input. Take the 3 sentinel tokens into account
            new_length = suffix.shape[0] + prefix.shape[0] + middle.shape[0] + 3
            diff = new_length - sample.shape[0]
            if diff > 0:  # too long
                if (
                    suffix.shape[0] <= diff
                ):  # if there's no space to truncate the suffix: stop and report it. atm i should have stopped this from happening
                    return sample, np_rng
                suffix = suffix[: suffix.shape[0] - diff]
            elif diff < 0:  # too short
                suffix = np.concatenate([suffix, np.full((-1 * diff), pad_tok_id)])

        if np_rng.binomial(1, fim_spm_rate):
            # SPM (variant 2 from FIM paper)
            new_sample = np.concatenate(
                [
                    [prefix_tok_id, suffix_tok_id],
                    suffix,
                    [middle_tok_id],
                    prefix,
                    middle,
                ]
            )
        else:
            # PSM
            new_sample = np.concatenate(
                [
                    [prefix_tok_id],
                    prefix,
                    [suffix_tok_id],
                    suffix,
                    [middle_tok_id],
                    middle,
                ]
            )
    else:
        # don't do FIM preproc
        new_sample = sample

    return new_sample, np_rng
