import argparse
import os
import shutil
import multiprocessing
import numpy as np
import pyarrow as pa
import pyarrow.dataset as pa_ds
import random
import json

from typing import List, Optional, Tuple
from torch.nn import functional as F
from torch.utils.data import Dataset
from tqdm import tqdm
from datasets import load_dataset, Dataset as DatasetsDataset
from transformers import AutoTokenizer

SCHEMA = pa.schema(
    [
        pa.field("inputs", pa.large_list(pa.int32())),
        pa.field("labels", pa.large_list(pa.int32())),
        pa.field("position_ids", pa.large_list(pa.int32())),
        pa.field("sequence_lengths", pa.large_list(pa.int64())),
    ]
)

DATASET_INFO = r"""{
  "citation": "",
  "description": "",
  "features": {
    "inputs": {
      "feature": {
        "dtype": "int32",
        "_type": "Value"
      },
      "_type": "LargeList"
    },
    "labels": {
      "feature": {
        "dtype": "int32",
        "_type": "Value"
      },
      "_type": "LargeList"
    },
    "position_ids": {
      "feature": {
        "dtype": "int32",
        "_type": "Value"
      },
      "_type": "LargeList"
    },
    "sequence_lengths": {
      "feature": {
        "dtype": "int64",
        "_type": "Value"
      },
      "_type": "LargeList"
    }
  },
  "homepage": "",
  "license": ""
}"""


def process_packing_shard(
    shard, args, tokenizer_pad_id, rank, world_size, epoch=0, rng_state=None
):
    """Pack a shard of the dataset.

    Args:
        epoch: Current epoch number (for logging)
        rng_state: Random state to restore for this shard. If provided, restores random state.
    """
    if rng_state is not None:
        # Restore the random state passed from main process
        random.setstate(rng_state)

    packer = PackedDataset(
        shard,
        max_seq_len=args.pack_to_sequence_length,
        padding_idx=tokenizer_pad_id,
        split_across_pack=not args.chat,
        show_pbar=rank == 0,
        packing_mode=args.packing_mode,
    )

    # Shuffle packs after packing (within this shard)
    random.shuffle(packer.packs)

    example = (
        {
            "inputs": packer.packs[0]["inputs"],
            "labels": packer.packs[0]["labels"],
            "position_ids": packer.packs[0]["position_ids"],
            "sequence_lengths": packer.packs[0]["sequence_lengths"],
        }
        if len(packer.packs) > 0
        else None
    )

    if args.save_to_disk:
        # create a schema that uses int64 for list sizes

        oriented_data = {
            "inputs": [pack["inputs"] for pack in packer.packs],
            "labels": [pack["labels"] for pack in packer.packs],
            "position_ids": [pack["position_ids"] for pack in packer.packs],
            "sequence_lengths": [pack["sequence_lengths"] for pack in packer.packs],
        }
        pa_table = pa.Table.from_pydict(oriented_data, schema=SCHEMA)
        del oriented_data

        pa_ds.write_dataset(
            pa_table,
            os.path.join(args.save_to_disk, str(rank)),
            format="arrow",
        )

        filename = f"data-{rank:05d}-of-{world_size:05d}.arrow"

        shutil.move(
            os.path.join(args.save_to_disk, str(rank), "part-0.arrow"),
            os.path.join(args.save_to_disk, filename),
        )

        os.rmdir(os.path.join(args.save_to_disk, str(rank)))
    else:
        filename = None

    return packer.total_tokens, packer.packed_tokens, packer.dropped, filename, example


# https://github.com/pytorch/torchtune/blob/9d91fe39f08661952da4180b9e7fb2eba5a7a5e7/torchtune/datasets/_packed.py
class PackedDataset(Dataset):
    """
    Performs greedy sample packing on a provided dataset. This is done as a single
    preprocessing step before training begins. Shuffling is done outside of this
    class on packed samples with a ``Sampler`` as part of the dataloader. Currently,
    this only supports in-memory map-style datasets.

    The class loads, tokenizes, and packs examples on initialization - no tokenization is done during training.

    The general flow on initialization is: load tokenized sample -> add to buffer ->
    when buffer is long enough, add to ``self.packs``.

    During training, returns self.packs[idx] as input, label, attention mask, and
    position ids. The attention mask is a lower triangular block mask to prevent
    samples from cross-attending within a pack. The position ids indicate the position
    of each token relative to its sample within a pack. These are all padded to max
    sequence length, so a batch-wise collator is not needed.

    A packed sample is made up of individual smaller sequence length samples jammed together
    within ``max_seq_len``. For example, if max_seq_len is 6 and there are varied
    length samples::

        tokens = [
            [S1, S1, S1, S2, S2, pad],
            [S3, S3, S4, S4, pad, pad],
            ...,
        ]

    To prevent cross-contamination, the following mask would be returned for the
    first pack in the example::

        mask = [
            [1, 0, 0, 0, 0, 0],
            [1, 1, 0, 0, 0, 0],
            [1, 1, 1, 0, 0, 0],
            [0, 0, 0, 1, 0, 0],
            [0, 0, 0, 1, 1, 0],
            [0, 0, 0, 0, 0, 1],
        ]

    The position ids would be::

        input_pos = [
            [0, 1, 2, 0, 1, 2],
            [0, 1, 0, 1, 2, 3],
            ...,
        ]

    The identity matrix is used in the mask for pad tokens instead of a causal mask.
    For position ids for pad tokens, we simply continue to increment from the previous
    sample normally.

    Args:
        ds (Dataset): dataset to sample pack. This should return a dictionary with field
            "tokens" and "labels" containing the tokenized and label samples.
        max_seq_len (int): Maximum number of tokens to pack
        padding_idx (int): padding index for the tokenizer. Default is 0.
        max_packs (Optional[int]): Maximum number of packs. Default is None, which will create as many
            packs as possible.
        split_across_pack (bool): if the last sample in a pack does not fit in ``max_seq_len``,
            split the sample into the next pack, or move it entirely to the beginning of the next pack.
            For pre-training, typically this is set to True for general text completion. For
            fine-tuning, typically this is set to False to avoid truncating sentences in instruct
            tuning. Default is False.
        packing_mode (str): Packing algorithm to use when split_across_pack=False.
            - "random_fit": (default) No sorting, random bin selection. Avoids length-based bias
              where long/short samples get grouped together.
            - "ffd": First-Fit Decreasing. Sorts by length for better packing efficiency but
              introduces correlation between sample length and pack position.
    """

    def __init__(
        self,
        ds: Dataset,
        *,
        max_seq_len: int,
        padding_idx: int = 0,
        max_packs: Optional[int] = None,
        split_across_pack: bool = False,
        group_size: int = 5000,
        show_pbar=True,
        packing_mode: str = "random_fit",
    ) -> None:
        self.ds = ds
        self.max_seq_len = max_seq_len
        self.padding_idx = padding_idx
        self.max_packs = max_packs
        self.split_across_pack = split_across_pack
        self.packs = []
        self.previous_sample_boundary: int = 0
        self.packed_tokens: int = 0
        self.total_tokens: int = 0
        self.dropped: int = 0
        self.show_pbar = show_pbar
        self.group_size = group_size
        if split_across_pack:
            self._pack_greedy()
        elif packing_mode == "ffd":
            self._pack_ffd()
        elif packing_mode == "random_fit":
            self._pack_random_fit()
        else:
            raise ValueError(
                f"Unknown packing_mode: {packing_mode}. Use 'ffd' or 'random_fit'"
            )

    def _get_empty_pack(self):

        return {
            "inputs": np.empty(0, dtype=np.int32),
            "labels": np.empty(0, dtype=np.int32),
            "position_ids": np.empty(0, dtype=np.int32),
            "sequence_lengths": [],
        }

    def _pack_ffd(self) -> None:
        ds_iterator = iter(self.ds)
        finished_iterating = False

        pbar = (
            tqdm(
                total=len(self.ds),
                desc="Packing dataset (FFD)",
                dynamic_ncols=True,
            )
            if self.show_pbar
            else None
        )

        while not finished_iterating:
            # 1. Fetch a large group of samples into memory.
            group = []
            try:
                for _ in range(self.group_size):
                    sample = next(ds_iterator)
                    seq_len = len(sample["inputs"])

                    if seq_len > self.max_seq_len:
                        self.dropped += 1
                        continue
                    # Store sample and its length for sorting
                    group.append({"sample": sample, "seq_len": seq_len})
            except StopIteration:
                finished_iterating = True

            if not group:
                break

            # 2. Sort the group by length in descending order (the "Decreasing" part of FFD).
            group.sort(key=lambda x: x["seq_len"], reverse=True)

            # 3. Pack this group using the "First-Fit" heuristic.
            # Each bin holds the samples it contains and its remaining space.
            bins = []  # List of {"samples": [], "remaining_space": max_seq_len}

            for item in group:
                placed = False
                # Try to place the item in the first available bin.
                for bin in bins:
                    if bin["remaining_space"] >= item["seq_len"]:
                        bin["samples"].append(item["sample"])
                        bin["remaining_space"] -= item["seq_len"]
                        placed = True
                        break

                # If no existing bin could accommodate the item, create a new one.
                if not placed:
                    bins.append(
                        {
                            "samples": [item["sample"]],
                            "remaining_space": self.max_seq_len - item["seq_len"],
                        }
                    )

            # 4. Convert the completed bins from this group into final, padded packs.
            for bin_info in bins:
                if self._should_stop_packing():
                    break

                current_pack = self._get_empty_pack()
                for sample in bin_info["samples"]:
                    tokens = np.array(sample["inputs"], dtype=np.int32)
                    labels = np.array(sample["labels"], dtype=np.int32)
                    seq_len = len(tokens)

                    current_pack["inputs"] = np.concatenate(
                        (current_pack["inputs"], tokens)
                    )
                    current_pack["labels"] = np.concatenate(
                        (current_pack["labels"], labels)
                    )
                    current_pack["position_ids"] = np.concatenate(
                        (
                            current_pack["position_ids"],
                            np.arange(seq_len, dtype=np.int32),
                        )
                    )
                    current_pack["sequence_lengths"].append(seq_len)

                self._add_pack(current_pack)

            if pbar:
                pbar.update(len(group))

            if self._should_stop_packing():
                # Ensure the outer loop breaks if max_packs is reached.
                break

        if pbar:
            # Manually set pbar to total to show 100% at the end
            pbar.n = pbar.total
            pbar.refresh()
            pbar.close()

    def _pack_random_fit(self) -> None:
        """Pack samples without length-based sorting to avoid bias.

        Unlike FFD which sorts by length (introducing correlation between
        sample length and pack position), this method:
        1. Processes samples in their original (pre-shuffled) order
        2. Randomly selects among bins that can fit each sample
        3. Shuffles resulting packs within each group

        Slightly less efficient than FFD but eliminates length-based bias.
        """
        ds_iterator = iter(self.ds)
        finished_iterating = False

        pbar = (
            tqdm(
                total=len(self.ds),
                desc="Packing dataset (random-fit)",
                dynamic_ncols=True,
            )
            if self.show_pbar
            else None
        )

        while not finished_iterating:
            # 1. Fetch a group of samples (no sorting!)
            group = []
            try:
                for _ in range(self.group_size):
                    sample = next(ds_iterator)
                    seq_len = len(sample["inputs"])

                    if seq_len > self.max_seq_len:
                        self.dropped += 1
                        continue
                    group.append({"sample": sample, "seq_len": seq_len})
            except StopIteration:
                finished_iterating = True

            if not group:
                break

            # 2. Pack using random-fit: for each sample, randomly pick a bin that fits
            bins = []  # List of {"samples": [], "remaining_space": max_seq_len}

            for item in group:
                # Find all bins that can fit this sample
                fitting_bins = [
                    (i, bin)
                    for i, bin in enumerate(bins)
                    if bin["remaining_space"] >= item["seq_len"]
                ]

                if fitting_bins:
                    # Randomly select among fitting bins (not first-fit)
                    idx, chosen_bin = random.choice(fitting_bins)
                    chosen_bin["samples"].append(item["sample"])
                    chosen_bin["remaining_space"] -= item["seq_len"]
                else:
                    # No bin fits, create new one
                    bins.append(
                        {
                            "samples": [item["sample"]],
                            "remaining_space": self.max_seq_len - item["seq_len"],
                        }
                    )

            # 3. Shuffle bins within this group to break any remaining order correlation
            random.shuffle(bins)

            # 4. Convert bins to packs
            for bin_info in bins:
                if self._should_stop_packing():
                    break

                current_pack = self._get_empty_pack()
                # Also shuffle samples within each pack
                random.shuffle(bin_info["samples"])

                for sample in bin_info["samples"]:
                    tokens = np.array(sample["inputs"], dtype=np.int32)
                    labels = np.array(sample["labels"], dtype=np.int32)
                    seq_len = len(tokens)

                    current_pack["inputs"] = np.concatenate(
                        (current_pack["inputs"], tokens)
                    )
                    current_pack["labels"] = np.concatenate(
                        (current_pack["labels"], labels)
                    )
                    current_pack["position_ids"] = np.concatenate(
                        (
                            current_pack["position_ids"],
                            np.arange(seq_len, dtype=np.int32),
                        )
                    )
                    current_pack["sequence_lengths"].append(seq_len)

                self._add_pack(current_pack)

            if pbar:
                pbar.update(len(group))

            if self._should_stop_packing():
                break

        if pbar:
            pbar.n = pbar.total
            pbar.refresh()
            pbar.close()

    def _pack_greedy(self) -> None:
        """Iterate through the dataset. Use a buffer to hold samples until max_seq_len,
        then append the buffer to self.packs as a single "packed" sample. Continue
        until max_packs or end of dataset."""

        current_pack = self._get_empty_pack()

        pbar = (
            tqdm(total=len(self.ds), desc="Packing dataset", dynamic_ncols=True)
            if self.show_pbar
            else None
        )

        for sample in self.ds:
            tokens = np.array(sample["inputs"], dtype=np.int32)
            labels = np.array(sample["labels"], dtype=np.int32)

            seq_len = len(tokens)
            if seq_len > self.max_seq_len and not self.split_across_pack:
                # print(
                #     f"Dropping sample that is too long ({seq_len} > {self.max_seq_len})"
                # )
                self.dropped += 1
                continue

            current_pack["inputs"] = np.concatenate((current_pack["inputs"], tokens))
            current_pack["labels"] = np.concatenate((current_pack["labels"], labels))

            position_ids = np.arange(seq_len, dtype=np.int32)
            current_pack["position_ids"] = np.concatenate(
                (current_pack["position_ids"], position_ids)
            )

            current_pack["sequence_lengths"] += [seq_len]

            while (
                len(current_pack["inputs"]) > self.max_seq_len
                and not self._should_stop_packing()
            ):
                current_pack = self._split_and_add_pack(current_pack)

            if pbar:
                pbar.update()

            self.previous_sample_boundary = len(current_pack["inputs"])

            if self._should_stop_packing():
                break

        if len(current_pack["inputs"]) > 0 and (
            self.max_packs is None or len(self.packs) < self.max_packs
        ):
            self._add_pack(current_pack)

    def _should_stop_packing(self) -> bool:
        """If max packs is set, stop packing when we reach that number."""

        if self.max_packs is not None and len(self.packs) == self.max_packs:
            return True
        return False

    def _split_and_add_pack(self, current_pack):
        """Splits the current pack at the boundary, processes it, adds it to ``self.packs`` and
        returns the start of the next pack."""

        if self.split_across_pack:
            boundary = self.max_seq_len
            # The last elem in ``seq_lens`` ensures that ``sum(seq_lens) == self.max_seq_len``
            leftover_seq_len = self.max_seq_len - sum(current_pack["seq_lens"][:-1])
            seq_len_padding = [leftover_seq_len] if leftover_seq_len > 0 else []
        else:
            boundary = self.previous_sample_boundary
            # If we aren't splitting across packs, we leave out the last sample b/c
            # it will go into the next pack
            seq_len_padding = []

        pack = {
            "inputs": current_pack["inputs"][:boundary],
            "labels": current_pack["labels"][:boundary],
            "position_ids": current_pack["position_ids"][:boundary],
            "sequence_lengths": current_pack["sequence_lengths"][:-1] + seq_len_padding,
        }

        self._add_pack(pack)

        # Return the length of the first sample in next pack if we are splitting across packs,
        # otherwise return the length of the last sample in the current pack
        next_seq_len = (
            len(current_pack["inputs"][boundary:])
            if self.split_across_pack
            else current_pack["sequence_lengths"][-1]
        )

        return {
            "inputs": current_pack["inputs"][boundary:],
            "labels": current_pack["labels"][boundary:],
            "position_ids": current_pack["position_ids"][boundary:],
            "sequence_lengths": [next_seq_len],
        }

    def _add_pack(self, pack) -> None:
        """Processes, pads and adds a pack to ``self.packs``."""
        pack = self._pad_pack(pack, padding_idx=self.padding_idx)
        self.packs.append(pack)

    def _pad_pack(self, pack, padding_idx: int):
        """Pads a pack to ``self.max_seq_len``."""
        num_tokens = len(pack["inputs"])
        num_padding_tokens = self.max_seq_len - num_tokens

        self.packed_tokens += num_tokens
        self.total_tokens += self.max_seq_len

        padded_inputs = np.pad(
            pack["inputs"], (0, num_padding_tokens), constant_values=self.padding_idx
        )
        padded_labels = np.pad(
            pack["labels"], (0, num_padding_tokens), constant_values=-100
        )

        if num_padding_tokens > 0:
            # don't care much about padded position_ids, but create them for consistency
            start_pos = int(pack["position_ids"][-1] + 1) if num_tokens > 0 else 0
            pad_positions = np.arange(
                start_pos, start_pos + num_padding_tokens, dtype=np.int32
            )
            padded_position_ids = np.concatenate((pack["position_ids"], pad_positions))
        else:
            padded_position_ids = pack["position_ids"]

        padded_seq_lens = pack["sequence_lengths"]
        if num_padding_tokens > 0:
            padded_seq_lens.append(num_padding_tokens)

        return {
            "inputs": padded_inputs,
            "labels": padded_labels,
            "position_ids": padded_position_ids,
            "sequence_lengths": padded_seq_lens,
        }

    def __len__(self) -> int:
        return len(self.packs)

    def __getitem__(self, idx: int) -> dict[str, np.ndarray]:
        return self.packs[idx]


def main(args):
    # Handle local files vs HuggingFace datasets
    if os.path.exists(args.dataset):
        # Local file - load as JSON/JSONL
        dataset = load_dataset("json", data_files=args.dataset, split="train")
    else:
        # HuggingFace dataset
        dataset = load_dataset(args.dataset, name=args.subset, split=args.split)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)

    def _tokenize(sample):
        # assumes "text" is the column
        inputs = tokenizer.batch_encode_plus(sample["text"]).input_ids
        for x in inputs:
            x.append(tokenizer.eos_token_id)
        return {"inputs": inputs}

    def _tokenize_chat(sample):
        inputs = []
        labels = []

        # Support both ShareGPT ("conversations") and OpenAI ("messages") column names
        if "conversations" in sample:
            conversations_list = sample["conversations"]
        elif "messages" in sample:
            conversations_list = sample["messages"]
        else:
            raise RuntimeError(
                f"Expected 'conversations' or 'messages' column, got: {list(sample.keys())}"
            )

        for conversation in conversations_list:
            for message in conversation:

                keys = list(message.keys())

                if "from" in keys and "value" in keys:
                    # sharegpt format
                    message_from = message.pop("from")
                    if message_from == "gpt":
                        message["role"] = "assistant"
                    elif message_from == "human":
                        message["role"] = "user"
                    else:
                        message["role"] = message_from

                    message["content"] = message.pop("value")
                elif "role" in keys and "content" in keys:
                    pass
                else:
                    raise RuntimeError(f"Unknown chat format, keys are {keys}")

            tokens = tokenizer.apply_chat_template(
                conversation, tokenize=True, add_generation_prompt=False
            )
            label = []

            current_len = 0
            for i in range(len(conversation)):
                if i + 1 == len(conversation):
                    next_tokens = tokenizer.apply_chat_template(
                        conversation, tokenize=True, add_generation_prompt=False
                    )[current_len:]
                else:
                    if "assistant" == conversation[i + 1]["role"]:
                        next_tokens = tokenizer.apply_chat_template(
                            conversation[: i + 1],
                            add_generation_prompt=True,
                            tokenize=True,
                        )[current_len:]
                    else:
                        next_tokens = tokenizer.apply_chat_template(
                            conversation[: i + 1],
                            tokenize=True,
                            add_generation_prompt=False,
                        )[current_len:]

                if conversation[i]["role"] == "assistant":
                    label.extend(next_tokens)
                else:
                    label.extend([-100] * len(next_tokens))

                current_len += len(next_tokens)

            inputs.append(tokens)
            labels.append(label)

        return {
            "inputs": inputs,
            "labels": labels,
        }

    # Note: Don't shuffle here - shuffling happens per-epoch during packing
    # to ensure each epoch has independent ordering
    if args.limit:
        dataset = dataset.select(range(args.limit))
    if args.chat and args.multiturn_only:
        # Support both ShareGPT ("conversations") and OpenAI ("messages") column names
        def _get_conversation_len(x):
            if "conversations" in x:
                return len(x["conversations"])
            elif "messages" in x:
                return len(x["messages"])
            return 0

        dataset = dataset.filter(lambda x: _get_conversation_len(x) > 3)

    original_column_names = list(dataset.features.keys())
    dataset = dataset.map(
        _tokenize_chat if args.chat else _tokenize,
        batched=True,
        batch_size=args.batch_size,
        num_proc=args.num_proc,
    )
    dataset = dataset.remove_columns(original_column_names)

    efficiency = 1.0
    dropped = 0
    if args.pack_to_sequence_length:
        num_shards = 64  # args.num_proc
        num_epochs = args.epochs

        # Seed random ONCE at the beginning - all randomness flows from here
        random.seed(args.seed)
        print(f"Initialized random seed: {args.seed}")

        all_examples = []
        total_tokens = 0
        packed_tokens = 0
        file_counter = 0

        for epoch in range(num_epochs):
            # Generate a unique shuffle seed for this epoch from the main RNG
            epoch_shuffle_seed = random.randint(0, 2**31 - 1)

            # Shuffle dataset for this epoch using generated seed
            epoch_dataset = dataset.shuffle(seed=epoch_shuffle_seed)

            shards = [
                epoch_dataset.shard(num_shards=num_shards, index=i)
                for i in range(num_shards)
            ]

            print(
                f"Packing epoch {epoch + 1}/{num_epochs} (shuffle_seed={epoch_shuffle_seed})..."
            )

            # Generate unique random states for each shard from main RNG
            shard_rng_states = [random.getstate() for _ in range(num_shards)]
            # Advance the RNG for each shard to ensure uniqueness
            for _ in range(num_shards):
                random.random()  # Advance state

            with multiprocessing.Pool(processes=num_shards) as pool:
                process_args = [
                    (
                        shard,
                        args,
                        tokenizer.pad_token_id,
                        index,
                        num_shards,
                        epoch,
                        shard_rng_states[index],
                    )
                    for index, shard in enumerate(shards)
                ]

                results = pool.starmap(process_packing_shard, process_args)

            # Collect results from this epoch
            epoch_filenames = []
            for total, packed, dropped_, filename, example in results:
                if example:
                    all_examples.append(example)
                if filename:
                    epoch_filenames.append(filename)
                total_tokens += total
                packed_tokens += packed
                dropped += dropped_

            # Shuffle filenames within this epoch using main RNG (no re-seeding)
            if args.save_to_disk and epoch_filenames:
                random.shuffle(epoch_filenames)

                # Rename files to global sequential order
                for old_filename in epoch_filenames:
                    old_path = os.path.join(args.save_to_disk, old_filename)
                    new_filename = f"data-{file_counter:05d}-of-TOTAL.arrow"
                    new_path = os.path.join(args.save_to_disk, new_filename)
                    os.rename(old_path, new_path)
                    file_counter += 1

        # Fix the "of-TOTAL" placeholders in filenames
        if args.save_to_disk:
            for i in range(file_counter):
                old_path = os.path.join(
                    args.save_to_disk, f"data-{i:05d}-of-TOTAL.arrow"
                )
                new_filename = f"data-{i:05d}-of-{file_counter:05d}.arrow"
                new_path = os.path.join(args.save_to_disk, new_filename)
                os.rename(old_path, new_path)

        if total_tokens > 0:
            efficiency = packed_tokens / total_tokens

        example = all_examples[0] if all_examples else None

        if args.save_to_disk:
            with open(os.path.join(args.save_to_disk, "dataset_info.json"), "wb") as f:
                f.write(DATASET_INFO.encode())

            # verify we can open and do any conversion needed
            final_dataset = load_dataset(args.save_to_disk, num_proc=args.num_proc)
            print(
                f"Created {len(final_dataset['train'])} packed samples across {num_epochs} epoch(s)"
            )
            if args.packing_mode == "ffd":
                print(
                    f"!!! Warning: FFD packing sorts by length, which may introduce bias. "
                    f"Consider using --packing-mode=random_fit or shuffle before using."
                )

    else:
        # No packing - just shuffle and save tokenized samples
        # Support multiple epochs: each epoch gets a different shuffle, then concatenated
        num_epochs = args.epochs

        # Seed random ONCE at the beginning - all randomness flows from here
        random.seed(args.seed)
        print(f"Initialized random seed: {args.seed}")

        if num_epochs == 1:
            dataset = dataset.shuffle(seed=args.seed)
        else:
            from datasets import concatenate_datasets

            epoch_datasets = []
            for epoch in range(num_epochs):
                # Generate a unique shuffle seed for this epoch from the main RNG
                epoch_shuffle_seed = random.randint(0, 2**31 - 1)
                print(
                    f"Shuffling epoch {epoch + 1}/{num_epochs} (shuffle_seed={epoch_shuffle_seed})..."
                )
                epoch_dataset = dataset.shuffle(seed=epoch_shuffle_seed)
                epoch_datasets.append(epoch_dataset)

            dataset = concatenate_datasets(epoch_datasets)
            print(f"Concatenated {num_epochs} epochs: {len(dataset)} total samples")

        if args.pad_to_and_drop_larger_than:
            max_seq_len = args.pad_to_and_drop_larger_than
            len_before = len(dataset)
            dataset = dataset.filter(lambda x: len(x["inputs"]) <= max_seq_len)
            dropped = len_before - len(dataset)

            # Pad samples and add position_ids/sequence_lengths
            def _pad_and_add_metadata(sample):
                seq_len = len(sample["inputs"])
                num_padding = max_seq_len - seq_len

                # Pad inputs with pad token
                padded_inputs = (
                    sample["inputs"] + [tokenizer.pad_token_id] * num_padding
                )

                # Pad labels with -100 (ignore index)
                if "labels" in sample:
                    padded_labels = sample["labels"] + [-100] * num_padding
                else:
                    # If no labels, create them from inputs and pad
                    padded_labels = sample["inputs"] + [-100] * num_padding

                # Position IDs: continue incrementing for padding
                position_ids = list(range(max_seq_len))

                # Sequence lengths: actual sequence + padding length
                sequence_lengths = [seq_len]
                if num_padding > 0:
                    sequence_lengths.append(num_padding)

                return {
                    "inputs": padded_inputs,
                    "labels": padded_labels,
                    "position_ids": position_ids,
                    "sequence_lengths": sequence_lengths,
                }

            dataset = dataset.map(
                _pad_and_add_metadata,
                num_proc=args.num_proc,
            )
        else:
            # Add position_ids and sequence_lengths for consistency with packed format
            def _add_position_ids_and_seq_lengths(sample):
                seq_len = len(sample["inputs"])
                return {
                    "position_ids": list(range(seq_len)),
                    "sequence_lengths": [seq_len],
                }

            dataset = dataset.map(
                _add_position_ids_and_seq_lengths,
                num_proc=args.num_proc,
            )

        if args.save_to_disk:
            print(f"Saving to {args.save_to_disk}")
            dataset.save_to_disk(args.save_to_disk)
        if args.push_to_hub:
            print(f"Pushing to Hugging Face repo {args.push_to_hub}")
            dataset.push_to_hub(args.save_to_disk, private=True)

        example = dataset[0]

    if args.show_example and example is not None:
        inputs = example["inputs"]
        labels = example["labels"] if "labels" in example else None
        position_ids = example["position_ids"] if "position_ids" in example else None

        example_out = ""
        for i in range(0, len(inputs)):
            token = inputs[i]
            label = labels[i] if labels is not None else token
            position_id = position_ids[i] if position_ids is not None else None

            decoded = tokenizer.decode([token])

            if label == -100:
                example_out += f"\033[31m{decoded}\033[0m({token}"
            else:
                example_out += f"\033[32m{decoded}\033[0m({token}"

            if position_id != None:
                example_out += f"@{position_id})"
            else:
                example_out += ")"

        print(example_out)

    if dropped > 0:
        print(f"Dropped {dropped} too-long samples")
    print(f"Efficiency: {efficiency * 100:.1f}%")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--subset", type=str)
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--tokenizer", type=str, required=True)
    parser.add_argument("--batch-size", type=int, default=1000)
    parser.add_argument("--num-proc", type=int)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--chat", action="store_true")
    parser.add_argument("--multiturn-only", action="store_true")
    parser.add_argument("--pack-to-sequence-length", type=int)
    parser.add_argument(
        "--epochs",
        type=int,
        default=1,
        help="Number of epochs to pack. Each epoch shuffles and packs independently, "
        "so the same sample appears in different packs across epochs.",
    )
    parser.add_argument(
        "--packing-mode",
        type=str,
        default="random_fit",
        choices=["ffd", "random_fit"],
        help="Packing algorithm: 'ffd' (First-Fit Decreasing, sorted by length) or "
        "'random_fit' (no sorting, random bin selection - default, avoids length bias)",
    )
    parser.add_argument("--pad-to-and-drop-larger-than", type=int)
    parser.add_argument("--save-to-disk", type=str)
    parser.add_argument("--push-to-hub", type=str)
    parser.add_argument("--show-example", action="store_true")
    args = parser.parse_args()

    main(args)
