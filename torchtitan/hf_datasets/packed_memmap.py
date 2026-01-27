# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import os
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
from torch.distributed.checkpoint.stateful import Stateful
from torch.utils.data import IterableDataset

from torchtitan.components.dataloader import ParallelAwareDataloader
from torchtitan.config import JobConfig
from torchtitan.tools.logging import logger


def pack_sequences_knapsack(
    sequences: List[Tuple[np.ndarray, np.ndarray]], seq_len: int
) -> Tuple[List[int], int]:
    """
    Use 0/1 knapsack DP to find optimal packing of sequences.

    Args:
        sequences: List of (tokens, masks) tuples
        seq_len: Maximum sequence length (capacity)

    Returns:
        indices: List of indices of sequences to pack (in original order)
        total_len: Total length of packed sequences
    """
    n = len(sequences)
    if n == 0:
        return [], 0

    # Get lengths (weights and values are the same - we want to maximize total length)
    lengths = [len(tokens) for tokens, _ in sequences]

    # DP table: dp[i][w] = max total length using first i items with capacity w
    # We only need current and previous row
    prev = [0] * (seq_len + 1)
    curr = [0] * (seq_len + 1)

    # Track which items to include for backtracking
    # keep[i][w] = True if item i is included for capacity w
    keep = [[False] * (seq_len + 1) for _ in range(n)]

    for i in range(n):
        item_len = lengths[i]
        for w in range(seq_len + 1):
            # Option 1: Don't take item i
            curr[w] = prev[w]
            keep[i][w] = False

            # Option 2: Take item i (if it fits)
            if item_len <= w:
                take_value = prev[w - item_len] + item_len
                if take_value > curr[w]:
                    curr[w] = take_value
                    keep[i][w] = True

        # Swap rows
        prev, curr = curr, prev

        # Early exit if we've achieved >= 95% efficiency
        if prev[seq_len] >= 0.95 * seq_len:
            break

    # Backtrack to find which items were selected
    selected = []
    w = seq_len
    for i in range(n - 1, -1, -1):
        if keep[i][w]:
            selected.append(i)
            w -= lengths[i]

    selected.reverse()
    total_len = prev[seq_len]

    return selected, total_len


# Hardcoded dataset paths
DATASETS = {
    "default": {
        "pretrain_mixin_paths": [
            "/home/dakota/github/image-posttraining/outputs/",
        ],
        "chat_data_paths": [
            "/home/shared/datasets/HoneyData15M_hermes4_14b/",
            "/home/shared/datasets/text-to-image-2M_hermes4_14b/",
            "/home/shared/datasets/OpenGPT-4o-Image_hermes4_14b/",
            "/home/shared/datasets/OmniEditFiltered1-2M_hermes4_14b/",
            "/home/dakota/github/image-posttraining/chat_outputs/",
        ],
    },
    "gen_only": {
        "pretrain_mixin_paths": [],
        "chat_data_paths": [
            "/home/shared/datasets/text-to-image-2M_hermes4_14b/",
            "/home/shared/datasets/OpenGPT-4o-Image_hermes4_14b/",
            "/home/shared/datasets/OmniEditFiltered1-2M_hermes4_14b/",
        ],
    },
    "multimodal_only": {
        "pretrain_mixin_paths": [],
        "chat_data_paths": [
            "/home/shared/datasets/HoneyData15M_hermes4_14b/",
            "/home/shared/datasets/text-to-image-2M_hermes4_14b/",
            "/home/shared/datasets/OpenGPT-4o-Image_hermes4_14b/",
            "/home/shared/datasets/OmniEditFiltered1-2M_hermes4_14b/",
        ],
    },
    "chat_only": {
        "pretrain_mixin_paths": [],
        "chat_data_paths": ["/home/dakota/github/torchtitan/tokenized/"],
    },
}


# Special tokens
EOS_TOKEN_ID = 151643  # <|endoftext|>


def discover_memmap_files(base_paths: List[str]) -> List[str]:
    """
    Walk through directories to find all _index.npy files.
    Returns list of prefixes (without _token/_mask/_index.npy suffix).
    """
    prefixes = []
    for base_path in base_paths:
        if not os.path.exists(base_path):
            logger.warning(f"Path does not exist: {base_path}")
            continue

        for root, dirs, files in os.walk(base_path):
            for file in files:
                if file.endswith("_index.npy"):
                    # Get the prefix by removing _index.npy
                    prefix = os.path.join(root, file[:-10])
                    prefixes.append(prefix)

    logger.info(f"Discovered {len(prefixes)} dataset files")
    return prefixes


class PackedMemmapDataset(IterableDataset, Stateful):
    """
    Dataset that loads and packs pre-tokenized sequences from memory-mapped .npy files.

    Handles two types of data:
    - Pretrain mixin: Can split sequences anywhere
    - Chat data: Packs sequences without splitting them, using a buffer

    Args:
        dataset_name: Name of the dataset to load (default: "default")
        seq_len: Maximum sequence length
        dp_rank: Data parallel rank
        dp_world_size: Data parallel world size
        infinite: Whether to loop the dataset infinitely
        shuffle: Whether to shuffle sequences
        seed: Random seed for shuffling
        pack_buffer_size: Number of sequences to buffer for packing (chat data only)
    """

    def __init__(
        self,
        dataset_name: str = "default",
        seq_len: int = 2048,
        dp_rank: int = 0,
        dp_world_size: int = 1,
        infinite: bool = False,
        shuffle: bool = True,
        seed: int = 42,
        pack_buffer_size: int = 32,
    ) -> None:
        self.seq_len = seq_len
        self.dp_rank = dp_rank
        self.dp_world_size = dp_world_size
        self.infinite = infinite
        self.shuffle = shuffle
        self.seed = seed
        self.pack_buffer_size = pack_buffer_size
        # Check to see if dataset name needs to be set to default
        if dataset_name not in list(DATASETS.keys()):
            logger.warning(
                f"Dataset name {dataset_name} not found in DATASETS, using default"
            )
            dataset_name = "default"

        self.dataset_name = dataset_name

        # Discover all dataset files
        pretrain_mixins = DATASETS[dataset_name]["pretrain_mixin_paths"]
        chat_data = DATASETS[dataset_name]["chat_data_paths"]
        self.pretrain_prefixes = discover_memmap_files(pretrain_mixins)
        self.chat_prefixes = discover_memmap_files(chat_data)

        # Copy files to /scratch for faster access
        logger.info(f"Rank {self.dp_rank}: Copying dataset files to /scratch...")
        self.pretrain_prefixes = self._copy_to_scratch(self.pretrain_prefixes)
        self.chat_prefixes = self._copy_to_scratch(self.chat_prefixes)
        logger.info(f"Rank {self.dp_rank}: Finished copying to /scratch")

        # Load all the memory-mapped files
        self._load_files()

        # Build global index mapping
        self._build_global_index()

        # Create shuffled order on rank 0 and broadcast
        self._create_and_broadcast_shuffle()

        # Calculate total sequences and split across DP ranks
        self._setup_data_split()

        # Checkpointing state
        self._current_iteration_seq = 0

    def _copy_to_scratch(self, prefixes: List[str]) -> List[str]:
        """
        Copy dataset files to /scratch for faster local access.
        One rank per node does the copying, others wait.
        Returns new prefixes pointing to /scratch.
        """
        import subprocess
        import time
        from concurrent.futures import as_completed, ThreadPoolExecutor

        if len(prefixes) == 0:
            logger.warning("No dataset prefixes provided, skipping copy...")
            return prefixes

        # Use local /scratch (it's per-node already)
        scratch_base = Path(f"/scratch/packed_memmapped_datasets/{self.dataset_name}")
        scratch_base.mkdir(parents=True, exist_ok=True)

        # Use a lock file so only one rank per node copies
        lock_file = scratch_base / ".copy_lock"
        is_leader = False

        # Try to create lock file (atomic operation)
        # If it fails, assume another rank is copying (or stale from failed run)
        try:
            lock_file.mkdir(exist_ok=False)
            is_leader = True
            logger.info(f"Rank {self.dp_rank}: Leader, copying files...")
        except FileExistsError:
            # Lock exists - either another rank is copying, or it's stale
            # Wait a bit, then if files exist, assume copy is done
            # If lock still exists after timeout, it's probably stale - proceed anyway
            logger.info(f"Rank {self.dp_rank}: Waiting for leader to finish copying...")
            max_wait = 1200  # 10 minutes max
            waited = 0
            while lock_file.exists() and waited < max_wait:
                time.sleep(1)
                waited += 1

            if lock_file.exists():
                logger.warning(
                    f"Rank {self.dp_rank}: Lock timeout - proceeding anyway (lock may be stale)"
                )

        def copy_file(src, dst):
            """Copy a single file with rsync."""
            if not Path(dst).exists():
                subprocess.run(["rsync", "-a", src, dst], check=True)
                return f"Copied {Path(src).name}"
            return f"Skipped {Path(src).name} (exists)"

        if is_leader:
            # Build list of all files to copy
            copy_tasks = []
            for i, prefix in enumerate(prefixes):
                scratch_prefix = scratch_base / f"data_{i}"

                token_file = f"{prefix}_token.npy"
                mask_file = f"{prefix}_mask.npy"
                index_file = f"{prefix}_index.npy"

                scratch_token = f"{scratch_prefix}_token.npy"
                scratch_mask = f"{scratch_prefix}_mask.npy"
                scratch_index = f"{scratch_prefix}_index.npy"

                copy_tasks.append((token_file, scratch_token))
                copy_tasks.append((mask_file, scratch_mask))
                copy_tasks.append((index_file, scratch_index))

            # Copy all files in parallel
            logger.info(
                f"Rank {self.dp_rank}: Copying {len(copy_tasks)} files in parallel..."
            )
            with ThreadPoolExecutor(max_workers=16) as executor:
                futures = [
                    executor.submit(copy_file, src, dst) for src, dst in copy_tasks
                ]
                for future in as_completed(futures):
                    result = future.result()
                    logger.info(f"Rank {self.dp_rank}: {result}")

        # Build new prefixes with simple incrementing
        new_prefixes = []
        for i in range(len(prefixes)):
            scratch_prefix = scratch_base / f"data_{i}"

            if not is_leader:
                # Non-leader waits for files to appear
                scratch_token = f"{scratch_prefix}_token.npy"
                scratch_mask = f"{scratch_prefix}_mask.npy"
                scratch_index = f"{scratch_prefix}_index.npy"

                while not (
                    Path(scratch_token).exists()
                    and Path(scratch_mask).exists()
                    and Path(scratch_index).exists()
                ):
                    time.sleep(0.5)

            new_prefixes.append(str(scratch_prefix))

        # Leader releases lock
        if is_leader:
            lock_file.rmdir()
            logger.info(f"Rank {self.dp_rank}: Finished copying")

        return new_prefixes

    def _load_files(self):
        """Load all memory-mapped numpy files."""
        self.tokens_list = []
        self.masks_list = []
        self.indices_list = []
        self.is_pretrain = []  # Track which files are pretrain vs chat
        self.num_sequences_per_file = []

        # Load pretrain files
        for prefix in self.pretrain_prefixes:
            self._load_single_file(prefix, is_pretrain=True)

        # Load chat files
        for prefix in self.chat_prefixes:
            self._load_single_file(prefix, is_pretrain=False)

        self.total_sequences = sum(self.num_sequences_per_file)
        logger.info(
            f"Total dataset: {len(self.pretrain_prefixes)} pretrain files, "
            f"{len(self.chat_prefixes)} chat files, "
            f"{self.total_sequences} sequences"
        )

    def _load_single_file(self, prefix: str, is_pretrain: bool):
        """Load a single set of memmap files."""
        token_file = f"{prefix}_token.npy"
        mask_file = f"{prefix}_mask.npy"
        index_file = f"{prefix}_index.npy"

        # Verify files exist
        for f in [token_file, mask_file, index_file]:
            if not Path(f).exists():
                logger.warning(f"Required file not found: {f}, skipping...")
                return

        # Memory-map the files (don't load into RAM!)
        tokens = np.load(token_file, mmap_mode="r")
        masks = np.load(mask_file, mmap_mode="r")
        indices = np.load(index_file, mmap_mode="r")

        # Verify shapes match
        if tokens.shape != masks.shape:
            logger.warning(
                f"Token and mask shapes don't match for {prefix}: "
                f"{tokens.shape} vs {masks.shape}, skipping..."
            )
            return

        self.tokens_list.append(tokens)
        self.masks_list.append(masks)
        self.indices_list.append(indices)
        self.is_pretrain.append(is_pretrain)
        self.num_sequences_per_file.append(len(indices))

        data_type = "pretrain" if is_pretrain else "chat"
        logger.info(
            f"Loaded {prefix} ({data_type}): {len(indices)} sequences, "
            f"{len(tokens)} tokens total"
        )

    def _build_global_index(self):
        """
        Build a mapping from global sequence index to (file_idx, seq_idx_in_file).
        This allows us to shuffle across all files.
        """
        self.global_index = []
        for file_idx, num_seqs in enumerate(self.num_sequences_per_file):
            for seq_idx in range(num_seqs):
                self.global_index.append((file_idx, seq_idx))

        # Convert to numpy array for easier manipulation
        self.global_index = np.array(self.global_index, dtype=np.int32)
        logger.info(f"Built global index with {len(self.global_index)} entries")

    def _create_and_broadcast_shuffle(self):
        """Create shuffled order on rank 0, save to file, other ranks load it."""
        shuffle_dir = Path("/home/dakota/tmp")
        shuffle_file = (
            shuffle_dir
            / f"shuffle_order_{self.dataset_name}_seed{self.seed}_{self.total_sequences}.npy"
        )

        if self.shuffle:
            if self.dp_rank == 0:
                # Create directory if it doesn't exist
                shuffle_dir.mkdir(parents=True, exist_ok=True)

                # Check if shuffle file already exists - if so, reuse it
                if shuffle_file.exists():
                    logger.info(f"Reusing existing shuffle file: {shuffle_file}")
                    self.shuffle_order = np.load(shuffle_file)
                    return

                # Create shuffled permutation on rank 0
                rng = np.random.RandomState(self.seed)
                self.shuffle_order = rng.permutation(self.total_sequences).astype(
                    np.int32
                )

                # Write to temp file first, then atomically rename
                # Note: np.save automatically adds .npy extension
                temp_file_base = (
                    shuffle_dir
                    / f".shuffle_order_seed{self.seed}_{self.total_sequences}.tmp"
                )
                np.save(temp_file_base, self.shuffle_order)

                # Give filesystem time to flush to disk
                import time

                time.sleep(30)

                # Now rename the fully-written file
                temp_file_actual = Path(str(temp_file_base) + ".npy")
                temp_file_actual.rename(shuffle_file)
                logger.info(
                    f"Created and saved shuffled order with seed {self.seed} to {shuffle_file}"
                )
            else:
                # Other ranks poll until the file exists
                import time

                max_wait = 180  # seconds
                waited = 0
                while not Path(shuffle_file).exists() and waited < max_wait:
                    time.sleep(0.5)
                    waited += 0.5

                if not Path(shuffle_file).exists():
                    raise RuntimeError(
                        f"Rank {self.dp_rank} waited {max_wait}s but shuffle file not found: {shuffle_file}"
                    )

                # Wait a bit more to ensure file is fully written and flushed
                logger.info(
                    "Found shuffle file, waiting 30s to ensure it's fully written..."
                )
                time.sleep(30)

                # Load from file
                self.shuffle_order = np.load(shuffle_file)
                logger.info(f"Loaded shuffled order from {shuffle_file}")
        else:
            # No shuffling, use sequential order
            self.shuffle_order = np.arange(self.total_sequences, dtype=np.int32)

    def _setup_data_split(self):
        """Split sequences across data parallel ranks."""
        # Calculate which sequences belong to this rank
        seqs_per_rank = self.total_sequences // self.dp_world_size
        remainder = self.total_sequences % self.dp_world_size

        # Distribute remainder across first few ranks
        if self.dp_rank < remainder:
            self.rank_start_seq = self.dp_rank * (seqs_per_rank + 1)
            self.rank_num_seqs = seqs_per_rank + 1
        else:
            self.rank_start_seq = self.dp_rank * seqs_per_rank + remainder
            self.rank_num_seqs = seqs_per_rank

        logger.info(
            f"Rank {self.dp_rank}/{self.dp_world_size}: "
            f"{self.rank_num_seqs} sequences (shuffled indices {self.rank_start_seq} "
            f"to {self.rank_start_seq + self.rank_num_seqs - 1})"
        )

    def _get_raw_sequence(self, global_idx: int) -> Tuple[np.ndarray, np.ndarray, bool]:
        """
        Get a raw sequence by global index.

        Returns:
            tokens: numpy array of tokens
            masks: numpy array of masks
            is_pretrain: whether this is pretrain data
        """
        # Look up which file and sequence index within that file
        file_idx, seq_idx_in_file = self.global_index[global_idx]

        # Get the start and end positions within the file
        indices = self.indices_list[file_idx]
        end_pos = int(indices[seq_idx_in_file])
        start_pos = 0 if seq_idx_in_file == 0 else int(indices[seq_idx_in_file - 1])

        # Extract tokens and masks (still memory-mapped, not loaded into RAM)
        tokens = self.tokens_list[file_idx][start_pos:end_pos]
        masks = self.masks_list[file_idx][start_pos:end_pos]
        is_pretrain = self.is_pretrain[file_idx]

        return tokens, masks, is_pretrain

    def _ensure_eos(
        self, tokens: np.ndarray, masks: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Ensure sequence ends with EOS token."""
        if len(tokens) == 0 or tokens[-1] != EOS_TOKEN_ID:
            tokens = np.append(tokens, EOS_TOKEN_ID)
            masks = np.append(masks, EOS_TOKEN_ID)
        return tokens, masks

    def _pack_sequences(
        self, buffer: List[Tuple[np.ndarray, np.ndarray]]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Pack sequences from buffer into a single sequence of seq_len (greedy approach).

        Returns:
            tokens: tensor of shape (seq_len,)
            labels: tensor of shape (seq_len,)
        """
        packed_tokens = []
        packed_masks = []
        current_len = 0

        for tokens, masks in buffer:
            if current_len + len(tokens) <= self.seq_len:
                packed_tokens.extend(tokens)
                packed_masks.extend(masks)
                current_len += len(tokens)
            else:
                # Can't fit this sequence, stop packing
                break

        # If we couldn't pack anything, just take the first sequence and truncate
        if current_len == 0 and len(buffer) > 0:
            tokens, masks = buffer[0]
            packed_tokens = tokens[-self.seq_len :].tolist()
            packed_masks = masks[-self.seq_len :].tolist()
            current_len = len(packed_tokens)

        # Convert to tensors
        tokens_tensor = torch.tensor(packed_tokens, dtype=torch.long)
        masks_tensor = torch.tensor(packed_masks, dtype=torch.long)

        # Pad if necessary
        if len(tokens_tensor) < self.seq_len:
            pad_len = self.seq_len - len(tokens_tensor)
            tokens_tensor = torch.cat(
                [tokens_tensor, torch.zeros(pad_len, dtype=torch.long)]
            )
            masks_tensor = torch.cat(
                [masks_tensor, torch.full((pad_len,), -100, dtype=torch.long)]
            )

        return tokens_tensor, masks_tensor

    def _pack_sequences_knapsack(
        self, buffer: List[Tuple[np.ndarray, np.ndarray]]
    ) -> Tuple[torch.Tensor, torch.Tensor, List[int]]:
        """
        Pack sequences from buffer using knapsack DP for optimal packing.

        Returns:
            tokens: tensor of shape (seq_len,)
            labels: tensor of shape (seq_len,)
            used_indices: list of buffer indices that were used
        """
        # Use knapsack to find optimal packing
        selected_indices, total_len = pack_sequences_knapsack(buffer, self.seq_len)

        # If knapsack couldn't select anything, just take the first sequence and truncate
        if len(selected_indices) == 0 and len(buffer) > 0:
            tokens, masks = buffer[0]
            packed_tokens = tokens[-self.seq_len :].tolist()
            packed_masks = masks[-self.seq_len :].tolist()
            used_indices = [0]
        else:
            # Pack the selected sequences
            packed_tokens = []
            packed_masks = []
            for idx in selected_indices:
                tokens, masks = buffer[idx]
                packed_tokens.extend(tokens.tolist())
                packed_masks.extend(masks.tolist())
            used_indices = selected_indices

        # Convert to tensors
        tokens_tensor = torch.tensor(packed_tokens, dtype=torch.long)
        masks_tensor = torch.tensor(packed_masks, dtype=torch.long)

        # Pad if necessary
        if len(tokens_tensor) < self.seq_len:
            pad_len = self.seq_len - len(tokens_tensor)
            tokens_tensor = torch.cat(
                [tokens_tensor, torch.zeros(pad_len, dtype=torch.long)]
            )
            masks_tensor = torch.cat(
                [masks_tensor, torch.full((pad_len,), -100, dtype=torch.long)]
            )

        return tokens_tensor, masks_tensor, used_indices

    def __iter__(self):
        """Iterate over sequences assigned to this rank."""
        chat_buffer = []  # Buffer for packing chat sequences
        pretrain_buffer = []  # Buffer for accumulating pretrain tokens
        total_toks_to_grab = self.seq_len + 1
        while True:
            # Get the shuffled global index for this rank
            shuffled_idx = self.rank_start_seq + self._current_iteration_seq
            global_idx = self.shuffle_order[shuffled_idx]

            # Get the raw sequence
            tokens, masks, is_pretrain = self._get_raw_sequence(global_idx)

            if is_pretrain:
                # Pretrain: accumulate tokens until we have >= seq_len
                # Convert to numpy arrays if they're memmap views
                tokens = np.array(tokens, dtype=np.int64)
                masks = np.array(masks, dtype=np.int64)

                # Add to pretrain buffer
                pretrain_buffer.extend(tokens.tolist())
                # For pretrain, masks should just be the tokens (no masking)
                # But keep whatever masks are in the file
                if len(pretrain_buffer) >= self.seq_len:
                    # We have enough tokens, yield a sequence
                    tokens_to_yield = pretrain_buffer[:total_toks_to_grab]
                    pretrain_buffer = pretrain_buffer[total_toks_to_grab:]

                    # Convert to tensor
                    tokens_tensor = torch.tensor(tokens_to_yield, dtype=torch.long)
                    # For pretrain, labels are just shifted tokens
                    masks_tensor = tokens_tensor.clone()

                    # Yield input and labels
                    yield {"input": tokens_tensor[:-1]}, masks_tensor[1:]

            else:
                # Chat: pack sequences
                # Convert to numpy arrays
                tokens = np.array(tokens, dtype=np.int64)
                masks = np.array(masks, dtype=np.int64)

                # Ensure EOS token
                tokens, masks = self._ensure_eos(tokens, masks)

                # Skip sequences that are too long (can't pack them without splitting)
                if len(tokens) > total_toks_to_grab:
                    logger.warning(
                        f"Skipping chat sequence of length {len(tokens)} > seq_len {self.seq_len}"
                    )
                    self._current_iteration_seq += 1
                    continue

                # Add to buffer
                chat_buffer.append((tokens, masks))

                # Check if buffer is full or if we can pack
                if len(chat_buffer) >= self.pack_buffer_size:
                    # Try to pack sequences using knapsack
                    (
                        packed_tokens,
                        packed_masks,
                        used_indices,
                    ) = self._pack_sequences_knapsack(chat_buffer)

                    # Remove used sequences from buffer (keep only non-used indices)
                    used_set = set(used_indices)
                    chat_buffer = [
                        seq for i, seq in enumerate(chat_buffer) if i not in used_set
                    ]

                    # Yield the packed sequence
                    yield {"input": packed_tokens[:-1]}, packed_masks[1:]

            self._current_iteration_seq += 1

            # Check if we've exhausted sequences for this rank
            if self._current_iteration_seq >= self.rank_num_seqs:
                # Flush remaining buffer for chat data
                while len(chat_buffer) > 0:
                    (
                        packed_tokens,
                        packed_masks,
                        used_indices,
                    ) = self._pack_sequences_knapsack(chat_buffer)

                    # Remove used sequences
                    used_set = set(used_indices)
                    chat_buffer = [
                        seq for i, seq in enumerate(chat_buffer) if i not in used_set
                    ]

                    yield {"input": packed_tokens[:-1]}, packed_masks[1:]

                if not self.infinite:
                    logger.info(
                        f"Rank {self.dp_rank} exhausted all {self.rank_num_seqs} sequences"
                    )
                    break
                else:
                    # Loop back to the beginning
                    logger.info(
                        f"Rank {self.dp_rank} re-looping dataset "
                        f"(completed {self._current_iteration_seq} sequences)"
                    )
                    self._current_iteration_seq = 0
                    chat_buffer = []

    def state_dict(self):
        """Save checkpoint state."""
        return {
            "current_iteration_seq": self._current_iteration_seq,
            "seed": self.seed,
        }

    def load_state_dict(self, state_dict):
        """Load checkpoint state."""
        self._current_iteration_seq = state_dict["current_iteration_seq"]
        # Verify seed matches
        if state_dict["seed"] != self.seed:
            logger.warning(
                f"Checkpoint seed {state_dict['seed']} doesn't match "
                f"current seed {self.seed}. Sequence order may differ."
            )


def build_packed_memmap_dataloader(
    dataset_name: str,
    batch_size: int,
    seq_len: int,
    dp_rank: int,
    dp_world_size: int,
    infinite: bool = True,
    shuffle: bool = True,
    seed: int = 42,
    pack_buffer_size: int = 32,
) -> ParallelAwareDataloader:
    """
    Build a dataloader for packed memory-mapped pre-tokenized datasets.

    Args:
        batch_size: Batch size per rank
        seq_len: Maximum sequence length
        dp_rank: Data parallel rank
        dp_world_size: Data parallel world size
        infinite: Whether to loop infinitely
        shuffle: Whether to shuffle sequences
        seed: Random seed for shuffling
        pack_buffer_size: Number of sequences to buffer for packing
    """
    dataset = PackedMemmapDataset(
        dataset_name=dataset_name,
        seq_len=seq_len,
        dp_rank=dp_rank,
        dp_world_size=dp_world_size,
        infinite=infinite,
        shuffle=shuffle,
        seed=seed,
        pack_buffer_size=pack_buffer_size,
    )

    return ParallelAwareDataloader(
        dataset=dataset,
        dp_rank=dp_rank,
        dp_world_size=dp_world_size,
        batch_size=batch_size,
    )


def build_memmap_text_dataloader(
    dp_world_size: int,
    dp_rank: int,
    tokenizer,  # Not used, but kept for interface compatibility
    job_config: JobConfig,
    infinite: bool = True,
) -> ParallelAwareDataloader:
    """
    Drop-in replacement for build_text_dataloader using pre-tokenized memmap files.

    This matches the interface of text_datasets.build_text_dataloader but uses
    pre-tokenized data from memory-mapped .npy files instead of tokenizing on the fly.

    Args:
        dp_world_size: Data parallel world size
        dp_rank: Data parallel rank
        tokenizer: Ignored (data is pre-tokenized)
        job_config: Job configuration with training.local_batch_size and training.seq_len
        infinite: Whether to loop infinitely
    """
    batch_size = job_config.training.local_batch_size
    seq_len = job_config.training.seq_len

    # Get seed from config if available, otherwise use default
    seed = getattr(job_config.training, "seed", None)
    if seed is None:
        seed = 42

    return build_packed_memmap_dataloader(
        dataset_name=job_config.training.dataset,
        batch_size=batch_size,
        seq_len=seq_len,
        dp_rank=dp_rank,
        dp_world_size=dp_world_size,
        infinite=infinite,
        shuffle=True,
        seed=seed,
        pack_buffer_size=32,
    )
