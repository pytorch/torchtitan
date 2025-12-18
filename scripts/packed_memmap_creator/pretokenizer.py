"""
Multiprocessed pretokenization pipeline for creating memory-mapped training data.

Creates three memory-mapped files:
- _index.npy: uint64 array of last token indexes for each sequence
- _token.npy: int64 array of concatenated tokens
- _mask.npy: int64 array of concatenated masks (-100 for user turns)
"""

import multiprocessing as mp
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any, Callable, List, Optional, Tuple

import numpy as np
from tqdm import tqdm


class Pretokenizer:
    """
    Multiprocessed pretokenizer that creates memory-mapped numpy arrays.

    The output format consists of three files:
    - {output_prefix}_index.npy: Array of uint64 indexes pointing to the last token of each sequence
    - {output_prefix}_token.npy: Concatenated tokens from all sequences
    - {output_prefix}_mask.npy: Concatenated masks (-100 over user turns, token IDs otherwise)

    Example:
        >>> pretokenizer = Pretokenizer(output_prefix="data/train", num_workers=8)
        >>>
        >>> # Process data (this would come from your data loader)
        >>> for batch in your_data_loader:
        >>>     pretokenizer.add_batch(batch)
        >>>
        >>> pretokenizer.finalize()

    Args:
        output_prefix: Prefix for output files (e.g., "data/train" -> data/train_token.npy)
        num_workers: Number of worker processes for multiprocessing
        temp_dir: Directory for temporary files (default: system temp)
        chunk_size: Number of sequences to accumulate before writing to disk
    """

    def __init__(
        self,
        output_prefix: str,
        num_workers: int = 8,
        temp_dir: Optional[str] = None,
        chunk_size: int = 10000,
    ):
        self.output_prefix = output_prefix
        self.num_workers = num_workers
        self.chunk_size = chunk_size

        # Create output directory if needed
        output_path = Path(output_prefix).parent
        output_path.mkdir(parents=True, exist_ok=True)

        # Setup temp directory for intermediate results
        if temp_dir is None:
            self.temp_dir = tempfile.mkdtemp(prefix="pretokenizer_")
        else:
            self.temp_dir = temp_dir
            Path(self.temp_dir).mkdir(parents=True, exist_ok=True)

        # Track total counts
        self.total_sequences = 0
        self.total_tokens = 0

        # Buffers for accumulating data before writing
        self.index_buffer: List[int] = []
        self.token_buffer: List[int] = []
        self.mask_buffer: List[int] = []

        # File counters for temporary chunks
        self.chunk_counter = 0
        self.temp_files: List[Tuple[str, str, str]] = []  # (index, token, mask) paths

        print("🚀 Pretokenizer initialized:")
        print(f"   Output prefix: {output_prefix}")
        print(f"   Workers: {num_workers}")
        print(f"   Temp dir: {self.temp_dir}")

    def add_sequences(
        self,
        sequences: List[Tuple[np.ndarray, np.ndarray]],
    ):
        """
        Add a batch of sequences to process.

        Args:
            sequences: List of (tokens, masks) tuples where:
                - tokens: np.ndarray of token IDs (int64)
                - masks: np.ndarray of masks (int64, -100 for user turns)

        Example:
            >>> sequences = [
            >>>     (np.array([1, 2, 3, 4]), np.array([-100, -100, 1, 2])),  # First sequence
            >>>     (np.array([5, 6, 7]), np.array([-100, 5, 6])),           # Second sequence
            >>> ]
            >>> pretokenizer.add_sequences(sequences)
        """
        for tokens, masks in sequences:
            # Validate inputs
            assert len(tokens) == len(
                masks
            ), f"Token and mask lengths must match: {len(tokens)} != {len(masks)}"
            assert (
                tokens.dtype == np.int64 or tokens.dtype == np.int32
            ), f"Tokens must be int64/int32, got {tokens.dtype}"
            assert (
                masks.dtype == np.int64 or masks.dtype == np.int32
            ), f"Masks must be int64/int32, got {masks.dtype}"

            # Add to buffers
            self.token_buffer.extend(tokens.tolist())
            self.mask_buffer.extend(masks.tolist())

            # Calculate the index of the last token (cumulative position)
            last_token_idx = self.total_tokens + len(tokens) - 1
            self.index_buffer.append(last_token_idx)

            self.total_tokens += len(tokens)
            self.total_sequences += 1

        # Write chunk if buffer is large enough
        if self.total_sequences % self.chunk_size == 0:
            self._write_chunk()

    def _write_chunk(self):
        """Write current buffers to temporary chunk files."""
        if not self.index_buffer:
            return

        # Create chunk files
        chunk_prefix = os.path.join(self.temp_dir, f"chunk_{self.chunk_counter:06d}")
        index_path = f"{chunk_prefix}_index.npy"
        token_path = f"{chunk_prefix}_token.npy"
        mask_path = f"{chunk_prefix}_mask.npy"

        # Save as numpy arrays
        np.save(index_path, np.array(self.index_buffer, dtype=np.uint64))
        np.save(token_path, np.array(self.token_buffer, dtype=np.int64))
        np.save(mask_path, np.array(self.mask_buffer, dtype=np.int64))

        self.temp_files.append((index_path, token_path, mask_path))
        self.chunk_counter += 1

        # Clear buffers
        self.index_buffer.clear()
        self.token_buffer.clear()
        self.mask_buffer.clear()

        print(
            f"   💾 Wrote chunk {self.chunk_counter} ({self.total_sequences:,} sequences, {self.total_tokens:,} tokens)"
        )

    def finalize(self):
        """
        Finalize the pretokenization by merging all chunks into final memory-mapped files.
        """
        print("\n✨ Finalizing pretokenization...")

        # Write any remaining buffered data
        self._write_chunk()

        if not self.temp_files:
            raise ValueError("No data was added to the pretokenizer!")

        print(f"   Merging {len(self.temp_files)} chunks...")

        # Final output paths
        index_output = f"{self.output_prefix}_index.npy"
        token_output = f"{self.output_prefix}_token.npy"
        mask_output = f"{self.output_prefix}_mask.npy"

        # Create memory-mapped arrays
        print("   Creating memory-mapped arrays...")
        print(f"      - {self.total_sequences:,} sequences")
        print(f"      - {self.total_tokens:,} total tokens")

        index_mmap = np.lib.format.open_memmap(
            index_output, mode="w+", dtype=np.uint64, shape=(self.total_sequences,)
        )
        token_mmap = np.lib.format.open_memmap(
            token_output, mode="w+", dtype=np.int64, shape=(self.total_tokens,)
        )
        mask_mmap = np.lib.format.open_memmap(
            mask_output, mode="w+", dtype=np.int64, shape=(self.total_tokens,)
        )

        # Merge chunks
        index_offset = 0
        token_offset = 0

        for idx_path, tok_path, mask_path in tqdm(
            self.temp_files, desc="Merging chunks"
        ):
            # Load chunk
            chunk_index = np.load(idx_path)
            chunk_tokens = np.load(tok_path)
            chunk_masks = np.load(mask_path)

            # Write to memory-mapped arrays
            index_mmap[index_offset : index_offset + len(chunk_index)] = chunk_index
            token_mmap[token_offset : token_offset + len(chunk_tokens)] = chunk_tokens
            mask_mmap[token_offset : token_offset + len(chunk_masks)] = chunk_masks

            index_offset += len(chunk_index)
            token_offset += len(chunk_tokens)

        # Flush to disk
        del index_mmap
        del token_mmap
        del mask_mmap

        # Clean up temp files
        print("   🧹 Cleaning up temp files...")
        shutil.rmtree(self.temp_dir)

        print("\n🎉 Pretokenization complete!")
        print("   Output files:")
        print(f"      - {index_output}")
        print(f"      - {token_output}")
        print(f"      - {mask_output}")
        print("   Stats:")
        print(f"      - {self.total_sequences:,} sequences")
        print(f"      - {self.total_tokens:,} tokens")
        print(
            f"      - {self.total_tokens / self.total_sequences:.1f} avg tokens/sequence"
        )


class MultiprocessPretokenizer(Pretokenizer):
    """
    Multiprocessed version of Pretokenizer that processes data in parallel.

    This is useful when you need to apply expensive preprocessing to your data
    before adding it to the pretokenizer.

    Example:
        >>> def process_conversation(conv_data):
        >>>     # Your expensive preprocessing here
        >>>     tokens = tokenize(conv_data['text'])
        >>>     masks = create_masks(conv_data['turns'])
        >>>     return (tokens, masks)
        >>>
        >>> pretokenizer = MultiprocessPretokenizer(
        >>>     output_prefix="data/train",
        >>>     num_workers=16,
        >>>     process_fn=process_conversation
        >>> )
        >>>
        >>> # Process in parallel
        >>> pretokenizer.process_data(raw_conversations)
        >>> pretokenizer.finalize()

    Args:
        output_prefix: Prefix for output files
        process_fn: Function that takes raw data and returns (tokens, masks) tuple
        num_workers: Number of worker processes
        temp_dir: Directory for temporary files
        chunk_size: Number of sequences before writing chunk
        batch_size: Number of items to process per worker batch
    """

    def __init__(
        self,
        output_prefix: str,
        process_fn: Callable[[Any], Tuple[np.ndarray, np.ndarray]],
        num_workers: int = 8,
        temp_dir: Optional[str] = None,
        chunk_size: int = 10000,
        batch_size: int = 100,
    ):
        super().__init__(output_prefix, num_workers, temp_dir, chunk_size)
        self.process_fn = process_fn
        self.batch_size = batch_size

    def process_data(self, data: List[Any], show_progress: bool = True):
        """
        Process raw data in parallel and add to pretokenizer.

        Args:
            data: List of raw data items to process
            show_progress: Whether to show progress bar
        """
        print(f"🔄 Processing {len(data):,} items with {self.num_workers} workers...")

        # Create batches
        batches = [
            data[i : i + self.batch_size] for i in range(0, len(data), self.batch_size)
        ]

        # Process in parallel
        with mp.Pool(self.num_workers) as pool:
            if show_progress:
                results = list(
                    tqdm(
                        pool.imap(self._process_batch, batches),
                        total=len(batches),
                        desc="Processing batches",
                    )
                )
            else:
                results = pool.map(self._process_batch, batches)

        # Add all results
        for batch_results in results:
            self.add_sequences(batch_results)

    def _process_batch(self, batch: List[Any]) -> List[Tuple[np.ndarray, np.ndarray]]:
        """Process a batch of items using the process function."""
        results = []
        for item in batch:
            try:
                tokens, masks = self.process_fn(item)
                results.append((tokens, masks))
            except Exception as e:
                print(f"⚠️  Error processing item: {e}")
                continue
        return results


def load_pretokenized_data(prefix: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load pretokenized data from memory-mapped files.

    Args:
        prefix: The output prefix used during pretokenization

    Returns:
        (index, tokens, masks) tuple of memory-mapped arrays

    Example:
        >>> index, tokens, masks = load_pretokenized_data("data/train")
        >>>
        >>> # Get the first sequence
        >>> start_idx = 0 if i == 0 else index[i-1] + 1
        >>> end_idx = index[0] + 1
        >>> first_seq_tokens = tokens[start_idx:end_idx]
        >>> first_seq_masks = masks[start_idx:end_idx]
    """
    index_path = f"{prefix}_index.npy"
    token_path = f"{prefix}_token.npy"
    mask_path = f"{prefix}_mask.npy"

    # Load as memory-mapped arrays
    index = np.load(index_path, mmap_mode="r")
    tokens = np.load(token_path, mmap_mode="r")
    masks = np.load(mask_path, mmap_mode="r")

    return index, tokens, masks


def get_sequence(
    index: np.ndarray, tokens: np.ndarray, masks: np.ndarray, sequence_idx: int
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract a specific sequence from pretokenized data.

    Args:
        index: Index array (from load_pretokenized_data)
        tokens: Token array (from load_pretokenized_data)
        masks: Mask array (from load_pretokenized_data)
        sequence_idx: Which sequence to extract

    Returns:
        (tokens, masks) for the requested sequence

    Example:
        >>> index, tokens, masks = load_pretokenized_data("data/train")
        >>> seq_tokens, seq_masks = get_sequence(index, tokens, masks, 0)
    """
    # Calculate start and end positions
    if sequence_idx == 0:
        start_idx = 0
    else:
        start_idx = index[sequence_idx - 1] + 1

    end_idx = index[sequence_idx] + 1

    return tokens[start_idx:end_idx], masks[start_idx:end_idx]


if __name__ == "__main__":
    # Example usage
    print("Creating example pretokenized data...\n")

    # Create some example data
    example_sequences = [
        # Sequence 1: User says "hello", assistant responds "hi there"
        # (tokens are fake IDs, masks are -100 for user turn)
        (
            np.array([1, 2, 3, 4, 5], dtype=np.int64),
            np.array([-100, -100, 3, 4, 5], dtype=np.int64),
        ),
        # Sequence 2: User asks question, assistant answers
        (
            np.array([10, 11, 12, 13, 14, 15, 16], dtype=np.int64),
            np.array([-100, -100, -100, 13, 14, 15, 16], dtype=np.int64),
        ),
        # Sequence 3: Short exchange
        (
            np.array([20, 21, 22], dtype=np.int64),
            np.array([-100, 21, 22], dtype=np.int64),
        ),
    ]

    # Create pretokenizer
    pretokenizer = Pretokenizer(
        output_prefix="test_output/example",
        num_workers=4,
        chunk_size=2,  # Small chunk size for demo
    )

    # Add sequences
    pretokenizer.add_sequences(example_sequences)

    # Finalize
    pretokenizer.finalize()

    # Load and verify
    print("\n📖 Loading and verifying data...")
    index, tokens, masks = load_pretokenized_data("test_output/example")

    print("\nLoaded data:")
    print(f"   Index shape: {index.shape}")
    print(f"   Tokens shape: {tokens.shape}")
    print(f"   Masks shape: {masks.shape}")

    print(f"\nIndex array: {index}")
    print(f"Tokens array: {tokens}")
    print(f"Masks array: {masks}")

    print("\n🔍 Extracting individual sequences:")
    for i in range(len(index)):
        seq_tokens, seq_masks = get_sequence(index, tokens, masks, i)
        print(f"\nSequence {i}:")
        print(f"   Tokens: {seq_tokens}")
        print(f"   Masks:  {seq_masks}")
