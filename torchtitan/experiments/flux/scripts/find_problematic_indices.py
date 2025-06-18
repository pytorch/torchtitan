# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import multiprocessing as mp
from multiprocessing import Pool
from tqdm import tqdm
from datasets import load_dataset
import argparse

def check_sample_batch(args):
    """Check a batch of samples for corruption."""
    dataset, start_idx, end_idx = args
    problems = []
    for i in range(start_idx, end_idx):
        try:
            _ = dataset[i]
        except (UnicodeDecodeError, SyntaxError, OSError, ZeroDivisionError, ValueError, TypeError):
            problems.append(i)
    return problems

def validate_dataset_multiprocessing(dataset, num_workers=None, batch_size=100):
    """
    Validate dataset using multiprocessing.
    
    Args:
        dataset: HuggingFace dataset to validate
        num_workers: Number of worker processes (default: CPU count)
        batch_size: Number of samples per batch for each worker
    
    Returns:
        List of problematic indices
    """
    if num_workers is None:
        num_workers = mp.cpu_count()
    
    dataset_len = len(dataset)
    print(f"Validating dataset with {dataset_len} samples using {num_workers} workers...")
    
    # Create batches for workers
    batches = []
    for start_idx in range(0, dataset_len, batch_size):
        end_idx = min(start_idx + batch_size, dataset_len)
        batches.append((dataset, start_idx, end_idx))
    
    problems = []
    
    # Use multiprocessing to check batches
    with Pool(num_workers) as pool:
        # Use tqdm to show progress
        results = list(tqdm(
            pool.imap(check_sample_batch, batches),
            total=len(batches),
            desc="Validating batches"
        ))
    
    # Flatten results
    for batch_problems in results:
        problems.extend(batch_problems)
    
    return sorted(problems)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Find problematic samples in a dataset')
    parser.add_argument('--num_workers', type=int, default=16,
                      help='Number of worker processes (default: 16)')
    parser.add_argument('--batch_size', type=int, default=1000,
                      help='Number of samples per batch (default: 1000)')
    args = parser.parse_args()

    dataset = load_dataset("pixparse/cc12m-wds", split="train", num_proc=args.num_workers)
    problems = validate_dataset_multiprocessing(
        dataset, 
        num_workers=args.num_workers,
        batch_size=args.batch_size
    )
    
    print(f"Found {len(problems)} problematic samples:")
    
    # Save the list of problematic indices to a file
    output_file = "problematic_indices.txt"
    with open(output_file, "w") as f:
        for idx in problems:
            f.write(f"{idx}\n")
    print(f"\nSaved {len(problems)} problematic indices to {output_file}")


